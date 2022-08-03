import click
from dataloader import CLASS_LABELS, ORIGINAL_IMAGE
from omegaconf import OmegaConf

def set_env(gpus: str = '-1'):
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=gpus
    os.environ["TF_GPU_THREAD_MODE"]="gpu_private"
    import tensorflow as tf
    print(f'CUDA AVAILABLE: {tf.test.is_gpu_available(cuda_only=True)}')
    print(f'TENSORFLOW VERSION: {tf.__version__}')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_devices = tf.config.experimental.list_logical_devices('GPU')
            print(f'PHYSICAL GPUs: {len(gpus)}')
            print(f'LOGICAL GPUs: {len(logical_devices)}')
        except RuntimeError as e:
            print(f'ERROR: {e}') # Memory growth must be set before GPUs have been initialized
    else: 
        print('WARNING: no GPU found!')

@click.group()
@click.pass_context
def main(ctx):
    """
    Training and evaluation
    """
    print("Mode:", ctx.invoked_subcommand)

@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "--cuda", 
    is_flag=True,
    help="Enable CUDA if available"
)
def train(config_path: str, cuda: bool = True):
    """
    Training ViT-PCM
    """
    CONFIG = OmegaConf.load(config_path)
    GPUs = CONFIG.EXP.NUM_GPUS if cuda else '-1'
    set_env(gpus=GPUs)
    import os
    import tensorflow as tf
    tf.random.set_seed(1234)
    from utils import create_folders
    from tensorflow.keras.optimizers import Adam
    from dataloader import DatasetVITPCMDistributed
    from tensorflow.summary import create_file_writer
    from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
    from model import get_model, ConfusionMatrixCallback
    GPUS = ["GPU:"+str(i) for i in range(len(GPUs.split(',')))]
    strategy = tf.distribute.MirroredStrategy(GPUS)
    REPLICAS = strategy.num_replicas_in_sync
    logdir, filepath = create_folders(CONFIG)
    writer = create_file_writer(logdir)
    writer.set_as_default()
    callbacks = TensorBoard(log_dir=logdir, histogram_freq=15, write_graph=False,
        write_images=False, update_freq='batch', profile_batch=(10),
        embeddings_freq=15, embeddings_metadata=None)
    checkpoint = ModelCheckpoint(filepath=filepath, save_weights_only=True,
                                 monitor='val_mean_iou', mode='max',
                                 save_best_only=True, verbose=1)
    confmatrix = ConfusionMatrixCallback(basedir=logdir)
    earlystop = EarlyStopping(monitor='val_mean_iou',patience=60,
        verbose=1,mode='max',restore_best_weights=True)

    with strategy.scope():
        model = get_model(CONFIG)
        model.warmup()
        model.compile(Adam(learning_rate=CONFIG.SOLVER.LR_WARMUP, name='adam'))
        train_set = DatasetVITPCMDistributed(CONFIG, 'train', REPLICAS).get_dataset()
        val_set = DatasetVITPCMDistributed(CONFIG, 'val', REPLICAS).get_dataset()
    _ = model.fit(train_set, epochs=CONFIG.SOLVER.EPOCHS_WARMUP, 
        validation_data=val_set,validation_freq=1, max_queue_size=1e10,
        callbacks=[callbacks, checkpoint, confmatrix, earlystop],
        use_multiprocessing=False, workers=CONFIG.DATALOADER.NUM_WORKERS)
    with strategy.scope():
        model.unfreeze()
        model.compile(Adam(learning_rate=CONFIG.SOLVER.LR, name='adam'))
        # train_set = DatasetVITPCMDistributed(config_path, 'train', REPLICAS)
    _ = model.fit(train_set, initial_epoch=CONFIG.SOLVER.EPOCHS_WARMUP, epochs=CONFIG.SOLVER.EPOCHS,
        validation_data=val_set,validation_freq=1, max_queue_size=1e10,
        callbacks=[callbacks, checkpoint, confmatrix, earlystop],
        use_multiprocessing=False, workers=CONFIG.DATALOADER.NUM_WORKERS)
    model.evaluate(val_set, batch_size=CONFIG.SOLVER.BATCH_SIZE.VAL*REPLICAS, callbacks=[callbacks])


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "-m",
    "--model-path",
    type=click.STRING,
    required=True,
    help="Model weights path",
)
@click.option(
    "--cuda", 
    is_flag=True,
    help="Enable CUDA if available"
)
def eval(config_path, model_path, cuda):
    """
    Fast distributed evaluation of ViT-PCM
    """
    CONFIG = OmegaConf.load(config_path)
    GPUs = CONFIG.EXP.NUM_GPUS if cuda else '-1'
    set_env(gpus=GPUs)
    import tensorflow as tf
    tf.random.set_seed(1234)
    from model import get_model
    from tensorflow.keras.optimizers import Adam
    from dataloader import DatasetVITPCMDistributed
    GPUS = ["GPU:"+str(i) for i in range(len(GPUs.split(',')))]
    strategy = tf.distribute.MirroredStrategy(GPUS)
    REPLICAS = strategy.num_replicas_in_sync
    CONFIG.MODEL.INIT_MODEL = model_path    
    with strategy.scope():
        model = get_model(CONFIG)
        model.compile(Adam(learning_rate=CONFIG.SOLVER.LR_WARMUP, name='adam'))
        model.trainable = False
        model.show()
        val_set = DatasetVITPCMDistributed(CONFIG, 'val', REPLICAS).get_dataset()
    results = model.evaluate(val_set, batch_size=CONFIG.SOLVER.BATCH_SIZE.VAL*REPLICAS, use_multiprocessing=True, workers=CONFIG.DATALOADER.NUM_WORKERS, return_dict=True)
    print(f'[INFO]: Evaluation \n{results}')
    print(f'[INFO]: mIoU \n{model.metrics[-1].summary()}')


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "-m",
    "--model-path",
    type=click.STRING,
    required=True,
    help="Model weights path",
)
@click.option(
    "--crf", 
    is_flag=True,
    help="Run CRF refinement after inference"
)
@click.option(
    "--cuda", 
    is_flag=True,
    help="Enable CUDA if available"
)
def test(config_path, model_path, crf, cuda):
    """
    Infer and evaluate (if seg labels are present) ViT-PCM on any specified set
    """
    CONFIG = OmegaConf.load(config_path)
    from os import makedirs
    from os.path import join, isfile
    out_dir = CONFIG.EXP.OUTPUT_DIR
    makedirs(out_dir,exist_ok=True)
    out_dir = join(out_dir,CONFIG.EXP.ID)
    makedirs(out_dir,exist_ok=True)
    out_dir = join(out_dir,CONFIG.EXP.RESULTS_DIR)
    makedirs(out_dir,exist_ok=True)
    out_dir = join(out_dir,CONFIG.DATASET.SPLIT.TEST)
    for p in [out_dir,join(out_dir,'preds'),join(out_dir,'segs')]: makedirs(p,exist_ok=True) 
    GPUs = CONFIG.EXP.NUM_GPUS if cuda else '-1'
    set_env(gpus=GPUs)
    import tensorflow as tf
    tf.random.set_seed(1234)
    from model import get_model, MeanIoUWrapper
    CONFIG.MODEL.INIT_MODEL = model_path
    model = get_model(CONFIG)
    model.trainable = False
    model.show()
    from dataloader import DatasetVITPCMDistributed, IMAGE, HEIGHT, WIDTH, IMAGE_NAME, CLASS_LABELS, ORIGINAL_LABEL
    data_set = DatasetVITPCMDistributed(CONFIG, 'test', 1).get_dataset()
    from utils import resize, create_pascal_label_colormap, preds2dict, dict2preds, ignore_labels, tensor
    import numpy as np
    from PIL import Image
    import json
    from tqdm import tqdm
    palette = create_pascal_label_colormap()
    miou = MeanIoUWrapper(CONFIG.DATASET.NUM_CLASSES)
    print('[INFO]: Running inference..')
    print(f'[INFO]: Output directory {out_dir}')
    for data in (bar := tqdm(iterable=iter(data_set),total=len(data_set))):
        image_name = str(data.get(IMAGE_NAME).numpy()[0].decode('UTF-8'))
        preds = model.infer(data.get(IMAGE), data.get(CLASS_LABELS) if CLASS_LABELS in data else None) # class labels is None on test set
        np.save(join(out_dir,'preds',image_name+'.npy'), preds2dict(preds.numpy()[0],data.get(CLASS_LABELS).numpy()[0] if CLASS_LABELS in data else None), allow_pickle=True)
        seg_preds = tf.argmax(resize(preds, (data.get(HEIGHT)[0], data.get(WIDTH)[0])), axis=-1)
        pil_seg = Image.fromarray(seg_preds.numpy().astype(np.uint8)[0])
        pil_seg.putpalette(palette)
        pil_seg.save(join(out_dir,'segs',image_name+'.png'))
        if ORIGINAL_LABEL in data:
            miou.update_state(ignore_labels(data.get(ORIGINAL_LABEL), ignore_value=CONFIG.DATASET.IGNORE_LABEL),\
                ignore_labels(seg_preds, data.get(ORIGINAL_LABEL), ignore_value=CONFIG.DATASET.IGNORE_LABEL))
            bar.set_description(f'[INFO]: mIoU {round(float(miou.result().numpy()),4)}')
    if np.any(miou.total_cm.numpy()):
        np.save(join(out_dir,'miou_matrix.npy'),miou.total_cm.numpy())
        with open(join(out_dir,'miou_summary.txt'), "w") as f:
            json.dump(miou.summary(), f, indent=4, sort_keys=True) 

    if crf:
        import joblib
        import warnings
        # suppress joblib warning (to be understood)
        warnings.filterwarnings("ignore")
        print('[INFO]: Running CRF post-processing..')
        from crf import DenseCRF
        makedirs(join(out_dir,'crf'),exist_ok=True)
        # CRF post-processor
        postprocessor = DenseCRF(
            iter_max=CONFIG.CRF.ITER_MAX,
            pos_xy_std=CONFIG.CRF.POS_XY_STD,
            pos_w=CONFIG.CRF.POS_W,
            bi_xy_std=CONFIG.CRF.BI_XY_STD,
            bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
            bi_w=CONFIG.CRF.BI_W,
        )
        iterator = data_set.as_numpy_iterator()
        # Process per sample
        def process(data):
            label = data.get(ORIGINAL_LABEL)[0] if ORIGINAL_LABEL in data else None
            image = data.get(ORIGINAL_IMAGE)[0].astype(np.uint8)
            image_name = str(data.get(IMAGE_NAME)[0].decode('UTF-8'))
            npy_path = join(out_dir,'preds',image_name+'.npy')
            if not isfile(npy_path): raise ValueError('[ERROR]: file doesn\'t exist')
            preds = dict2preds(np.load(npy_path, allow_pickle=True).item(), CONFIG.DATASET.NUM_CLASSES)
            preds = resize(preds,(data.get(HEIGHT)[0], data.get(WIDTH)[0]))
            crf_preds = np.transpose(postprocessor(image, np.transpose(preds,(2,0,1))),(1,2,0)) # h w c -> c h w
            crf_segs = np.argmax(crf_preds, axis=-1).astype(np.uint8)
            pil_seg = Image.fromarray(crf_segs)
            pil_seg.putpalette(palette)
            pil_seg.save(join(out_dir,'crf',image_name+'.png'))
            return crf_segs, label, image_name
        # CRF in multi-process
        results = joblib.Parallel(n_jobs=CONFIG.DATALOADER.NUM_WORKERS, verbose=10, pre_dispatch="all")(
            [joblib.delayed(process)(next(iterator)) for i in range(len(data_set))]
        )
        seg_preds, labels, image_names = zip(*results)

        if not all(isinstance(l,type(None)) for l in labels):
            # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
            miou = MeanIoUWrapper(CONFIG.DATASET.NUM_CLASSES)
            print('[INFO]: Compute stats after post-processing..')
            for l, p in zip(labels, seg_preds):
                miou.update_state(ignore_labels(tensor(l), ignore_value=CONFIG.DATASET.IGNORE_LABEL),\
                    ignore_labels(tensor(p), tensor(l), ignore_value=CONFIG.DATASET.IGNORE_LABEL))
            np.save(join(out_dir,'miou_matrix_crf.npy'),miou.total_cm.numpy())
            with open(join(out_dir,f'miou_summary_crf.txt'), "w") as f:
                json.dump(miou.summary(), f, indent=4, sort_keys=True)
            print(f'[INFO]: mIoU {round(float(miou.result().numpy()),4)}')
            


if __name__ == "__main__":
    main()