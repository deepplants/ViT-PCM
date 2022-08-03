from genericpath import isdir
from multiprocessing.sharedctypes import Value
import os
import json
import numpy as np
import skimage as ski
from PIL import Image
from tqdm import tqdm
from typing import Dict, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from omegaconf import OmegaConf
from sklearn.metrics import ConfusionMatrixDisplay
tf.set_random_seed(1234)


######################
###### GENERAL #######
######################

def create_folders(config: OmegaConf):
    ''' create directories and subdirectories to store experiments '''
    os.makedirs(config.EXP.OUTPUT_DIR, exist_ok=True)
    folder = os.path.join(config.EXP.OUTPUT_DIR,config.EXP.ID)
    if os.path.isdir(folder):
        raise ValueError(f'[ERROR]: {folder} already exists, change OUTPUT_DIR in config file.')
    os.makedirs(folder, exist_ok=True)
    logdir = os.path.join(folder, config.EXP.LOGS_DIR)
    os.makedirs(logdir, exist_ok=True)
    filepath = os.path.join(folder, config.EXP.WEIGHTS_DIR)
    os.makedirs(filepath, exist_ok=True)
    # copy config file into experiment folder
    with open(f'configs/{config.DATASET.NAME}.yaml', mode='r') as in_file, open(f'{folder}/{config.DATASET.NAME}.yaml', mode='w') as out_file:
        out_file.write(in_file.read())
    modelpath = os.path.join(filepath,"model.{epoch:02d}-{val_mean_iou:.4f}.h5")
    return logdir, modelpath

def create_pascal_label_colormap(N=256, normalized=False):
    ''' map PascalVOC class labels to RGB colors '''
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    cmap = cmap/255 if normalized else cmap
    return cmap

def inverse_function(input, output, func, *args, **kwargs):
    ''' use tape to compute the inverse of a function and apply it to output tensor '''
    _input = tf.identity(input)
    _input = tf.cast(_input, tf.float32) if _input.dtype.is_integer else _input
    output_dtype = output.dtype
    output = tf.cast(output, tf.float32) if output.dtype.is_integer else output
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(_input)
        _output = func(_input, *args, **kwargs)
    normalize_grad = tape.gradient(_output, _input)
    result_grad = tape.gradient(_output, _input, output_gradients=output)
    return tf.cast(tf.math.divide_no_nan(result_grad,normalize_grad),output_dtype)

def preds2dict(preds: np.ndarray, class_labels: Optional[np.ndarray] = None):
    ''' convert numpy prediction into dict to save space (avoid storing zeros) '''
    if not isinstance(class_labels, np.ndarray): class_labels = np.ones(preds.shape[-1]) # take all
    assert preds.ndim == 3, 'preds must be shape H x W x C'
    assert class_labels.ndim == 1, 'class_labels must be shape C'
    assert preds.shape[-1] == class_labels.shape[-1], 'preds and class_labels must have same last dimension C'
    preds_dict = dict()
    for l in np.where(class_labels)[0]:
        preds_dict[l] = preds[...,l]
    return preds_dict

def dict2preds(dictnp: Dict[int,np.ndarray], num_classes: int):
    ''' convert dict back to numpy '''
    elem = list(dictnp.values())[0]
    preds = np.zeros((elem.shape[0], elem.shape[1], num_classes),elem.dtype)
    for k, v in dictnp.items():
        preds[...,k] = v
    return preds

def fix_json_ndarray(func):
    ''' decorator to fix json not accept ndarray '''
    def fix(d):
        for k,v in d.items():
            if type(v) == dict:
                d[k] = fix(v)
            elif isinstance(v,np.floating):
                d[k] = float(v)
            elif isinstance(v,np.integer):
                d[k] = int(v)
            else:
                pass
        return d
    def fixed(*args, **kwargs):
        return fix(func(*args, **kwargs))

    return fixed

#####################
####### IMAGE #######
#####################

def imsize(image_size_arg) -> Tuple[int, int]:
    """From vit-keras: Process the image_size argument whether a tuple or int."""
    if isinstance(image_size_arg, int):
        return (image_size_arg, image_size_arg)
    if (
        isinstance(image_size_arg, tuple)
        and len(image_size_arg) == 2
        and all(map(lambda v: isinstance(v, int), image_size_arg))
    ):
        return image_size_arg
    raise ValueError(
        f"The image_size argument must be a tuple of 2 integers or a single integer. Received: {image_size_arg}"
    )
    
def tensor(input, dtype=tf.float32):    
    ''' convert input to tensor '''
    return tf.cast(tf.convert_to_tensor(input),dtype)

def normalize(image, mean=0.0, std=255.0, dtype=tf.float32):
    ''' for all image in X: image->(image - mean(X))/stdv(image) '''
    return tf.cast(( image - tensor(mean,dtype=image.dtype) ) / tensor(std,dtype=image.dtype), dtype)

def denormalize(image, mean=0.0, std=255.0, dtype=tf.float32):
    ''' for all image in X: image-> image * stdv(image) + mean(X) '''
    return tf.cast(image * tensor(std) + tensor(mean), dtype)
                
def pil2vit(image):
    '''
    Normalize in [-1,1], same as:
        tf.keras.applications.imagenet_utils.preprocess_input(
            image, data_format=None, mode="tf")
    '''
    return normalize(image, mean=127.5, std=127.5, dtype=tf.float32)

def vit2pil(image):
    '''
    Denormalize in [-1,1], opposite to:
        tf.keras.applications.imagenet_utils.preprocess_input(
            image, data_format=None, mode="tf")
    '''
    return denormalize(image, mean=127.5, std=127.5, dtype=tf.uint8)

def extract_patches(image, size=16, stride=16, patch_first=True):
    ''' transform image into patch list '''
    patches = tf.extract_image_patches(
        image,
        (1, size, size, 1),
        (1, stride, stride, 1),
        (1, 1, 1, 1),
        padding="VALID"
    )
    return tf.reshape(patches, [-1 , size, size, tf.shape(image)[-1]]) if patch_first else patches

def extract_patches_inverse(image, patched_image, size=16, stride=16, patch_first=True):
    ''' transform patch list back into image '''
    return inverse_function(image, patched_image, extract_patches, size=size, stride=stride, patch_first=patch_first)

def extract_nonoverlapping_patches(image, size=16, patch_first=True):
    ''' transform image into non overlapping patch list '''
    return extract_patches(image, size=size, stride=size, patch_first=patch_first)

def extract_nonoverlapping_patches_inverse(image, patched_image, size=16, patch_first=True):
    ''' transform non overlapping patch list back into image '''
    return inverse_function(image, patched_image, extract_nonoverlapping_patches, size=size, patch_first=patch_first)

def get_square_root(x):
    ''' get square root of input number '''
    y = tf.cast(x,tf.float32)
    y = tf.math.sqrt(y)
    return tf.cast(y, x.dtype)

def p2i(x):
    ''' patch sequence to square grid '''
    C = tf.shape(x)[-1]
    P = tf.shape(x)[-2]
    G = get_square_root(P)
    return tf.reshape(x,shape=(-1,G,G,C))

def i2p(x):
    ''' square grid to patch sequence '''
    C = tf.shape(x)[-1]
    G = tf.shape(x)[-2]
    return tf.reshape(x,shape=(-1,G*G,C))

def iT(x):
    ''' square grid of shape B x H x W x C transpose '''
    return tf.transpose(x, perm=(0,2,1,3))

def pT(x):
    ''' patch sequence of shape B x (H x W) x C transpose '''
    return i2p(iT(p2i(x)))    

def flatten(x):
    ''' make a tensor mono-dimensional '''
    return tf.reshape(x,(-1,))

def unsqueeze(x, axis):
    ''' add a dimension to the tensor on the specified axis '''
    return tf.expand_dims(x,axis=axis)

def entry_stop_gradients(tensor, stop_mask):
    ''' stop the gradient for masked entries '''
    return tf.stop_gradient(stop_mask * tensor) + tf.abs(1.0 - stop_mask) * tensor

def ignore_labels(masked: tf.Tensor, mask: Optional[tf.Tensor] = None, ignore_value: int = 255):
    ''' flattenize and gate the tensor entries removing the ignore_value '''
    if not tf.is_tensor(mask):
        mask = masked
    indices = tf.where(tf.not_equal(flatten(mask),ignore_value))
    return flatten(tf.gather(flatten(masked),indices))
 
def seg2class(seg_labels, ignore_label: Optional[int] = None):
    ''' segmentation labels to class multilabels '''
    class_labels, _ = tf.unique(flatten(seg_labels)) 
    if isinstance(ignore_label, int):
        class_labels = ignore_labels(class_labels, ignore_value=ignore_label)
    return class_labels

def class2multihot(class_labels, num_classes):
    ''' convert class labels to multihot encoding '''
    onehot = tf.one_hot(class_labels, num_classes)
    multihot = tf.reduce_max(onehot, axis=0)
    return multihot

def seg2multihot(seg_labels, num_classes, ignore_label=255):
    ''' convert segmentation labels to multihot encoding '''
    class_labels = seg2class(seg_labels, ignore_label)
    multionehot = class2multihot(class_labels, num_classes)
    return multionehot

def seg2multihot_batch(seg_labels, num_classes, ignore_label):
    ''' convert batch of segmentation labels to multihot encoding '''
    copies = tf.ones((tf.shape(seg_labels)[0],), tf.int32)
    num_classes = copies * num_classes
    ignore_label = copies * ignore_label
    return tf.map_fn(seg2multihot, (seg_labels, num_classes, ignore_label), seg_labels.dtype)

def resize(image, size):
    ''' resize the input image to the input size '''
    if tf.is_tensor(image):
        return tf.image.resize(
            image, size, method = get_tf_resize_method(image), 
            align_corners=True, preserve_aspect_ratio=False, name='resize'
        )
    elif isinstance(image, np.ndarray):
        return ski.transform.resize(
            image, output_shape=size, 
            order=get_skimage_resize_method(image)
            )
    else:
      raise ValueError('input type not match supported types.')

def get_tf_resize_method(input):
    '''
    # From Tensorflow Garden
    Returns the resize method of input depending on input dtype.
    Args:
      input: Groundtruth input tensor.
    Returns:
      tf.image.ResizeMethod.BILINEAR, if input dtype is floating.
      tf.image.ResizeMethod.NEAREST_NEIGHBOR, if input dtype is integer.
    Raises:
      ValueError: If input is neither floating nor integer.
    '''
    if input.dtype.is_floating:
      return tf.image.ResizeMethod.BILINEAR
    elif input.dtype.is_integer:
      return tf.image.ResizeMethod.NEAREST_NEIGHBOR
    else:
      raise ValueError('input type must be either floating or integer.')


def get_skimage_resize_method(input):
    '''
    get resize method from numpy array using scikit-image:
    
    The order of interpolation. The order has to be in the range 0-5:

            0: Nearest-neighbor

            1: Bi-linear (default)

            2: Bi-quadratic

            3: Bi-cubic

            4: Bi-quartic

            5: Bi-quintic

        Default is 0 if image.dtype is bool and 1 otherwise.

    '''
    if issubclass(input.dtype.type, np.floating):
      return 1
    elif issubclass(input.dtype.type, np.integer):
      return 0
    else:
      raise ValueError('input type must be either floating or integer.')


######################
##### DATALOADER #####
######################

def affine_transform(image: tf.Tensor, label: Optional[tf.Tensor], \
    prob_rot90: float = 0.5, prob_flip_ud: float = 0.5, \
    prob_flip_lr: float = 0.5, prob_crop: float = 0.5, \
    patch_size: int = 16, crop_size: int = 128, scale_size: int = 192):
    ''' apply affine transformations to input image and label '''
    image_size = tf.shape(image)[0]

    ##### BBOX FOR CROP #####  
    crop_size = tensor(crop_size)
    crop_size = tf.cond(tf.equal(tf.rank(crop_size),0), \
        lambda: tf.expand_dims(crop_size,0), lambda: crop_size)
    # random choice among multiple crop dimensions if given
    crop_size = tf.squeeze(random_uniform_choice(crop_size,1))  
    bbox = tf.cond(
        tf.math.logical_and(
            tf.greater(crop_size,tf.zeros((),crop_size.dtype)), # only second branch
            tf.less_equal(tf.random_uniform(()), prob_crop)
        ),
        lambda: bbox_for_crop(image_size=image_size,delta_factor=patch_size,crop_size=crop_size),
        lambda: tf.zeros((4,),dtype=tf.float32)
    )

    ##### RESIZE TO STANDARD DIM #####
    resize_size = tf.cond(
            tf.greater(crop_size,tf.zeros((),crop_size.dtype)), # only second branch          
            lambda: tf.cond(
                tf.math.logical_and(
                    tf.reduce_any(tf.greater(bbox,tf.zeros((4,),bbox.dtype))),
                    tf.equal(crop_size,scale_size)
                ),
                lambda: tf.zeros((2,),bbox.dtype),
                lambda: tensor([scale_size, scale_size],bbox.dtype)
            ),
            lambda: tf.zeros((2,),bbox.dtype),
    )

    ##### FLIP #####    
    fliplr = tf.cast(tf.less_equal(tf.random_uniform((1,)), prob_flip_lr), tf.float32)
    flipud = tf.cast(tf.less_equal(tf.random_uniform((1,)), prob_flip_ud), tf.float32)
    
    ##### ROTATE #####
    transpose = tf.cast(tf.less_equal(tf.random_uniform((1,)), prob_rot90),tf.float32)

    ##### APPLY #####
    affine = tf.cast(tf.concat([fliplr,flipud,transpose,bbox,resize_size],axis=0), tf.float32) # transformation summary
    image = apply_affine_transform(image, affine)
    label = apply_affine_transform(label, affine) if tf.is_tensor(label) else -1
    
    return image, label, affine

def apply_affine_transform(image, affine):   
    ''' apply given affine transformation to the input: only second branch images are affected '''
    bbox, resize_size = affine[...,3:-2], affine[...,-2:]
    # crop before reduces memory requirements
    image = tf.cond(tf.reduce_any(tf.greater(bbox,0.0)), lambda: crop_image(image, tf.cast(bbox,tf.int32)), lambda: image)  
    image = tf.cond(tf.reduce_any(tf.greater(resize_size,0.0)), lambda: tf.squeeze(resize(tf.expand_dims(image,0), tf.cast(resize_size,tf.int32)),0), lambda: image)    
    image = apply_affine_transform_invertible(image, affine)
    return  image

def apply_affine_transform_invertible(image, affine):
    ''' apply given affine transformation to the input: all images are affected '''
    fliplr, flipud, transpose = tf.split(affine[...,:3], num_or_size_splits=3, axis=-1)
    # crop before reduces memory requirements  
    image = tf.cond(tf.cast(fliplr,tf.bool), lambda: tf.image.flip_left_right(image), lambda: image)
    image = tf.cond(tf.cast(flipud,tf.bool), lambda: tf.image.flip_up_down(image), lambda: image)
    image = tf.cond(tf.cast(transpose,tf.bool), lambda: tf.transpose(image,perm=(1,0,2)), lambda: image)
    
    return image

def affine_patch_level(affine, patch_dim=16):
    ''' convert affine transformation to patch sizes '''
    return tf.concat([affine[...,:3],affine[...,3:]//patch_dim],axis=-1)

def revert_affine_transform(image, affine):
    ''' revert the applied affine transformations.
    inverse_function can be used only for double invertible functions, it
    gives LookupError for those who have not:
     > LookupError: No gradient defined for operation'gradients/.../ResizeBilinear_grad/ResizeBilinearGrad' (op type: ResizeBilinearGrad).
     We might define the gradient instead:
    @tf.RegisterGradient("ResizeBilinearGrad")
    def _ResizeBilinearGrad_grad(op, grad):
        up = tf.image.resize(grad,tf.shape(op.inputs[0])[1:-1])
        return up,None 
    but it is not necessary since we prefer to compute et loss without inverting the resize and cropping. '''
    return  inverse_function(tf.zeros_like(image), image, apply_affine_transform_invertible, affine=affine)

def apply_affine_transform_batch(image, affine):
    ''' apply affine transformations to a batch '''
    image = tf.map_fn(
        lambda inputs: apply_affine_transform(*inputs),
        [image, affine],
        fn_output_signature=image.dtype
        )
        
    return image

def revert_affine_transform_batch(image, affine):
    ''' revert affine transformations to a batch '''
    image = tf.map_fn(
        lambda inputs: revert_affine_transform(*inputs),
        [image, affine],
        fn_output_signature=image.dtype
        )    
    return image

# def apply_affine_t_clean(image, affine):
#     (fliplr, flipud, transpose), bbox, resize_size = tf.split(affine[...,:3], num_or_size_splits=3, axis=-1), affine[...,3:-2], affine[...,-2:]
#     # crop before reduces memory requirements
#     # image = tf.cond(tf.reduce_any(tf.greater(bbox,0.0)), lambda: crop_image(image, tf.cast(bbox,tf.int32)), lambda: image)
#     # resize gives trouble in backprop  
#     image = tf.cond(tf.cast(fliplr,tf.bool), lambda: tf.image.flip_left_right(image), lambda: image)
#     image = tf.cond(tf.cast(flipud,tf.bool), lambda: tf.image.flip_up_down(image), lambda: image)
#     image = tf.cond(tf.cast(transpose,tf.bool), lambda: tf.transpose(image,perm=(1,0,2)), lambda: image)
    
#     return  image

# def apply_affine_t_batch_clean(image, affine):
#     ''' not use tf methods which do no support
#     2nd gradient i.e. resize or crop '''
#     image = tf.map_fn(
#         lambda inputs: apply_affine_t_clean(*inputs),
#         [image, affine],
#         fn_output_signature=image.dtype
#         )
        
#     return image

# def revert_affine_t_batch_clean(image, affine):
    
#     return inverse_function(tf.zeros_like(image), image, apply_affine_t_batch_clean, affine=affine)

def color_transform(image, max_delta_bright=0.2, 
    lower_sat=0.5, upper_sat=1.5, lower_cont=0.5, 
    upper_cont=2.0, max_delta_hue=0.1, gray_prob=0.1):
    ''' apply color transformations to input image '''
    image = tf.image.random_brightness(image, max_delta=max_delta_bright)
    image = tf.image.random_contrast(image, lower=lower_cont, upper=upper_cont)
    image = tf.image.random_saturation(image, lower=lower_sat, upper=upper_sat)
    image = tf.image.random_hue(image, max_delta=max_delta_hue)
    image = tf.cond(tf.less_equal(tf.random_uniform(()), gray_prob), \
        lambda: tf.tile(tf.image.rgb_to_grayscale(image),(1,1,3)), lambda: image)
    return image

def random_uniform_choice(inputs, n_samples=1):
    '''
    # Web source
    With replacement.
    Params:
      inputs (Tensor): Shape [n_states, n_features]
      n_samples (int): The number of random samples to take.
    Returns:
      sampled_inputs (Tensor): Shape [n_samples, n_features]
    '''
    uniform_log_prob = tf.expand_dims(tf.zeros(tf.shape(inputs)[0]), 0)
    ind = tf.random.categorical(uniform_log_prob, n_samples)
    ind = tf.squeeze(ind, 0, name="random_uniform_choice_ind")
    return tf.gather(inputs, ind, name="random_uniform_choice")

def bbox_for_crop(image_size: tf.Tensor, crop_size: int = 128, delta_factor: int = 16):
    '''
    We use top-left/right-bottom 
    convention for bounding boxes:
    (x1,y1) +-------+
            |       |
            |       |
            +-------+
                    (x2,y2)
    '''    
    start_options = tf.range(0,image_size+1,delta_factor,dtype=tf.int32)
    deltas = tensor(crop_size, start_options.dtype)
    start = tf.minimum(random_uniform_choice(start_options, n_samples=2),image_size-deltas)
    bbox = tf.concat([start, start + deltas], -1) 
    
    return tf.cast(bbox,tf.float32)

def crop_image(image, bbox):
    '''
    We use top-left/right-bottom 
    convention for bounding boxes:
    (x1,y1) +-------+
            |       |
            |       |
            +-------+
                    (x2,y2)
    '''
    offsets = bbox[0:2]
    deltas = tf.abs(bbox[2:4] - offsets) # y2-y1, ..
    return tf.image.crop_to_bounding_box(image, \
        offsets[0], offsets[1], deltas[0], deltas[1])


#####################
###### RESULTS ######
#####################

# def save_n_results(model, dataset, n_samples=10, verbose=True, save_mask=False, save_fig=False, save_npy=False, save_iou_scores=False, dest_folder='dict_npy', dest_path=cfg.SAMPLES, mask_pred_with_class_labels=False, source_path=None, source_folder='npy', save_psa_format=False, show_metrics=True, save_irn_format=False, ignore_index=cfg.IGNORE_LABEL):

#     n_samples = min(len(dataset),n_samples)
#     if save_iou_scores or show_metrics:
#         mean_iou = tf.keras.metrics.MeanIoU(cfg.NUM_CLASSES, name='mean_iou')
#     os.makedirs(name=dest_path, exist_ok=True)
#     for index, data in (bar := tqdm(iterable=enumerate(iter(dataset)),total=n_samples)):
#         if index >= n_samples: break

#         labels, preds, iou = save_result(model, data, verbose, save_mask, save_fig, save_npy, dest_folder, dest_path, mask_pred_with_class_labels, source_path, source_folder, save_psa_format, return_stats=save_iou_scores or show_metrics, save_irn_format=save_irn_format, ignore_index=ignore_index)
#         if save_iou_scores or show_metrics:
#             mean_iou.update_state(labels, preds) #, sample_weight=sample_weight) # include background
#             total_mean_iou = round(mean_iou.result().numpy().round(4),4)
#             iou = np.array2string(iou, formatter={'float_kind':lambda x: "%.4f" % iou})
#             total_iou = np.array2string(total_mean_iou, formatter={'float_kind':lambda x: "%.4f" % total_mean_iou})
#             bar.set_description(f'current_meanIOU: {iou}, total_meanIOU: {total_iou} ')
#         if save_iou_scores: 
#             name = str(data["image_name"].numpy()[0].decode('UTF-8'))
#             with open(os.path.join(dest_path,'per_sample_meanIOU.txt'),"a") as f:
#                 f.write(name+"\t"+iou+"\n")
    
#     if save_iou_scores:        
#         plot_and_save_confusion_matrix(total_cm=mean_iou.total_cm.numpy(),result=mean_iou.result().numpy(),save_path=dest_path)
    
#     return dict(meaniou=mean_iou.result(), cm=mean_iou.total_cm) if save_iou_scores or show_metrics else None

# def save_result(model, data, verbose=True, save_mask=False, save_fig=False, save_npy=False, dest_folder='dict_npy', dest_path=cfg.SAMPLES, mask_pred_with_class_labels=False, source_path=None, source_folder='npy',save_psa_format=False, return_stats=True, save_irn_format=False, ignore_index=cfg.IGNORE_LABEL):
#     image = data['image']
#     original_image = data['original_image']
#     original_label = data['original_label'][0]
#     labels = data['label']
#     height = data['height'][0]
#     width = data['width'][0]
#     name = str(data["image_name"].numpy()[0].decode('UTF-8'))
#     if not ('png' in source_folder or 'seg' in source_folder):
#         if model is not None:
#             output = model(image)
#         elif os.path.exists(os.path.join(os.path.join(source_path,source_folder), name+'.npy')):
#             npy_preds = np.load(os.path.join(os.path.join(source_path,source_folder), name+'.npy'),allow_pickle=True)
#             npy_preds = npy_preds.item() if npy_preds.dtype == np.dtype('object') else npy_preds
#             if isinstance(npy_preds, dict):
#                 npy_preds_keys = list(npy_preds.keys())
#                 npy_preds_values = np.stack(list(npy_preds.values()),axis=-1)
#                 if source_folder=='dict_npy' or source_folder=='dict_crf_npy': # dict_npy has 20 classes
#                     npy_preds = np.zeros(npy_preds_values.shape[:2]+(cfg.NUM_CLASSES,))
#                     npy_preds[...,npy_preds_keys]=npy_preds_values
#                     npy_preds = np.concatenate([1 - np.max(npy_preds,axis=-1,keepdims=True),npy_preds] , axis=-1)
#                 else:
#                     npy_preds = np.zeros(npy_preds_values.shape[:2]+(cfg.NUM_CLASSES,))
#                     npy_preds[...,npy_preds_keys]=npy_preds_values
#                 npy_preds = npy_preds[None]
#             npy_preds = tensor(npy_preds)
#             output = dict(preds=npy_preds, pooled_preds=tf.reduce_max(npy_preds,axis=(1,2)))
#         else:
#             raise ValueError('you should pass a model or a valid npy scores file in dict format or 3D matrix.')
        
#         pred_grid = output.get('preds')

#         mask_l =  tf.cast(tf.tile(labels[:,None,None],(1,tf.shape(pred_grid)[1],tf.shape(pred_grid)[2],1)),pred_grid.dtype)

#         pred_grid = mask_pred(pred_grid,mask_l) if mask_pred_with_class_labels else pred_grid
#         if save_psa_format:
#             path = os.path.join(dest_path,dest_folder)
#             os.makedirs(name=path, exist_ok=True)
#             dict_npy=dict()
#             for i,k in enumerate(labels[0]): # psa don't want background 
#                 if float(k)>1e-5:
#                     dict_npy[i]=pred_grid[0,...,i].numpy()
#             if ignore_index!=None:
#                 dict_npy[ignore_index]=pred_grid[0,...,-1].numpy()
#             np.save(os.path.join(path, name+'.npy'), dict_npy)
#         preds_patch = resize(pred_grid,(height,width))[0]
#         if save_irn_format:
#             strided_size = get_strided_size((height,width), 4)
#             strided_up_size = get_strided_up_size((height,width), 16)
#             strided_cam = resize(pred_grid,strided_size)[0]
#             highres_cam = resize(pred_grid,strided_up_size)[0,:height,:width]
#             strided_cam = tf.math.divide_no_nan(strided_cam, tf.reduce_max(strided_cam,axis=(1,2),keepdims=True))
#             highres_cam = tf.math.divide_no_nan(highres_cam, tf.reduce_max(highres_cam,axis=(1,2),keepdims=True))
#             path = os.path.join(dest_path,dest_folder)
#             os.makedirs(name=path, exist_ok=True)
#             dict_npy_s=dict()
#             dict_npy_h=dict()
#             for i,k in enumerate(labels[0,1:]): # psa don't want background 
#                 if float(k)>1e-5:
#                     dict_npy_s[i]=strided_cam[...,i+1].numpy()
#                     dict_npy_h[i]=highres_cam[...,i+1].numpy()
#             valid_cat = np.array(list(dict_npy_s.keys()))
#             strided_cam = np.array(list(dict_npy_s.values()))
#             highres_cam = np.array(list(dict_npy_h.values()))
#             np.save(os.path.join(path, name + '.npy'),
#                     {"keys": valid_cat, "cam": strided_cam, "high_res": highres_cam})

#         if save_npy: 
#             path = os.path.join(dest_path,'npy' if dest_folder==None else dest_folder)
#             os.makedirs(name=path, exist_ok=True)
#             np.save(os.path.join(path, name+'.npy'), pred_grid, allow_pickle=True, fix_imports=True)
        

#         preds_mask = tf.argmax(preds_patch,axis=-1).numpy().astype(np.uint8)
#     else:
#         preds_mask = np.array(Image.open(os.path.join(source_path,source_folder,name+'.png'))).astype(np.uint8)
#     if save_mask:
#         pil_image = Image.fromarray(preds_mask.astype(np.uint8))
#         pil_image.putpalette(create_pascal_label_colormap())
#         path = os.path.join(dest_path,'preds')
#         os.makedirs(name=path, exist_ok=True) 
#         pil_image.save(os.path.join(path,name+".png"))
#     if save_fig or verbose:
        
#         preds_classes = tf.where(tf.round(output.get('pooled_preds'))[0] if labels is None else labels[0])[:,0]
#         fig, ax = plt.subplots(1, len(preds_classes)+4, figsize=(15,3))
#         for a in ax:
#             a.tick_params(left = False, right = False , labelleft = False ,
#                         labelbottom = False, bottom = False)
#         ax[0].imshow(original_image[0])
#         ax[0].set_title('input')
#         ax[1].imshow(label2color_image(original_label[...,0].numpy()).astype(np.uint8))
#         ax[1].set_title(',\n'.join([cfg.LABEL_NAMES[cc] for cc in tf.where(labels[0])[:,0]]),fontsize=8)
#         ax[2].imshow(label2color_image(preds_mask).astype(np.uint8))
#         ax[2].set_title('predictions',fontsize=8)
#         for i,pred in enumerate(preds_classes):
#             ax[i+3].imshow(preds_patch[...,pred])
#             ax[i+3].set_title(cfg.LABEL_NAMES[pred],fontsize=8)
#         ax[-1].imshow(preds_patch[...,-1])
#         ax[-1].set_title('Ignore label',fontsize=8)
#         if verbose: plt.show()
#         if save_fig: 
#             path = os.path.join(dest_path,'fig')
#             os.makedirs(name=path, exist_ok=True) 
#             fig.savefig(os.path.join(path,name+".png"), dpi=200)
#         plt.close(fig)
#     if return_stats:
#         labels, preds, iou = evaluate_model_single_image(preds_mask,original_label,ignore_index=ignore_index)
#         return labels, preds, iou
#     else:
#         return [], [], []


# def evaluate_model_single_image(preds_label, original_label,ignore_index=None):
#     mean_iou = tf.keras.metrics.MeanIoU(cfg.NUM_CLASSES)
#     # preds = tf.argmax(preds,axis=-1,output_type=tf.int32)[...,None]
#     labels = tf.reshape(tensor(original_label), (-1,))
#     preds = tf.reshape(tensor(preds_label), (-1,))
#     preds = tf.gather(preds,tf.where(tf.not_equal(labels,cfg.IGNORE_LABEL)))[...,0]
#     labels = tf.gather(labels,tf.where(tf.not_equal(labels,cfg.IGNORE_LABEL)))[...,0]
#     mean_iou.update_state(labels, preds)
#     iou = mean_iou.result().numpy().round(4)
#     return labels, preds, iou

# def compute_cm_stats(total_cm):
#     hist = total_cm
#     n_class = total_cm.shape[0]
#     acc = np.diag(hist).sum() / hist.sum()
#     acc_cls = np.diag(hist) / hist.sum(axis=1)
#     acc_cls = np.nanmean(acc_cls)
#     iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
#     valid = hist.sum(axis=1) > 0  # added
#     mean_iu = np.nanmean(iu[valid])
#     freq = hist.sum(axis=1) / hist.sum()
#     fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
#     cls_iu = dict(zip(range(n_class), iu))
#     res = {
#         "Pixel Accuracy": acc,
#         "Mean Accuracy": acc_cls,
#         "Frequency Weighted IoU": fwavacc,
#         "Mean IoU": mean_iu,
#         "Class IoU": cls_iu,
#     }
#     return res



# def plot_and_save_confusion_matrix(total_cm, result, save_path=None):
#     if save_path: os.makedirs(name=save_path, exist_ok=True) 
#     if save_path: np.save(file=os.path.join(save_path,'confusion_matrix_'+str(result.round(4))+'.npy'), arr=total_cm)
#     if save_path: 
#         score = fix(compute_cm_stats(total_cm))
#         scoreiou = round(score["Mean IoU"],4)
#         with open(os.path.join(save_path,f'meanIOU_{scoreiou}.txt'), "w") as f:
#             json.dump(score, f, indent=4, sort_keys=True) 
#     fig, ax = plt.subplots(1,1,figsize=(20,20))
#     cmd = ConfusionMatrixDisplay((total_cm/tf.reduce_sum(total_cm,axis=0)).numpy().round(3),display_labels=cfg.LABEL_NAMES)
#     cmd.plot(ax=ax,xticks_rotation=45)
#     if save_path: plt.savefig(os.path.join(save_path,'confusion_matrix_precision_'+str(result.round(4))+'.png'))
#     plt.close(fig)
#     fig, ax = plt.subplots(1,1,figsize=(20,20))
#     cmd = ConfusionMatrixDisplay((total_cm/tf.reduce_sum(total_cm,axis=1)).numpy().round(3),display_labels=cfg.LABEL_NAMES)
#     cmd.plot(ax=ax,xticks_rotation=45)
#     if save_path: plt.savefig(os.path.join(save_path,'confusion_matrix_recall_'+str(result.round(4))+'.png'))
#     plt.close(fig)