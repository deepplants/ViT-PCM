#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 20:47:01 2021

@author: simone

"""
import os
import numpy as np
import tensorflow as tf
from vit_keras import vit
import tensorflow_addons as tfa
from omegaconf import OmegaConf
from tensorflow.keras import Model
from typing import List, Optional, Tuple
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, \
    Dropout, GlobalMaxPooling2D, Activation, \
    TimeDistributed, LSTM, Bidirectional, Layer
from tensorflow.keras.losses import Reduction
from tensorflow.keras.initializers import RandomNormal
from utils import resize, ignore_labels, \
        p2i, flatten, iT, affine_patch_level, \
        apply_affine_transform_batch, revert_affine_transform_batch, \
        extract_nonoverlapping_patches_inverse, extract_nonoverlapping_patches, fix_json_ndarray
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from dataloader import IMAGE, IMAGE_ET, CLASS_LABELS, SEG_LABELS, AFFINE, AFFINE_ET

def get_model(config: OmegaConf, *args, **kwargs):
    '''
    Return VIT-PCM for WSSS task
    '''
    backbone = ViT(config, name='patch_encoder')
    features = p2i(backbone.output)
    features_hv = HVBiLSTM(hidden_dim=config.MODEL.HIDDEN_SIZE,
        dropout=config.SOLVER.DROPOUT,  \
        name='patch_conditioning', *args, **kwargs)(features)
    output = PC(num_categories=config.DATASET.NUM_CLASSES,
        dropout=config.SOLVER.DROPOUT,  \
        name='patch_classifier', *args, **kwargs)(features_hv)
    model = VITPCM(inputs=backbone.input, outputs=output, \
        config=config, name='vit_pcm', *args, **kwargs)
    if config.MODEL.INIT_MODEL is not None:
        print(f'[INFO]: load model from {config.MODEL.INIT_MODEL}')
        model.load_weights(
            config.MODEL.INIT_MODEL,
            by_name=True,skip_mismatch=True)
    return model

def ViT(config: OmegaConf, *args, **kwargs):
    vit_builder = dict(ViT_L16=vit.vit_l16,ViT_B16=vit.vit_b16,ViT_S16=vit.vit_s16,ViT_B8=vit.vit_b8)
    backbone = vit_builder[config.MODEL.NAME](image_size = 384,
        pretrained = True, include_top = False, pretrained_top = False,
        weights = "imagenet21k+imagenet2012")
    # remove [cls] token
    features = backbone.get_layer('Transformer/encoder_norm').output[:,1:]
    return Model(inputs=backbone.input, outputs=features, *args, **kwargs)

class VITPCM(Model):
    '''
    This class implements a Wrapper for the VITPCM for WSSS problem
    '''
    def __init__(self, config: OmegaConf, *args, **kwargs):
        super(VITPCM, self).__init__(*args, **kwargs)
        self.config = config
        f1_macro = tfa.metrics.F1Score(
            num_classes=self.config.DATASET.NUM_CLASSES-1, # no background
            average='macro', threshold=0.5,name= 'f1_macro')
        f1_micro = tfa.metrics.F1Score(
            num_classes=self.config.DATASET.NUM_CLASSES-1, # no background
            average='micro', threshold=0.5,name= 'f1_micro')
        mean_iou = MeanIoUWrapper(self.config.DATASET.NUM_CLASSES,name='mean_iou')
        self.metrics_list = [f1_macro, f1_micro, mean_iou]

    def compile(self, optimizer):
        super(VITPCM, self).compile()
        self.optimizer = optimizer

    def train_step(self, data):
        images = data.get(IMAGE)
        class_labels = data.get(CLASS_LABELS)
        if self.config.SOLVER.LOSS_ET:    
            # inference: stop gradient, no dropout
            outputs_sg = self(images,training=False)
        with tf.GradientTape() as tape:
            outputs = self(images,training=True)
            record = dict()
            if self.config.SOLVER.LOSS_MCE:
                record.update(compute_mce_loss(outputs, class_labels))
            if self.config.SOLVER.LOSS_REG:
                record.update(compute_reg_loss(self))
            if self.config.SOLVER.LOSS_ET:
                outputs_et = self.merged_step(data.get(IMAGE_ET), training=True)           
                record.update(compute_cce_loss(outputs_sg, outputs_et, \
                    data.get(AFFINE), data.get(AFFINE_ET), class_labels, self.config.MODEL.PATCH_SIZE, \
                    self.config.IMAGE.SIZE.SCALE))
            record.update(dict(loss=sum(record.values())))
        gradients = tape.gradient(record.get('loss'), self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        record.update(compute_metrics(self.metrics, \
            outputs_sg if self.config.SOLVER.LOSS_ET else outputs, class_labels, \
            data.get(SEG_LABELS), self.config.DATASET.IGNORE_LABEL))
        record.update(dict(lr=self.optimizer.lr))
        return record

    def test_step(self, data):
        outputs = self(data.get(IMAGE),training=False)
        record = compute_mce_loss(outputs, data.get(CLASS_LABELS))
        record.update(compute_metrics(self.metrics, outputs, \
            data.get(CLASS_LABELS), data.get(SEG_LABELS), \
            self.config.DATASET.IGNORE_LABEL))
        return record
    
    def infer(self, image, class_labels: Optional[tf.Tensor] = None) -> tf.Tensor:
        preds = self(image,training=False).get('preds')
        return preds if class_labels is None else merge_class_labels(preds, class_labels)

    def merged_step(self, batch, training=True):
        B = tf.shape(batch)[0]
        D = self.config.IMAGE.SIZE.SCALE
        merged_batch = extract_nonoverlapping_patches_inverse(
            tf.ones((B//4,D*2,D*2,3)),batch[:4*B//4],size=D)
        merged_output = self(merged_batch, training=training)
        return merged_output

    @property
    def metrics(self):
        return self.metrics_list

    def warmup(self):
        self.freeze(False)
        self.get_layer('patch_conditioning').trainable = True
        self.get_layer('patch_classifier').trainable = True
        self.show()

    def unfreeze(self):
        n_block = sum([1 for layer in self.layers if 'encoderblock' in layer.name])
        trainables = ['Transformer/encoder_norm'] + [f'Transformer/encoderblock_{i}' \
                for i in range(n_block-self.config.MODEL.NUM_BLOCK_UNFREEZE,n_block)]
        for layer_name in trainables:
            self.get_layer(layer_name).trainable = True
        self.show()
    
    def freeze(self, summary=True):
        for layer in self.layers:
            self.get_layer(layer.name).trainable = False
        self.show()

    def show(self):
        self.summary(expand_nested=True, show_trainable=True)

class HVBiLSTM(Layer):
    '''
    HVBiLSTM for patch conditioning
    '''
    def __init__(self, 
        hidden_dim: int = 384, 
        dropout: float = 0.1, 
        *args, **kwargs):
        super(HVBiLSTM, self).__init__(*args, **kwargs)
        self.drop = Dropout(dropout)
        self.HBiLSTM = TimeDistributed(
            Bidirectional(
                LSTM(hidden_dim,dropout=dropout,
                return_sequences=True,name='lstm_rows'),
                name='bidirection_rows'),
                name='temporal_rows')
        self.VBiLSTM = TimeDistributed(
            Bidirectional(
                LSTM(hidden_dim,dropout=dropout,
                return_sequences=True,name='lstm_cols'),
                name='bidirection_cols'),
                name='temporal_cols')

    def build(self, input_shape):
        self.HBiLSTM.build(input_shape)
        self.VBiLSTM.build(input_shape)

    def call(self, features, training=True):
        features_grid = self.drop(features,training=training) # B x (Gh x Gw) x F -> B x Gh x Gw x F
        features_grid_h = self.HBiLSTM(features_grid) # B x Gh x Gw x F -> B x Gh x Gw x D1
        features_grid_v = iT(self.VBiLSTM(iT(features_grid))) # B x Gh x Gw x F -> B x Gw x Gh x D2
        features_hv = tf.concat([features_grid_h, features_grid_v],axis=-1) # B x Gh x Gw x D
        return features_hv

class PC(Layer):
    '''
    Patch Classifier and Max Pooling
    '''
    def __init__(self,
        num_categories: int = 21,
        dropout: float = 0.1,
        *args,**kwargs):
        super(PC, self).__init__(*args,**kwargs)
        self.drop = Dropout(dropout)
        self.layer = Dense(units=num_categories, 
            kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
            kernel_regularizer=L2(0.01),use_bias=False,activation=None,name='classifier')
        self.softmax = Activation('softmax', name='preds')
        self.pool = GlobalMaxPooling2D(data_format='channels_last',keepdims=False)

    def build(self, input_shape):
        self.layer.build(input_shape)

    def call(self, features, training=True):
        drop = self.drop(features, training=training)
        logits = self.layer(drop)
        preds = self.softmax(logits)
        pooled_preds = self.pool(preds)
        return dict(logits=logits, preds=preds, pooled_preds=pooled_preds, features=features)

def compute_reg_loss(model):
    loss = tf.reduce_mean(model.get_layer('patch_classifier').losses)
    return {f"reg_loss": loss}

def compute_cce_loss(outputs, outputs_t, affine, affine_t, labels, patch_size, scale_size):
    affine_t = affine_patch_level(affine_t, patch_size)
    affine = affine_patch_level(affine, patch_size)
    preds = apply_affine_transform_batch(revert_affine_transform_batch(outputs.get('preds'), affine),affine_t)
    preds_t = extract_nonoverlapping_patches(outputs_t.get('preds'),size=scale_size//patch_size)
    cce_loss = CategoricalCrossentropy(reduction=Reduction.NONE,name='cce')
    loss = tf.reduce_mean(cce_loss(y_true=tf.stop_gradient(merge_class_labels(preds, labels)), y_pred=preds_t))
    return {f"cce_loss": loss}

def compute_mce_loss(outputs, labels):
    bce_loss = BinaryCrossentropy(reduction=Reduction.NONE,name='bce')
    loss = tf.reduce_mean(bce_loss(y_true=labels, y_pred=outputs.get('pooled_preds')))
    return {f"bce_loss": loss}

def merge_class_labels(preds, labels):
    H, W = tf.shape(preds)[1], tf.shape(preds)[2]
    # incorporate gt class labels observation -> push to 0 absent categories
    preds *= tf.cast(tf.tile(labels[:,None,None],(1,H,W,1)),preds.dtype)
    # incorporate gt class labels observation -> push to 1 max prob for present categories
    preds = tf.math.divide_no_nan(preds, tf.reduce_max(preds, axis=(1,2), keepdims=True))
    # map again into categorical distribution
    preds = tf.math.divide_no_nan(preds, tf.reduce_sum(preds, axis=-1, keepdims=True))
    return preds

def compute_metrics(metrics, outputs, class_labels, seg_labels, ignore_label):
    '''
    This method permits to compute metrics for nested structure
    '''
    for m in metrics:
        if 'iou' in m.name:
            preds = merge_class_labels(outputs.get('preds'), class_labels)
            preds = resize(preds,(tf.shape(seg_labels)[1],tf.shape(seg_labels)[2]))
            preds = tf.argmax(preds,axis=-1,output_type=seg_labels.dtype)
            preds, seg_labels = flatten(preds), flatten(seg_labels)
            preds = ignore_labels(preds, seg_labels, ignore_label)
            seg_labels = ignore_labels(seg_labels, seg_labels, ignore_label)
            m.update_state(seg_labels, preds)
        else:
            # exclude background
            m.update_state(class_labels[...,1:],outputs.get('pooled_preds')[...,1:])
    return {m.name: m.result() for m in metrics}

class ConfusionMatrixCallback(Callback):
    def __init__(self, basedir=''):
        super(ConfusionMatrixCallback, self).__init__()
        self.basedir = os.path.join(basedir,'conf_matrix')
        self.epoch = 0
        os.makedirs(self.basedir, exist_ok = True)
    
    def on_test_begin(self, batch, logs=None):
        self.epoch+=1

    def on_test_end(self, batch, logs=None):
        miou = [m for m in self.model.metrics if 'iou' in m.name][0]
        filepath = os.path.join(self.basedir,\
            f'cm_{self.epoch}_miou_{miou.result().numpy().round(4)}.npy')
        np.save(file=filepath, arr=miou.total_cm.numpy())
        tf.print(f'Saved confusion matrix: {filepath}.')

class MeanIoUWrapper(MeanIoU):
    @fix_json_ndarray
    def summary(self):
        ''' adapted from https://github.com/kazuto1011/deeplab-pytorch/blob/master/libs/utils/metric.py'''
        hist = self.total_cm.numpy()
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        valid = hist.sum(axis=1) > 0  # added
        mean_iu = np.nanmean(iu[valid])
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.num_classes), iu))
        return {
            "Pixel Accuracy": acc,
            "Mean Accuracy": acc_cls,
            "Frequency Weighted IoU": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
        }

