# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrapper for providing semantic segmentaion data.

The SegmentationDataset class provides both images and annotations (semantic
segmentation and/or instance segmentation) for TensorFlow. Currently, we
support the following datasets:

1. PASCAL VOC 2012 (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

PASCAL VOC 2012 semantic segmentation dataset annotates 20 foreground objects
(e.g., bike, person, and so on) and leaves all the other semantic classes as
one background class. The dataset contains 1464, 1449, and 1456 annotated
images for the training, validation and test respectively.

2. Cityscapes dataset (https://www.cityscapes-dataset.com)

The Cityscapes dataset contains 19 semantic labels (such as road, person, car,
and so on) for urban street scenes.

3. ADE20K dataset (http://groups.csail.mit.edu/vision/datasets/ADE20K)

The ADE20K dataset contains 150 semantic labels both urban street scenes and
indoor scenes.

References:
  M. Everingham, S. M. A. Eslami, L. V. Gool, C. K. I. Williams, J. Winn,
  and A. Zisserman, The pascal visual object classes challenge a retrospective.
  IJCV, 2014.

  M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson,
  U. Franke, S. Roth, and B. Schiele, "The cityscapes dataset for semantic urban
  scene understanding," In Proc. of CVPR, 2016.

  B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso, A. Torralba, "Scene Parsing
  through ADE20K dataset", In Proc. of CVPR, 2017.
"""

import os
import collections
from typing import List, Optional, Tuple
from tqdm import tqdm
import tensorflow.compat.v1 as tf
from utils import pil2vit, seg2multihot, \
    resize, affine_transform, color_transform, imsize
from omegaconf import OmegaConf

IMAGE = 'image'
IMAGE_NAME = 'image_name'
CLASS_LABELS = 'class_label'
SEG_LABELS = 'seg_label'
ORIGINAL_IMAGE = 'original_image'
ORIGINAL_LABEL = 'original_label'
AFFINE = 'affine'
IMAGE_ET = 'image_et'
AFFINE_ET = 'affine_et'
HEIGHT = 'height'
WIDTH = 'width'
# Test set name.
TEST_SET = 'test'

DTYPES = {
  IMAGE : tf.float32,
  IMAGE_NAME : tf.string,
  CLASS_LABELS : tf.int32,
  SEG_LABELS : tf.int32,
  ORIGINAL_IMAGE : tf.float32,
  ORIGINAL_LABEL : tf.int32,
  AFFINE : tf.float32,
  IMAGE_ET : tf.float32,
  AFFINE_ET : tf.float32,
  HEIGHT : tf.int32,
  WIDTH : tf.int32,
}

# Named tuple to describe the dataset properties.
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    [
        'splits_to_sizes',  # Splits of the dataset into training, val and test.
        'num_classes',  # Number of semantic classes, including the
                        # background class (if exists). For example, there
                        # are 20 foreground classes + 1 background class in
                        # the PASCAL VOC 2012 dataset. Thus, we set
                        # num_classes=21.
        'ignore_label',  # Ignore label value.
    ])

_VOC12_SEG_INFO = DatasetDescriptor(
    splits_to_sizes={
        'train': 1464,
        'trainaug': 10582,
        'trainval': 2913,
        'val': 1449,
        'test': 1456,
    },
    num_classes=21,
    ignore_label=255,
)

_COCO14_VOC12_SEG_INFO = DatasetDescriptor(
    splits_to_sizes={
        'train': 94571,
    },
    num_classes=21,
    ignore_label=255,
)

_COCO14_SEG_INFO = DatasetDescriptor(
    splits_to_sizes={
        'train': 81477,
        'val': 39884,
    },
    num_classes=81,
    ignore_label=255,
)

_DATASETS_INFORMATION = {
    'voc12': _VOC12_SEG_INFO,
    'coco14_voc12': _COCO14_VOC12_SEG_INFO,
    'coco14': _COCO14_SEG_INFO
}

# Default file pattern of TFRecord of TensorFlow Example.
_FILE_PATTERN = '%s-*'

class DatasetVITPCM(object):
    """Represents dataset for vit model."""

    def __init__(self,
        dataset_name: List[str] = ['voc12'],
        split_name: List[str] = ['trainaug'],
        dataset_dir: List[str] = ['datasets/voc12'],
        batch_size: int = 1,
        input_size: int = 384,
        patch_size: int = 16,
        crop_size: int = 128,
        scale_size: int = 192,
        num_workers: int = 1,
        should_augment: bool = False,
        should_shuffle: bool = False,
        should_repeat: bool = False,
        prob_rot90: float = 0.5, 
        prob_flip_ud: float = 0.5,
        prob_flip_lr: float = 0.5, 
        prob_crop: float = 0.5,
        replicas: int = 1):
        """Initializes the dataset.
        
        Args:
          dataset_name: The list of directory of the dataset split names.
          split_name: A train/val list split names.
          dataset_dir: The directory of the dataset sources. 
          batch_size: Batch size.
          input_size: The size used to resize the image and label.
          crop_size: The size used to crop the image and label.
          scale_size: The size used to scale the image and label.
          patch_size: The size used to shift the crop of the image and label.
          num_workers: Number of readers for data provider.
          should_augment: Boolean, if dataset is for training or not.
          should_shuffle: Boolean, if should shuffle the input data.
          should_repeat: Boolean, if should repeat the input data.
        Raises:
          ValueError: Dataset name and split name are not supported.
        """
        
        if len(dataset_name) != len(split_name) != len(dataset_dir):
          raise ValueError('you should provide a list of dataset names, splits and root directories.')
        
        for name in dataset_name:
          if name not in _DATASETS_INFORMATION:
            raise ValueError('The specified dataset is not supported yet.')
        self.dataset_name = dataset_name
        
        splits_to_sizes = []
        for name in dataset_name:
          splits_to_sizes.extend(_DATASETS_INFORMATION[name].splits_to_sizes)

        for split in split_name:
          if split not in splits_to_sizes:
            raise ValueError('data split name %s not recognized' % split)

        self.split_name = split_name
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_workers = num_workers
        self.should_augment = should_augment
        self.should_shuffle = should_shuffle
        self.should_repeat = should_repeat
        self.replicas = replicas
        self.crop_size = crop_size
        self.scale_size = scale_size
        self.patch_size = patch_size
        self.prob_crop = prob_crop
        self.prob_rot90 = prob_rot90
        self.prob_flip_ud = prob_flip_ud
        self.prob_flip_lr = prob_flip_lr
        
        classes = [_DATASETS_INFORMATION[n].num_classes for n in self.dataset_name]
        if not classes.count(classes[0]) == len(classes):
          raise ValueError('datasets must have same number of classes')
        self.num_classes = classes[0]

        ignore_label = [_DATASETS_INFORMATION[n].ignore_label for n in self.dataset_name]
        if not ignore_label.count(ignore_label[0]) == len(ignore_label):
          raise ValueError('datasets must have same ignore_label')
        self.ignore_label = ignore_label[0]
        

    def get_one_shot_iterator(self):
        """Gets an iterator that iterates across the dataset once.

        Returns:
          An iterator of dype tf.data.Iterator.
        """
        return self.get_dataset().make_one_shot_iterator()
  
    def get_dataset(self):
        """Gets the dataset.

        Returns:
          The TFRecord dataset.
        """
        list_files = []
        for (dataset,split) in zip(self.dataset_dir,self.split_name):
          list_files.extend(self._get_all_files(dataset,split))

        dataset = tf.data.TFRecordDataset(list_files, num_parallel_reads=self.num_workers)

        length = (sum(1 for fn in list_files for _ in tf.python_io.tf_record_iterator(fn)) // self.batch_size) * self.batch_size

        if self.should_shuffle:
          dataset = dataset.shuffle(buffer_size=length)

        if self.should_repeat:
          dataset = dataset.repeat()  # Repeat forever for training.
        else:
          dataset = dataset.repeat(1)

        dataset = (
          dataset.map(self._parse_function, num_parallel_calls=self.num_workers)
          .map(self._preprocess_image, num_parallel_calls=self.num_workers)
          )

        if not self.should_repeat:
          dataset = dataset.apply(tf.data.experimental.assert_cardinality(length))
        
        dataset = dataset.batch(self.batch_size).prefetch(self.batch_size)

        return dataset

    def _preprocess_image(self, sample):
        """Preprocesses the image and label.
        
        Args:
          sample: A sample containing image and label.
        
        Returns:
          sample: Sample with preprocessed image and label.
        
        Raises:
          ValueError: Ground truth label not provided during training.
        """
        image = sample[IMAGE]
        label = sample[CLASS_LABELS]
                
        if self.batch_size==1:
            ##### Test case  #####
            # Original image is only used during visualization.
            sample[ORIGINAL_IMAGE] = tf.identity(image)
            if label is not None:
              sample[ORIGINAL_LABEL] = tf.identity(label)
        
        ##### Resize to model input size #####
        image = resize(tf.cast(image,tf.float32), imsize(self.input_size))
        if label is not None:
          label = resize(tf.cast(label,tf.int32), imsize(self.input_size))

        if self.should_augment and label is not None:
            ##### Augmentation #####
            # It is extremely important here to never use segmentation mask to
            # produce different labels than the ones in the ground truth.
            # no shift, only rotation, shift would imply we should check for labels consistency
            # which could be opinable as approach, since ideally we should run this 
            # algothm on datasets with no segmentation at all, in that case this approach wouldn't be possible
            ########################
            image_t1, label_t1, affine_t1 = affine_transform(image=color_transform(tf.identity(image)), 
                                                            label=tf.identity(label), patch_size = self.patch_size, \
                                                            prob_rot90 = self.prob_rot90, prob_flip_ud = self.prob_flip_ud, \
                                                            prob_flip_lr = self.prob_flip_lr, prob_crop = self.prob_crop, \
                                                            crop_size = 0, scale_size = 0) 
            sample[IMAGE] = pil2vit(image_t1)
            sample[CLASS_LABELS] = seg2multihot(tf.identity(label), self.num_classes, self.ignore_label) # 2D labels to 1D
            sample[AFFINE] = affine_t1
            sample[SEG_LABELS] = label_t1 # for visualization and eval test only, not for training
                              
            # create a transformed version of the image
            # to self train patch classification
            image_t2, _, affine_t2 = affine_transform(image=color_transform(tf.identity(image)), 
                                                            label=tf.identity(label), patch_size = self.patch_size, \
                                                            prob_rot90 = self.prob_rot90, prob_flip_ud = self.prob_flip_ud, \
                                                            prob_flip_lr = self.prob_flip_lr, prob_crop = self.prob_crop, \
                                                            crop_size = self.crop_size, scale_size = self.scale_size) 
            sample[IMAGE_ET] = pil2vit(image_t2)
            sample[AFFINE_ET] = affine_t2
        else:
            ##### No augmentation ######
            sample[IMAGE] = pil2vit(tf.identity(image))     
            if label is not None:   
              sample[SEG_LABELS] = label
              sample[CLASS_LABELS] = seg2multihot(label, self.num_classes, self.ignore_label) if label is not None else None
        
        return sample

    def _get_all_files(self, dataset_dir, split_name):
        """Gets all the files to read data from.

        Returns:
          A list of input files.
        """
        print(f'[INFO]: Load from {dataset_dir}, split {split_name}')
        file_pattern = _FILE_PATTERN
        file_pattern = os.path.join(dataset_dir,
                                    file_pattern % split_name)
        return tf.gfile.Glob(file_pattern)


    def _parse_function(self, example_proto):
        """Function to parse the example proto.

        Args:
          example_proto: Proto in the format of tf.Example.

        Returns:
          A dictionary with parsed image, label, height, width and image name.

        Raises:
          ValueError: Label is of wrong shape.
        """

        # Currently only supports jpeg and png.
        # Need to use this logic because the shape is not known for
        # tf.image.decode_image and we rely on this info to
        # extend label if necessary.
        def _decode_image(content, channels):
          return tf.cond(
              tf.image.is_jpeg(content),
              lambda: tf.image.decode_jpeg(content, channels),
              lambda: tf.image.decode_png(content, channels))

        seg_dict = {}
        if all(name != TEST_SET for name in self.split_name):
          seg_dict = {
            'image/segmentation/class/encoded':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/segmentation/class/format':
                tf.FixedLenFeature((), tf.string, default_value='png'),
          }

        features = {
            'image/encoded':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/filename':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format':
                tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/height':
                tf.FixedLenFeature((), tf.int64, default_value=0),
            'image/width':
                tf.FixedLenFeature((), tf.int64, default_value=0),
            **seg_dict
        }

        parsed_features = tf.parse_single_example(example_proto, features)

        image = _decode_image(parsed_features['image/encoded'], channels=3)

        label = None
        if all(name != TEST_SET for name in self.split_name):
          label = _decode_image(
              parsed_features['image/segmentation/class/encoded'], channels=1)

        image_name = parsed_features['image/filename']
        if image_name is None:
          image_name = tf.constant('')

        sample = {
            IMAGE: image,
            IMAGE_NAME: image_name,
            HEIGHT: parsed_features['image/height'],
            WIDTH: parsed_features['image/width'],
        }

        if label is not None:
          if label.get_shape().ndims == 2:
            label = tf.expand_dims(label, 2)
          elif label.get_shape().ndims == 3 and label.shape.dims[2] == 1:
            pass
          else:
            raise ValueError('Input label shape must be [height, width], or '
                            '[height, width, 1].')

          label.set_shape([None, None, 1])

        sample[CLASS_LABELS] = label

        return sample

class DatasetVITPCMDistributed(DatasetVITPCM):
    """Represents distributed dataset for vit model."""

    def __init__(self, config: OmegaConf, set_name: str = 'train', num_replicas: int = 1):
        TRAIN_DICT = dict(
            split_name=config.DATASET.SPLIT.TRAIN.split(','), 
            batch_size=config.SOLVER.BATCH_SIZE.TRAIN * num_replicas,
            input_size=config.IMAGE.SIZE.TRAIN, 
            crop_size=config.IMAGE.SIZE.CROP,
            scale_size=config.IMAGE.SIZE.SCALE,
            should_augment=True,
            should_shuffle=True,
            prob_crop=config.DATASET.AUGMENTATIONS.PROB_CROP,
            prob_flip_lr=config.DATASET.AUGMENTATIONS.PROB_FLIP_LR,
            prob_flip_ud=config.DATASET.AUGMENTATIONS.PROB_FLIP_UD,
            prob_rot90=config.DATASET.AUGMENTATIONS.PROB_ROT90,
        )
        VAL_DICT = dict(
            split_name=config.DATASET.SPLIT.VAL.split(','),
            batch_size=config.SOLVER.BATCH_SIZE.VAL * num_replicas,
            input_size=config.IMAGE.SIZE.VAL, 
            should_augment=False,
            should_shuffle=False,
        )
        TEST_DICT = dict(
            split_name=config.DATASET.SPLIT.TEST.split(','),
            batch_size=config.SOLVER.BATCH_SIZE.TEST,
            input_size=config.IMAGE.SIZE.TEST, 
            should_augment=False,
            should_shuffle=False,
        )
        DICT = dict(train=TRAIN_DICT, val=VAL_DICT, test=TEST_DICT)
        super(DatasetVITPCMDistributed, self).__init__(
            dataset_name=config.DATASET.NAME.split(','),
            dataset_dir=config.DATASET.ROOT.split(','), 
            patch_size=config.MODEL.PATCH_SIZE,
            num_workers=config.DATALOADER.NUM_WORKERS,
            replicas=num_replicas, **DICT.get(set_name)
        )