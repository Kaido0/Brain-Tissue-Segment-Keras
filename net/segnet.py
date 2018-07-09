from __future__ import print_function
import tensorflow as tf
# import packages
from functools import partial

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
session=tf.Session(config=config)
import numpy as np
from keras.layers import Input
from keras.layers.core import Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import np_utils
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Layer

# import configurations
import configs
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

# init configs
image_rows = configs.VOLUME_ROWS
image_cols = configs.VOLUME_COLS
image_depth = configs.VOLUME_DEPS
num_classes = configs.NUM_CLASSES

# patch extraction parameters
patch_size = configs.PATCH_SIZE
BASE = configs.BASE
smooth = configs.SMOOTH

# compute dsc
def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)
'''
# proposed loss function
def dice_coef_loss(y_true, y_pred):
    distance = 0
    for label_index in range(num_classes):
        dice_coef_class = dice_coef(y_true[:,:,:,label_index], y_pred[:, :,:,label_index])
        distance = 1 - dice_coef_class + distance
    return distance
'''
# dsc per class
def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coef(y_true[:,:,:,label_index], y_pred[:, :,:,label_index])

# get label dsc
def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f

def segnet(classes=num_classes):

    metrics = dice_coef
    include_label_wise_dice_coefficients = True;
    img_input = Input((patch_size, patch_size, 1))

    x = img_input
    # Encoder
    x = Convolution2D(64, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(128, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(256, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(512, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # Decoder
    x = Convolution2D(512, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(256, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(128, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(64, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Convolution2D(num_classes, (1, 1), activation='softmax')(x)

    model = Model(img_input, x)

    if not isinstance(metrics, list):
        metrics = [metrics]
    
    if include_label_wise_dice_coefficients and num_classes > 1:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(num_classes)]
        if metrics:
            metrics = metrics + label_wise_dice_metrics
        else:
            metrics = label_wise_dice_metrics
             
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=metrics)

    return model