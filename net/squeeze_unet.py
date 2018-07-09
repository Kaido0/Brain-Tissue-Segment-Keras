# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
session=tf.Session(config=config)
from functools import partial
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout,Input
from keras.layers import concatenate, Conv2DTranspose, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
import keras.backend.tensorflow_backend as KTF
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



def fire_module(x, fire_id, squeeze=16, expand=64):
    f_name = "fire{0}/{1}"
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(squeeze, (1, 1), activation='relu', padding='same', name=f_name.format(fire_id, "squeeze1x1"))(x)
    x = BatchNormalization(axis=channel_axis)(x)

    left = Conv2D(expand, (1, 1), activation='relu', padding='same', name=f_name.format(fire_id, "expand1x1"))(x)
    right = Conv2D(expand, (3, 3), activation='relu', padding='same', name=f_name.format(fire_id, "expand3x3"))(x)
    x = concatenate([left, right], axis=channel_axis, name=f_name.format(fire_id, "concat"))
    return x


def SqueezeUNet(deconv_ksize=3, dropout=0.5):
    """SqueezeUNet is a implementation based in SqueezeNetv1.1 and unet for semantic segmentation
    :param inputs: input layer.
    :param num_classes: number of classes.
    :param deconv_ksize: (width and height) or integer of the 2D deconvolution window.
    :param dropout: dropout rate
    :param activation: type of activation at the top layer.
    :returns: SqueezeUNet model
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    #channel_axis=-1
    metrics = dice_coef
    include_label_wise_dice_coefficients = True;
    inputs = Input((patch_size, patch_size, 1))

    x01 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu', name='conv1')(inputs)
    x02 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1', padding='same')(x01)

    x03 = fire_module(x02, fire_id=2, squeeze=16, expand=64)
    x04 = fire_module(x03, fire_id=3, squeeze=16, expand=64)
    x05 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3', padding="same")(x04)

    x06 = fire_module(x05, fire_id=4, squeeze=32, expand=128)
    x07 = fire_module(x06, fire_id=5, squeeze=32, expand=128)
    x08 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5', padding="same")(x07)

    x09 = fire_module(x08, fire_id=6, squeeze=48, expand=192)
    x10 = fire_module(x09, fire_id=7, squeeze=48, expand=192)
    x11 = fire_module(x10, fire_id=8, squeeze=64, expand=256)
    x12 = fire_module(x11, fire_id=9, squeeze=64, expand=256)

    if dropout != 0.0:
        x12 = Dropout(dropout)(x12)

    up1 = concatenate([
        Conv2DTranspose(192, deconv_ksize, strides=(1, 1), padding='same')(x12),
        x10,
    ], axis=channel_axis)
    up1 = fire_module(up1, fire_id=10, squeeze=48, expand=192)

    up2 = concatenate([
        Conv2DTranspose(128, deconv_ksize, strides=(1, 1), padding='same')(up1),
        x08,
    ], axis=channel_axis)
    up2 = fire_module(up2, fire_id=11, squeeze=32, expand=128)

    up3 = concatenate([
        Conv2DTranspose(64, deconv_ksize, strides=(2, 2), padding='same')(up2),
        x05,
    ], axis=channel_axis)
    up3 = fire_module(up3, fire_id=12, squeeze=16, expand=64)

    up4 = concatenate([
        Conv2DTranspose(32, deconv_ksize, strides=(2, 2), padding='same')(up3),
        x02,
    ], axis=channel_axis)
    up4 = fire_module(up4, fire_id=13, squeeze=16, expand=32)
    up4 = UpSampling2D(size=(2, 2))(up4)

    x = concatenate([up4, x01], axis=channel_axis)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(num_classes, (1, 1), activation='softmax')(x)

    model = Model(inputs=[inputs], outputs=[x])
    
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
    