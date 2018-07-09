from __future__ import print_function
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
session=tf.Session(config=config)
from functools import partial
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation
from keras.layers import BatchNormalization, Add, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras import backend as K

import keras.backend.tensorflow_backend as KTF 
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
    
    
def fcn_8s():

    metrics = dice_coef
    include_label_wise_dice_coefficients = True;
    inputs = Input((patch_size, patch_size, 1))
    
    x = Conv2D(32, (3, 3), activation=None, padding='same', name='block1_conv1')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), activation=None, padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    p1 = x
    # Block 2
    x = Conv2D(64, (3, 3), activation=None, padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), activation=None, padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    p2 = x

    # Block 3
    x = Conv2D(128, (3, 3), activation=None, padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), activation=None, padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), activation=None, padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    p3 = x

    # Block 4
    x = Conv2D(256, (3, 3), activation=None, padding='same', name='block4_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), activation=None, padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), activation=None, padding='same', name='block4_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    p4 = x

    # Block 5
    # x = Conv2D(512, (3, 3), activation=None, padding='same', name='block5_conv1')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Conv2D(512, (3, 3), activation=None, padding='same', name='block5_conv2')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Conv2D(512, (3, 3), activation=None, padding='same', name='block5_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    # p5 = x

    vgg  = Model(inputs , x)
    
    ### pool4 (16,32,256) --> up1 (32,64,256)
    o = p4
    o = Conv2D(256, (5, 5), activation=None, padding='same')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(256, (3, 3), activation=None, padding='same')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), padding='same')(o)

    ### pool3 (32,64,128) --> (32,64,256)
    o2 = p3
    o2 = Conv2D(256, (3, 3), activation=None, padding='same')(o2)
    o2 = BatchNormalization()(o2)
    o2 = Activation('relu')(o2)
    
    ### concat1 [(32,64,256), (32,64,256)]
    o = concatenate([o, o2], axis=3)
    o = Conv2D(256, (3, 3), padding='same')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    ### up2 (32,64,512) --> (64,128,128)
    o = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same')(o)
    o = Conv2D(128, (3, 3), padding='same')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    ### pool2 (64,128,64) --> (64,128,128)
    o2 = p2
    o2 = Conv2D(128, (3, 3), activation=None, padding='same')(o2)
    o2 = BatchNormalization()(o2)
    o2 = Activation('relu')(o2)

    ### concat2 [(64,128,128), (64,128,128)]
    o = concatenate([o, o2], axis=3)
    
    ### up3 (64,128,256) --> (128,256,64)
    o = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding='same')(o)
    o = Conv2D(64, (3, 3), padding='same')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    ### pool1 (128,256,64) --> (128,256,32)
    o2 = p1
    o2 = Conv2D(32, (3, 3), activation=None, padding='same')(o2)
    o2 = BatchNormalization()(o2)
    o2 = Activation('relu')(o2)

    ### concat3 [(128,256,64), (128,256,32)] --> (128,256,32)
    o = concatenate([o, o2], axis=3)
    o = Conv2D(32, (3, 3), padding='same')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    ### up (128,256,32) --> (256,512,32)
    o = Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding='same')(o)
    ### mask out (128,256,32) --> (256,512,3)
    o = Conv2D(num_classes, (3, 3), padding='same')(o)
    o = Activation('softmax')(o)
    
    model = Model(inputs=[inputs], outputs=[o])
    
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