'''
=================
traditional unet
=================
get_unet_default
get_unet_reduced
get_unet_extended
get_unet_dense
get_unet_res

'''
from __future__ import print_function
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
session=tf.Session(config=config)
# import packages
from functools import partial
import os
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import callbacks
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

# 2D U-net depth=5
def get_unet_default():
    metrics = dice_coef
    include_label_wise_dice_coefficients = True;
    inputs = Input((patch_size, patch_size, 1))
    conv1 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(BASE*8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(BASE*4, (2, 2),strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(BASE*2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(BASE, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    
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

# 2D U-net depth=4
def get_unet_reduced():
    metrics = dice_coef
    include_label_wise_dice_coefficients = True;
    inputs = Input((patch_size, patch_size, 1))
    conv1 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(conv4)


    up7 = concatenate([Conv2DTranspose(BASE*4, (2, 2),strides=(2, 2), padding='same')(conv4), conv3], axis=3)
    conv7 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(BASE*2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(BASE, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    #model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    
    if not isinstance(metrics, list):
        metrics = [metrics]

    if include_label_wise_dice_coefficients and num_classes > 1:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(num_classes)]
        if metrics:
            metrics = metrics + label_wise_dice_metrics
        else:
            metrics = label_wise_dice_metrics

    model.compile(optimizer=Adam(lr=1), loss='categorical_crossentropy', metrics=metrics)
    return model

# 2D U-net depth=6
def get_unet_extended():
    metrics = dice_coef
    include_label_wise_dice_coefficients = True;
    inputs = Input((patch_size, patch_size, 1))
    conv1 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    
    conv5_extend = Conv2D(BASE*32, (3, 3), activation='relu', padding='same')(pool5)
    conv5_extend = Conv2D(BASE*32, (3, 3), activation='relu', padding='same')(conv5_extend)

    up6_extend = concatenate([Conv2DTranspose(BASE*16, (2, 2), strides=(2, 2), padding='same')(conv5_extend), conv5], axis=3)
    conv6_extend = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(up6_extend)
    conv6_extend = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(conv6_extend)
    
    up6 = concatenate([Conv2DTranspose(BASE*8, (2, 2), strides=(2, 2), padding='same')(conv6_extend), conv4], axis=3)
    conv6 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(BASE*4, (2, 2),strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(BASE*2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(BASE, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    
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
    
def get_unet_res():
    metrics = dice_coef
    include_label_wise_dice_coefficients = True;
    inputs = Input((patch_size, patch_size, 1))
    conv1 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(conv1)
    conc1 = concatenate([inputs, conv1], axis=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conc1)

    conv2 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(conv2)
    conc2 = concatenate([pool1, conv2], axis=3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conc2)

    conv3 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(conv3)
    conc3 = concatenate([pool2, conv3], axis=3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conc3)

    conv4 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(conv4)
    conc4 = concatenate([pool3, conv4], axis=3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conc4)

    conv5 = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(conv5)
    conc5 = concatenate([pool4, conv5], axis=3)

    up6 = concatenate([Conv2DTranspose(BASE*8, (2, 2), strides=(2, 2), padding='same')(conc5), conv4], axis=3)
    conv6 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(conv6)
    conc6 = concatenate([up6, conv6], axis=3)

    up7 = concatenate([Conv2DTranspose(BASE*4, (2, 2), strides=(2, 2), padding='same')(conc6), conv3], axis=3)
    conv7 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(conv7)
    conc7 = concatenate([up7, conv7], axis=3)

    up8 = concatenate([Conv2DTranspose(BASE*2, (2, 2), strides=(2, 2), padding='same')(conc7), conv2], axis=3)
    conv8 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(conv8)
    conc8 = concatenate([up8, conv8], axis=3)

    up9 = concatenate([Conv2DTranspose(BASE, (2, 2), strides=(2, 2), padding='same')(conc8), conv1], axis=3)
    conv9 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(conv9)
    conc9 = concatenate([up9, conv9], axis=3)

    conv10 = Conv2D(num_classes, (1, 1), activation='sigmoid')(conc9)

    model = Model(inputs=[inputs], outputs=[conv10])
    
    if not isinstance(metrics, list):
        metrics = [metrics]

    if include_label_wise_dice_coefficients and num_classes > 1:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(num_classes)]
        if metrics:
            metrics = metrics + label_wise_dice_metrics
        else:
            metrics = label_wise_dice_metrics
            
    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=metrics)
    return model

def get_unet_dense():
    metrics = dice_coef
    include_label_wise_dice_coefficients = True;
    inputs = Input((patch_size, patch_size, 1))
    conv11 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(inputs)
    conv11 = BatchNormalization()(conv11)
    conc11 = concatenate([inputs, conv11], axis=3)
    conv12 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(conc11)
    conv12 = BatchNormalization()(conv12)
    conc12 = concatenate([inputs, conv12], axis=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conc12)

    conv21 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(pool1)
    conv21 = BatchNormalization()(conv21)
    conc21 = concatenate([pool1, conv21], axis=3)
    conv22 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(conc21)
    conv22 = BatchNormalization()(conv22)
    conc22 = concatenate([pool1, conv22], axis=3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conc22)

    conv31 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(pool2)
    conv31 = BatchNormalization()(conv31)
    conc31 = concatenate([pool2, conv31], axis=3)
    conv32 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(conc31)
    conv32 = BatchNormalization()(conv32)
    conc32 = concatenate([pool2, conv32], axis=3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conc32)

    conv41 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(pool3)
    conv41 = BatchNormalization()(conv41)
    conc41 = concatenate([pool3, conv41], axis=3)
    conv42 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(conc41)
    conv42 = BatchNormalization()(conv42)
    conc42 = concatenate([pool3, conv42], axis=3)
    '''
    pool4 = MaxPooling2D(pool_size=(2, 2))(conc42)

    conv51 = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(pool4)
    conv51 = BatchNormalization()(conv51)
    conc51 = concatenate([pool4, conv51], axis=3)
    conv52 = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(conc51)
    conv52 = BatchNormalization()(conv52)
    conc52 = concatenate([pool4, conv52], axis=3)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conc52), conc42], axis=3)
    conv61 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(up6)
    conv61 = BatchNormalization()(conv61)
    conc61 = concatenate([up6, conv61], axis=3)
    conv62 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(conc61)
    conv62 = BatchNormalization()(conv62)
    conc62 = concatenate([up6, conv62], axis=3)
    '''
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conc42), conv32], axis=3)
    conv71 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(up7)
    conv71 = BatchNormalization()(conv71)
    conc71 = concatenate([up7, conv71], axis=3)
    conv72 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(conc71)
    conv72 = BatchNormalization()(conv72)
    conc72 = concatenate([up7, conv72], axis=3)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conc72), conv22], axis=3)
    conv81 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(up8)
    conv81 = BatchNormalization()(conv81)
    conc81 = concatenate([up8, conv81], axis=3)
    conv82 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(conc81)
    conv82 = BatchNormalization()(conv82)
    conc82 = concatenate([up8, conv82], axis=3)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conc82), conv12], axis=3)
    conv91 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(up9)
    conv91 = BatchNormalization()(conv91)
    conc91 = concatenate([up9, conv91], axis=3)
    conv92 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(conc91)
    conv92 = BatchNormalization()(conv92)
    conc92 = concatenate([up9, conv92], axis=3)
    
    conv10 = Conv2D(num_classes, (1, 1), activation='softmax')(conc92)

    model = Model(inputs=[inputs], outputs=[conv10])
    
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
