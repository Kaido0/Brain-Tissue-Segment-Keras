# -*- coding: utf-8 -*-
"""
Created on 2018
@author: ZhangWeiting
"""
from __future__ import print_function
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
session=tf.Session(config=config)
from functools import partial
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras import callbacks
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
from keras.utils import plot_model
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as KTF

import configs
from data_handling_2d_patch import load_train_data
# import model
from net.unet import get_unet_default,get_unet_reduced,get_unet_extended,get_unet_dense,get_unet_res
from net.res_unet import build_res_unet
from net.fcn import fcn_8s
from net.squeeze_unet import SqueezeUNet
from net.segnet import segnet

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
image_type = configs.IMAGE_TYPE

# patch extraction parameters
nb_epochs  = configs.NUM_EPOCHS
batch_size  = configs.BATCH_SIZE
unet_model_type = configs.MODEL
PATIENCE = configs.PATIENCE


def train():
    print('Loading data...')
    imgs_train, imgs_gtruth_train ,imgs_val, imgs_gtruth_val= load_train_data()
    print('train shape:',imgs_train.shape)
    print('val shape:',imgs_val.shape) 
    print('Load data done!!!')
    print('-'*30)
    print('Creating and compiling model...')
    
    print('model type = '+unet_model_type)
    if unet_model_type == 'default':
        model = get_unet_default()
    elif unet_model_type == 'reduced':
        model = get_unet_reduced()
    elif unet_model_type == 'extended':
        model = get_unet_extended()  
    elif unet_model_type=='res':
        model = build_res_unet()  
        #model = get_unet_res()
    elif unet_model_type=='dense':
        model = get_unet_dense()  
    elif unet_model_type=='fcn':
        model = fcn_8s()  
    elif unet_model_type=='senet':
        model = SqueezeUNet()
    elif unet_model_type=='segnet':
        model = segnet()
    model.summary() 
    plot_model(model,to_file=unet_model_type+'_model.png',show_shapes=True)       
    tensorboard = TensorBoard()

    #============================================================================
    print('training starting..')
    log_filename = 'outputs/' + unet_model_type +'_model_train.csv' 
    #Callback that streams epoch results to a csv file.
    
    csv_log = callbacks.CSVLogger(log_filename, separator=',', append=True)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=PATIENCE, verbose=0, mode='min')
    checkpoint_filepath = 'outputs/' + unet_model_type +"_best_weight_model_{epoch:03d}_{val_loss:.4f}.h5"
    checkpoint = callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [tensorboard,csv_log, early_stopping, checkpoint]

    #============================================================================
    history=model.fit(imgs_train, imgs_gtruth_train, batch_size=batch_size, nb_epoch=nb_epochs, verbose=1, validation_data=(imgs_val,imgs_gtruth_val), shuffle=True, callbacks=callbacks_list) 
    
    print('Saving model...')
    model_name = 'outputs/' + unet_model_type + '_model_last.h5'
    model.save(model_name)

    print(history.history.keys())
    '''
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    '''
    # summarize history for loss plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('model loss')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper left') 
    plt.savefig(unet_model_type+'_loss.png')
    plt.show()
	
# main
if __name__ == '__main__':
    if 'outputs' not in os.listdir(os.curdir):
        os.mkdir('outputs')   
    train()
