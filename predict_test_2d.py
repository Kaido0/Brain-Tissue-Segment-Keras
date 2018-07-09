# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 00:07:41 2017

@author: Vu Hoang Minh
"""

from __future__ import print_function

# import packages
from functools import partial
import os, time
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import callbacks
from keras import backend as K
from keras.utils import plot_model
import nibabel as nib
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import extract_patches

# import load data
from data_handling_2d_patch import load_train_data
from net.unet import *
from net.res_unet import *
from net.fcn import *
from net.squeeze_unet import *
from net.segnet import *

# import configurations
import configs

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

image_type = configs.IMAGE_TYPE

# init configs
image_rows = configs.VOLUME_ROWS
image_cols = configs.VOLUME_COLS
image_depth = configs.VOLUME_DEPS
num_classes = configs.NUM_CLASSES

# patch extraction parameters
patch_size = configs.PATCH_SIZE
BASE = configs.BASE
smooth = configs.SMOOTH
nb_epochs  = configs.NUM_EPOCHS
batch_size  = configs.BATCH_SIZE
unet_model_type = configs.MODEL
extraction_step = 1

extraction_reconstruct_step = configs.extraction_reconstruct_step

checkpoint_filename=unet_model_type + '_model_last.h5'

print('weight file: ', checkpoint_filename)

# for each slice estract patches and stack
def create_slice_testing(slice_number, img_dir_name):
    # empty matrix to hold patches
    patches_training_imgs_2d = np.empty(shape=[0, patch_size, patch_size], dtype='int16')
    patches_training_gtruth_2d = np.empty(shape=[0, patch_size, patch_size, num_classes], dtype='int16')
    
    img_data=np.load(img_dir_name+'test_x.npy')
    img_data=img_data[0]
    img_mask_data=np.load(img_dir_name+'test_mask.npy')
    img_mask_data=img_mask_data[0]
    
    patches_training_imgs_2d_temp = np.empty(shape=[0, patch_size, patch_size], dtype='int16')
    patches_training_gtruth_2d_temp = np.empty(shape=[0, patch_size, patch_size, num_classes], dtype='int16')

    rows = []; cols = []
    if np.count_nonzero(img_mask_data[slice_number, :, :]) and np.count_nonzero(img_data[slice_number,:, :]):
        # extract patches of the jth volume image
        imgs_patches, rows, cols = extract_2d_patches_one_slice(img_data[slice_number,:, :],
                                                                img_mask_data[slice_number,:, :])

        # update database
        patches_training_imgs_2d_temp = np.append(patches_training_imgs_2d_temp, imgs_patches, axis=0)

    patches_training_imgs_2d = np.append(patches_training_imgs_2d, patches_training_imgs_2d_temp, axis=0)

    X = patches_training_imgs_2d.shape
    Y = patches_training_gtruth_2d.shape

    # convert to single precision
    patches_training_imgs_2d = patches_training_imgs_2d.astype('float32')
    patches_training_imgs_2d = np.expand_dims(patches_training_imgs_2d, axis=3)

    S = patches_training_imgs_2d.shape
    label_predicted = np.zeros((img_data.shape[0], img_data.shape[1]), dtype=np.uint8)

    return label_predicted, patches_training_imgs_2d, rows, cols


# extract patches in one slice
def extract_2d_patches_one_slice(img_data, mask_data):
    patch_shape = (patch_size, patch_size)

    # empty matrix to hold patches
    imgs_patches_per_slice = np.empty(shape=[0, patch_size, patch_size], dtype='int16')

    img_patches = extract_patches(img_data, patch_shape, extraction_reconstruct_step)
    mask_patches = extract_patches(mask_data, patch_shape, extraction_reconstruct_step)
    print(mask_patches.shape)
    Sum = np.sum(mask_patches, axis=(2,3))
    rows, cols = np.nonzero(Sum)
   
    N = len(rows)
    # select non-zero patches index
    selected_img_patches = img_patches[rows, cols, :, :]

    # update database
    imgs_patches_per_slice = np.append(imgs_patches_per_slice, selected_img_patches, axis=0)
    return imgs_patches_per_slice, rows, cols

# write predicted label to the final result
def write_slice_predict(imgs_valid_predict, rows, cols):
    label_predicted_filled = np.zeros((image_rows, image_cols, num_classes))
    label_final = np.zeros((image_rows, image_cols))

    Count = len(rows)
    count_write = len(rows)

    for index in range(0, len(rows)):
        row = rows[index]; col = cols[index]
        start_row = row * extraction_reconstruct_step
        start_col = col * extraction_reconstruct_step
        patch_volume = imgs_valid_predict[index, :, :, :]
        for i in range(0, patch_size):
            for j in range(0, patch_size):
                prob_class0_new = patch_volume[i][j][0]
                prob_class1_new = patch_volume[i][j][1]
                prob_class2_new = patch_volume[i][j][2]
                prob_class3_new = patch_volume[i][j][3]

                label_predicted_filled[start_row + i][start_col + j][0] = prob_class0_new
                label_predicted_filled[start_row + i][start_col + j][1] = prob_class1_new
                label_predicted_filled[start_row + i][start_col + j][2] = prob_class2_new
                label_predicted_filled[start_row + i][start_col + j][3] = prob_class3_new
    
    for i in range(0, 512):
        for j in range(0, 512):
                prob_class0 = label_predicted_filled[i][j][0]
                prob_class1 = label_predicted_filled[i][j][1]
                prob_class2 = label_predicted_filled[i][j][2]
                prob_class3 = label_predicted_filled[i][j][3]

                prob_max = max(prob_class0, prob_class1, prob_class2, prob_class3)
                if prob_class0 == prob_max:
                    label_final[i][j] = 0
                elif prob_class1 == prob_max:
                    label_final[i][j] = 1
                elif prob_class2 == prob_max:
                    label_final[i][j] = 2
                else:
                    label_final[i][j] = 3

    print('Number of processed patches: ', count_write)
    print('Number of extracted patches: ', Count)
    return label_final

# predict function
def predict(img_dir_name):
    print(unet_model_type+'.'*30)
    if unet_model_type == 'default':
        model = get_unet_default()
    elif unet_model_type == 'reduced':
        model = get_unet_reduced()
    elif unet_model_type == 'extended':
        model = get_unet_extended()  
    elif unet_model_type=='res':
        #model = build_res_unet()  
        model = get_unet_res()
    elif unet_model_type=='dense':
        model = get_unet_dense()  
    elif unet_model_type=='fcn':
        model = fcn_8s()  
    elif unet_model_type=='senet':
        model = SqueezeUNet() 
    elif unet_model_type=='segnet':
        model = segnet()
        
        
    checkpoint_filepath = 'outputs/' + checkpoint_filename
    model.load_weights(checkpoint_filepath)  
    model.summary()
    
    SegmentedVolume = np.zeros((image_depth,image_rows,image_cols))

    img_mask_data=np.load('npy_data/test_mask.npy')
    img_mask_data=img_mask_data[0]
    res=[]
    
	# for each slice, extract patches and predict
    for iSlice in range(26,36):
        print('Now predicting '+str(iSlice)+' layer.'+'.'*10)
        mask = img_mask_data[iSlice]

        if np.sum(mask, axis=(0,1))>0:
            print('-' * 30)
            print('Slice number: ', iSlice)
            label_predicted, patches_training_imgs_2d, rows, cols = create_slice_testing(iSlice, img_dir_name)
            imgs_valid_predict = model.predict(patches_training_imgs_2d)
            label_predicted_filled = write_slice_predict(imgs_valid_predict, rows, cols)
            
            for i in range(0, SegmentedVolume.shape[1]):
                for j in range(0, SegmentedVolume.shape[2]):
                        if img_mask_data.item((iSlice,i, j)) == 1:
                            SegmentedVolume.itemset((iSlice,i,j), label_predicted_filled.item((i, j)))
                        else:
                            label_predicted_filled.itemset((i, j), 0)
        print(np.unique(SegmentedVolume))
        data = SegmentedVolume
        kkk=np.array(data)

        print('saving test image layer='+str(iSlice)+'.'*20)
        misc.imsave('predict/'+unet_model_type+'_'+str(iSlice)+"layer_test.bmp",kkk[iSlice]*80)
        print ('saving done!')
        
        print('calculating dice...'+'.'*20)
        test_label=np.load(img_dir_name+'test_y.npy')
        test_label=test_label[0,iSlice]  
        
        dice_gm,dice_wm,dice_csf=dice(kkk[iSlice],test_label)
        
        b=[iSlice,dice_gm,dice_wm,dice_csf]
        res.append(b)
        '''
        plt.imshow(kkk[iSlice],cmap='gray')
        print('saving test_label image layer='+str(iSlice)+'.'*20)
        misc.imsave('predict/'+unet_model_type+'_'+str(iSlice)+"layer_test_label.bmp",test_label)
        print ('saving done!')
        '''
    
    head=['iSlice','dice_gm','dice_wm','dice_csf']
    dataframe = pd.DataFrame(res,columns=head)
    print(dataframe.shape)
    dataframe.to_csv("predict/"+unet_model_type+"_result.csv")
    

def dice(gt,testlabel): #(2341,1)
    gt=gt.flatten()
    gt=gt.reshape(len(gt),1)
    
    testlabel=testlabel.flatten()
    testlabel=testlabel.reshape(len(testlabel),1)
    
    a=np.sum(gt[:,0]==1)
    b=np.sum(testlabel[:,0]==1)
    m1g=np.where(gt[:,0]==1,1,0)
    m1l=np.where(testlabel[:,0]==1,1,0)
    m1=m1g+m1l
    c1=np.sum(m1==2)
    dice_gm=c1*2/(a+b)
    
    a=np.sum(gt[:,0]==2)
    b=np.sum(testlabel[:,0]==2)
    m2g=np.where(gt[:,0]==2,1,0)
    m2l=np.where(testlabel[:,0]==2,1,0)
    m2=m2g+m2l
    c2=np.sum(m2==2)
    dice_wm=c2*2/(a+b)
    
    a=np.sum(gt[:,0]==3)
    b=np.sum(testlabel[:,0]==3)
    m3g=np.where(gt[:,0]==3,1,0)
    m3l=np.where(testlabel[:,0]==3,1,0)
    m3=m3g+m3l
    c3=np.sum(m3==2)
    dice_csf=c3*2/(a+b)
    
   # total=np.sum(gt)
   # dice_total=(c1*2+c2*2+c3*2)/total
    print("dice_gm:"+str(dice_gm))
    print("dice_wm:"+str(dice_wm))
    print("dice_csf:"+str(dice_csf))
  #  print("dice_total:"+str(dice_total))
    return dice_gm,dice_wm,dice_csf


# main
if __name__ == '__main__':
    # folder to hold outputs
    if 'predict' not in os.listdir(os.curdir):
        os.mkdir('predict')  
    img_dir_name="npy_data/"

    print('*' * 50)
    predict(img_dir_name)
    
