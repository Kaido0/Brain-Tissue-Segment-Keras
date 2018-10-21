'''
Created on 2018
@author:ZhangWeiting
'''
import numpy as np
from scipy import misc
from PIL import Image
import random
import configs
patch_size=configs.PATCH_SIZE
SLICE=range(26,36)

def crop_data(train_path,label_path,image_num=1200):
    new_train=np.zeros
    train=np.load(train_path)
    label=np.load(label_path)
    image_each=image_num//(train.shape[0]*len(SLICE))
    new_train=np.zeros((train.shape[0],image_each*len(SLICE),patch_size,patch_size))
    new_label=np.zeros((train.shape[0],image_each*len(SLICE),patch_size,patch_size))

    for geti in range(train.shape[0]):
        print(geti)
        for iSlice in SLICE:
            image=train[geti,iSlice]
            image_label=label[geti,iSlice]
            w,h=image.shape[0],image.shape[1]
            count=0
            index=0
            while count<image_each:
                random_width=random.randint(0,w-patch_size-1)
                random_height=random.randint(0,h-patch_size-1)
                image_ogi = image[random_height: random_height + patch_size, random_width: random_width + patch_size]
                label_ogi = image_label[random_height: random_height + patch_size, random_width: random_width + patch_size]
                new_train[geti,index:index+image_each]=image_ogi
                new_label[geti,index:index+image_each]=label_ogi
                index=index+image_each
                count+=1
    np.save('train256.npy',new_train)
    np.save('label256.npy',new_label)
            

            
            
            
if __name__=='__main__':
    train_path='npy_data/train_x.npy'
    label_path='npy_data/train_y.npy'
    crop_data(train_path,label_path)
