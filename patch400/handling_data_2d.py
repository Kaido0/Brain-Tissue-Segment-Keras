from __future__ import print_function

# import packages
import time, os, random
import numpy as np
import nibabel as nib
from sklearn.feature_extraction.image import extract_patches
# import configurations
import configs

# init configs
num_classes = configs.NUM_CLASSES
patch_size = configs.PATCH_SIZE

# create npy data
def create_npy_data(is_train,trainpath):
    train_x=np.load(trainpath+'train256.npy')
    train_y=np.load(trainpath+'label256.npy')
    
    patch_train = np.empty(shape=[0,patch_size,patch_size], dtype='int16')
    patch_train_label = np.empty(shape=[0,patch_size,patch_size,num_classes], dtype='int16')
    
    if is_train:
        vol=[0,1,2,3,4]
        print('Ready to Train'+'.'*10)
    else:
        vol=[5]
        print('Ready to Val'+'.'*10)
    j=0
    for i in vol:
        print('Processing: volume {0} / 6 volume',(j+1))
        patch_train=np.append(patch_train,train_x[i],axis=0)
        temp=separate_labels(train_y[i])
        patch_train_label=np.append(patch_train_label,temp,axis=0)

        j+=1
        X  = patch_train.shape
        Y  = patch_train_label.shape
        print('shape im: [{0} , {1} , {2}]'.format(X[0], X[1], X[2])) 
        print('shape gt: [{0} , {1} , {2}, {3}]'.format(Y[0], Y[1], Y[2], Y[3]))

    patch_train = patch_train.astype('float32')
    patch_train = np.expand_dims(patch_train, axis=3)
    print(patch_train.shape)
    
    # save train or validation
    if is_train:
        print('NOW tarin data'+'.'*20)
        np.save('train_256/trainImage'+str(patch_size)+'.npy', patch_train)
        np.save('train_256/trainLabel'+str(patch_size)+'.npy', patch_train_label)
    else:
        print('NOW:val data'+'.'*20)
        np.save('train_256/valImage'+str(patch_size)+'.npy', patch_train)
        np.save('train_256/valLabel'+str(patch_size)+'.npy', patch_train_label)        
    print('npy done!!!')


# separate labels
def separate_labels(patch_3d_volume):
    result =np.empty(shape=[0,patch_size,patch_size,num_classes], dtype='int16')
    N = patch_3d_volume.shape[0]
    # for each class do:
    for V in range(N):
        V_patch = patch_3d_volume[V , :, :]
        U  = np.unique(V_patch)
        unique_values = list(U)
        result_v =np.empty(shape=[patch_size,patch_size,0], dtype='int16')
        for label in range(0,4):
            if label in unique_values:
                im_patch = V_patch == label
                im_patch = im_patch*1
            else:
                im_patch = np.zeros((V_patch.shape))
             
            im_patch = np.expand_dims(im_patch, axis=2) 
            result_v  = np.append(result_v,im_patch, axis=2)
        result_v = np.expand_dims(result_v, axis=0) 
        result  = np.append(result,result_v, axis=0)
    return result


def dataloader(trainpath,testpath):
    TestImage=np.load(testpath+'test_x.npy')
    TestMask=np.load(testpath+'test_y.npy')
    TrainImage=np.load(trainpath+'train256.npy')
    TrainLabel=np.load(trainpath+'label256.npy')
    return TrainImage,TrainLabel,TestImage,TestMask

if __name__ == '__main__':
    '''
    if '2d_patch' not in os.listdir(os.curdir):
        os.mkdir('2d_patch')
    '''
    create_npy_data(0,'train_256/')
    exit()
    trainx,trainy,testx,testy=dataloader('train_256/','npy_data/')  
