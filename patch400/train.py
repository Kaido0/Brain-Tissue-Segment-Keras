import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import pickle
from keras.utils import plot_model
from keras.optimizers import SGD, Adam, Nadam
from keras.callbacks import *
from keras.objectives import *
from keras.metrics import binary_accuracy
from keras.models import load_model
import keras.backend as K
from model import SegModel
from dataLoader import load_train_data,load_test_data
from keras.preprocessing.image import *
from keras.utils.np_utils import to_categorical
#import keras.utils.visualize_util as vis_util
import argparse
from keras import callbacks
from keras import optimizers
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as KTF
from PIL import Image
from scipy import misc
       
def train(opt):  

    input_size=(opt.row,opt.col,opt.ch)
    train_img,train_mask,val_img,val_mask=load_train_data('train_256/')
    SegM=SegModel(input_size,opt.classes)
    model=SegM.model
    model.summary() 
    plot_model(model,to_file='PDCNet.png',show_shapes=True)       
    tensorboard = TensorBoard()
    print('training starting..')
    
    
    if 'outputs' not in os.listdir(os.curdir):
        os.mkdir('outputs')
    log_filename = 'outputs/' + 'PDC_model_train.csv' 
    csv_log = callbacks.CSVLogger(log_filename, separator=',', append=True)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=opt.PATIENCE, verbose=0, mode='min')
    checkpoint_filepath = 'outputs/' +"PDCNet_best_weight_model_{epoch:03d}_{val_loss:.4f}.h5"
    checkpoint = callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [tensorboard,csv_log, early_stopping, checkpoint]

    _adam=optimizers.Adam(lr=opt.lr, beta_1=0.9, beta_2=0.999, decay=0.0)
    model.compile(loss='categorical_crossentropy',optimizer = _adam,metrics=['accuracy'])


    
    history=model.fit(train_img,train_mask,batch_size=opt.batch_size, nb_epoch=opt.epochs,verbose=1,\
    validation_data=(val_img,val_mask),shuffle=True, callbacks=callbacks_list)
    
    print('Saving model...')
    model_name = 'outputs/' +'PDCmodel_last.h5'
    model.save(model_name)

    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper left') 
    plt.savefig(unet_model_type+'_loss.png')
    plt.show()
    

def predict(opt):
    if 'predict' not in os.listdir(os.curdir):
        os.mkdir('predict')
    
    test_img,test_mask=load_test_data('npy_data/')
    
    #test_img=test_img[:,:,70:test_img.shape[2]-70,70:test_img.shape[3]-70]
    #test_mask=test_mask[:,:,68:test_mask.shape[2]-68,68:test_mask.shape[3]-68]

    input_size=(test_img.shape[2],test_img.shape[3],opt.ch)
    SegM=SegModel(input_size,opt.classes)
    model=SegM.model
    checkpoint_filepath = 'outputs/PDCNet.h5'
    model.load_weights(checkpoint_filepath)  
#   model.summary()
    for iSlice in range(26,36):
        print('Now predicting '+str(iSlice)+' layer.'+'.'*10)
        new_test = np.expand_dims(test_img[0,iSlice], axis=3)
        new_test =np.expand_dims(new_test,axis=0)
        img_predict = model.predict(new_test)
        misc.imsave('predict/'+'PDCNet_'+str(iSlice)+"layer_2prob_test.bmp",img_predict[0,:,:,1]*80)
        misc.imsave('predict/'+'PDCNet_'+str(iSlice)+"layer_3prob_test.bmp",img_predict[0,:,:,2]*80)
        misc.imsave('predict/'+'PDCNet_'+str(iSlice)+"layer_4prob_test.bmp",img_predict[0,:,:,3]*80)
        label_final=result(img_predict[0],opt)
        misc.imsave('predict/'+'PDCNet_'+str(iSlice)+"layer_final_test.bmp",label_final*80)
        misc.imsave('predict/'+str(iSlice)+"layer_label.bmp",test_mask[0,iSlice])
   
        dice_gm,dice_wm,dice_csf=dice(label_final,test_mask[0,iSlice])
        print('End predicting '+str(iSlice)+' layer.'+'.'*10)
    

def result(img_prob,opt):
    label_final=np.zeros((img_prob.shape[0],img_prob.shape[0]))
    for i in range(0,img_prob.shape[0]):
        for j in range(0,img_prob.shape[0]):
            prob_class0 = img_prob[i][j][0]
            prob_class1 = img_prob[i][j][1]
            prob_class2 = img_prob[i][j][2]
            prob_class3 = img_prob[i][j][3]

            prob_max = max(prob_class0, prob_class1, prob_class2, prob_class3)
            if prob_class0 == prob_max:
                label_final[i][j] = 0
            elif prob_class1 == prob_max:
                label_final[i][j] = 1
            elif prob_class2 == prob_max:
                label_final[i][j] = 2
            else:
                label_final[i][j] = 3   
    return label_final         

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

    print("dice_gm:"+str(dice_gm))
    print("dice_wm:"+str(dice_wm))
    print("dice_csf:"+str(dice_csf))
    return dice_gm,dice_wm,dice_csf

           
if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--chekp', type=str, default='')
    parser.add_argument('--row', type=int, default=256)
    parser.add_argument('--col', type=int, default=256)
    parser.add_argument('--ch', type=int, default=1)
    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--PATIENCE', type=int, default=20)

    opt = parser.parse_args()
    #train(opt)            
    predict(opt)
    print('Done!')