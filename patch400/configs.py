# -*- coding: utf-8 -*-

# number of classes
NUM_CLASSES = 4
PATCH_SIZE = 400
    
# training configs
UNET_MODEL = 3
if UNET_MODEL==0:
    MODEL = 'default'    
elif UNET_MODEL==1:
    MODEL = 'reduced'
elif UNET_MODEL==2:
    MODEL = 'extended' 
elif UNET_MODEL==3:
    MODEL = 'res' 
elif UNET_MODEL==4:
    MODEL = 'dense'   
elif UNET_MODEL==5:
    MODEL='fcn'  
elif UNET_MODEL==6:
    MODEL='senet'   
elif UNET_MODEL==7:
    MODEL='segnet'
    
BASE = PATCH_SIZE
SMOOTH = 1.
NUM_EPOCHS  = 300
BATCH_SIZE  = 64

extraction_reconstruct_step = PATCH_SIZE

if PATCH_SIZE==64:
    PATIENCE = 10  
elif PATCH_SIZE==32:
    PATIENCE = 20  
elif PATCH_SIZE==16:
    PATIENCE = 10
elif PATCH_SIZE==400:
    PATIENCE = 10

# output
IMAGE_TYPE = '2d_whole_image'    
