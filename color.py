from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import os
path='predict/'


def color(img_name):
    img=misc.imread(path+img_name)
    color=np.ones([img.shape[0],img.shape[1],3])
    color[img==0]=[0,0,0]
    color[img==85]=[255,0,0]
    color[img==170]=[0,255,0]
    color[img==255]=[0,0,255]
    misc.imsave("predict/"+img_name.split('.bmp')[0]+".png",color)
    
if __name__=='__main__':
    list=os.listdir(path)
    for img in list:
      color(img)