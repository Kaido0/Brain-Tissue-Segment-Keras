import os
from glob import glob
import matplotlib.pyplot as plt
from os import listdir
from PIL import Image as PImage
import numpy as np
import pandas as pd
from scipy import ndimage, misc
from PIL import Image

from skimage.io import imread

def load_train_data(trainpath):
    ValImage=np.load(trainpath+'valImage256.npy')
    ValLabel=np.load(trainpath+'valLabel256.npy')
    TrainImage=np.load(trainpath+'trainImage256.npy')
    TrainLabel=np.load(trainpath+'trainLabel256.npy')
     
    return TrainImage,TrainLabel,ValImage,ValLabel

def load_test_data(testpath):
    TestImage=np.load(testpath+'test_x.npy')
    TestLabel=np.load(testpath+'test_y.npy')
    return TestImage,TestLabel