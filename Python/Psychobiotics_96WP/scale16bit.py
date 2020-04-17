#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:05:23 2020

@author: sm5911
"""
#%% Imports

import cv2 #os, glob
import numpy as np
#from matplotlib import pyplot as plt
from PIL import Image
from helper import lookforfiles

#%% Variables

path = "/Users/sm5911/Documents/fluorescence_data_local_copy_focussed/"

upper_thresh = 99.9
lower_thresh = 0.1

#%% Main

#filelist = glob.glob(os.path.join(path, "*/*/*.tif"))
filelist = lookforfiles(root_dir=path, regex=".tif$")

arr_uppers = np.zeros(len(filelist))
arr_lowers = np.zeros(len(filelist))

for i, file in enumerate(filelist):
    # read image and convert to greyscale
    im = np.array(Image.open(file)) # .convert('L')
        
    upper_percentile, lower_percentile = np.percentile(im, (upper_thresh, lower_thresh))
    
    arr_uppers[i], arr_lowers[i] = upper_percentile, lower_percentile

# TODO: use medians here, and clip
max_upper = arr_uppers.max()
min_lower = arr_lowers.min()

#%% Test

img16bit = cv2.imread(file).astype(np.uint16)

# Apply linear stretch - set min to 0, and max to 255
img8bit = (img16bit.astype(float) - min_lower) * (255/(max_upper-min_lower))

im8 = (im.astype(float) - min_lower) * (255/(max_upper-min_lower))

#cv2.resize?
#cv2.imshow?
