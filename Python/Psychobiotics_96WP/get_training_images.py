#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract training and test images from directory of image files 

@author: sm5911
@date: 01/04/2020

"""

#%% Imports

import os, glob, time
import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copyfile

#%% Main

if __name__ == "__main__":
    
    #%% Params
    
    image_root_dir = "/Users/sm5911/Documents/fluorescence_data_local_copy_focussed/"
    train_outdir = "/Users/sm5911/Documents/CellProfiler/WormToolBox/data/fluorescence_data_training_images/"
    test_outdir = "/Users/sm5911/Documents/CellProfiler/WormToolBox/data/fluorescence_data_test_images/"
    
    fileregex = "*/*/*.tif"

    #%% Sample images
    
    tic = time.time()   
    image_list = glob.glob(os.path.join(image_root_dir, fileregex), recursive=True)
    images = pd.Series(image_list)
    images.name = 'filename'

    training_images, test_images = train_test_split(images,\
                                                    train_size=200,\
                                                    test_size=800,\
                                                    random_state=42,\
                                                    shuffle=True,\
                                                    stratify=None)
    
    #%% Copy training images    
    n = len(training_images)
    print("\nExtracting training images..")
    for i, imPath in enumerate(sorted(training_images)):
        if (i+1) % 10 == 0:
            print("%d/%d (%.1f%%)" % (i+1, n, ((i+1)/n)*100))
        outPath = imPath.replace(image_root_dir, train_outdir)
        if not os.path.exists(os.path.dirname(outPath)):
            os.makedirs(os.path.dirname(outPath))
        copyfile(imPath, outPath)
        
    # Save filenames of training images
    outPath = os.path.join(train_outdir, "training_image_filenames.csv")
    training_images.name = 'filename'
    training_images.to_csv(outPath, index=False)
    
    #%% Copy test images    
    n = len(test_images)
    print("\nExtracting test images..")
    for i, imPath in enumerate(sorted(test_images)):
        if (i+1) % 10 == 0:
            print("%d/%d (%.1f%%)" % (i+1, n, ((i+1)/n)*100))
        outPath = imPath.replace(image_root_dir, test_outdir)
        if not os.path.exists(os.path.dirname(outPath)):
            os.makedirs(os.path.dirname(outPath))
        copyfile(imPath, outPath)        

    # Save filenames of test images
    outPath = os.path.join(test_outdir, "test_image_filenames.csv")
    test_images.name = 'filename'
    test_images.to_csv(outPath, index=False)

    #%%   
    toc=time.time()
    print("Complete! (Time taken: %.2f seconds)" % (toc-tic))
    
