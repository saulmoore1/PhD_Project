#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract training and test images from directory of image files 

@author: sm5911
@date: 01/04/2020

"""

#%% Imports

import time
import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copyfile
from pathlib import Path
from tqdm import tqdm

#%% Main

if __name__ == "__main__":
    
    # input params
    image_root_dir = Path("/Users/sm5911/Documents/PanGenome/data/200827_dev_assay_optimisation_focussed")
    train_outdir = Path("/Users/sm5911/Documents/PanGenome/data/200827_dev_assay_optimisation_training_images")
    test_outdir = Path("/Users/sm5911/Documents/PanGenome/data/200827_dev_assay_optimisation_test_images")
    
    # sample train/test images from image_root_dir with train_test_split
    tic = time.time()  
    for fileregex in ["*.tif","*.npy"]:
        image_list = list(image_root_dir.rglob(fileregex))
        
    images = pd.Series(image_list)
    images.name = 'filename'
    training_images, test_images = train_test_split(images,\
                                                    train_size=100,\
                                                    test_size=200,\
                                                    random_state=11092020,\
                                                    shuffle=True,\
                                                    stratify=None)
    # copy training images    
    n = len(training_images)
    print("\nExtracting training images..")
    for imPath in tqdm(sorted(training_images)):
        outPath = train_outdir / imPath.name
        outPath.parent.mkdir(parents=True, exist_ok=True)
        copyfile(imPath, outPath)
        
    # Save filenames of training images
    outPath = train_outdir / "training_image_filenames.csv"
    training_images.name = 'filename'
    training_images.to_csv(outPath, index=False)
    
    # copy test images    
    n = len(test_images)
    print("\nExtracting test images..")
    for imPath in tqdm(sorted(test_images)):
        outPath = test_outdir / imPath.name
        outPath.parent.mkdir(parents=True, exist_ok=True)
        copyfile(imPath, outPath)

    # save filenames of test images
    outPath = test_outdir / "test_image_filenames.csv"
    test_images.name = 'filename'
    test_images.to_csv(outPath, index=False)

    toc=time.time()
    print("Complete! (Time taken: %.2f seconds)" % (toc-tic))
    
