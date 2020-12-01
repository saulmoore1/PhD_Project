#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script to convert '.npy' image files into '.tif' images compatible with
Ilastik software.

@author: sm5911
@date: 11/09/2020

"""

import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def npy_to_tif(dirpath_npy):
    """ Find NPY image files in directory, convert to TIF images, and save 
        in a new directory titled, 'tif_images' """
    npy_list = list(dirpath_npy.rglob('*.npy'))
    
    for img_path in tqdm(npy_list):
        img = np.load(img_path).squeeze()
    
        outdir = img_path.parent / "tif_images"
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / img_path.with_suffix('.tif').name
        
        cv2.imwrite(str(outpath), img)       

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        dirpath_npy = Path(sys.argv[1])
    else:
        dirpath_npy = Path("/Users/sm5911/Documents/PanGenome/data/200827_dev_assay_optimisation_focussed")
        print("WARNING: No directory provided! Using default path: %s" % dirpath_npy)
    
    npy_to_tif(dirpath_npy)
    print("\nDone!")