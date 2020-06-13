#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple python script to convert Ilastik HDF5 probability maps into 
CellProfiler-compatible TIF images

@author: sm5911
@date: 24/04/2020

"""

import os, sys, tables, cv2
import numpy as np
from my_helper import lookforfiles
from tqdm import tqdm

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dirpath_ilastik_h5 = sys.argv[1]
    else:
        # Default
        dirpath_ilastik_h5 = "/Users/sm5911/Documents/PanGenomeGFP/results/probabilities/rep3"
    
    h5_list = lookforfiles(dirpath_ilastik_h5, "_Probabilities.h5")
        
    for h5 in tqdm(h5_list):
        
        # Read file
        with tables.File(h5, mode='r') as fid:
            img = fid.get_node('/exported_data').read()
        assert (np.sum(img, axis=2) == 1).all()
        img = img[0,0,0,:,:]
        assert img.ndim == 2
        
        # Rescale image
        img *= 65535.0
        img = img.astype(np.uint16)
        
        # Save files in new folder in within image directory(s)
        fname = os.path.basename(h5)
        outpath = h5.replace(fname, "probabilities_tif/" + fname)
        outpath = outpath.replace(".h5", ".tif")
        
        if not os.path.exists(os.path.dirname(outpath)):
            os.makedirs(os.path.dirname(outpath))
        cv2.imwrite(outpath, img)
