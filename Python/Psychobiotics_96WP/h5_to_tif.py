#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple python script to convert Ilastik HDF5 probability maps into TIF images
compatible with CellProfiler/Ilastik software.

@author: sm5911
@date: 24/04/2020

"""

import sys, tables, cv2 #os
import numpy as np
from pathlib import Path
from tqdm import tqdm

def h5_to_tif(dirpath_h5):
    h5_list = list(dirpath_h5.rglob("*.h5"))
        
    for h5 in tqdm(h5_list):
        # Read file
        with tables.File(h5, mode='r') as fid:
            img = fid.get_node('/exported_data').read()
        #assert (np.sum(img, axis=2) == 1).all()
        
        if len(img.shape) == 5:
            # 'tzcyx' - eg. ilastik object probabilities
            img = img[0,0,0,:,:]
        elif len(img.shape) == 3:
            # 'cyx' - eg. ilastik pixel probabilities
            img = img[:,:,0]
        assert img.ndim == 2
        
        # Rescale image
        img *= 65535.0
        img = img.astype(np.uint16)
        
        # Save files in new folder named 'tif_images'
        outpath = Path(str(h5.parent)) / "tif_images" / h5.with_suffix('.tif').name
        outpath.parent.mkdir(exist_ok=True, parents=True)
        
        cv2.imwrite(str(outpath), img)
        
if __name__ == "__main__":
    if len(sys.argv) > 1:
        dirpath_h5 = sys.argv[1]
    else:
        # Default
        dirpath_h5 = "/Users/sm5911/Documents/PanGenome/results/object_probabilities"
 
    h5_to_tif(Path(dirpath_h5))
    print("/nDone!")