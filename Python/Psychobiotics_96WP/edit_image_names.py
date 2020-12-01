#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

PanGenomeGFP - Edit series names to convert to well names in image metadata

@author: sm5911
@date: 30/04/2020

"""

import sys
import os
import re
import numpy as np
from pathlib import Path

def edit_image_names(image_dir, remove_duplicate_0th=False):
    """ Edit image filenames to replace series ID with the corresponding well ID """
    
    # Look for '.tif' or '.npy' files
    file_list = list(image_dir.rglob("*.tif"))
    file_list.extend(list(image_dir.rglob("*.npy")))
    
    # Create well mapping dictionary
    well_IDs = [(i+str(j+1)) for i in 'ABCDEFGH' for j in range(12)]

    if remove_duplicate_0th:
        series_IDs = np.arange(1,97,1) # series 0 is a duplicate image to be removed
    else:
        series_IDs = np.arange(0,96,1)
        
    well_mapping_dict = {key:value for key, value in zip(series_IDs, well_IDs)}
    
    # Edit image names to change seriesID into wellID
    for file in file_list:
        seriesID = re.findall(r'.*_s([0-9]+?)z.*', str(file))
        assert len(seriesID) <= 1
        
        if len(seriesID) == 1:
            seriesID = int(seriesID[0])
            
            if remove_duplicate_0th and seriesID == 0:
                # Well A1 was imaged twice, so can omit TIF images where series ID == 0
                os.remove(str(file))
                print("\nRemoved initial duplicate image: %s" % file.name)
        
            else:
                try:
                    well_ID = well_mapping_dict[int(seriesID)]
                    
                    newfile = str(file).replace("_s{0}z".format(seriesID),\
                                                "_w{0}z".format(well_ID))
                    #print("Renaming %s as %s" % (str(file), newfile))
                    os.rename(str(file), newfile)
                except Exception as EE:
                    print(EE)

if __name__ == "__main__":
    print("Running script %s" % sys.argv[0])
    if len(sys.argv) > 1:
        image_dir = Path(sys.argv[1]) # str
    else:
        #image_dir = Path('/Users/sm5911/Documents/PanGenomeGFP/data/fluorescence_data_local_copy_focussed/200908_acs-2-GFP_metformin_rep4')
        image_dir = Path("/Users/sm5911/Documents/PanGenome/data/200924_dev_assay_optimisation2_focussed")
        print("WARNING: No directory provided! Using default path: %s" % image_dir)
    if len(sys.argv) > 2:
        remove_duplicate_0th = sys.argv[2].lower() == 'true' # create boolean from string input       
    else:
        # Default is to NOT remove duplicate images when renaming
        print("WARNING: Assuming first series image (s0) is first well of plate (A1)")
        remove_duplicate_0th = False
    
    edit_image_names(image_dir, remove_duplicate_0th=remove_duplicate_0th)
    
    print("\nDone!")