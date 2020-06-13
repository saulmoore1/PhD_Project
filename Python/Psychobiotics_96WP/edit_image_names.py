#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

PanGenomeGFP - Edit series names to convert to well names in image metadata

@author: sm5911
@date: 30/04/2020

"""

import os, re
import numpy as np
from my_helper import lookforfiles

image_dir = "/Users/sm5911/Documents/PanGenomeGFP/data/fluorescence_data_local_copy_focussed"

file_list = lookforfiles(image_dir, ".tif")

well_IDs = [(i+str(j+1)) for i in 'ABCDEFGH' for j in range(12)]
series_IDs = np.arange(1,192,2)
well_mapping_dict = {key:value for key, value in zip(series_IDs, well_IDs)}

# Edit image names to change seriesID into wellID
for file in file_list:
    print(file)
    seriesID = re.findall(r'.*_s([0-9]+?)z.*', file)
    assert len(seriesID) <= 1
    
    if len(seriesID) == 1:
        seriesID = int(seriesID[0])
        
        # Well A1 was imaged twice, so can omit TIF images where series ID == 0
        if seriesID == 0:
            os.remove(file)
            print("Removed initial duplicate image: %s" % os.path.basename(file))
        
        else:
            try:
                well_ID = well_mapping_dict[int(seriesID)]
                
                newfile = file.replace("_s{0}z".format(seriesID),\
                                       "_w{0}z".format(well_ID))
                print("Renaming %s as %s" % (file, newfile))
                os.rename(file, newfile)
            except Exception as EE:
                print(EE)
