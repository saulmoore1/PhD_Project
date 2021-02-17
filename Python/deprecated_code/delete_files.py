#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 18:09:26 2020

@author: sm5911
"""


import os
from my_helper import lookforfiles

dirpath = "/Users/sm5911/Documents/PanGenomeGFP/ilastik/ilastik_training_images/object_h5"

files2delete = lookforfiles(dirpath, "_table.csv")

for file in files2delete:
    os.remove(file)
    print("Removed: %s" % file)



#%%

dir1 = "/Users/sm5911/Documents/PanGenomeGFP/data/fluorescence_data_local_copy_focussed/200218_acs-2-GFP_metformin_rep1"
dir2 = "/Users/sm5911/Documents/PanGenomeGFP/results/rep1/intensities"

test_images = lookforfiles(dir1, ".tif")
len(test_images)
test_h5 = lookforfiles(dir2, "_intensities_table.csv")
len(test_h5)

names_images = [os.path.basename(file).split(".tif")[0] for file in test_images]
names_h5 = [os.path.basename(file).split("_intensities_table.csv")[0] for file in test_h5]

images2keep_dir1 = [img for img in names_images if img in names_h5]
len(images2keep_dir1)

images2delete_dir2 = [img for img in names_h5 if img not in names_images]
