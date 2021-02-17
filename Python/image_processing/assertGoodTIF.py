#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 21:17:43 2020

@author: sm5911
"""


import tifffile

good_img_path = '/Users/sm5911/Documents/PanGenomeGFP/data/fluorescence_data_local_copy_focussed/200218_acs-2-GFP_metformin_rep1/PG1_0mM_GFP_1pm_wA1z1.tif'
bad_img_path = '/Users/sm5911/Documents/PanGenomeGFP/data/fluorescence_data_local_copy/200908_acs-2-GFP_metformin_rep4_focussed/PG6_50mM_GFP/PG6_50mM_GFP_s39z11.tif'

# tifffile.imread
good_img = tifffile.imread(good_img_path)
bad_img = tifffile.imread(bad_img_path)

assert good_img.dtype == bad_img.dtype

assert len(good_img.shape) == len(bad_img.shape)

assert good_img.shape[0] % 2 == good_img.shape[0] % 2 ==\
       good_img.shape[1] % 2 == good_img.shape[1] % 2 == 0
       
assert bad_img.shape[0] % 2 == bad_img.shape[0] % 2 ==\
       bad_img.shape[1] % 2 == bad_img.shape[1] % 2 == 0
       
for img_path in [good_img_path, bad_img_path]:
    with tifffile.TiffFile(img_path) as tif:
        imagej_hyperstack = tif.asarray()
        imagej_metadata = tif.imagej_metadata
        print(imagej_metadata)
        print(imagej_hyperstack.shape)
       