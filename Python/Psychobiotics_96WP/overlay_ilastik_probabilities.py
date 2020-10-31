#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay ilastik probability map outlines (TIF) onto raw images (TIF)

@author: sm5911
@date: 14/09/2020

"""

# Imports
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt

# Globals (user-defined)
RAW_TIF_DIR = Path("/Users/sm5911/Documents/PanGenome/data/200924_dev_assay_optimisation2_focussed/tif_images")
PROB_MAP_DIR = Path("/Users/sm5911/Documents/PanGenome/results/200924_dev_assay_optimisation2_results/object_probabilities/tif_images")
SAVE_DIR = Path("/Users/sm5911/Documents/PanGenome/results/200924_dev_assay_optimisation2_results/object_overlays")
 
# Functions
def pair_input_files(RAW_TIF_DIR, PROB_MAP_DIR):
    """ Pair raw image filepaths with associated probability map filepaths """
    
    raw_img_list = list(RAW_TIF_DIR.rglob("*.tif"))
    prob_map_list = list(PROB_MAP_DIR.rglob("*.tif"))
    
    missing_prob_map_path_list = []
    filepaths_tuple_list = []
    for raw_img_path in raw_img_list:
        prob_map_name = raw_img_path.name.replace(".tif",
                                                  "_Object Predictions.tif")
        prob_map_path = PROB_MAP_DIR / prob_map_name
         
        if not prob_map_path in prob_map_list:
            print("WARNING! No probability map file found for: \n%s" %\
                  raw_img_path)
            missing_prob_map_path_list.append(prob_map_path)
        else:
            filepaths_tuple_list.append((raw_img_path, prob_map_path))
    
    return filepaths_tuple_list

def overlay_probability_map_outline(raw_img_path, prob_map_path, save_dir=None, 
                                    show=False, ax=None, **kwargs):
    """ Overlay pixel/object probability maps onto raw images """
    
    # load image
    raw_img = np.array(Image.open(raw_img_path))
    assert np.ndim(raw_img) == 2

    # load probability map
    prob_map = np.array(Image.open(prob_map_path))
    assert np.ndim(prob_map) == 2

    if show and not ax:
        fig, ax = plt.subplots(**kwargs)
    
    # get outline of mask - erode and subtract from original mask
    prob_map_eroded = cv2.erode(prob_map, kernel=np.ones((5,5), np.uint8), 
                                iterations=1)
    prob_map_outline = cv2.subtract(prob_map, prob_map_eroded)
    
    # overlay probability map outline onto raw image
    img = cv2.add(raw_img, prob_map_outline)
    
    if show:
        ax.imshow(img)
        plt.pause(2); plt.close()
    
    if save_dir:
        outPath = Path(save_dir) / raw_img_path.name
        outPath.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(outPath), img)
        
    return img

# Main
if __name__ == "__main__":
           
    paired_files = pair_input_files(RAW_TIF_DIR, PROB_MAP_DIR)
    
    for (raw_img_path, prob_map_path) in tqdm(paired_files):
        img = overlay_probability_map_outline(raw_img_path = raw_img_path, 
                                              prob_map_path = prob_map_path,
                                              save_dir = SAVE_DIR)
