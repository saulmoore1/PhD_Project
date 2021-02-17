#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay ilastik probability map outlines (TIF) onto raw images (TIF)

@author: sm5911
@date: 14/09/2020

"""

# Imports
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
 
# Functions
def pair_input_files(RAW_TIF_DIR, PROB_MAP_DIR):
    """ Pair raw image filepaths with associated probability map filepaths 
        (pixel or object probabilities) """
    
    raw_img_list = list(RAW_TIF_DIR.rglob("*.tif"))
    prob_map_list = list(PROB_MAP_DIR.rglob("*.tif"))
    
    missing_prob_map_path_list = []
    filepaths_tuple_list = []
    for raw_img_path in raw_img_list:
        
        # Look for object probability map files
        prob_map_name = raw_img_path.name.replace(".tif",
                                                  "_Object Probabilities.tif")
        prob_map_path = PROB_MAP_DIR / prob_map_name
        
        if prob_map_path in prob_map_list:
            filepaths_tuple_list.append((raw_img_path, prob_map_path))
        else:
            # Look for object prediction map files
            prob_map_name = raw_img_path.name.replace(".tif",
                                                  "_Object Predictions.tif")
            prob_map_path = PROB_MAP_DIR / prob_map_name
        
            if prob_map_path in prob_map_list:
                filepaths_tuple_list.append((raw_img_path, prob_map_path))
            else:
                # Look for pixel probability map files
                prob_map_name = raw_img_path.name.replace(".tif",
                                                          "_Probabilities.tif")
                prob_map_path = PROB_MAP_DIR / prob_map_name
                
                if prob_map_path in prob_map_list:
                    filepaths_tuple_list.append((raw_img_path, prob_map_path))
                else:
                    #print("WARNING! No probability map file found for: \n%s" % raw_img_path)
                    missing_prob_map_path_list.append(prob_map_path)
    
    if len(missing_prob_map_path_list) > 0:
        print("\nWARNING: Probability map files missing for %d images" % len(missing_prob_map_path_list))
        
    return filepaths_tuple_list

def image_rescale_255(img_mat):
    """ Rescale image between 0-255 """
    
    img_mat = img_mat - img_mat.min()
    img_mat = img_mat.astype(float) / img_mat.max()
    img_mat = np.uint8(img_mat * 255)
    
    return img_mat
    
def overlay_probability_map_outline(raw_img_path, prob_map_path, save_dir=None, 
                                    show=False, ax=None, alpha=0.2, **kwargs):
    """ Overlay pixel/object probability maps onto raw images """
    
    # load image
    raw_img = np.array(Image.open(raw_img_path))
    assert np.ndim(raw_img) == 2

    # load probability map
    prob_map = np.array(Image.open(prob_map_path))
    assert np.ndim(prob_map) == 2    
    if not prob_map.dtype == np.uint16:
        prob_map = prob_map.astype(np.uint16)
        
    if show and not ax:
        fig, ax = plt.subplots(**kwargs)
    
    # get outline of mask - erode and subtract from original mask
    prob_map_mask = (prob_map > 0).astype(np.uint8)
    prob_map_mask_eroded = cv2.erode(prob_map_mask, kernel=np.ones((5,5), np.uint8), 
                                     iterations=1)
    prob_map_mask_outline = cv2.subtract(prob_map_mask, prob_map_mask_eroded)
    prob_map_outline = prob_map.copy()
    prob_map_outline[prob_map_mask_outline == 0] = 0
    
    # rescale raw image + prob maps
    raw_img = image_rescale_255(raw_img)
    prob_map_outline = image_rescale_255(prob_map_outline)   
    
    # Create RBG tricolor image
    bgr_image = np.concatenate([raw_img[...,None]]*3, axis=2)
    overlay_img = np.concatenate((np.zeros(prob_map_outline.shape, dtype=np.uint8)[...,None],
                                  np.zeros(prob_map_outline.shape, dtype=np.uint8)[...,None], 
                                  prob_map_outline.copy()[...,None]), axis=2)
    #overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_GRAY2BGR)
    
    # overlay probability map outline onto raw image
    cv2.addWeighted(overlay_img, alpha, bgr_image, (1-alpha), 0, bgr_image)
    
    if show:
        ax.imshow(bgr_image[...,::-1]) # flip in third axis to plot RGB and not BGR
        plt.pause(2); plt.close()
    
    if save_dir:
        outPath = Path(save_dir) / raw_img_path.name
        outPath.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(outPath), bgr_image)
        
    return bgr_image

# Main
if __name__ == "__main__":
    # command-line args
    parser = argparse.ArgumentParser(description='Overlay ilastik probability maps onto raw tif images and save')
    parser.add_argument('--raw_tif_dir', help='Path to raw TIF image directory') 
    parser.add_argument('--prob_map_dir', help='Path to Ilastik probability map directory (pixel or object)')
    parser.add_argument('--save_dir', help='Path to output directory')
    args = parser.parse_args()
    
    RAW_TIF_DIR = Path(args.raw_tif_dir)
    PROB_MAP_DIR = Path(args.prob_map_dir)
    SAVE_DIR = Path(args.save_dir)
    print('\nRaw TIFF directory: %s' % str(RAW_TIF_DIR))
    print('\nProbability map directory: %s' % str(PROB_MAP_DIR))
    print('\nSaving to: %s\n' % str(SAVE_DIR))

    paired_files = pair_input_files(RAW_TIF_DIR, PROB_MAP_DIR)
    
    for (raw_img_path, prob_map_path) in tqdm(paired_files):
        img = overlay_probability_map_outline(raw_img_path = raw_img_path, 
                                              prob_map_path = prob_map_path,
                                              save_dir = SAVE_DIR,
                                              alpha=0.2,
                                              show=False)
    plt.close('all')
    print("\nDone!")
