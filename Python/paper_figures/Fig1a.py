#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAPER FIGURE 1a

A script to plot the full 96-well plate, a single camera view and a single well for paper figure 
Provide a featuresN filepath from Tierpsy filenames summaries and a plot will be produced of:
    1. the entire 96-well plate (imaged under 6 cameras simultaneously)
    2. a given camera view
    3. a single well 
for the first frame of the video

@author: sm5911
@date: 30/01/2023

"""

#%% Imports 

import sys
import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt

sys.path.insert(0, "/Users/sm5911/Tierpsy_Versions/tierpsy-tracker") # path to tierpsy tracker repo
from tierpsy.analysis.split_fov.helper import CAM2CH_df, serial2channel, parse_camera_serial

#%% Globals

# Channel-to-plate mapping dictionary {'channel' : ((ax array location), rotate)}
CH2PLATE_dict = {'Ch1':((0,0),True),
                 'Ch2':((1,0),False),
                 'Ch3':((0,1),True),
                 'Ch4':((1,1),False),
                 'Ch5':((0,2),True),
                 'Ch6':((1,2),False)}

FEAT_FILE_PATH = "/Volumes/hermes$/KeioScreen_96WP/Results/20210420/keio_rep3_run7_prestim_20210420_145642.22956805/metadata_featuresN.hdf5"

SAVE_DIR = "/Users/sm5911/OneDrive - Imperial College London/Publications/Keio_Paper/Figures/Fig1"
FRAME = 14
DPI = 900

#%% Functions

def get_video_set(featurefilepath):
    """ Get the set of filenames of the featuresN results files that belong to
        the same 96-well plate that was imaged under that rig """
        
    dirpath = Path(featurefilepath).parent
    maskedfilepath = Path(str(dirpath).replace("Results/","MaskedVideos/"))
    
    # get camera serial from filename
    camera_serial = parse_camera_serial(featurefilepath)
    
    # get list of camera serials for that hydra rig
    hydra_rig = CAM2CH_df.loc[CAM2CH_df['camera_serial'] == camera_serial, 'rig']
    rig_df = CAM2CH_df[CAM2CH_df['rig'] == hydra_rig.values[0]]
    camera_serial_list = list(rig_df['camera_serial'])
   
    # extract filename stem 
    file_stem = str(maskedfilepath).split('.' + camera_serial)[0]
    
    file_dict = {}
    for camera_serial in camera_serial_list:
        channel = serial2channel(camera_serial)
        _loc, rotate = CH2PLATE_dict[channel]
        
        # get path to masked video file
        maskedfilepath = Path(file_stem + '.' + camera_serial) / "metadata.hdf5"
        featurefilepath = Path(str(maskedfilepath.parent).replace("MaskedVideos/", "Results/")) /\
            'metadata_featuresN.hdf5'
        
        file_dict[channel] = (maskedfilepath, featurefilepath)
        
    return file_dict
    
def plot_plate(featurefilepath, save_path=None, frame=0, dpi=900): # frame = 'all'
    """ Tile plots and merge into a single plot for the 
        entire 96-well plate, correcting for camera orientation. """

    file_dict = get_video_set(featurefilepath)
    
    # define multi-panel figure
    columns = 3
    rows = 2
    h_in = 4
    x_off_abs = (3600-3036) / 3036 * h_in
    x = columns * h_in + x_off_abs
    y = rows * h_in
    
    x_offset = x_off_abs / x        # for bottom left image
    width = (1-x_offset) / columns  # for all but top left image
    width_tl = width + x_offset     # for top left image
    height = 1/rows                 # for all images
    
    plt.close('all')
    fig, axs = plt.subplots(rows, columns, figsize=[x,y])

    errlog = []    
    for channel, (maskedfilepath, featurefilepath) in tqdm(file_dict.items()):
        
        _loc, rotate = CH2PLATE_dict[channel]
        _ri, _ci = _loc

        # create bbox for image layout in figure
        if (_ri == 0) and (_ci == 0):
            # first image, bbox slightly shifted
            bbox = [0, height, width_tl, height]
        else:
            # other images
            bbox = [x_offset + width * _ci, height * (rows - (_ri + 1)), width, height]   
        
        # get location of subplot for camera
        ax = axs[_loc]
        
        try:
            # plot first frame of video
            with h5py.File(maskedfilepath, 'r') as f:
                if frame == 'all':
                    # extract all frames and take the brightest pixels over time
                    img = f['full_data'][:]
                    img = img.max(axis=0)
                else:
                    # extract a given frame (index)
                    img = f['full_data'][frame]            
            
            if rotate:
                img = np.rot90(img, 2)   
                
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            
        except Exception as e:
            print("WARNING: Could not plot video file: '%s'\n%s" % (maskedfilepath, e))
            errlog.append(maskedfilepath)
        
        # set image position in figure
        ax.set_position(bbox)
        
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        
        # store filepath for final camera video
        if _ri == rows - 1 and _ci == columns - 1:
            bottom_left_featurefilepath = featurefilepath
        
    if len(errlog) > 0:
        print(errlog)
    
    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_path,
                    bbox_inches='tight',
                    dpi=dpi,
                    pad_inches=0,
                    transparent=True)
            
    return bottom_left_featurefilepath

def plot_camera_wells(featurefilepath, save_path=None, frame=0, dpi=900):
    """ Plot single camera view """

    plt.close('all')
    fig, ax = plt.subplots()

    file_dict = get_video_set(featurefilepath)
        
    for _channel, (_maskedfilepath, _featurefilepath) in file_dict.items():
        if str(_featurefilepath) == str(featurefilepath):
            maskedfilepath = _maskedfilepath
            _loc, rotate = CH2PLATE_dict[_channel]
            break
    
    with h5py.File(maskedfilepath, 'r') as f:
        if frame == 'all':
            # extract all frames and take the brightest pixels over time
            img = f['full_data'][:]
            img = img.max(axis=0)
        else:
            # extract a given frame (index)
            img = f['full_data'][frame]       

    if rotate:
        img = np.rot90(img, 2)
        
    # plot image
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
        
    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_path,
                    bbox_inches='tight',
                    dpi=dpi,
                    pad_inches=0,
                    transparent=True)
    
    return


#%% Main
    
if __name__ == "__main__":
            
    print("Plotting plate for %s" % str(FEAT_FILE_PATH))
    bottom_left_featurefilepath = plot_plate(featurefilepath=FEAT_FILE_PATH, 
                                             save_path=Path(SAVE_DIR) / "Fig1a_plate.png",
                                             frame=FRAME,
                                             dpi=DPI)
    
    print(bottom_left_featurefilepath)
    # /Volumes/hermes$/KeioScreen_96WP/Results/20210420/keio_rep3_run7_prestim_20210420_145642.22956832/metadata_featuresN.hdf5
    plot_camera_wells(featurefilepath=bottom_left_featurefilepath, 
                      save_path=Path(SAVE_DIR) / "Fig1b_camera.png",
                      frame=FRAME,
                      dpi=DPI)
    
  