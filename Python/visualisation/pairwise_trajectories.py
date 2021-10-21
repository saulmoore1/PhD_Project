#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Side-by-side trajectory plots for strain vs control

@author: sm5911
@date: 20/10/2021

"""

#%% Imports 

import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from visualisation.plate_trajectories import CH2PLATE_dict, get_video_set, plot_trajectory

from tierpsy.analysis.split_fov.FOVMultiWellsSplitter import FOVMultiWellsSplitter

#%% Globals

METADATA_PATH = '/Users/sm5911/Documents/Keio_Conf_Screen/metadata.csv'
FEATURES_PATH = '/Users/sm5911/Documents/Keio_Conf_Screen/features.csv'

SAVE_DIR = '/Users/sm5911/Documents/Keio_Conf_Screen/pairwise_trajectories'

#%% Functions

def plot_pairwise_trajectory(metadata, strain_colname, strain, control, downsample=10, saveDir=None):
    """ Plot pairwise trajectory plots of strain vs control """
    
    assert strain in metadata[strain_colname].unique()
    assert control in metadata[strain_colname].unique()
    
    # subset for strain and control
    strain_meta = metadata[metadata[strain_colname]==strain]
    control_meta = metadata[metadata[strain_colname]==control]
    
    assert strain_meta.shape[0] == len(strain_meta['featuresN_filename'].unique())
    
    # for each strain well, pair with random control well of the same plate
    strain_info = []
    control_info = []
    for i in strain_meta.index:
        s_file = strain_meta.loc[i, 'featuresN_filename']
        s_well = strain_meta.loc[i, 'well_name']
        
        matched_control = control_meta[np.logical_and(
            control_meta['date_yyyymmdd'] == strain_meta.loc[i, 'date_yyyymmdd'],
            control_meta['imaging_plate_id'] == strain_meta.loc[i, 'imaging_plate_id'])]
        
        if matched_control.shape[0] == 0:
            print("WARNING: No control well found for %s in file %s well %s" % (strain, s_file, s_well))
            
        # grab random control well data
        rand_idx = np.random.choice(matched_control.index, size=1, replace=True)[0]
        c_file = matched_control.loc[rand_idx, 'featuresN_filename']
        c_well = matched_control.loc[rand_idx, 'well_name']
        
        strain_info.append((s_file, s_well))
        control_info.append((c_file, c_well))
        
    for s, c in zip(control_info, strain_info):
        s_file, s_well = s
        c_file, c_well = c
        
    
    file_dict = get_video_set(featurefilepath)
    
    # define multi-panel figure
    columns = 3
    rows = 2
    x = 25.5
    y = 16
    plt.ioff() if saveDir else plt.ion()
    plt.close('all')
    fig, axs = plt.subplots(rows,columns,figsize=[x,y])
    
    x_offset = 1.5 / x  # for bottom left image
    width = 0.3137      # for all but top left image
    width_tl = 0.3725   # for top left image
    height = 0.5        # for all images

    errlog = []    
    for channel, (maskedfilepath, featurefilepath) in file_dict.items():

        if saveDir:
            saveName = maskedfilepath.parent.stem + ('_filtered.png' if filter_trajectories else '.png')
            savePath = Path(saveDir) / saveName
            if savePath.exists():
                if del_if_exists:
                    os.remove(savePath)
                else:
                    print("Skipping file '%s' (already exists)" % savePath.name)
                    continue
        
        _loc, rotate = CH2PLATE_dict[channel]
        _ri, _ci = _loc

        # create bbox for image layout in figure
        if (_ri == 0) and (_ci == 0):
            # first image (with well names), bbox slightly shifted
            bbox = [0, height, width_tl, height]
        else:
            # other images
            bbox = [x_offset + width * _ci, height * (rows - (_ri + 1)), width, height]   
        
        # get location of subplot for camera
        ax = axs[_loc]
        
        try:
            # plot first frame of video + annotate wells
            FOVsplitter = FOVMultiWellsSplitter(maskedfilepath)
            FOVsplitter.plot_wells(is_rotate180=rotate, ax=ax, line_thickness=10)
            
            # plot worm trajectories
            plot_trajectory(featurefilepath, 
                            downsample=downsample,
                            filter_trajectories=filter_trajectories,
                            mark_endpoints=mark_endpoints,
                            rotate=rotate,
                            img_shape=FOVsplitter.img_shape,
                            legend=False, 
                            ax=ax)
        except Exception as e:
            print("WARNING: Could not plot video file: '%s'\n%s" % (maskedfilepath, e))
            errlog.append(maskedfilepath)
        
        # set image position in figure
        ax.set_position(bbox)
    
    if saveDir:
        if savePath.exists():
            print("Skipping file '%s' (already exists)" % savePath.name)
        else:
            Path(saveDir).mkdir(exist_ok=True, parents=True)
            fig.savefig(savePath,
                        bbox_inches='tight',
                        dpi=300,
                        pad_inches=0,
                        transparent=True)  

    
    

#%% Main

if __name__ == "__main__":
    
    np.random.seed(0)
    
    # load metadata and features
    metadata = pd.read_csv(METADATA_PATH, dtype={"comments":str})
    features = pd.read_csv(FEATURES_PATH)
    
    # plot pairwise trajectory plots
    plot_pairwise_trajectory(metadata,
                             strain='fepB',
                             strain_colname='gene_name',
                             control='wild_type',
                             downsample=10,
                             saveDir=Path(SAVE_DIR))