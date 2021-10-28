#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Side-by-side trajectory plots for strain vs control

@author: sm5911
@date: 20/10/2021

"""

#%% Imports 

import os
import sys
import h5py
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

from filter_data.filter_trajectories import filter_worm_trajectories
from visualisation.plate_trajectories import CH2PLATE_dict, plot_trajectory

from tierpsytools.read_data.get_timeseries import read_timeseries

sys.path.insert(0, "/Users/sm5911/Tierpsy_Versions/tierpsy-tracker") # path to tierpsy tracker repo
from tierpsy.analysis.split_fov.FOVMultiWellsSplitter import FOVMultiWellsSplitter
from tierpsy.analysis.split_fov.helper import serial2channel, parse_camera_serial
from tierpsy.helper.params.read_attrs import read_microns_per_pixel, read_fps

#%% Globals

METADATA_PATH = '/Users/sm5911/Documents/Keio_Conf_Screen/metadata.csv'
FEATURES_PATH = '/Users/sm5911/Documents/Keio_Conf_Screen/features.csv'
SAVE_DIR = '/Users/sm5911/Documents/Keio_Conf_Screen/pairwise_trajectories'

STRAIN_LIST = ['fepB']
MAX_N_PAIRS = 5

# set seed for reproducibility 
np.random.seed(0)

# Params for filter trajectories
THRESHOLD_DISTANCE_PIXELS = 10
THRESHOLD_DURATION_FRAMES = 25


#%% Functions

def plot_well_trajectory(featuresfilepath, maskedvideopath, well_name, downsample=10, 
                         filter_trajectories=False, ax=None, verbose=True, **kwargs):
    """ Plot centroid coordinates for worms in a given well """
    
    # from read_data.read import get_fov_well_data
    # fov_wells = get_fov_well_data(featuresfilepath)
    # well_fov = fov_wells[fov_wells['well_name']==well_name]
    # assert well_fov.shape[0] == 1
    # well_fov = well_fov.iloc[0]
        
    # # get camera serial + channel from filename
    # camera_serial = parse_camera_serial(featuresfilepath)
    # channel = serial2channel(camera_serial)
    
    # # get whether to rotate plot for video
    # _, rotate = CH2PLATE_dict[channel]
            
    # plot first frame of video for sample well
    FOVsplitter = FOVMultiWellsSplitter(maskedvideopath)
    fov_wells = FOVsplitter.wells
    well_fov = fov_wells[fov_wells['well_name']==well_name]
    
    if well_fov.iloc[0]['is_good_well'] != 1:
        print("WARNING: Bad well data for:\n%s\t%s" % (featuresfilepath, well_name))
        return
    
    if not ax:
        fig, ax = plt.subplots(**kwargs)
        
    #FOVsplitter.plot_wells(is_rotate180=rotate, ax=ax, line_thickness=10)
    img_list = FOVsplitter.tile_FOV(FOVsplitter.img)
    well_img = [i[1] for i in img_list if i[0] == well_name][0]
    
    ax.imshow(well_img)

    df = read_timeseries(featuresfilepath, 
                         names=['worm_index','timestamp','well_name','coord_x_body','coord_y_body'], 
                         only_wells=[well_name])
    
    microns_per_pixel = read_microns_per_pixel(featuresfilepath)
    df['x'] = df['coord_x_body'] / microns_per_pixel
    df['y'] = df['coord_y_body'] / microns_per_pixel

    # subtract x,y offset to set bottom left coords of well as plot origin for trajectory plot
    df['x'] = df['x'] - well_fov.iloc[0]['x']
    df['y'] = df['y'] - well_fov.iloc[0]['y']

    # Optional - filter trajectories using global movement/time threshold parameters
    if filter_trajectories:
        df, _ = filter_worm_trajectories(df, 
                                         threshold_move=THRESHOLD_DISTANCE_PIXELS, 
                                         threshold_time=THRESHOLD_DURATION_FRAMES,
                                         fps=read_fps(featuresfilepath), 
                                         microns_per_pixel=microns_per_pixel, 
                                         timestamp_col='timestamp',
                                         worm_id_col='worm_index', 
                                         x_coord_col='x', 
                                         y_coord_col='y',
                                         verbose=verbose)
            
    # # Rotate trajectories if necessary (for tiling 96-well plate)
    # if rotate:
    #     img_shape = well_img.shape
    #     height, width = img_shape[0], img_shape[1]
    #     df['x'] = width - df['x']
    #     df['y'] = height - df['y']
            
    # Plot trajectory
    if downsample is not None:
        # Downsample frames for plotting
        downsample = 1 if downsample < 1 else downsample
        
        ax.scatter(x=df['x'][::downsample], y=df['y'][::downsample], 
                   c=df['timestamp'][::downsample], cmap='plasma', s=10)
    else:
        ax.scatter(x=df['x'], y=df['y'], c=df['timestamp'], cmap='plasma', s=10)
        
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)    
    ax.autoscale(enable=True, axis='x', tight=True) # re-scaling axes
    ax.autoscale(enable=True, axis='y', tight=True)
    
    return

def plot_pairwise_trajectory(metadata, strain_colname, strain, control, downsample=10, 
                             filter_trajectories=False, stim_type='bluelight', saveAs=None, 
                             del_if_exists=True, verbose=True):
    """ Plot pairwise trajectory plots of strain vs control """
    
    assert strain in metadata[strain_colname].unique() and control in metadata[strain_colname].unique()
    
    # subset for strain and control
    strain_meta = metadata[metadata[strain_colname]==strain]
    control_meta = metadata[metadata[strain_colname]==control]
    
    assert strain_meta.shape[0] == len(strain_meta['featuresN_filename'].unique())

    # subset for bluelight condition
    #if stim_type is not None:
    assert stim_type in ['prestim','bluelight','poststim']
            
    # for each strain well, pair with random control well of the same plate
    strain_info = []
    control_info = []
    for i in strain_meta.index:
        s_imgstore = strain_meta.loc[i, 'imgstore_name_{}'.format(stim_type)]
        s_well = strain_meta.loc[i, 'well_name']

        # get path to strain featuresN results for a given stimulus type
        s_file = strain_meta.loc[i, 'featuresN_filename']
        _day = s_imgstore.split('/')[0]
        _res_dir = s_file.split(_day)[0]
        s_file = Path(_res_dir) / s_imgstore / 'metadata_featuresN.hdf5'
        assert s_file.exists()
        
        # get path to strain maskedvideo file
        _mask_dir = _res_dir.replace('/Results/','/MaskedVideos/')
        s_mask = Path(_mask_dir) / s_imgstore / 'metadata.hdf5'
        assert s_mask.exists()
        
        matched_control = control_meta[np.logical_and(
            control_meta['date_yyyymmdd'] == strain_meta.loc[i, 'date_yyyymmdd'],
            control_meta['imaging_plate_id'] == strain_meta.loc[i, 'imaging_plate_id'])]
        
        assert matched_control.shape[0] != 0
            
        # grab random control well data
        rand_idx = np.random.choice(matched_control.index, size=1, replace=True)[0]
        c_imgstore = matched_control.loc[rand_idx, 'imgstore_name_{}'.format(stim_type)]
        c_well = matched_control.loc[rand_idx, 'well_name']   
        
        c_file = Path(_res_dir) / c_imgstore / 'metadata_featuresN.hdf5'
        assert c_file.exists()
        
        c_mask = Path(_mask_dir) / c_imgstore / 'metadata.hdf5'
        assert c_mask.exists()
        
        strain_info.append([s_file, s_mask, s_well])
        control_info.append([c_file, c_mask, c_well])

    # sample n sample vs control pairs from list
    n = min(MAX_N_PAIRS, len(strain_info))
    idxs = np.random.choice(range(len(strain_info)), n, replace=False)
    strain_info = np.array(strain_info)[idxs].tolist()
    control_info = np.array(control_info)[idxs].tolist()

    if saveAs is not None and saveAs.exists():
        if verbose:
            print("Skipping file '%s' (already exists)" % saveAs.name)
    else:           
        # define multi-panel figure
        plt.close('all')
        plt.ioff() if saveAs else plt.ion()
        fig, axs = plt.subplots(nrows=n, ncols=2) # figsize=(5, len(strain_info)*2)
        
        errlog = []
        for i, (s, c) in enumerate(zip(strain_info, control_info)):
            s_file, s_mask, s_well = s
            c_file, c_mask, c_well = c
            
            try:
                # plot worm trajectories in sample well
                plot_well_trajectory(s_file,
                                     s_mask,
                                     well_name=s_well,
                                     downsample=downsample,
                                     filter_trajectories=filter_trajectories,
                                     ax=axs[i,0],
                                     verbose=verbose)
                
                # plot worm trajectories in control well
                plot_well_trajectory(c_file, 
                                     c_mask,
                                     well_name=c_well,
                                     downsample=downsample,
                                     filter_trajectories=filter_trajectories,
                                     ax=axs[i,1],
                                     verbose=verbose)
            except Exception as e:
                print("WARNING: Could not plot video file!\n%s\n%s\n%s" % (s, c, e))
                errlog.append((s, c, e))
        
        if saveAs is not None:
            Path(saveAs).parent.mkdir(exist_ok=True, parents=True)
            fig.savefig(saveAs,
                        bbox_inches='tight',
                        dpi=300,
                        pad_inches=0,
                        transparent=True)  

#%% Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot pairwise trajectories for each strain vs \
                                     control (aligned bluelight only)")
    parser.add_argument('--metadata_path', help="Path to metadata file containing columns: \
                        ['featuresN_filename','well_name','date_yyyymmdd','imaging_plate_id']",
                        default=METADATA_PATH)
    parser.add_argument('--features_path', help="Path to compiled features summaries file",
                        default=FEATURES_PATH)
    parser.add_argument('--control', help="Control strain to compare with", default='wild_type')
    parser.add_argument('--strain_list', help="List of strains to compare against control", 
                        nargs='+', default=STRAIN_LIST)
    parser.add_argument('-strain_colname', help="Column name containing strain and control variables",
                        default='gene_name')
    parser.add_argument('--save_dir', help="Directory path to save pairwise trajectory plots",
                        default=SAVE_DIR)
    parser.add_argument('--downsample', help="Downsample video by selecting only every nth frame",
                        default=10)
    parser.add_argument('--filter_trajectories', help="Filter trajectories by global distance and \
                        duration parameters", default=False)
    args = parser.parse_args()
        
    # load metadata and features
    metadata = pd.read_csv(args.metadata_path, dtype={"comments":str})
    features = pd.read_csv(args.features_path)
    
    assert all(s in metadata[args.strain_colname].unique() for s in args.strain_list)
    
    # plot pairwise trajectory plots
    for strain in args.strain_list:
        print("Plotting pairwise trajectory plots for '%s'" % strain)
        plot_pairwise_trajectory(metadata,
                                 strain=strain,
                                 strain_colname=args.strain_colname,
                                 control=args.control,
                                 stim_type='bluelight',
                                 downsample=args.downsample,
                                 filter_trajectories=args.filter_trajectories,
                                 saveAs=Path(args.save_dir) / (strain + '_trajectories.pdf'),
                                 verbose=True)
        
        