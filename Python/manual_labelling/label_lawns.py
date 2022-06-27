#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual Labelling of Food Regions for On/Off Food Calculation

@author: sm5911
@date: 01/03/2021

"""

# TODO: Check compatibility with Hydra camera 16-well videos + maintain backwards compatibility
# TODO: Automate food labelling using (U-Net) ML algorithm -- pre-trained already by Luigi

#%% Imports 

import h5py
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from matplotlib import patches
from matplotlib import pyplot as plt

from visualisation.plotting_helper import hexcolours
from visualisation.plate_trajectories import plot_trajectory   

# sys.path.insert(0, "/Users/sm5911/Tierpsy_Versions/tierpsy-tracker") # path to tierpsy tracker repo
# from tierpsy.analysis.split_fov.FOVMultiWellsSplitter import FOVMultiWellsSplitter                                       

#%% Globals

# food choice assay (6-well):
# EXAMPLE_FILE = "/Volumes/behavgenom$/Priota/Data/FoodChoiceAssay/MaskedVideos/20181101/PC1/Set1/Set1_Ch1_01112018_103218.hdf5"
# confirmation screen (96-well):
# EXAMPLE_FILE = "/Volumes/hermes$/KeioScreen2_96WP/MaskedVideos/20210914/keio_confirm_screen_run1_bluelight_20210914_135936.22956809/metadata.hdf5"
# single worm assay (6-well):
EXAMPLE_FILE = "/Volumes/hermes$/Keio_Acute_Single_Worm/MaskedVideos/20220206/acute_single_worm_run1_bluelight_20220206_101625.22956818/metadata.hdf5"

PROJECT_DIR = "/Volumes/hermes$/Keio_Supplements"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Supplements"
METADATA_PATH = "/Users/sm5911/Documents/Keio_Supplements/metadata.csv"


SKIP = False # Skip file if coordinates of food lawn are already saved

#%% Functions

def plot_brightfield(maskedfilepath, frame, fig=None, ax=None, **kwargs):
    """ Plot the first bright-field image of a given masked HDF5 video. """
        
    with h5py.File(maskedfilepath, 'r') as f:
        if frame == 'all':
            # Extract all frames and take the brightest pixels over time
            brightfield = f['full_data'][:]
            brightfield = brightfield.max(axis=0)
        else:
            # Extract a given frame (index)
            brightfield = f['full_data'][frame]
            
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
        
    plt.imshow(brightfield, cmap='gray', vmin=0, vmax=255)
    
    return(fig, ax)

def draw_polygon(ax, n_poly=1, alpha=0.5):
    """ A function to record coordinates from user mouse input when drawing a 
        number of polygons (eg. oulining food patches) on an image and return 
        them as a dictionary (keys=polygons,values=xycoords). 
        NB: For assigning labels to polygons, see 'label_polygon'. """
        
    colours = hexcolours(n_poly)

    poly_dict = {}
    for i, polygon in enumerate(range(n_poly)):
        print("Draw polygon %d" % (polygon + 1))
        input_coords = plt.ginput(n=-1, timeout=0, mouse_add=1, mouse_pop=3, mouse_stop=2)
        
        poly_dict['{0}'.format('Poly_' + str(polygon))] = input_coords
        
        print("Plotting polygon..")
        polygon = patches.Polygon(input_coords, 
                                  closed=True, 
                                  color=colours[i], # 'k'
                                  alpha=alpha)
        ax.add_patch(polygon)
        
    print("Done.")
        
    return(ax, poly_dict)

# def label_polygon(poly_dict, ax):
#     """ A function that accepts keyboard input from the user to assign labels 
#         (stored as dictionary keys) to each polygon (set of x,y coords). """
        
#     labels = list(poly_dict.keys())
#     colours = hexcolours(len(labels))
    
#     for i, key in enumerate(labels):
#         polygon = patches.Polygon(poly_dict[key], closed=True, color=colours[i], alpha=0.5)
#         ax.add_patch(polygon); plt.show()
#         label = input("Assign name to {0}: ".format(key))
#         poly_dict['{0}'.format(label.upper())] = poly_dict.pop(key)
        
#     return(poly_dict)

def annotate_lawns(maskedfilepath, 
                   save_dir,
                   n_poly=1, 
                   skip=True):
    """ A function written to assist with the manual labelling of food regions in the worm
        food choice assay video recordings. It has subsequently been adapted to label lawns in 
        6-well experiments with a single food lawn, for estimating lawn leaving rate. Food regions 
        are given labels and coordinates are saved to file. If a coordinate file already exists for 
        the video, the file will be skipped.
        - Plot brightfield image
        - Accept user-input to outline and assign labels to food regions in image + save coordinates
        - Plot labelled trajectory overlay 
        - Save plot (default, save=True)
        - Skips files which have already been labelled (default, skip=True)
    """
        
    if type(maskedfilepath) is not str:
        maskedfilepath = str(maskedfilepath)
    
    # construct save paths
    featuresNfilepath = maskedfilepath.replace('MaskedVideos','Results').replace('.hdf5','_featuresN.hdf5')
    imgstorepath = maskedfilepath.split('/MaskedVideos')[-1]
    fname = maskedfilepath.split('/')[-1]
    save_path = str(save_dir) + imgstorepath.replace(fname,'lawn_coordinates.csv')
    
    # Avoid re-labelling videos
    if Path(save_path).exists() and skip: 
        print("Skipping file. Lawns coordinates already saved.")
    else:
        # Plot the first brightfield image
        print("\nProcessing file: %s" % maskedfilepath)    
            
        plt.close("all")
        fig, ax = plt.subplots(figsize=(10,10))
        plt.ion()
        
        # plot first frame of video
        fig, ax = plot_brightfield(maskedfilepath, frame=0, fig=fig, ax=ax)

        # USER INPUT: draw polygons around food regions in the assay
        print("Manually outline food regions.\nLeft click - add a point.\n\
               Right click - remove a point.\nMiddle click - next")
        ax, poly_dict = draw_polygon(ax, n_poly=n_poly, alpha=0.4)
        # fig.canvas.draw()

        # TODO: Check that assay not bumped by plotting last frame at the end 
        # TODO: well name as key for 96-well plates by comparing coords with FOVsplitter?
        
        # USER INPUT: assign labels to food regions
        # poly_dict = label_polygon(poly_dict, ax)
        
        # Save coordinates to file
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        fid = open(save_path, 'w')
        print(poly_dict, file=fid)
        fid.close()
        print("Coords successfully saved!")
        
        # Plot trajectory overlay
        ax = plot_trajectory(featuresNfilepath, downsample=10, ax=ax)
        # fig.canvas.draw()
        # fig.canvas.flush_events()
        
        # Save plot overlay
        plt.savefig(save_path.replace('lawn_coordinates.csv','overlay_plot.png'), dpi=300)
        
        # Return food coordinates
        return(poly_dict)
        
def annotate_lawns_from_metadata_6wp(metadata, 
                                     save_dir,
                                     group_by='treatment', 
                                     group_subset=None, 
                                     max_videos_per_group=30):
    """ Label lawns using masked video filenames from metadata for given treatment groups """
    
    metadata.groupby('treatment').count()
    
    if group_subset is not None:
        assert all(g in metadata[group_by].unique() for g in group_subset)
        metadata = metadata[metadata[group_by].isin(group_subset)]
    else:
        group_subset = metadata[group_by].unique().tolist()
        
    grouped = metadata.groupby(group_by)
    for group in group_subset:
        group_meta = grouped.get_group(group)
        if group_meta.shape[0] > max_videos_per_group:
            print("Annotating the first %d videos for '%s'" % (max_videos_per_group, group))
            group_meta = group_meta.loc[group_meta.index[:max_videos_per_group],:]
        else:
            print("Annotating %d videos for '%s'" % (group_meta.shape[0], group))
    
        for featuresfilepath in tqdm(group_meta['featuresN_filename']):
            maskedfilepath = featuresfilepath.replace('/Results/','/MaskedVideos/')
            maskedfilepath = maskedfilepath.replace('_featuresN.hdf5','.hdf5')
            try:
                annotate_lawns(maskedfilepath, 
                               save_dir=save_dir, 
                               n_poly=1, 
                               skip=True)
            except Exception as error:
                print("WARNING: Could not read file!\n", error)
            
    return


#%% Main

if __name__ == '__main__':
    # Accept MaskedVideo list from command line for labelling food regions in multiple videos
    parser = argparse.ArgumentParser(description='Manual labelling of lawns in MaskedVideos')
    parser.add_argument('--metadata_path', help="Path to project metadata to annotate", type=str,
                        default=METADATA_PATH)
    parser.add_argument('--maskedvideo_dir', help="Directory of MaskedVideos to annotate",
                        default=Path(PROJECT_DIR) / 'MaskedVideos')
    parser.add_argument('--maskedvideo_list', help="List of MaskedVideo filepaths for videos to \
                        annotate lawns", nargs='+', default=None)
    parser.add_argument('--save_dir', help="Directory to save lawn coordinates", 
                        default=SAVE_DIR)
    args = parser.parse_args()

    tic = time.time()
    
    # annotate given treatment groups in metadata
    if args.metadata_path is not None:
        # load project metadata
        metadata = pd.read_csv(args.metadata_path, header=0, index_col=None, dtype={'comments':str})

        ##### SUPPLEMENT ANALYSIS #####
        if 'Supplements' in str(args.metadata_path):
            # subset for metadata for a single window (so there are no duplicate filenames)
            metadata = metadata[metadata['window']==0]
    
            # subset for paraquat results only
            metadata = metadata[metadata['drug_type'].isin(['paraquat','none'])]
            
            # treatment names for experiment conditions
            metadata['treatment'] = metadata[['food_type','drug_type','imaging_plate_drug_conc']].astype(str).agg('-'.join, axis=1)
            
            # user-label lawn regions in first frame image of each video
            annotate_lawns_from_metadata_6wp(metadata,
                                             save_dir=Path(args.save_dir) / 'lawn_leaving',
                                             group_by='treatment',
                                             group_subset=None,
                                             max_videos_per_group=10)
        
    else:
        # will annotate all videos found in MaskedVideo directory if no masked video list is given
        if args.maskedvideo_list is not None:
            maskedfilelist = args.maskedvideo_list
        elif args.maskedvideo_dir is not None:
            # Find masked HDF5 video files (for labelling) 
            maskedfilelist = list(Path(args.maskedvideo_dir).rglob('*.hdf5'))
        else:
            maskedfilelist = [EXAMPLE_FILE]
            args.save_dir = EXAMPLE_FILE.split("MaskedVideos")[0]
    
        assert type(args.maskedvideo_list) == list
        maskedfilelist = args.maskedvideo_list
            
        print("Manual labelling: %d masked videos found.." % len(maskedfilelist))
        
        # Interactive plotting (for user input when labelling plots)
        for i in range(len(maskedfilelist)):
            maskedfilepath = maskedfilelist[i]
            
            # manually outline food regions, assign labels and save coordinates and trajectory overlay
            annotate_lawns(maskedfilepath,                       
                           save_dir=Path(args.save_dir) / 'lawn_leaving', 
                           n_poly=1, 
                           skip=SKIP)
            
    print("All Done!\n(Time taken: %d seconds.)\n" % (time.time() - tic))
