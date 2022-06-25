#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keio screen lawn-leaving assay

@author: sm5911
@date: 25/06/22
"""

#%% Imports

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from pathlib import Path
from matplotlib import path as mpath

from filter_data.filter_trajectories import filter_worm_trajectories

#%% Globals

PROJECT_DIR = "/Users/sm5911/Documents/Keio_Supplements" # local

# Filter parameters
THRESHOLD_DURATION = 25 # threshold trajectory length (n frames) / 25 fps => 1 second
THRESHOLD_MOVEMENT = 10 # threshold movement (n pixels) * 12.4 microns per pixel => 124 microns

#%% Functions

def _on_food(traj_df, poly_dict):
    
    for key, values in poly_dict.items():
        polygon = mpath.Path(values, closed=True)
        traj_df[key] = polygon.contains_points(traj_df[['x','y']])
    
    return traj_df

def fraction_on_food(metadata, 
                     food_coords_dir, 
                     threshold_duration=None, 
                     threshold_movement=None,
                     max_videos_per_group=30):
    """ Calculate the mean fraction of worms on food in each timestamp of each video (6-well only) """
    
    coordsfilelist = [str(food_coords_dir) + s.split('RawVideos')[-1] + '/lawn_coordinates.csv' 
                      for s in metadata['filename']]
    
    # subset for files that have lawn coordinates annotated
    mask = [Path(file).exists() for file in coordsfilelist]
    n_coords = sum(mask)
    if n_coords < metadata.shape[0]:
        print("%d files found with no annotations" % (metadata.shape[0] - n_coords))
    
    metadata = metadata.loc[metadata.index[mask],:]
    maskedfilelist = [s.replace('RawVideos','MaskedVideos') + '/metadata.hdf5' for s in 
                      metadata['filename']]
    coordsfilelist = np.array(coordsfilelist)[mask].tolist()

    assert all(str(Path(i).parent).split('MaskedVideos')[-1] == 
               str(Path(j).parent).split(str(food_coords_dir))[-1] 
               for i, j in zip(maskedfilelist, coordsfilelist))
    
    video_frac_list = []
    print("Calculating fraction on/off food")
    for i, (maskedfile, featurefile, coordsfile) in enumerate(tqdm(zip(
            maskedfilelist, metadata['featuresN_filename'], coordsfilelist), total=metadata.shape[0])):
        
        assert (str(Path(maskedfile).parent).split('MaskedVideos')[-1] == 
                str(Path(coordsfile).parent).split(str(food_coords_dir))[-1])
        
        # load coordinates of food lawns (user labelled)
        f = open(coordsfile, 'r').read()
        poly_dict = eval(f) # use 'evaluate' to read as dictionary not string
        
        # load coordinates of worm trajectories
        with h5py.File(featurefile, 'r') as f:
            traj_df = pd.DataFrame({'x': f['trajectories_data']['coord_x'],
                                    'y': f['trajectories_data']['coord_y'],
                                    'frame_number': f['trajectories_data']['frame_number'],
                                    'worm_id': f['trajectories_data']['worm_index_joined']})
        # from tierpsytools.read_data.get_timeseries import read_timeseries
        # traj_df = read_timeseries(featurefile, names=None, only_wells=None)
                
        # filter based on thresholds of trajectory movement/duration
        traj_df, _stats = filter_worm_trajectories(traj_df,
                                                   threshold_move=threshold_movement, 
                                                   threshold_time=threshold_duration,
                                                   microns_per_pixel=12.4)
        # TODO: Throw warning if stats show a high number of bad worm trajectories??
        
        # Compute whether each wormID in each timestamp is on or off food + append results
        traj_df = _on_food(traj_df, poly_dict)
        
        grouped_frame = traj_df.groupby('frame_number')
        on_food_frac = grouped_frame['Poly_0'].sum() / grouped_frame['Poly_0'].count()
        on_food_frac.name = featurefile
        
        video_frac_list.append(on_food_frac)
    
    video_frac_df = pd.concat(video_frac_list, axis=1)
            
    return video_frac_df


def lawn_leaving_rate(metadata, 
                      food_coords_dir, 
                      threshold_duration=None, 
                      threshold_movement=None):
    
    print("Estimating time spent on vs off the lawn in each video..")
    
    video_frac_df = fraction_on_food(metadata, 
                                     food_coords_dir, 
                                     threshold_duration, 
                                     threshold_movement)
    
    return video_frac_df

#%% Main

if __name__ == '__main__':
    toc = time()

    metadata_path = Path(PROJECT_DIR) / "metadata.csv"
    food_coords_dir = Path(PROJECT_DIR) / "lawn_leaving"

    # load project metadata
    metadata = pd.read_csv(metadata_path, header=0, index_col=None)
    
    ##### SUPPLEMENT ANALYSIS #####
    if 'Supplements' in PROJECT_DIR:
        metadata = metadata[metadata['drug_type'].isin(['paraquat','none'])]
        metadata['treatment'] = metadata[['food_type','drug_type','imaging_plate_drug_conc']
                                         ].astype(str).agg('-'.join, axis=1)
        
    video_frac_df = lawn_leaving_rate(metadata,
                                      food_coords_dir=food_coords_dir,
                                      threshold_movement=THRESHOLD_MOVEMENT,
                                      threshold_duration=THRESHOLD_DURATION)
    print(video_frac_df.shape)
    
    print("Done! (Time taken: %.1f seconds" % (time() - toc))
    
