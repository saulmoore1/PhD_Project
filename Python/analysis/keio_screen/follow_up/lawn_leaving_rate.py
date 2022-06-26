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
THRESHOLD_LEAVING_DURATION = 50 # n frames a worm has to leave food for to be called a true leaving event

#%% Functions

def on_food(traj_df, poly_dict):
    
    for key, values in poly_dict.items():
        polygon = mpath.Path(values, closed=True)
        traj_df[key] = polygon.contains_points(traj_df[['x','y']])
    
    return traj_df

def leaving_events(df, 
                   on_food_col='Poly_0', 
                   threshold_n_frames=50): # 50 frames = 2 seconds (25fps)
    """ Calculate leaving events as when a worm midbody centroid leaves the annotated coordinates 
        of the lawn 
    """
    
    leaving_event_list = []
    
    wormIDs = df['worm_id'].unique()
    grouped_worm = df.groupby('worm_id')
    for worm in wormIDs:
        worm_df = grouped_worm.get_group(worm)
        
        _leave_mask = np.where(worm_df[on_food_col].astype(int).diff() == -1)[0]
        _enter_mask = np.where(worm_df[on_food_col].astype(int).diff() == 1)[0]
        leaving = worm_df.iloc[_leave_mask].index
        entering = worm_df.iloc[_enter_mask].index
        
        # if there is a leaving event
        if len(leaving) > 0:
            # if the worm does not return
            if len(entering) == 0:
                # compare to end of trajectory
                entering = np.array([worm_df.index[-1]])
            # if there is also an entering event
            elif len(entering) > 0:
                # if worm leaves/enters an equal number of times
                if len(leaving) == len(entering):
                    # if worm enters first, then leaves
                    if entering[0] < leaving[0]:
                        # ignore first entering event + compare leaving duration to end of trajectory 
                        # by adding end of trajectory index as last 'entering' event
                        entering = entering[1:]
                        entering = np.insert(entering, len(entering), worm_df.index[-1])
                    # if the worm returned to the food
                    elif leaving[0] < entering[0]:
                        pass
                # if worm leaves/enters an unequal number of times
                elif len(leaving) != len(entering):
                    # if worm leaves first
                    if leaving[0] < entering[0]:
                        # compare to end (add end of trajectory index as last 'entering event')
                        entering = np.insert(entering, len(entering), worm_df.index[-1])
                    # if worm enters first
                    elif entering[0] < leaving[0]:
                        # ignore first entering event
                        entering = entering[1:]
               
            # calculate leaving duration
            leaving_duration = entering - leaving
            
            # -1 to index the frame just before the leaving event to know which food it left from
            leaving_df = worm_df.loc[leaving - 1] 

            # append columns for leaving duration to leaving frame data for worm on food
            leaving_df['duration_n_frames'] = pd.Series(leaving_duration, 
                                                               index=leaving_df.index)
            # append as rows to out-dataframe
            leaving_event_list.append(leaving_df)
            
    # compile leaving events for all worms in video
    leaving_events_df = pd.concat(leaving_event_list, axis=0)
    
    # Filter for worms that left food for longer than threshold_n_frames (n frames after leaving)
    long_leaving_df = leaving_events_df[leaving_events_df['duration_n_frames'] >= threshold_n_frames]
    short_leaving_df = leaving_events_df[leaving_events_df['duration_n_frames'] < threshold_n_frames]
    print("Removed %d (%.1f%%) leaving events < %d frames" % (short_leaving_df.shape[0], 
          (short_leaving_df.shape[0]/leaving_events_df.shape[0])*100, threshold_n_frames))
    # TODO: could also filter by distance from food edge (spatial thresholding)

    return long_leaving_df

def fraction_on_food(metadata, 
                     food_coords_dir, 
                     threshold_duration=None, 
                     threshold_movement=None,
                     threshold_leaving_duration=50):
    """ Calculate the mean fraction of worms on food in each timestamp of each video (6-well only) """
    
    video_frac_path = Path(food_coords_dir) / 'fraction_on_food.csv'
    leaving_events_path = Path(food_coords_dir) / 'leaving_events.csv'
    
    if video_frac_path.exists() and leaving_events_path.exists():
        print("Found compiled information for the fraction of worms on food")
        video_frac_df = pd.read_csv(video_frac_path, header=0, index_col=0)
        leaving_events_df = pd.read_csv(leaving_events_path, header=0, index_col=0)
        
    else:
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
        leaving_events_list = []
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
            # traj_df = read_timeseries(featurefile, names=None, only_wells=None) ### SLOW!
                    
            # filter based on thresholds of trajectory movement/duration
            traj_df, _stats = filter_worm_trajectories(traj_df,
                                                       threshold_move=threshold_movement, 
                                                       threshold_time=threshold_duration,
                                                       microns_per_pixel=12.4)
            # TODO: store stats / investigate number of bad worm trajectories?
            
            # compute whether each wormID in each timestamp is on or off food + append results
            traj_df = on_food(traj_df, poly_dict)
            
            # calculate fraction of worms on food in each timestamp
            grouped_frame = traj_df.groupby('frame_number')
            on_food_frac = grouped_frame['Poly_0'].sum() / grouped_frame['Poly_0'].count()
            on_food_frac.name = featurefile
            
            video_frac_list.append(on_food_frac)
            
            # compute leaving events for each wormID in video
            leaving_df = leaving_events(traj_df, threshold_n_frames=threshold_leaving_duration)

            # append file info
            leaving_df['masked_video_path'] = maskedfile
            leaving_events_list.append(leaving_df[
                ['frame_number','worm_id','duration_n_frames','masked_video_path']])
                
        # save fraction of worms in each timestamp of each video to file
        video_frac_df = pd.concat(video_frac_list, axis=1)
        video_frac_df.to_csv(video_frac_path, index=True, header=True)
        
        # save leaving event information to file
        leaving_events_df = pd.concat(leaving_events_list, axis=0)
        leaving_events_df.to_csv(leaving_events_path, index=False, header=True)
            
    return video_frac_df, leaving_events_df

def lawn_leaving_rate(metadata, 
                      food_coords_dir, 
                      threshold_duration=None, 
                      threshold_movement=None,
                      threshold_leaving_duration=50):
    
    print("Estimating time spent on vs off the lawn in each video..")
    video_frac_df, leaving_events_df = fraction_on_food(metadata, 
                                                        food_coords_dir, 
                                                        threshold_duration, 
                                                        threshold_movement,
                                                        threshold_leaving_duration)
            
    return video_frac_df, leaving_events_df

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
        
    video_frac_df, leaving_events_df = lawn_leaving_rate(metadata,
                                                         food_coords_dir=food_coords_dir,
                                                         threshold_movement=THRESHOLD_MOVEMENT,
                                                         threshold_duration=THRESHOLD_DURATION,
                                                         threshold_leaving_duration=THRESHOLD_LEAVING_DURATION)
    
    print("Done! (Time taken: %.1f seconds" % (time() - toc))
    
