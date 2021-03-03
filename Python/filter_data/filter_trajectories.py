#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter Tierpsy worm trajectory data (Phenix only)

@author: sm5911
@date: 01/03/2021

"""

# TODO: Check compatibility with Hydra camera 16-well videos + maintain backwards compatibility

#%% Functions

def filter_worm_trajectories(trajectory_df, threshold_move=10, threshold_time=25,
                             fps=25, microns_per_pixel=10, verbose=True):
    """ A function to filter Tierpsy trajectory data worm IDs to remove unwanted
        tracked entities (ie. not worms), based on threshold parameters for the 
        duration (in frames) and amount of movement (in pixels) of their trajectories
        
        Parameters
        ----------
        trajectory_df : pd.DataFrame
            Dataframe of xy coords for worm trajectories in each frame of the video
        threshold_move : int
            Threshold minimum length of trajectory (pixels) to keep wormID
        threshold_time : int
            Threshold minimum duration of trajectory (frames) to keep wormID
        fps : int
            Frame rate of video (frames per second)
        microns_per_pixel : int
            Micron to pixel ratio of video (resolution)
        verbose : bool
            Print statements to std out
            
        Returns
        -------
        Filtered trajectory dataframe
    """
        
    group_worm = trajectory_df.groupby('worm_id')
    n_worms = group_worm.count().shape[0]
    
    # Filter by TRAJECTORY LENGTH
    filterTime_df = group_worm.filter(lambda x: x['frame_number'].count() > threshold_time)
    
    # Re-group by worm id
    group_worm = filterTime_df.groupby('worm_id')
    n_worms_time = group_worm.count().shape[0]
    
    if verbose:
        print("%d worm IDs filtered that existed for less than %d frames (%.1f seconds)." %\
              (n_worms - n_worms_time, threshold_time, threshold_time / fps))
            
    # Filter by MOVEMENT
    filterMove_df = group_worm.filter((lambda x: x['x'].ptp() > threshold_move or 
                                                 x['y'].ptp() > threshold_move))
    
    # Re-group by worm id
    group_worm = filterMove_df.groupby('worm_id')
    n_worms_alive = group_worm.count().shape[0]
    
    if verbose:
        print("%d worm IDs filtered that moved less than %d pixels (%d microns)." %\
              (n_worms_time - n_worms_alive, threshold_move, threshold_move * microns_per_pixel))
        print("%d/%d worm IDs filtered in total." % (n_worms - n_worms_alive, n_worms))
        
    return(filterMove_df)
