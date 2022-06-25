#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter Tierpsy worm trajectory data (Phenix only)

@author: sm5911
@date: 01/03/2021

"""

#%% Functions

def filter_worm_trajectories(trajectory_df, threshold_move=10, threshold_time=25,
                             fps=25, microns_per_pixel=10, worm_id_col='worm_id', 
                             timestamp_col='frame_number', x_coord_col='x', y_coord_col='y', 
                             verbose=True):
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
        worm_id_col : str
            Name of worm id column in trajectory dataframe, default = 'worm_id'
        x_coord_col, y_coord_col : str
            Same as above for x and y coord column names in trajectory dataframe
        verbose : bool
            Print statements to std out
            
        Returns
        -------
        Filtered trajectory dataframe
    """
        
    import numpy as np
    
    group_worm = trajectory_df.groupby(worm_id_col)
    n_worms = group_worm.count().shape[0]
    
    # Filter by TRAJECTORY LENGTH
    trajectory_df = group_worm.filter(lambda x: x[timestamp_col].count() > threshold_time)
    
    # Re-group by worm id
    group_worm = trajectory_df.groupby(worm_id_col)
    n_worms_time = group_worm.count().shape[0]
    
    if verbose:
        print("%d worm IDs filtered that existed for less than %d frames (%.1f seconds)." %\
              (n_worms - n_worms_time, threshold_time, threshold_time / fps))
            
    # Filter by MOVEMENT
    trajectory_df = group_worm.filter((lambda x: np.ptp(x[x_coord_col]) > threshold_move or 
                                                 np.ptp(x[y_coord_col]) > threshold_move))
    
    # Re-group by worm id
    group_worm = trajectory_df.groupby(worm_id_col)
    n_worms_alive = group_worm.count().shape[0]
    
    if verbose:
        print("%d worm IDs filtered that moved less than %d pixels (%d microns)." %\
              (n_worms_time - n_worms_alive, threshold_move, threshold_move * microns_per_pixel))
        print("%d/%d (%.1f%%) worm IDs filtered in total." % (n_worms-n_worms_alive, n_worms,
                                                              (n_worms-n_worms_alive)/n_worms*100))
        
    stats = {"short_worm_trajectories" : (n_worms - n_worms_time),
             "stationary_worm_trajectories" : (n_worms_time - n_worms_alive),
             "total_worm_trajectories" : n_worms}

    return (trajectory_df, stats)
