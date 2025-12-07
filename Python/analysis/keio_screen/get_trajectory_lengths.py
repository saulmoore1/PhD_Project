#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract trajectories data and calculate trajectory length for each worm ID from featuresN file

@author: sm5911
@date: 3/7/24
"""

import argparse
import pandas as pd
from pathlib import Path
from time import time

#EXAMPLE_FILE = "/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Plate_Tap/Results/20220803/keio_tap_run1_bluelight_20220803_151821.22956805/metadata_featuresN.hdf5"

def read_trajectories(featuresN_file, names=None, only_wells=None):
    """
    Read trajectories from a *featuresN.hdf5 file from tierpsy.
    Input:
        featuresN_file = name of file.
        names = names of series data to read.
                If none, all series data will be read.
        only_wells = list of well_names to read from the featuresN.hdf5 file.
                     If None, the data will not be filtered by well (good for legacy data).
    Output: trajectories_data series data.
    
    NB: This function is taken from tierpsytools.read_data.get_timeseries import read_timeseries
        and has been edited to extract trajectories instead of timeseries

    """
    
    if only_wells is not None:
      assert isinstance(only_wells, list), 'only_wells must be a list, or None'
      assert all(isinstance(well, str) for well in only_wells), \
          'only_wells must be a list of strings'
          
    with pd.HDFStore(featuresN_file, 'r') as f:
        if only_wells is None:
            series = f['trajectories_data']
        else:
            query_str = 'well_name in {}'.format(only_wells)
            series = f['trajectories_data'].query(query_str)
            
    if names is None:
        return series
    else:
        return series[names]


def trajectory_lengths(featuresN_file, names=None, only_wells=None):
    """
    Extract trajectories data from a *featuresN.hdf5 file and calculate worm trajectory lengths for
    each unique worm ID ('worm_index_joined') in the video/well(s).
    Input: 
        featuresN_file = name of file.
        names = names of series data to read.
                If none, all series data will be read.
        only_wells = list of well_names to read from the featuresN.hdf5 file.
                     If None, the data will not be filtered by well (good for legacy data).
    Output: 
        worm_trajectory_lengths = pandas.DataFrame of trajectory lengths (number of frames) for
        ach unique worm ID ('worm_index_joined').
    """
    
    # read trajectories data
    try:
        trajectories_df = read_trajectories(featuresN_file,
                                            names=names,
                                            only_wells=only_wells)        
    except Exception as e:
        print("\nWARNING! Could not read trajectories data from file:\n%s\n%s" % (featuresN_file, e))
        
    # extract trajectory lengths for each worm ID (worm_index_joined)
    worm_ids = trajectories_df.worm_index_joined.unique()
    
    grouped = trajectories_df.groupby('worm_index_joined')
    
    tick = time()
    worm_trajectory_lengths = {}
    for worm in worm_ids:
        traj_length = grouped.get_group(worm).shape[0]
        worm_trajectory_lengths[worm] = traj_length
    worm_trajectory_lengths = pd.DataFrame.from_dict(worm_trajectory_lengths, 
                                                     orient='index', 
                                                     columns=['length_n_frames']
                                                     ).reset_index(drop=False).rename(
                                                         columns={'index': 'worm_index_joined'})
    print('Trajectories data extracted in %.3f seconds' %(time()-tick))
    
    # tick=time()
    # worm_trajectory_lengths = pd.DataFrame(index=range(len(worm_ids)), 
    #                                        columns=['worm_index_joined','length_n_frames'])
    # for i, worm in enumerate(worm_ids):
    #     traj_length = grouped.get_group(worm).shape[0]
    #     worm_trajectory_lengths.iloc[i] = [worm, traj_length]
    # print('Done in %f seconds' %(time()-tick))
    # NOTE: Dictionary method is 2x faster - convert to dataframe afterwards
    
    return worm_trajectory_lengths
                                   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--featuresN_file',
                        help="Path to featuresN HDF5 file to extract trajectories data from.",
                        default=None, type=str) #default=EXAMPLE_FILE
    parser.add_argument('--names',
                        help="Names of series data to read. If none, all series data will be read.",
                        default=None, type=list)
    parser.add_argument('--only_wells',
                        help="List of well names to read from the featuresN file. If None, the data will not be filtered by well.",
                        default=None, type=list)
    args = parser.parse_args()
    assert args.featuresN_file is not None and Path(args.featuresN_file).is_file()
    
    worm_trajectory_lengths = trajectory_lengths(args.featuresN_file, args.names, args.only_wells)
    print("Trajectory lengths calculated for %d worms" % worm_trajectory_lengths.shape[0])
    
    
    