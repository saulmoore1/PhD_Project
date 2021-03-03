#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read Trajectory Data

@author: sm5911
@date: 02/01/2021

"""

# FIXME: Compatibility with Hydra camera 16-well videos + maintain backwards compatibility

#%% Functions

def get_trajectory_data(featuresfilepath, rig='Phenix'):
    """ A function to read Tierpsy-generated featuresN file trajectories data
        and extract the following info as a dataframe:
        ['coord_x', 'coord_y', 'frame_number', 'worm_index_joined'] """

    import h5py
    import pandas as pd

    if rig.lower() == 'phenix':
        # Read HDF5 file + extract info
        with h5py.File(featuresfilepath, 'r') as f:
            df = pd.DataFrame({'x': f['trajectories_data']['coord_x'],\
                               'y': f['trajectories_data']['coord_y'],\
                               'frame_number': f['trajectories_data']['frame_number'],\
                               'worm_id': f['trajectories_data']['worm_index_joined']})
        # {'midbody_speed': f['timeseries_data']['speed_midbody']}
    elif rig.lower() == 'hydra':
        raise IOError('FIXME: Update get trajectory data for Hydra videos')
        
    return(df)
