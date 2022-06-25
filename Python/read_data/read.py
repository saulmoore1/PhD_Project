#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read data from file

@author: sm5911
@date: 03/03/2021

"""

#%% Functions

class dict2obj:
    """ A simple class to convert a dictionary into an object """
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
def load_json(json_path):
    """ Read parameters from JSON file and convert to a Python object """
    
    import json

    raw_json = open(json_path, 'r').read()
    args = json.loads(raw_json)
    
    args = dict2obj(**args)
    
    return args
    
def load_topfeats(topfeats_path, add_bluelight=True, remove_path_curvature=True, verbose=True, header=0):
    """ Load Tierpsy Top256 set of features describing N2 behaviour on E. coli OP50 bacteria 
        
        Parameters
        ----------
        topfeats_path : str
            Path to Tierpsy feature set containing a list of features to use foor the analysis
        add_bluelight : bool
            Append feature set separately for each bluelight condition
        remove_path_curvature : bool
            Omit path curvature-related feature from analysis
        verbose : bool
            Print progress to std out
            
        Returns
        -------
        feature list
    """   
    import pandas as pd
    
    topfeats_df = pd.read_csv(topfeats_path, header=header)
    toplist = list(topfeats_df[topfeats_df.columns[0]])
    n = len(toplist)
    if verbose:
        print("Feature list loaded (n=%d features)" % n)

    # Remove features from Top256 that are path curvature related
    toplist = [feat for feat in toplist if "path_curvature" not in feat]
    n_feats_after = len(toplist)
    if verbose:
        print("Dropped %d features from Top%d that are related to path curvature"\
              % ((n - n_feats_after), n))
        
    if add_bluelight:
        bluelight_suffix = ['_prestim','_bluelight','_poststim']
        toplist = [col + suffix for suffix in bluelight_suffix for col in toplist]

    return toplist


def read_list_from_file(filepath):
    """ Read a multi-line list from text file """   
    
    list_from_file = []
    with open(filepath, 'r') as fid:
        for line in fid:
            list_from_file.append(line.strip('\n'))
    
    return list_from_file      

def get_fov_well_data(featuresfilepath):
    """ Read Tierpsy-generated featuresN HDF5 file data for 'fov_wells' and return as a dataframe """

    import h5py
    import pandas as pd
    
    # Read HDF5 file and extract fov_well info
    with h5py.File(featuresfilepath, 'r') as f:
        df = pd.DataFrame({'x_min': f['fov_wells']['x_min'],
                           'x_max': f['fov_wells']['x_max'],
                           'y_min': f['fov_wells']['y_min'],
                           'y_max': f['fov_wells']['y_max'],
                           'well_name': f['fov_wells']['well_name'],
                           'is_good_well': f['fov_wells']['is_good_well']})
        df['well_name'] = [i.decode("utf-8") for i in df['well_name']]
        
    return df

def get_trajectory_data(featuresfilepath):
    """ A function to read Tierpsy-generated featuresN HDF5 file trajectories data
        and extract the following info as a dataframe:
        ['coord_x', 'coord_y', 'frame_number', 'worm_index_joined'] """

    # TODO: deprecated replace with tierpsytools
    # from tierpsytools.read_data.get_timeseries import read_timeseries
    
    import h5py
    import pandas as pd

    # Read HDF5 file + extract info
    with h5py.File(featuresfilepath, 'r') as f:
        df = pd.DataFrame({'x': f['trajectories_data']['coord_x'],\
                           'y': f['trajectories_data']['coord_y'],\
                           'frame_number': f['trajectories_data']['frame_number'],\
                           'worm_id': f['trajectories_data']['worm_index_joined']})
    # {'midbody_speed': f['timeseries_data']['speed_midbody']}
    
    return(df)

def get_skeleton_data(skeletonfilepath, rig='Phenix', dataset='trajectories_data'):
    """ A function to read Tierpsy-generated skeleton file data and extract the dataset information 
        in dataframe format 
    """
    
    import h5py
    import tables
    import pandas as pd

    # Read HDF5 file + extract info    
    try:
        if rig.lower() == 'phenix':
            with h5py.File(skeletonfilepath, 'r') as f:
                df = pd.DataFrame({'roi_size': f[dataset]['midbody_speed']})
                
        elif rig.lower() == 'hydra':
            with pd.HDFStore(skeletonfilepath, 'r') as f:
                df = pd.DataFrame(f[dataset])
               
    except Exception as E:
            print("\n\nWARNING\n\n", E)
            print("\nUnable to read file with 'h5py', trying to read with 'PyTables..")
            f = tables.open_file(skeletonfilepath, mode='r')
            print(f)
            
    return(df)

def get_auxiliary_data(maskedvideopath, sheet=0, rig='Phenix'):
    """ A function to retrieve auxiliary file data for a given masked HDF5 video. """

    import os
    import numpy as np
    import pandas as pd
    
    if rig.lower() == 'phenix':
        # TODO: from tierpsytools.hydra.compile_metadata import get_camera_serial
        # Extract set + camera info from filename string
        Set = int(maskedvideopath.split('/')[-2].split('Set')[-1])
        Camera = int(maskedvideopath.split('/')[-1].split('_')[1].split('Ch')[-1])
        # Locate + read auxiliary file (Excel Workbook)
        auxfilepath = os.path.join(maskedvideopath.replace('MaskedVideos','AuxiliaryFiles').split('/PC')[0],\
                                                       'ExperimentDetails.xlsx')
        aux_workbook = pd.ExcelFile(auxfilepath, engine='openpyxl')
        worksheet = aux_workbook.sheet_names[sheet]
        
        if sheet == 0:
            aux_info = aux_workbook.parse(worksheet, skiprows=5, header=0, index_col=None)
            # Locate row in auxiliary data corresponding to queried video file using 
            # set number and camera number, then slice just that row + return it
            aux_info = aux_info[np.logical_and(aux_info['Camera_N']==Camera, aux_info['Set_N']==Set)]
        elif sheet == 1:
            aux_info = aux_workbook.parse(worksheet, header=None, index_col=None, squeeze=True)
    elif rig.lower() == 'hydra':
        # FIXME: Compatibility with Hydra camera 16-well videos + maintain backwards compatibility
        raise IOError('FIXME: Update get trajectory data for Hydra videos')
        
    return(aux_info)
