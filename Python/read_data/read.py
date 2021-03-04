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
    """ A function for reading parameters from JSON and converting to a Python object """
    
    import json

    raw_json = open(json_path, 'r').read()
    args = json.loads(raw_json)
    
    args = dict2obj(**args)
    
    return args
    
def load_top256(top256_path, add_bluelight=True, remove_path_curvature=True, verbose=True):
    """ Load Tierpsy Top256 set of features describing N2 behaviour on E. coli OP50 bacteria 
        
        Parameters
        ----------
        top256_path : str
            Path to Tierpsy Top256 feature list
        add_bluelight : bool
            Append feature set separately for each bluelight condition
        remove_path_curvature : bool
            Omit path curvature-related feature from analysis
        verbose : bool
            Print progress to std out
            
        Returns
        -------
        top256 feature list
    """   
    import pandas as pd
    
    top256_df = pd.read_csv(top256_path, header=0)
    top256 = list(top256_df[top256_df.columns[0]])
    n = len(top256)
    if verbose:
        print("Feature list loaded (n=%d features)" % n)

    # Remove features from Top256 that are path curvature related
    top256 = [feat for feat in top256 if "path_curvature" not in feat]
    n_feats_after = len(top256)
    if verbose:
        print("Dropped %d features from Top%d that are related to path curvature"\
              % ((n - n_feats_after), n))
        
    if add_bluelight:
        bluelight_suffix = ['_prestim','_bluelight','_poststim']
        top256 = [col + suffix for suffix in bluelight_suffix for col in top256]

    return top256


def read_list_from_file(filepath):
    """ Read a multi-line list from text file """   
    
    list_from_file = []
    with open(filepath, 'r') as fid:
        for line in fid:
            info = line[:-1]
            list_from_file.append(info)
    
    return list_from_file      

def get_trajectory_data(featuresfilepath):
    """ A function to read Tierpsy-generated featuresN file trajectories data
        and extract the following info as a dataframe:
        ['coord_x', 'coord_y', 'frame_number', 'worm_index_joined'] """

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

def get_skeleton_data(skeletonfilepath, rig='Phenix'):
    """ A function to read Tierpsy-generated skeleton file data and extract the
        following information as a dataframe:
        ['roi_size', 'midbody_speed'] 
    """
    
    import h5py
    import tables
    import pandas as pd

    if rig.lower() == 'phenix':
        # Read HDF5 file + extract info
        # TODO: Fix this
        try:
            with h5py.File(skeletonfilepath, 'r') as f:
                df = pd.DataFrame({'roi_size': f['trajectories_data']['midbody_speed']})
        except:
            print("Unable to read file with 'h5py', trying to read with 'PyTables..")
            f = tables.open()
    elif rig.lower() == 'hydra':
        # FIXME: Compatibility with Hydra camera 16-well videos + maintain backwards compatibility
        raise IOError('FIXME: Update get trajectory data for Hydra videos')
            
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
