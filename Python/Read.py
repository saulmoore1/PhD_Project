#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULE: READ

@author: sm5911
@date: 27/02/2019

"""

# IMPORTS
import os, h5py
import pandas as pd
import numpy as np

#%% FUNCTIONS
def gettrajdata(featuresfilepath):
    """ A function to read Tierpsy-generated featuresN file trajectories data
        and extract the following info as a dataframe:
        ['coord_x', 'coord_y', 'frame_number', 'worm_index_joined'] """
    # Read HDF5 file + extract info
    with h5py.File(featuresfilepath, 'r') as f:
        df = pd.DataFrame({'x': f['trajectories_data']['coord_x'],\
                           'y': f['trajectories_data']['coord_y'],\
                           'frame_number': f['trajectories_data']['frame_number'],\
                           'worm_id': f['trajectories_data']['worm_index_joined']})
    # {'midbody_speed': f['timeseries_data']['speed_midbody']}
    return(df)

#%%    
def getskeldata(skeletonfilepath):
    """ A function to read Tierpsy-generated skeleton file data and extract the
        following information as a dataframe:
        ['roi_size', 'midbody_speed'] 
    """
    # Read HDF5 file + extract info
    with h5py.File(skeletonfilepath, 'r') as f:
        df = pd.DataFrame({'roi_size': f['trajectories_data']['midbody_speed']})
    return(df)

#%%
def getauxinfo(maskedvideopath, sheet=0):
    """ A function to retrieve auxiliary file data for a given masked HDF5 video. """
    # Extract set + camera info from filename string
    Set = int(maskedvideopath.split('/')[-2].split('Set')[-1])
    Camera = int(maskedvideopath.split('/')[-1].split('_')[1].split('Ch')[-1])
    # Locate + read auxiliary file (Excel Workbook)
    auxfilepath = os.path.join(maskedvideopath.replace('MaskedVideos','AuxiliaryFiles').split('/PC')[0],\
                                                   'ExperimentDetails.xlsx')
    aux_workbook = pd.ExcelFile(auxfilepath)
    worksheet = aux_workbook.sheet_names[sheet]
    if sheet == 0:
        aux_info = aux_workbook.parse(worksheet, skiprows=5, header=0, index_col=None)
        # Locate row in auxiliary data corresponding to queried video file using 
        # set number and camera number, then slice just that row + return it
        aux_info = aux_info[np.logical_and(aux_info['Camera_N']==Camera, aux_info['Set_N']==Set)]
    elif sheet == 1:
        aux_info = aux_workbook.parse(worksheet, header=None, index_col=None, squeeze=True)
    return(aux_info)

#%%