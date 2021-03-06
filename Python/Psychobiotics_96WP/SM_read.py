#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULE: READ

@author: sm5911
@date: 27/02/2019

"""

# IMPORTS
import os, h5py, tables
import pandas as pd
import numpy as np
import warnings

# CUSTOM IMPORTS
from SM_find import lookforfiles, listdiff

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
#                              {'midbody_speed': f['timeseries_data']['speed_midbody']}
    return(df)
#        f = tables.open_file(featuresfilepath)
#        table = f.get
        

#%%    
def getskeldata(skeletonfilepath):
    """ A function to read Tierpsy-generated skeleton file data and extract the
        following information as a dataframe:
        ['roi_size', 'midbody_speed'] 
    """
    # Read HDF5 file + extract info
    try:
        with h5py.File(skeletonfilepath, 'r') as f:
            df = pd.DataFrame({'roi_size': f['trajectories_data']['midbody_speed']})
    except:
        print("Unable to read file with 'H5py', trying to read with 'PyTables..")
        f = tables.open()
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
def getfeatsums(directory):
    """ A function to load feature summary data from a given directory and return
        a dataframe of features along with a corresponding dataframe of file names.
        A unique ID key is used to maintain file identity. """
        
    # Obtain all feature summaries and filename summaries in given directory
    file_summary_list = lookforfiles(directory, '^filenames_summary*', depth=1)
    feat_summary_list = lookforfiles(directory, '^features_summary*', depth=1)
        
    # CHECK: Match file and features summaries
    matched_summaries = []
    for file in file_summary_list:
        feat = file.replace('filenames_summary', 'features_summary')
        if feat in feat_summary_list:
            matched_summaries.append([file,feat])
        else:
            warnings.warn('No match found for: \n%s' % file)
            
    if len(matched_summaries) > 1:
        print("ERROR: Multiple feature summary files found in directory: '%s'" % directory)
    elif len(matched_summaries) < 1:
        print("ERROR: No feature summary results found.")

    # Read matched feature summaries and filename summaries
    file, feat = matched_summaries[0]
    files_df = pd.read_csv(file)
    feats_df = pd.read_csv(feat)
        
    # Remove entries from file summary data where corresponding feature summary data is missing
    missing_featfiles = listdiff(files_df['file_id'], feats_df['file_id'])
    files_df = files_df[np.logical_not(files_df['file_id'].isin(missing_featfiles))]
        
    files_df.reset_index(drop=True, inplace=True)
    feats_df.reset_index(drop=True, inplace=True)
                
    # Check that file_id column matches
    if (files_df['file_id'] != feats_df['file_id'].unique()).any():
        print("ERROR: Features summary and filenames summary do not match!")
    else:
        return files_df, feats_df

