#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compile wiindow summaries

@author: sm5911
@date: 25/11/2021

"""

#%% Imports

import argparse
import pandas as pd
from pathlib import Path
from time import time
from tqdm import tqdm

from tierpsytools.read_data.get_feat_summaries_helper import read_tierpsy_feat_summaries

#%% Globals

RESULTS_DIR = '/Volumes/hermes$/Keio_Fast_Effect/Results'
IMAGING_DATES = ['20211109']
N_WELLS = 6

#%% Functions

def parse_window_number(fname):
    """
    Parse the filename to find the number between 'window_' and '.csv'
    (with .csv being at the end of the name)
    """
    import re    
    
    regex = r'(?<=window_)\d+(?=\.csv)'
    window_str = re.findall(regex, fname.name)[0]
    
    return int(window_str)

def find_window_summaries(results_dir, dates=None):
    """ 
    Search project root directory for windows summary files 
    """
    
    results_dir = Path(results_dir)
    assert results_dir.exists()

    # find windows summary files
    windows_file_list = []
    if dates is not None and type(dates) == list:
        for date in dates:
            windows_files = list((results_dir / date).glob('*summary*window*csv'))
            windows_file_list.extend(windows_files)
    else:
        windows_files = list(results_dir.rglob('*summary*window*csv'))
        windows_file_list.append()
    
    filenames_summary_files = [f for f in windows_file_list if str(f.name).startswith('filenames')]  
    features_summary_files = [f for f in windows_file_list if str(f.name).startswith('features')]
    
    # match filenames and features summaries with their respective windows
    matched = []
    for fname_file in filenames_summary_files:
        window = parse_window_number(fname_file) # get window number
        
        # find matching feature summary file for window
        feat_file_list = [f for f in features_summary_files if parse_window_number(f)==window]
        
        if len(feat_file_list) == 0:
            print('\nERROR: Cannot match filenames summary file: %s' % fname_file)
            raise OSError('No features summary file found for window %d' % window)
        elif len(feat_file_list) > 1:
            print('\nERROR: Multiple matched found for filenames summary file: %s' % fname_file)
            raise OSError('Multiple features summary files found for window %s' % window)   
            
        feat_file = feat_file_list[0]
        
        matched.append((fname_file, feat_file))
    
    # extract filenames and features windows summary files from matched
    filenames_summary_files = []
    features_summary_files = []
    for (fname_file, feat_file) in sorted(matched):
        filenames_summary_files.append(fname_file)
        features_summary_files.append(feat_file)
    
    return filenames_summary_files, features_summary_files

def get_channels_from_filenames(filenames):
    """ Extract camera serial number from each file in filenames pd.Series object
    
        Inputs
        ------
        filenames : pd.Series
            filenames column from tierpsy filenames summaries file, containing list of filenames
            
        Returns
        -------
        camera_serials : pd.Series
            list of extracted camera serial number for each file in filenames
    """
    from tierpsytools.hydra import CAM2CH_df

    CAM2CH_DICT = {s:(c.split('Ch')[-1],r) for (s,c,r) in zip(CAM2CH_df['camera_serial'], 
                                                              CAM2CH_df['channel'], 
                                                              CAM2CH_df['rig'])}
    
    camera_serials = [Path(file).parent.name.split('.')[-1] for file in filenames]
    channels = [CAM2CH_DICT[s][0] for s in camera_serials]

    return pd.Series(channels)

def compile_window_summaries(fname_files, 
                             feat_files, 
                             compiled_fnames_path, 
                             compiled_feats_path, 
                             results_dir, 
                             window_list=None,
                             n_wells=96):
    """ Compile window summaries files from matching lists of filenames and features windows 
        summary files 
    """

    if window_list is not None:
        assert type(window_list) == list
    else:
        window_list = sorted([parse_window_number(f) for f in fname_files])
    
    window_dict = {parse_window_number(fname):(fname,feat) for fname,feat in zip(fname_files,feat_files)}
    
    filenames_summaries_list = []
    features_summaries_list = []
    for window in tqdm(window_list):
        fname_file, feat_file = window_dict[window]
        
        # read filenames/features summary for window and append to list of dataframes
        filenames_df, features_df = read_tierpsy_feat_summaries(feat_file, fname_file)
        filenames_df = filenames_df[['file_id','filename','is_good']]
        
        assert all(str(results_dir) in str(f) for f in filenames_df['filename'])
        assert all(i == j for i, j in zip(filenames_df['file_id'], features_df['file_id']))

        # store window number (unique identifier = file_id + window)
        filenames_df['window'] = window
        features_df['window'] = window
        
        # add well name as channel number if n_wells == 6
        if n_wells == 6:
            channels = get_channels_from_filenames(filenames_df['filename'])
            filenames_df['well_name'] = channels
            features_df['well_name'] = channels
       
        # append to list of dataframes
        filenames_summaries_list.append(filenames_df)
        features_summaries_list.append(features_df)
       
    # compile full filenames/features summaries from list of dataframes
    compiled_filenames_summaries = pd.concat(filenames_summaries_list, axis=0, sort=False)
    compiled_features_summaries = pd.concat(features_summaries_list, axis=0, sort=False)
    
    # save compiled window summaries to csv
    print("\nSaving window summaries to file..")
    Path(compiled_fnames_path).parent.mkdir(parents=True, exist_ok=True)
    compiled_filenames_summaries.to_csv(compiled_fnames_path, index=False)
    
    Path(compiled_feats_path).parent.mkdir(parents=True, exist_ok=True)
    compiled_features_summaries.to_csv(compiled_feats_path, index=False)
    
    return compiled_filenames_summaries, compiled_features_summaries

#%% Main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compile window summaries for project results \
    directory and imaging dates provided')
    parser.add_argument('-r','--results_dir', help="Path to project results directory, containing \
    'YYYYMMDD' imaging date folders with windows summary files to be compiled", default=RESULTS_DIR)
    parser.add_argument('-d','--imaging_dates', help="Selected imaging dates to compile window summaries",
                        default=IMAGING_DATES)
    parser.add_argument('--window_numbers', help="List of window numbers to compile results for",
                        nargs='+', default=None)
    parser.add_argument('--n_wells', help="Number of wells imaged under each Hydra rig \
    (choose between 96 or 6 wells)", default=N_WELLS)
    args = parser.parse_args()
    
    tic = time()
    
    # find window summaries files
    print("\nFinding window summaries files..")
    fname_files, feat_files = find_window_summaries(results_dir=args.results_dir, dates=args.imaging_dates)
    
    # compile window summaries files
    print("\nCompiling window summaries..")
    compiled_fnames_path = Path(args.results_dir) / 'full_window_filenames.csv'
    compiled_feats_path = Path(args.results_dir) / 'full_window_features.csv'
    
    compiled_filenames, compiled_features = compile_window_summaries(fname_files, 
                                                                     feat_files,
                                                                     compiled_fnames_path=compiled_fnames_path,
                                                                     compiled_feats_path=compiled_feats_path,
                                                                     results_dir=args.results_dir, 
                                                                     window_list=args.window_numbers,
                                                                     n_wells=args.n_wells)
    
    print("\nDone! (%.1f seconds)" % (time()-tic))
