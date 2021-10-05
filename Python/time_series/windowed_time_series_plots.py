#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windowed time-series plots

@author: sm5911
@date: 01/10/2021

"""

#%% Imports

import pandas as pd
from pathlib import Path
from time import time
from tqdm import tqdm

from read_data.read import load_json, load_topfeats, read_list_from_file

from tierpsytools.read_data.hydra_metadata import read_hydra_metadata, align_bluelight_conditions
from tierpsytools.read_data.compile_features_summaries import compile_tierpsy_summaries
    
#%% Globals

JSON_PARAMETERS_PATH = "analysis/20210719_parameters_keio_screen.json"

STRAIN_LIST_PATH = "/Volumes/hermes$/KeioScreen_96WP/Analysis/52_selected_strains_from_initial_keio_top100_lowest_pval_tierpsy16.txt"

DATE = '20210914'

#%% Functions

def find_window(window_summary_path):
    """ Function to return window number from window feature summary filepath for sorting """
    
    window_number = str(window_summary_path).split('_')[-1].split('.csv')[0]
    
    return int(window_number)

def find_window_summaries(window_dir, pattern='*_window_*'):
    """ Recursive search the provided directory for window summaries features/filenames files and 
        return matched lists of window features summaries and window filenames summaries
    """

    # glob search to find window summaries files
    window_summary_paths = list(window_dir.rglob(pattern))
    window_feats = [f for f in window_summary_paths if 'features' in str(f)]
    window_feats.sort(key=find_window)
    window_files = [f for f in window_summary_paths if 'filenames' in str(f)]
    window_files.sort(key=find_window)
    
    assert all(find_window(f[0]) == find_window(f[1]) for f in zip(window_feats, window_files))
    
    return window_feats, window_files

#%% Main

if __name__ == "__main__":
    tic = time()
    
    args = load_json(JSON_PARAMETERS_PATH)
    
    strain_list = read_list_from_file(STRAIN_LIST_PATH)
    
    window_dir = Path(args.project_dir) / 'Results'
    window_date_dir = window_dir / str(DATE)
    
    window_feats, window_files = find_window_summaries((window_date_dir if DATE is not None 
                                                        else window_dir), pattern='*_window_*')
    
    window_list = [find_window(p) for p in window_feats]
    
    for w in tqdm(window_list):
        
        compiled_feat_path = window_dir / 'features_summary_compiled_window_{}.csv'.format(str(w))
        compiled_file_path = window_dir / 'filenames_summary_compiled_window_{}.csv'.format(str(w))
        
        w_feat = [f for f in window_feats if find_window(f) == w]
        w_file = [f for f in window_files if find_window(f) == w]
        
        # assert that only one windows file exists for that window
        assert len(w_feat) == 1 and len(w_file) == 1
        
        # compile features/filenames summaries for window
        compile_tierpsy_summaries(feat_files = w_feat, 
                                  fname_files = w_file,
                                  compiled_feat_file = compiled_feat_path,
                                  compiled_fname_file = compiled_file_path)
        
    compiled_window_feats, compiled_window_files = find_window_summaries(window_dir, pattern='*_compiled_window_*.csv')

    features_path = Path(args.project_dir) / 
    metadata_path = Path()
    
    # load metadata
    features, metadata = read_hydra_metadata()
        
    toc = time()
    print("Done in %.1f seconds" % (toc - tic))

