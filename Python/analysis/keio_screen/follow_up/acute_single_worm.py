#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compile acute single worm metadata and feature summaries

@author: sm5911
@date: 30/03/2022

"""

#%% Imports

import pandas as pd
from pathlib import Path
from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from statistical_testing.stats_helper import single_feature_window_stats

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Keio_Acute_Single_Worm"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Acute_Single_Worm"

IMAGING_DATES = ['20220206','20220209','20220212']
N_WELLS = 6

CONTROL_STRAIN = 'BW'

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
MIN_NSKEL_PER_VIDEO = None
MIN_NSKEL_SUM = 50
PVAL_THRESH = 0.05

WINDOW_DICT_SECONDS = {0:(290,300), 1:(305,315), 2:(315,325), 
                       3:(590,600), 4:(605,615), 5:(615,625), 
                       6:(890,900), 7:(905,915), 8:(915,925), 
                       9:(1190,1200), 10:(1205,1215), 11:(1215,1225), 
                       12:(1490,1500), 13:(1505.1515), 14:(1515,1525)}

WINDOW_NUMBERS = [12,13,14]

#%% Functions

def acute_single_worm(metadata, features, project_dir, save_dir, window_list):
    
    
    assert all(w in WINDOW_DICT_SECONDS.keys() for w in window_list)
    grouped_window = metadata.groupby('window')
    for window in window_list:
        window_metadata = grouped_window.get_group(window)
        
    

#%% Main

if __name__ == "__main__":
    
    AUX_DIR = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    RES_DIR = Path(PROJECT_DIR) / 'Results'

    META_PATH = Path(SAVE_DIR) / 'metadata.csv'
    FEAT_PATH = Path(SAVE_DIR) / 'features.csv'
    
    if not META_PATH.exists() and FEAT_PATH.exists():
    
        # compile metadata
        metadata, metadata_path = compile_metadata(aux_dir=AUX_DIR,
                                                   imaging_dates=IMAGING_DATES,
                                                   add_well_annotations=N_WELLS==96,
                                                   n_wells=N_WELLS,
                                                   from_source_plate=False)
            
        # compile window summaries
        features, metadata = process_feature_summaries(metadata_path,
                                                       results_dir=RES_DIR,
                                                       compile_day_summaries=True,
                                                       imaging_dates=IMAGING_DATES,
                                                       align_bluelight=False,
                                                       window_summaries=True)
        
        # clean results
        features, metadata = clean_summary_results(features, 
                                                   metadata,
                                                   feature_columns=None,
                                                   nan_threshold_row=NAN_THRESHOLD_ROW,
                                                   nan_threshold_col=NAN_THRESHOLD_COL,
                                                   max_value_cap=1e15,
                                                   imputeNaN=True,
                                                   min_nskel_per_video=MIN_NSKEL_PER_VIDEO,
                                                   min_nskel_sum=MIN_NSKEL_SUM,
                                                   drop_size_related_feats=False,
                                                   norm_feats_only=False,
                                                   percentile_to_use=None)
    
        assert not features.isna().sum(axis=1).any()
        assert not (features.std(axis=1) == 0).any()

        # rename 'bacteria_strain' column to 'gene_name' in metadata
        metadata = metadata.rename(columns={'bacteria_strain': 'gene_name'})
        
        # save clean metadata and features
        metadata.to_csv(META_PATH, index=False)
        features.to_csv(FEAT_PATH, index=False) 
    
    else:
        # load clean metadata and features
        metadata = pd.read_csv(META_PATH, dtype={'comments':str, 'source_plate_id':str})
        features = pd.read_csv(FEAT_PATH, index_col=None)
            
    # statistics: perform pairwise t-tests comparing fepD vs BW at each window 
    single_feature_window_stats(metadata,
                                features,
                                group_by='gene_name',
                                control=CONTROL_STRAIN,
                                save_dir=SAVE_DIR,
                                windows=WINDOW_NUMBERS, #sorted(WINDOW_DICT_SECONDS.keys()),
                                pvalue_threshold=PVAL_THRESH)
    
    # plotting: pairwise box plots comparing fepD vs BW at each timepoint
    #acute_single_worm(metadata, features, PROJECT_DIR, SAVE_DIR)

