#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keio Mutant Worm Screen - Response to BW vs fepD bacteria


@author: sm5911
@date: 30/03/2022

"""

#%% Imports

from pathlib import Path
from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Keio_Mutant_Worm_6WP"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Mutant_Worm"

IMAGING_DATES = ['20220305','20220314','20220321']
N_WELLS = 6

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
MIN_NSKEL_PER_VIDEO = None
MIN_NSKEL_SUM = 500

WINDOW_DICT = {0:(1790,1800), 1:(1805,1815), 2:(1815,1825),
               3:(1850,1860), 4:(1865,1875), 5:(1875,1885),
               6:(1910,1920), 7:(1925,1935), 8:(1935,1945)}

# 1790:1800, 1805:1815, 1815:1825, 1850:1860, 1865:1875, 1875:1885, 1910:1920, 1925:1935, 1935:1945

#%% Functions

def mutant_worm():
    
    return

#%% Main

if __name__ == '__main__':
    
    AUX_DIR = Path(PROJECT_DIR) / "AuxiliaryFiles"
    RES_DIR = Path(PROJECT_DIR) / "Results"
    
    # compile metadata
    metadata, metadata_path = compile_metadata(aux_dir=AUX_DIR, 
                                               imaging_dates=IMAGING_DATES, 
                                               n_wells=N_WELLS,
                                               add_well_annotations=N_WELLS==96,
                                               from_source_plate=False)
    
    # compile window summaries
    features, metadata = process_feature_summaries(metadata_path=metadata_path, 
                                                   results_dir=RES_DIR, 
                                                   compile_day_summaries=True,
                                                   imaging_dates=IMAGING_DATES, 
                                                   align_bluelight=False,
                                                   window_summaries=True,
                                                   n_wells=N_WELLS)
    
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

    # save features
    features_path = Path(SAVE_DIR) / 'features.csv'
    features.to_csv(features_path, index=False) 

    # Save metadata
    metadata_path = Path(SAVE_DIR) / 'metadata.csv'
    metadata.to_csv(metadata_path, index=False)
    
    #mutant_worm(metadata, features)
    
    
