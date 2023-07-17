#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse bacterial mutants that were created based on results of fepD proteomics analysis in order
to identify the pathway(s) responsible for arousal on fepD

@author: sm5911
@date: 11/07/2023

"""

#%% Imports

import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from tierpsytools.plot.plot_plate_from_raw_video import plot_plates_from_metadata

#%% Globals 

PROJECT_DIR = "/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Bacteria_Mutants"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Bacteria_Mutants"

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
MIN_NSKEL_PER_VIDEO = None
MIN_NSKEL_SUM = 500
P_VALUE_THRESHOLD = 0.05
FDR_METHOD = 'fdr_bh'

FEATURE = 'speed_50th'
N_WELLS = 6
DPI = 600

BLUELIGHT_TIMEPOINTS_SECONDS = [(60, 70),(160, 170),(260, 270)]
WINDOW_DICT = {0:(290,300)}
WINDOW_NAME_DICT = {0:"20-30 seconds after blue light 3"}

#%% Functions 

def main():
    
    metadata_path_local = Path(SAVE_DIR) / 'metadata.csv'
    features_path_local = Path(SAVE_DIR) / 'features.csv'
    
    aux_dir = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    res_dir = Path(PROJECT_DIR) / 'Results'
    
    if not metadata_path_local.exists() or not features_path_local.exists():

        # compile metadata and feature summaries
        metadata, metadata_path = compile_metadata(aux_dir, 
                                                   n_wells=N_WELLS, 
                                                   add_well_annotations=False,
                                                   from_source_plate=True)
        
        features, metadata = process_feature_summaries(metadata_path, 
                                                       results_dir=res_dir, 
                                                       compile_day_summaries=True, 
                                                       imaging_dates=None, 
                                                       align_bluelight=False, 
                                                       window_summaries=True,
                                                       n_wells=N_WELLS)

        # Clean results - Remove bad well data + features with too many NaNs/zero std + impute NaNs
        features, metadata = clean_summary_results(features, 
                                                   metadata,
                                                   feature_columns=None,
                                                   nan_threshold_row=NAN_THRESHOLD_ROW,
                                                   nan_threshold_col=NAN_THRESHOLD_COL,
                                                   max_value_cap=1e15,
                                                   imputeNaN=True,
                                                   min_nskel_per_video=None,
                                                   min_nskel_sum=None,
                                                   drop_size_related_feats=False,
                                                   norm_feats_only=False)
        
        # save clean metadata and features
        metadata.to_csv(metadata_path_local, index=False)
        features.to_csv(features_path_local, index=False)
        
    else:
        metadata = pd.read_csv(metadata_path_local, header=0, index_col=None, dtype={'comments':str})
        features = pd.read_csv(features_path_local, header=0, index_col=None)

    assert not features.isna().sum(axis=1).any()
    assert not (features.std(axis=1) == 0).any()    
    
    #plot_plates_from_metadata(metadata, save_dir=Path(SAVE_DIR), dpi=600)
    
    return

#%% Main

if __name__ == '__main__':
    main()
