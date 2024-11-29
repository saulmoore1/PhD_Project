#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse Initial Keio Screen 

- compile metadata and feature summaries
- clean metadata and feature summaries
- run analysis

@author: sm5911
@date: 28/11/2024

"""

#%% Imports

import pandas as pd
from pathlib import Path

from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from preprocessing.append_supplementary_info import load_supplementary_7, append_supplementary_7


#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Screen_Initial"
SAVE_DIR = "/Users/sm5911/Documents/PhD_DLBG/3_Keio_Screen_Initial"

PATH_SUP_INFO = Path(PROJECT_DIR) / "AuxiliaryFiles/Baba_et_al_2006/Supporting_Information/Supplementary_Table_7.xls"

EXPERIMENT_DATES = ["20210406", "20210413", "20210420", "20210427", "20210504", "20210511"]
N_TOP_FEATS = 256
MIN_NSKEL_SUM = 6000

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
P_VALUE_THRESHOLD = 0.05
FDR_METHOD = 'fdr_bh' # fdr_by
N_WELLS = 96
N_SIG_FEATS = 50

RENAME_DICT = {"FECE" : "fecE",
               "AroP" : "aroP",
               "TnaB" : "tnaB"}

#BLUELIGHT_TIMEPOINTS_SECONDS = [(60, 70),(160, 170),(260, 270)]

#%% Functions

def stats(metadata,
          features,
          save_dir=None,
          feature_list=None,
          p_value_threshold=0.05,
          fdr_method='fdr_by',
          group_by='gene_name',
          control='BW'):
    
    """ Perform ANOVA and t-tests to compare worms on each bacterial food vs BW25113 control 
        for each Tierpsy feature in feature_list """

    assert all(metadata.index == features.index)
    
    if feature_list is None:
        feature_list = list(features.columns)
    else:
        assert type(feature_list) == list
        assert all(f in features.columns for f in feature_list)
        features = features[feature_list]
        
    n_strains = metadata[group_by].nunique()
    
    metadata.groupby(group_by)
    
    
    return anova_results, ttest_results

def main():
    
    metadata_path_local = Path(SAVE_DIR) / 'metadata.csv'
    features_path_local = Path(SAVE_DIR) / 'features.csv'
    
    aux_dir = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    res_dir = Path(PROJECT_DIR) / 'Results'
    
    if not metadata_path_local.exists() or not features_path_local.exists():

        # compile metadata and feature summaries        
        metadata, metadata_path = compile_metadata(aux_dir, 
                                                   imaging_dates=EXPERIMENT_DATES,
                                                   n_wells=N_WELLS, 
                                                   add_well_annotations=True,
                                                   from_source_plate=False)
        
        features, metadata = process_feature_summaries(metadata_path, 
                                                       results_dir=res_dir, 
                                                       compile_day_summaries=True, 
                                                       imaging_dates=EXPERIMENT_DATES, 
                                                       align_bluelight=True, 
                                                       window_summaries=False,
                                                       n_wells=N_WELLS)

        # Fill in 'gene_name' for control as 'wild_type' (for control plates where gene name is missing)
        metadata.loc[metadata['source_plate_id'] == "BW", 'gene_name'] = "wild_type"

        # Rename gene names in metadata
        for k, v in RENAME_DICT.items():
            metadata.loc[metadata['gene_name'] == k, 'gene_name'] = v

        # Clean results - Remove bad well data + features with too many NaNs/zero std + impute NaNs
        features, metadata = clean_summary_results(features,
                                                   metadata,
                                                   feature_columns=None,
                                                   nan_threshold_row=NAN_THRESHOLD_ROW,
                                                   nan_threshold_col=NAN_THRESHOLD_COL,
                                                   max_value_cap=1e15,
                                                   imputeNaN=True,
                                                   min_nskel_per_video=None,
                                                   min_nskel_sum=MIN_NSKEL_SUM,
                                                   drop_size_related_feats=False,
                                                   norm_feats_only=False,
                                                   percentile_to_use=None,
                                                   no_nan_cols=['worm_strain','gene_name'])
        
        # Add COG category info from Baba et al. (2006) supplementary info to metadata
        if not 'COG_category' in metadata.columns:
            supplementary_7 = load_supplementary_7(PATH_SUP_INFO)
            metadata = append_supplementary_7(metadata, supplementary_7, column_name='gene_name')
            assert set(metadata.index) == set(features.index)    
        COG_families = {'Information storage and processing' : ['J', 'K', 'L', 'D', 'O'], 
                        'Cellular processes' : ['M', 'N', 'P', 'T', 'C', 'G', 'E'], 
                        'Metabolism' : ['F', 'H', 'I', 'Q', 'R'], 
                        'Poorly characterised' : ['S', 'U', 'V']}
        COG_mapping_dict = {i : k for (k, v) in COG_families.items() for i in v}        
        COG_info = []
        for i in metadata['COG_category']:
            try:
                COG_info.append(COG_mapping_dict[i])
            except:
                COG_info.append('Unknown') # np.nan
        metadata['COG_info'] = COG_info
        
        # save clean metadata and features
        metadata.to_csv(metadata_path_local, index=False)
        features.to_csv(features_path_local, index=False)
        
    else:
        metadata = pd.read_csv(metadata_path_local, header=0, index_col=None, dtype={'comments':str})
        features = pd.read_csv(features_path_local, header=0, index_col=None)


    return

#%% Main

if __name__ == '__main__':
    main()
