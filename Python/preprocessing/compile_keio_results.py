#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compile Keio Screen metadata and feature summaries

@author: sm5911
@date: 21/05/2021
"""

#%% IMPORTS

import argparse
from time import time
from pathlib import Path
from read_data.read import load_json
from preprocessing.compile_hydra_data import process_metadata, process_feature_summaries
from preprocessing.append_supplementary_info import load_supplementary_7, append_supplementary_7
from filter_data.clean_feature_summaries import clean_summary_results
from tierpsytools.hydra.platechecker import fix_dtypes

#%% GLOBALS

JSON_PARAMETERS_PATH = "analysis/20210406_parameters_keio_screen.json"

#%% FUNCTIONS

def compile_keio_results(args):
    
    assert args.project_dir is not None
    AUX_DIR = Path(args.project_dir) / "AuxiliaryFiles"
    RESULTS_DIR = Path(args.project_dir) / "Results"
    
    ##### Compile results #####
        
    # Process metadata 
    metadata, metadata_path = process_metadata(aux_dir=AUX_DIR,
                                               imaging_dates=args.dates, # 20210420, 20210504
                                               add_well_annotations=args.add_well_annotations,
                                               update_day_meta=False)
            
    # Process feature summary results
    features, metadata = process_feature_summaries(metadata_path,
                                                   RESULTS_DIR,
                                                   compile_day_summaries=args.compile_day_summaries,
                                                   imaging_dates=args.dates,
                                                   align_bluelight=args.align_bluelight)
    
    # Fix data types
    metadata = fix_dtypes(metadata)
    
    # Fill in 'gene_name' for control as 'wild_type'
    metadata.loc[metadata['source_plate_id'] == "BW", 'gene_name'] = "wild_type"
    
    # Subset results (rows) to remove entries for wells with unknown strain data for 'gene_name'
    metadata = metadata.loc[~metadata['gene_name'].isna(),:]
    features = features.reindex(metadata.index)

    # Add COG category info to metadata
    if not 'COG category' in metadata.columns:
        supplementary_7 = load_supplementary_7(args.path_sup_info)
        metadata = append_supplementary_7(metadata, supplementary_7, column_name='gene_name')
        assert set(metadata.index) == set(features.index)

    # # Calculate duration on food + duration in L1 diapause
    # metadata = duration_on_food(metadata) 
    # metadata = duration_L1_diapause(metadata)

    ##### Clean results #####
    
    # Remove bad well data + features with too many NaNs/zero std + impute remaining NaNs
    features, metadata = clean_summary_results(features, 
                                               metadata,
                                               feature_columns=None,
                                               nan_threshold_row=args.nan_threshold_row,
                                               nan_threshold_col=args.nan_threshold_col,
                                               max_value_cap=args.max_value_cap,
                                               imputeNaN=args.impute_nans,
                                               min_nskel_per_video=args.min_nskel_per_video,
                                               drop_size_related_feats=args.drop_size_features,
                                               norm_feats_only=args.norm_features_only,
                                               percentile_to_use=args.percentile_to_use)

    return features, metadata

#%% MAIN

if __name__ == "__main__":
    tic = time()
    parser = argparse.ArgumentParser(description="Compile Keio screen metadata & feature summaries")
    parser.add_argument('-j', '--json', help="Path to JSON parameters file for analysis",
                        default=JSON_PARAMETERS_PATH, type=str)
    parser.add_argument('--save_dir', help="Path to save metadata and features results",
                        default=None)
    args = parser.parse_args()
    
    json_args = load_json(args.json)
    if not args.save_dir:
        args.save_dir = json_args.save_dir
    
    features, metadata = compile_keio_results(json_args)
    
    # Save features to file
    features_path = Path(args.save_dir) / 'features.csv'
    features.to_csv(features_path, index=False) 

    # Save metadata to file
    metadata_path = Path(args.save_dir) / 'metadata.csv'
    metadata.to_csv(metadata_path, index=False)
    
    toc = time()
    print("\nDone in %.1f seconds (%.1f minutes)" % (toc-tic, (toc-tic)/60))
