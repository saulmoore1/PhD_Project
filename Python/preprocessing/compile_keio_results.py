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

from tierpsytools.hydra.platechecker import fix_dtypes

from preprocessing.compile_hydra_data import process_metadata, process_feature_summaries
from preprocessing.append_supplementary_info import load_supplementary_7, append_supplementary_7
from filter_data.clean_feature_summaries import clean_summary_results

#%% GLOBALS

JSON_PARAMETERS_PATH = "analysis/20210719_parameters_keio_screen.json"

#%% FUNCTIONS

def compile_keio_results(args):
    """ Compile metadata (from day meta if provided) and feature summary results (from day summaries),
        for dates in imaging_dates if provided, else all dates found in AuxiliaryFiles directory. 
        Appends COG info for Keio strains to metadata and then cleans feature summary results to:
        Remove:
            - Samples with < min_nskel_per_video number of skeletons tracked throughout the video
            - Samples with > nan_threshold_row proportion of NaN features,
            - Features with > nan_threshold_col proportion of NaN values across all samples
            - Features that are size related (if drop_size_features=True)
            - Features that are not normalised by length (if norm_features_only=True)
            - Features from other percentiles of the distribution (if percentile_to_use is not None)
        Replace:
            - Feature values > max_value_cap (if max_value_cap is not None)
            - Remaining NaN feature values with global mean of all samples (if impute_nans=True)
            
        Input
        -----
        args : Python object containing required variables
        
        Returns
        -------
        features, metadata
    """
    
    assert args.project_dir is not None
    aux_dir = Path(args.project_dir) / "AuxiliaryFiles"
    results_dir = Path(args.project_dir) / "Results"
    
    ##### Compile results #####
        
    # Process metadata 
    metadata, metadata_path = process_metadata(aux_dir=aux_dir,
                                               imaging_dates=args.dates, # 20210420, 20210504
                                               add_well_annotations=args.add_well_annotations,
                                               update_day_meta=False)
            
    # Process feature summary results
    features, metadata = process_feature_summaries(metadata_path,
                                                   results_dir,
                                                   compile_day_summaries=args.compile_day_summaries,
                                                   imaging_dates=args.dates,
                                                   align_bluelight=args.align_bluelight)

    # Fix data types
    metadata = fix_dtypes(metadata)
    
    # Fill in 'gene_name' for control as 'wild_type'
    if aux_dir.parent.name == 'KeioScreen_96WP':
        metadata.loc[metadata['source_plate_id'] == "BW", 'gene_name'] = "wild_type"
    elif aux_dir.parent.name == 'KeioScreen2_96WP':
        metadata.loc[metadata['gene_name'] == "BW", 'gene_name'] = "wild_type"
    
    # Subset results (rows) to remove entries for wells with unknown strain data for 'gene_name'
    metadata = metadata.loc[~metadata['gene_name'].isna(),:]
    features = features.reindex(metadata.index)

    # Add COG category info to metadata
    if not 'COG category' in metadata.columns:
        supplementary_7 = load_supplementary_7(args.path_sup_info)
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
            COG_info.append('Unknown')
    metadata['COG_info'] = COG_info

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
                                               min_nskel_sum=args.min_nskel_sum,
                                               drop_size_related_feats=args.drop_size_features,
                                               norm_feats_only=args.norm_features_only,
                                               percentile_to_use=args.percentile_to_use)

    assert not features.isna().sum(axis=1).any()
    assert not (features.std(axis=1) == 0).any()
    
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
    print("\nDone in %.1f seconds (%.1f minutes)" % (toc - tic, (toc - tic) / 60))
