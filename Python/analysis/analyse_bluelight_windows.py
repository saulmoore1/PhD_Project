#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse bluelight windows

@author: sm5911
@date: 19/08/1993

"""

#%% Imports
import argparse
from time import time
from pathlib import Path
from read_data.read import load_json

from preprocessing.compile_hydra_data import process_metadata, process_feature_summaries
from preprocessing.append_supplementary_info import load_supplementary_7, append_supplementary_7
from filter_data.clean_feature_summaries import clean_summary_results

# from tierpsytools.hydra.platechecker import fix_dtypes

#%% Globals
JSON_PARAMETERS_PATH = "analysis/20210819_parameters_keio_screen_windows.json"
METADATA_PATH = "/Users/sm5911/Documents/Keio_Screen/metadata.csv"

#%% Functions

#%% Main
if __name__ == "__main__":
    tic = time()
    parser = argparse.ArgumentParser(description="Compile Keio screen metadata & feature summaries")
    parser.add_argument('-j', '--json', help="Path to JSON parameters file for analysis",
                        default=JSON_PARAMETERS_PATH, type=str)
    args = parser.parse_args()    
    args = load_json(args.json)
    
    assert args.project_dir is not None
    aux_dir = Path(args.project_dir) / "AuxiliaryFiles"
    results_dir = Path(args.project_dir) / "Results"
    
    metadata, metadata_path = process_metadata(aux_dir=aux_dir,
                                               imaging_dates=args.dates,
                                               add_well_annotations=True)
    
    metadata, features = process_feature_summaries(METADATA_PATH, results_dir, 
                                                   compile_day_summaries=False, 
                                                   imaging_dates=args.dates, 
                                                   align_bluelight=False)
    
    features, metadata = compile_bluelight_window_results(args)
    
    # Save features to file
    features_path = Path(args.save_dir) / 'features.csv'
    features.to_csv(features_path, index=False) 

    # Save metadata to file
    metadata_path = Path(args.save_dir) / 'metadata.csv'
    metadata.to_csv(metadata_path, index=False)
    
    toc = time()
    print("\nDone in %.1f seconds (%.1f minutes)" % (toc - tic, (toc - tic) / 60))

