#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keio Screen Analysis

@author: sm5911
@date: 19/11/2020

"""

# Imports
import time
import argparse
from pathlib import Path

# Functions

# Main 
if __name__ == "__main__":
    
    # record scrupt start time
    tic = time.time()
    
    # accept command-line arguments for inputs
    parser = argparse.ArgumentParser(description='Analyse Keio results using\
                                     provided directory paths for metadata and\
                                     featuresN files')
    parser.add_argument("--root_dir", help="project root directory path, eg. /Volumes/hermes$/KeioScreen_96WP")
    parser.add_argument("--dates", help="Experiment dates to analyse", default=None)
    parser.add_argument("--timepoint", help="Experiment timepoint to analyse", default=None)
    parser.add_argument("--omit_food", help="Bacterial strains to exclude from analysis", default=None)
    parser.add_argument("--control", help="Control bacterial strain", default="WT")
    
    # known_args = parser.parse_known_args()
    # parser.add_argument("--featuresN_dir", help="featuresN results directory path",
    #                     default=Path(known_args[0].metadata_dir).parent / "results")
    args = parser.parse_args()
       
    PROJECT_ROOT_DIR = Path('/Volumes/hermes$/KeioScreen_96WP')
    IMAGING_DATES = ['20200303']
    STRAINS_TO_EXCLUDE = ["0"]    
    CONTROL_STRAIN = "WT"
    variables_list = ["imaging_run_number", "imaging_plate_id",\
                      "master_stock_plate_ID", "instrument_name", "well_name",\
                      "dispense_method"]
    TIMEPOINT = None

    
    print("Metadata directory:", args.metadata_dir)
    print("Features file directory:", args.featuresN_dir)
    print("Output directory", args.save_dir)
    
    print("Analysing Keio data..")
    
    # read metadata
    
    # locate reuslts files
    
    # 

    
    toc = time.time()
    print("Done! (%.1f seconds)" % (toc - tic))