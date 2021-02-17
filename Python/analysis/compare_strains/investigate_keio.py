#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@date: 12/03/2020
@author: Saul Moore (sm5911)

Investigate Keio screening results



"""

#%% Imports

import os, sys
import numpy as np
import pandas as pd

# Path to Github/local helper functions (USER-DEFINED: Path to local copy of my Github repo)
PATH = '/Users/sm5911/Documents/GitHub/PhD_Project/Python/Psychobiotics_96WP'
if PATH not in sys.path:
    sys.path.insert(0, PATH)

from process_metadata_96wp import compile_from_day_metadata,\
     find_metadata_filenames #calculate_L1_diapause, add_timepoints, foodUppercase
     
from process_feature_summary_96wp import processfeatsums #getfeatsums, listdiff

#%% Functions


#%% Main
if __name__ == "__main__":
    
    # Params
    if len(sys.argv) > 1:
        PROJECT_ROOT_DIR = sys.argv[1]
    else:
        PROJECT_ROOT_DIR = "/Volumes/hermes$/KeioScreen_96WP"
        print("No project directory provided. \nUsing default: %s" % PROJECT_ROOT_DIR)
        
    IMAGING_DATES = ['20200303']
    metadata_dir = os.path.join(PROJECT_ROOT_DIR, 'AuxiliaryFiles')
    control_strain = "WT"
    no_strain_data = "0"

    # Compile metadata from day-metadata
    metadata = compile_from_day_metadata(metadata_dir, IMAGING_DATES)
    metadata_path = os.path.join(metadata_dir, 'metadata.csv')  
    
    # Find filenames
    metadata = find_metadata_filenames(metadata, PROJECT_ROOT_DIR, IMAGING_DATES=None)
        
    # Save metadata
    metadata.to_csv(metadata_path, index=False)
    
    # Subset metadata to remove remaining entries with missing filepaths
    is_filename = [isinstance(path, str) for path in metadata['filename']]
    if any(list(~np.array(is_filename))):
        print("WARNING: Could not find filepaths for %d entries in metadata.\n\
        Omitting these files from analysis..." % sum(list(~np.array(is_filename))))
        metadata = metadata[list(np.array(is_filename))]
        # Reset index
        metadata.reset_index(drop=True, inplace=True)
    
    # Read list of important features (highlighted by previous research - see Javer, 2018 paper)
    featslistpath = os.path.join(PROJECT_ROOT_DIR, 'AuxiliaryFiles',\
                    'top256_tierpsy_no_blob_no_eigen_only_abs_no_norm.csv')
    top256features = pd.read_csv(featslistpath)
    
    # Take first set of 256 features (it does not matter which set is chosen)
    top256features = list(top256features[top256features.columns[0]])
    
    # Remove features from Top256 that are path curvature related
    n_feats_before = len(top256features)
    top256features = [feat for feat in top256features if "path_curvature" not in feat]
    n_feats_after = len(top256features)
    print("Dropped %d features from Top256 that are related to path curvature" % (n_feats_before - n_feats_after))
 
    # Process feature summaries
    results_df = processfeatsums(metadata_path, save=True)
    
    # Subset to remove data for "0" strain (No data)
    results_df = results_df[results_df['food_type']!=no_strain_data]
    
    # Subset + save control strain data
    control_df = results_df[results_df['food_type']==control_strain]
    control_path = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Control', 'control_results.csv')
    directory = os.path.dirname(control_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    control_df.to_csv(control_path, index=False)
        