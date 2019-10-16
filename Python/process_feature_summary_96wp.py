#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sm5911
@date: 11/08/2019

PROCESS FEATURES SUMMARY RESULTS

A script written to process microbiome assay project feature summary results. 
INPUTS:
    [1]  PATH to METADATA file providing meta-info for feature summaries
    [2]  PATH to OUTPUT directory (including desired filename)
    [3:] OPTIONAL: An unpacked list of folders to look in for Tierpsy feature 
         summaries (eg. specific imaging dates)
The script does the following:
    1. Read feature summaries in given directory
    2. Constructs a full dataframe of feature summary results
    3. Saves results to CSV file

"""

#%% IMPORTS

# General imports
import os, sys, time, json
import numpy as np
import pandas as pd

# Custom imports
sys.path.insert(0, '/Users/sm5911/Documents/GitHub/PhD_Project/Python')
from SM_read import getfeatsums

#%% MAIN

if __name__ == '__main__':
    # Record script start time
    tic = time.time()
        
#%% INPUT HANDLING + READ METADATA
    
    IMAGING_DATES = None
    if len(sys.argv) < 2:
        print("ERROR: No metadata file path provided!")
#        metadata_filepath = "/Volumes/behavgenom$/Saul/MicrobiomeAssay96WP/AuxiliaryFiles/metadata_updated.csv"
#        print("WARNING: No metadata file path provided!\nUsing default path: '%s'" % metadata_filepath)
    elif len(sys.argv) >= 2:
        print("\nRunning script", sys.argv[0], "...")
        metadata_filepath = sys.argv[1]
        if len(sys.argv) > 2:
            IMAGING_DATES = list(sys.argv[2:])
    
    PROJECT_ROOT_DIR = metadata_filepath.split("/AuxiliaryFiles/")[0]
        
    # Read metadata
    metadata = pd.read_csv(metadata_filepath)
    print("\nMetadata file loaded.")

    # Subset metadata to remove remaining entries with missing filepaths
    is_filename = [isinstance(path, str) for path in metadata['filename']]
    if any(list(~np.array(is_filename))):
        print("WARNING: Could not find filepaths for %d entries in metadata!\n\t These files will be omitted from further analyses." \
              % sum(list(~np.array(is_filename))))
        metadata = metadata[list(np.array(is_filename))]
        # Reset index
        metadata.reset_index(drop=True, inplace=True)
        
    if not IMAGING_DATES:
        IMAGING_DATES = list(metadata['date_recording_yyyymmdd'].unique())
        
#%%     
    # Pre-allocate full dataframe combining results and metadata
    print("\nGetting features summaries...\n")    
    full_features_df = pd.DataFrame()
    
    index_offset = 0
    for date in IMAGING_DATES:
        results_dir = os.path.join(PROJECT_ROOT_DIR, "Results/{0}".format(date))

        # Get file names summaries (files_df) + featuresN summaries (feats_df)
        files_df, feats_df = getfeatsums(results_dir)

        feats_df['date'] = date
        feats_df['file_name'] = ''
        for i, fid in enumerate(feats_df['file_id']):
            feats_df.loc[i,'file_name'] = files_df[files_df['file_id']==fid]['file_name'].values[0]
            
        # Combine dataframes across imaging dates to construct full dataframe of results
        feats_df.index = feats_df.index + index_offset
        full_features_df = pd.concat([full_features_df, feats_df], axis=0, sort=False)
        index_offset += len(feats_df.index)
        
    non_data_cols = ['file_id','well_name','date','file_name']
    feature_colnames = [feat for feat in full_features_df.columns if feat not in non_data_cols]
    
    out_columns = list(metadata.columns)
    out_columns.extend(feature_colnames)
    
    # Use reindex to generate a full results dataframe for storing metadata + feature summary results
    full_results_df = metadata.reindex(columns=out_columns)

# TODO: Fix well-filename mappings for feature summary stats!
    
#    for i, featuresfile in     
    for i in full_results_df.index:
        metainfo = full_results_df.iloc[i]
        masked_dirname = metainfo['filename']
        well_number = metainfo['well_number']
        feat_filename = os.path.join(masked_dirname, "metadata_featuresN.hdf5").replace("MaskedVideos", "Results")

        try:
            featsum_data = full_features_df.iloc[np.where(np.logical_and(full_features_df['file_name']==feat_filename,\
                                                                         full_features_df['well_name']==well_number))[0][0]]
            full_results_df.loc[i,feature_colnames] = featsum_data[feature_colnames].values
            
        except Exception as EE:
            print("WARNING: Cannot locate results for %s (well: %s)" % (masked_dirname, well_number))
            print(EE)
        
        
    # Save full feature summary results to CSV
#    results_outpath = metadata_filepath.replace("/metadata.csv", "/Results/fullresults.csv")
#    directory = os.path.dirname(results_outpath)
#    if not os.path.exists(directory):
#        os.makedirs(directory)
#    full_results_df.to_csv(results_outpath, index=False)

#%% Load JSON files and extract hydra imaging rig temperature and humidity info
    
    e = os.path.join(PROJECT_ROOT_DIR, 'RawVideos/20191003/microbiome_96wp_20191003_114718.22956806/000000.extra_data.json')
    with open(e) as fid:
        extras = json.load(fid)
        rig_data = pd.DataFrame(index=range(len(extras)), columns=['frame_number','humidity_percent','temperature_C'])
        for d, dictionary in enumerate(extras):
            rig_data.iloc[d] = dictionary['frame_index'], dictionary['humidity'], dictionary['tempo']


#%%    
    toc = time.time()
    print("Complete! Feature summary results + metadata info saved to file.\n(Time taken: %.1f seconds)" % (toc - tic))
