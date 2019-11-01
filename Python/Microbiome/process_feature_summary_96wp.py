#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROCESS FEATURES SUMMARY RESULTS

A script written to process microbiome assay project feature summary results. 
INPUTS:
    [1] PATH to METADATA file providing meta-info for feature summaries
    [2] OPTIONAL: Unpacked list of imaging dates to process
The script does the following:
    1. Read feature summaries in given directory
    2. Constructs a full dataframe of feature summary results
    3. Saves results to CSV file

@author: sm5911
@date: 21/10/2019

"""

#%% IMPORTS

# General imports
import os, sys, time
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
    elif len(sys.argv) >= 2:
        print("\nRunning script", sys.argv[0], "...")    
        metafilepath = sys.argv[1]                                             # METADATA FILEPATH        
        if len(sys.argv) > 2:            
            IMAGING_DATES = list(sys.argv[2:])                                 # IMAGING DATES
    
    PROJECT_ROOT_DIR = metafilepath.split("/AuxiliaryFiles/")[0]               # PROJECT ROOT DIRECTORY
        
    # Read metadata
    metadata = pd.read_csv(metafilepath)                                       # READ METADATA
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
        try:
            IMAGING_DATES = list(metadata['date_recording_yyyymmdd'].unique().astype(str))
            print("Found the following imaging dates in metadata: %s" % IMAGING_DATES)
        except Exception as EE:
            print("ERROR: Could not read imaging dates from metadata.\n\
                   Please provide them when calling this script, or include them\n\
                   in metadata under the column name: 'date_recording_yyyymmdd'")
            print(EE)
        
#%%     
    # Pre-allocate full dataframe combining results and metadata
    print("\nGetting features summaries...\n")    
    full_feats_list = []
    
    index_offset = 0
    for date in IMAGING_DATES:
        print("Fetching results for date: %d" % int(date))
        results_dir = os.path.join(PROJECT_ROOT_DIR, "Results/{0}".format(int(date)))

        # Get file names summaries (files_df) + featuresN summaries (feats_df)
        files_df, feats_df = getfeatsums(results_dir)

        feats_df.insert(0,'date',date)
        feats_df.insert(0,'file_name','')
        for i, fid in enumerate(feats_df['file_id']):
            feats_df.loc[i,'file_name'] = files_df[files_df['file_id']==fid]['file_name'].values[0]
            
        # Compile a list of dataframes across imaging dates to concat into full dataframe of results
        feats_df.index = feats_df.index + index_offset
        full_feats_list.append(feats_df)
    
    full_feats_df = pd.concat(full_feats_list, axis=0, ignore_index=True, sort=False)

#%%
    # Use reindex to generate a full results dataframe for storing metadata + feature summary results
    non_data_cols = ['file_name', 'date', 'file_id', 'well_name']
    feature_colnames = [feat for feat in full_feats_df.columns if feat not in non_data_cols]
    out_columns = list(metadata.columns)
    out_columns.extend(feature_colnames)   
    full_results_df = metadata.reindex(columns=out_columns)

#%% 
    
# TODO: Vectorise! - Do not use a loop, instead sort datframes by the combined values (uniqueID) of 2 columns
    
    # Retrieve feature summary stats using filename and well number in metadata
    error_log = []
    for i in full_results_df.index:
        if i % 10 == 0:
            print("Processing full results file: %d/%d" % (i+1, max(full_results_df.index)+1))
        resultinfo = full_results_df.iloc[i]
        masked_dirname = resultinfo['filename']
        well_number = resultinfo['well_number']
        feat_filename = os.path.join(masked_dirname, "metadata_featuresN.hdf5").replace("MaskedVideos", "Results")
        try:
            featsum_data = full_feats_df.iloc[np.where(np.logical_and(full_feats_df['file_name']==feat_filename,\
                                                                      full_feats_df['well_name']==well_number))[0][0]]
            full_results_df.loc[i,feature_colnames] = featsum_data[feature_colnames].values
            
        except Exception as EE:
            error_log.append([masked_dirname, well_number])
            print("WARNING: Cannot locate results for %s (well: %s)" % (masked_dirname, well_number))
            print(EE)

    # Save full feature summary results to CSV
    results_outpath = os.path.join(PROJECT_ROOT_DIR, "Results/fullresults.csv")
    full_results_df.to_csv(results_outpath, index=False)
    
    # Save error log of wells with no tracking results (likely no worms were dispensed into those wells)
    if error_log:
        error_log_path = os.path.join(PROJECT_ROOT_DIR, "Results/errorlog_well_noresults_videopaths.txt")
        fid = open(error_log_path, 'w')
        print(error_log, file=fid)
        fid.close()
        print("No feature summary results were found for %d/%d entries in metadata"\
              % (len(error_log), len(~np.array(is_filename))))
    else:
        print("WOOHOO! Worm behaviour successfully tracked in all wells!")

#%%    
    toc = time.time()
    print("Complete! Feature summary results + metadata info saved to file.\n(Time taken: %.1f seconds)" % (toc - tic))
