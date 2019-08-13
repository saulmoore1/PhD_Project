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
import os, sys, time
import numpy as np
import pandas as pd

# Custom imports
#sys.path.insert(0, '/Users/sm5911/Documents/GitHub/PhD_Project/Python') # OPTIONAL: Path to GitHub functions
from SM_read import getfeatsums

#%% MAIN

if __name__ == '__main__':
    # Record script start time
    tic = time.time()
    
    IMAGING_DATES = None
    if len(sys.argv) < 2:
        print("ERROR: Please provide path to metadata file (CSV)!")
    elif len(sys.argv) >= 2:
        metadata_filepath = sys.argv[1]
        print("\nRunning script", sys.argv[0], "...")
        if len(sys.argv) > 2:
            IMAGING_DATES = list(sys.argv[2:])
    
    # Read metadata
    metadata = pd.read_csv(metadata_filepath)
    print("\nMetadata file loaded.")

    # Subset metadata to remove remaining entries with missing filepaths
    is_filename = [isinstance(path, str) for path in metadata['filename']]
    if any(list(~np.array(is_filename))):
        print("WARNING: Could not find filepaths for %d entries in metadata.\n\t These files will be omitted from further analyses!" \
              % sum(list(~np.array(is_filename))))
        metadata = metadata[list(np.array(is_filename))]
        # Reset index
        metadata.reset_index(drop=True, inplace=True)
        
    if not IMAGING_DATES:
        IMAGING_DATES = list(metadata['date_yyyymmdd'].unique())
    
    print("\nGetting features summaries...\n")    
    full_results_df = pd.DataFrame()
    
    for date in IMAGING_DATES:
        results_dir = metadata_filepath.replace("/Saul/MicrobiomeAssay/metadata.csv", 
                                                "/Priota/Data/MicrobiomeAssay/Results/{0}".format(date))
        
        ##### Get files summaries and features summaries #####
        # NB: Ignores empty video snippets at end of some assay recordings 
        files_df, feats_df = getfeatsums(results_dir)
    
        # Pre-allocate full dataframe combining results and metadata to allow for subsetting by treatment group + statistical analyses
        metadata_dirnames = [os.path.dirname(file) for file in files_df['file_name']]
        metadata_colnames = list(metadata.columns)
        results_date_df = pd.DataFrame(index=range(len(metadata_dirnames)), columns=metadata_colnames)
        
        for i, dirname in enumerate(metadata_dirnames):
            # Add metadata data to results dataframe for each entry in files_df
            results_date_df.iloc[i] = metadata[metadata['filename'] == dirname.replace('/Results/', '/MaskedVideos/')].values
            
            # In results dataframe, replace folder name (from metadata) with full file name (from files_df)
            results_date_df.iloc[i]['filename'] = files_df.iloc[i]['file_name']
        
        # OPTION 1: Add just 'file_id' column to results_date_df
        #           results_date_df.insert(0, column='file_id', value=files_df['file_id'], allow_duplicates=False)
        # OPTION 2: Combine results and metadata into single results dataframe for that imaging date
        #           NB: This loop will be slow, as it involves growing dataframes on-the-fly,
        #               resulting in continuous re-allocation in memory under the hood
        results_date_df = pd.concat([results_date_df, feats_df], axis=1)
    
        try: # Maintain unique file_ids across imaging dates
            # Add max value of unique file IDs of results of previous imaging date to the unique IDs of the next
            results_date_df['file_id'] = results_date_df['file_id'] + full_results_df['file_id'].max()
        except: 
            # If empty, or does not contain 'file_id'...
            results_date_df['file_id'] = results_date_df['file_id'] + full_results_df.shape[0]
        
        # Combine dataframes across imaging dates to construct full dataframe of results
        full_results_df = pd.concat([full_results_df, results_date_df], axis=0, sort=False).reset_index(drop=True)
        
    # Save full feature summary results to CSV
    results_outpath = metadata_filepath.replace("/metadata.csv", "/Results/fullresults.csv")
    directory = os.path.dirname(results_outpath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    full_results_df.to_csv(results_outpath, index=False)
    
    toc = time.time()
    print("Complete! Feature summary results + metadata info saved to file.\n(Time taken: %.1f seconds)" % (toc - tic))
