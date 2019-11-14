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

# Path to Github/local helper functions (USER-DEFINED: Path to local copy of my Github repo)
PATH = '/Users/sm5911/Documents/GitHub/PhD_Project/Python/Psychobiotics_96WP'
if PATH not in sys.path:
    sys.path.insert(0, PATH)
    
# Custom imports
from helper import lookforfiles, listdiff

#%% FUNCTIONS

def getfeatsums(directory):
    """ A function to load feature summary data from a given directory and return
        a dataframe of features along with a corresponding dataframe of file names.
        A unique ID key is used to maintain file identity. """
        
    # Obtain all feature summaries and filename summaries in given directory
    file_summary_list = lookforfiles(directory, '^filenames_summary*', depth=1)
    feat_summary_list = lookforfiles(directory, '^features_summary*', depth=1)
        
    # CHECK: Match file and features summaries
    matched_summaries = []
    for file in file_summary_list:
        feat = file.replace('filenames_summary', 'features_summary')
        if feat in feat_summary_list:
            matched_summaries.append([file,feat])
        else:
            print('No match found for: \n%s' % file)
            
    if len(matched_summaries) > 1:
        print("ERROR: Multiple feature summary files found in directory: '%s'" % directory)
    elif len(matched_summaries) < 1:
        print("ERROR: No feature summary results found.")

    # Read matched feature summaries and filename summaries
    file, feat = matched_summaries[0]
    files_df = pd.read_csv(file)
    feats_df = pd.read_csv(feat)
        
    # Remove entries from file summary data where corresponding feature summary data is missing
    missing_featfiles = listdiff(files_df['file_id'], feats_df['file_id'])
    files_df = files_df[np.logical_not(files_df['file_id'].isin(missing_featfiles))]
        
    files_df.reset_index(drop=True, inplace=True)
    feats_df.reset_index(drop=True, inplace=True)
                
    # Check that file_id column matches
    if (files_df['file_id'] != feats_df['file_id'].unique()).any():
        print("ERROR: Features summary and filenames summary do not match!")
    else:
        return files_df, feats_df
  
#%% MAIN
        
def processfeatsums(COMPILED_METADATA_FILEPATH, IMAGING_DATES=None):
    
    PROJECT_ROOT_DIR = COMPILED_METADATA_FILEPATH.split("/AuxiliaryFiles/")[0]

    # Read metadata
    metadata = pd.read_csv(COMPILED_METADATA_FILEPATH)                                       # READ METADATA
    print("\nMetadata file loaded.")

    # Subset metadata to remove remaining entries with missing filepaths
    is_filename = [isinstance(path, str) for path in metadata['filename']]
    if any(list(~np.array(is_filename))):
        print("""WARNING: Could not find filepaths for %d entries in metadata!
        These files will be omitted from further analyses.""" % sum(list(~np.array(is_filename))))
        metadata = metadata[list(np.array(is_filename))]
        # Reset index
        metadata.reset_index(drop=True, inplace=True)
        
    if not IMAGING_DATES:
        try:
            IMAGING_DATES = sorted(list(metadata['date_recording_yyyymmdd'].dropna().astype(int).unique().astype(str)))
            print("Found the following imaging dates in metadata: %s" % IMAGING_DATES)
        except Exception as EE:
            print("""ERROR: Could not read imaging dates from metadata.
            Please provide them when calling this script, or include them
            in metadata under the column name: 'date_recording_yyyymmdd'""")
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
    # Convert results file path to maskedvideo path to match with metadata
    full_feats_df['file_name'] = [file.split("/metadata_featuresN.hdf5")[0].replace("Results","MaskedVideos")\
                                  for file in full_feats_df['file_name']]
        
    # Merge metadata and results dataframes using uniqueID column (created using filename and well number)
    out_columns = list(metadata.columns)
    
    metadata['uniqueID'] = metadata['filename'] + '__' + metadata['well_number']
    full_feats_df['uniqueID'] = full_feats_df['file_name'] + '__' + full_feats_df['well_name']
    
    non_data_cols = ['file_name', 'date', 'file_id', 'well_name', 'uniqueID']
    feature_colnames = [feat for feat in full_feats_df.columns if feat not in non_data_cols]    
    out_columns.extend(feature_colnames)
    
    # Sort filename column for both metadata and results and merge together
    # NB: Keep data for metadata entries where reuslts are missing + join results to metadata where uniqueID matches
    full_results_df = pd.merge(left=metadata, right=full_feats_df, how='left',\
                               left_on='uniqueID', right_on='uniqueID')
    
    # Drop rows and record uniqueIDs for entries with missing results    
    no_results_indices = np.where(full_results_df[feature_colnames].isna().all(axis=1))[0]
    errorlog_no_results_uniqueIDs = list(full_results_df.loc[no_results_indices, 'uniqueID'])
    print("Dropped %d entries with missing results (empty wells)." % len(errorlog_no_results_uniqueIDs))
   
    # Save error log of entries (wells) with no tracking results (likely no worms were dispensed into those wells)
    n_missing_results = len(errorlog_no_results_uniqueIDs)
    if n_missing_results > 0:
        errlog_outpath = os.path.join(PROJECT_ROOT_DIR, "Results/errorlog_empty_wells.txt")
        with open(errlog_outpath, 'w') as fid:
            print(errorlog_no_results_uniqueIDs, file=fid)
            print("No feature summary results were found for %d/%d entries in metadata"\
                  % (n_missing_results, len(~np.array(is_filename))))
    else:
        print("WOOHOO! Worm behaviour successfully tracked in all wells!")
    
    # Use reindex to obtain full results dataframe (only necessary columns)
    full_results_df = full_results_df.reindex(columns=out_columns)
        
    # Save full feature summary results to CSV
    results_outpath = os.path.join(PROJECT_ROOT_DIR, "Results/fullresults.csv")
    if os.path.exists(results_outpath):
        print("Overwriting existing results file: '%s'" % results_outpath)
    full_results_df.to_csv(results_outpath, index=False)

    print("Complete!\nFull results saved to file: %s" % results_outpath)

#%%    
if __name__ == '__main__':
    # START
    tic = time.time()
    
    # INPUT HANDLING
    
    print("\nRunning script", sys.argv[0], "...")
    if len(sys.argv) > 1:
        COMPILED_METADATA_FILEPATH = sys.argv[1]  
                
    IMAGING_DATES = None
    if len(sys.argv) > 2:
        IMAGING_DATES = list(sys.argv[2:])
        print("Using %d imaging dates provided: %s" % (len(IMAGING_DATES), IMAGING_DATES))   
        
    # MAIN
    processfeatsums(COMPILED_METADATA_FILEPATH, IMAGING_DATES)
    
    # END
    toc = time.time()
    print("\n(Time taken: %.1f seconds)\n" % (toc - tic))

