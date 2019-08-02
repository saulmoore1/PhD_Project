#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bacterial affects on Caenorhabditis elegans Behaviour - Microbiome Analysis

This script reads Tierpsy results files for experimental data collected during 
preliminary screening of Schulenberg Lab bacterial strains isolated from the C. 
elegans gut microbiome. 

The script does the following: 
    - Reads the project metadata file, and completes missing filepath info
    - Checks for results files (features/skeletons/intensities)
    - Extracts summary features of interest and compiles a dataframe for visualisation
The script WILL DO the following:
    - Principal components analysis (PCA) to extract most important features
    - Visualisation of extracted features
    - Comparison of these features between N2 worms on different foods

@author: sm5911
@date: 07/07/2019

"""


#%% IMPORTS

# General imports
#import datetime
#sys.path.insert(0, '/Users/sm5911/Documents/GitHub/PhD_Project/Python') # OPTIONAL: Path to GitHub functions
import os, sys, time, decimal
import subprocess as sp
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats import multitest as smm
from scipy.stats import zscore
from sklearn.decomposition import PCA

# Custom imports
from SM_find import changepath
from SM_read import gettrajdata, getfeatsums
from SM_calculate import pcainfo, ranksumtest
#from SM_plot import manuallabelling
from SM_save import savefig


#%% PRE-AMBLE

# Global variables
PROJECT_NAME = 'MicrobiomeAssay'
PROJECT_ROOT_DIR = '/Volumes/behavgenom$/Saul/' + PROJECT_NAME
DATA_DIR = '/Volumes/behavgenom$/Priota/Data/' + PROJECT_NAME


#%% PREPROCESS METADATA

# Select imaging date(s) for analysis
IMAGING_DATES = ['20190704','20190705','20190711','20190712']

# Use subprocess to call 'process_metadata.py', passing imaging dates as arguments to the script
print("\nProcessing metadata file...")
process_metadata = sp.Popen([sys.executable, "process_metadata.py", *IMAGING_DATES])
process_metadata.communicate()


#%% READ METADATA

# Read metadata (CSV file)
metafilepath = os.path.join(PROJECT_ROOT_DIR, "metadata.csv")
metadata = pd.read_csv(metafilepath)
print("\nMetadata file loaded.")


#%% REMOVE MISSING FILEPATHS FROM METADATA

# Subset metadata to remove remaining entries with missing filepaths
is_filename = [isinstance(path, str) for path in metadata['filename']]
if any(list(~np.array(is_filename))):
    print("WARNING: Could not find filepaths for %d entries in metadata.\n\t These files will be omitted from further analyses!" \
          % sum(list(~np.array(is_filename))))
    metadata = metadata[list(np.array(is_filename))]
    # Reset index
    metadata.reset_index(drop=True, inplace=True)
    

#%% CHECK FEATURES N RESULTS - TRY TO READ TRAJECTORY DATA

ERROR_LIST = []
for i, maskedvideodir in enumerate(metadata.filename):
    if i % 10 == 0:
        print("Checking for missing/corrupt results files: %d/%d" % (i, len(metadata.filename)))
    try:
        featuresfilepath = changepath(maskedvideodir + '/000000.hdf5', returnpath='features')
        data = gettrajdata(featuresfilepath)
#        if data.shape[0] > 1:
#            print(data.head())
    except Exception as EE:
        print("ERROR:", EE)
        ERROR_LIST.append(maskedvideodir)

# Save error log to file
if ERROR_LIST:
    print("WARNING: %d maskedvideos found with incomplete or missing results files!" % len(ERROR_LIST))
    error_outpath = os.path.join(PROJECT_ROOT_DIR, 'Error_logs', 'Unprocessed_MaskedVideos.txt')
    directory = os.path.dirname(error_outpath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fid = open(error_outpath, 'w')
    print(ERROR_LIST, file=fid)
    fid.close()
else:
    print("Complete! All results files are present.")

#%% GET FEATURES SUMMARIES

print("Getting features summaries...")    
full_results_df = pd.DataFrame()

tic = time.time()
for date in IMAGING_DATES:
    results_dir = os.path.join(DATA_DIR, "Results", date)
    
    ##### Get files summaries and features summaries #####
    # NB: Ignores empty video snippets at end of some assay recordings 
    #     2hrs = 10 x 12min video segments (+/- a few frames)
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
    
    # Maintain unique file_ids across imaging dates
    try: # Add max value of unique file IDs of results of previous imaging date to the unique IDs of the next
        results_date_df['file_id'] = results_date_df['file_id'] + full_results_df['file_id'].max()
    except: # If empty, or does not contain 'file_id'..
        results_date_df['file_id'] = results_date_df['file_id'] + full_results_df.shape[0]
    
    # Combine dataframes across imaging dates to construct full dataframe of results
    full_results_df = pd.concat([full_results_df, results_date_df], axis=0, sort=False).reset_index(drop=True)
    
# Save full results dataframe to CSV
results_outpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'fullresults.csv')
directory = os.path.dirname(results_outpath)
if not os.path.exists(directory):
    os.makedirs(directory)
full_results_df.to_csv(results_outpath)

toc = time.time()
print("Complete! Feature summary results + metadata info saved to file.\n(Time taken: %.1f seconds)" % (toc - tic))

print(full_results_df[['date_yyyymmdd','file_id','food_type','speed_50th']].head())
  
#%% PERFORM T-TESTS / RANK SUM TESTS
# - Look at first segment only  
# - Start with L4-prefed worms (long exposure to food)
# - Any sig. diff's between foods?  
# - Which feats are different (stats- t-test/rank sum)

tests = [ttest_ind, ranksumtest]
nan_threshold = 0.75
p_value_threshold = 0.05
verbose = False

# Filter feature summary results to look at 1st video snippets only
first_snippets_df = full_results_df[full_results_df['filename'].str.contains('000000_featuresN.hdf5')]

# Filter feature summary results to look at L4-prefed worms (long food exposure) only
L4_1st_snippets_df = first_snippets_df[first_snippets_df['preconditioned_from_L4'].str.lower() == "yes"]

# Perform t-tests to look for any significant differences in any features between foods
# (comparing each strain to OP50 control)
test_bacteria = list(np.unique(L4_1st_snippets_df['food_type'].str.lower()))
test_bacteria.remove('op50')

# Filter feature summary results for OP50 control
OP50_control_df = L4_1st_snippets_df[L4_1st_snippets_df['food_type'].str.lower() == "op50"]

# Select non-data columns to drop for statistics
non_data_columns = L4_1st_snippets_df.columns[0:25]

# Record a list of feature column names
feature_colnames = L4_1st_snippets_df.columns[25:]

# Record a list of bacterial strains to compare against OP50 control strain
n_foods2compare = len(test_bacteria)

for test in tests:
    # Pre-allocate dataframes for storing test statistics and p-values
    test_stats_df = pd.DataFrame(index=list(test_bacteria), columns=feature_colnames)
    test_pvalues_df = pd.DataFrame(index=list(test_bacteria), columns=feature_colnames)
    
    sigdifffeats_df = pd.DataFrame(index=test_pvalues_df.index, columns=['N_sigdiff_noBF','N_sigdiff_withBF'])
    
    # Record number of decimal places of requested pvalue - for print statements to std_out
    p_decims = abs(int(decimal.Decimal(str(p_value_threshold)).as_tuple().exponent))
    
    # Compare each strain to OP50: compute t-test statistics for each feature
    for food in test_bacteria:
        if test == ranksumtest:
            print("Computing ranksum tests for %s vs OP50" % food.upper())
        else:
            print("Computing t-tests for %s vs OP50" % food.upper())
            
        # Filter feature summary results for that food
        test_food_df = L4_1st_snippets_df[L4_1st_snippets_df['food_type'].str.lower() == food]
        
        # Drop non-data columns
        test_data = test_food_df.drop(columns=non_data_columns)
        control_data = OP50_control_df.drop(columns=non_data_columns)
        
        # Drop columns with too many nan values
        n_cols = len(test_data.columns)
        test_data.dropna(axis=1, thresh=nan_threshold, inplace=True)
        control_data.dropna(axis=1, thresh=nan_threshold, inplace=True)
        nan_cols = n_cols - min(len(test_data.columns), len(control_data.columns))
        
        # Drop columns that contain only zeros
        test_data.drop(columns=test_data.columns[(test_data == 0).all()], inplace=True)
        control_data.drop(columns=control_data.columns[(control_data == 0).all()], inplace=True)
        zero_cols = n_cols - nan_cols - min(len(test_data.columns), len(control_data.columns))
        
        # Impute missing values by replacing with mean value
        test_data.fillna(test_data.mean(axis=0), inplace=True)
        control_data.fillna(control_data.mean(axis=0), inplace=True)
        
        # Use only shared feature summaries between control data and test data
        shared_colnames = control_data.columns.intersection(test_data.columns)
        test_data = test_data[shared_colnames]
        control_data = control_data[shared_colnames]
        
        total_cols_dropped = n_cols - min(len(test_data.columns), len(control_data.columns))
        if verbose:
            print("%d features dropped (%d containing too many NaNs (> 0.75), %d containing only zeros)" % (total_cols_dropped, nan_cols, zero_cols))
            print("WARNING: Remaining missing values (NaNs) were imputed by replacing with mean value")
        
        # Perform t-tests/ranksums comparing between foods for each feature (max features = 4539)
        test_stats, test_pvalues = test(test_data, control_data)
        
        # Add t-test/ranksum results to out-dataframe
        test_stats_df.loc[food][shared_colnames] = test_stats
        test_pvalues_df.loc[food][shared_colnames] = test_pvalues
            
        sigdiff_feats = feature_colnames[np.where(test_pvalues < p_value_threshold)]
        sigdifffeats_df.loc[food,'N_sigdiff_noBF'] = len(sigdiff_feats)
        
        if verbose:
            print("%d features found for worms on %s that differ from worms on OP50 (p<%.{0}f, before Bonferroni correction for multiple comparisons)\n".format(p_decims) %\
                  (len(sigdiff_feats), food.upper(), p_value_threshold))
        
    # Bonferroni corrections for multiple comparisons (CREDIT: Thanks to Ida for her help here)
    test_pvalues_corrected_df = pd.DataFrame(index=test_pvalues_df.index, columns = test_pvalues_df.columns)
    
    
    sigdifffeatslist = []
    for food in test_pvalues_df.index:
        # Locate pvalue results (row) for food
        food_pvals_df = test_pvalues_df.loc[food] # pd.Series object
        
        # Perform Bonferroni correction for multiple comparisons on t-test pvalues
        _corrArray = smm.multipletests(food_pvals_df.values, alpha=p_value_threshold, method='fdr_bh',\
                                       is_sorted=False, returnsorted=False)
        
        # Get pvalues for features that passed the Benjamini/Hochberg (non-negative) correlation test
        pvalues_corrected = _corrArray[1][_corrArray[0]]
        
        # Add pvalues to dataframe of corrected ttest pvalues
        test_pvalues_corrected_df.loc[food, _corrArray[0]] = pvalues_corrected
    
        sigdiff_feats = test_pvalues_corrected_df.columns[np.where(_corrArray[1] < p_value_threshold)]
        sigdifffeatslist.append(sigdiff_feats)
        sigdifffeats_df.loc[food,'N_sigdiff_withBF'] = len(sigdiff_feats)
        
        if verbose:
            print("%d features found for worms on %s that differ from worms on OP50 (p<%.{0}f, after Bonferroni correction for multiple comparisons)\n".format(p_decims) %\
                  (len(sigdiff_feats), food.upper(), p_value_threshold))
       
    print(sigdifffeats_df)


#%% VISUALISE / PLOT SUMMARY FEATURES
# - Investigate Avelino's top 256 features to look for any differences between foods (see paper: Javer et al, 2018)
# - Plot features that show significant differences from behaviour on OP50

# Read list of important features (highlighted by previous research - see Javer, 2018 paper)
featslistpath = os.path.join(PROJECT_ROOT_DIR, 'top256_tierpsy_no_blob_no_eigen_only_abs_no_norm.csv')
top256features = pd.read_csv(featslistpath)

# Take first set of 256 features (it does not matter which set is chosen)
top256features = top256features[top256features.columns[0]]

# Filter feature summary results to look at 1st video snippets only
first_snippets_df = full_results_df[full_results_df['filename'].str.contains('000000_featuresN.hdf5')]

# Filter feature summary results to look at L4-prefed worms (long food exposure) only
L4_1st_snippets_df = first_snippets_df[first_snippets_df['preconditioned_from_L4'].str.lower() == "yes"]

test_bacteria = list(np.unique(L4_1st_snippets_df['food_type'].str.lower()))

# Only investigate features for which we have results
features2plot = [feature for feature in top256features if feature in test_pvalues_corrected_df.columns]

## Filter for features displaying significant differences between any food and OP50 control
#features2plot = [feature for feature in features2plot if (test_pvalues_corrected_df[feature] < p_value_threshold).any()]

# Make directory for storing plots of each feature (n=256)
plotdir_out = os.path.join(PROJECT_ROOT_DIR, "Results", "Plots", "Top256")
if not os.path.exists(plotdir_out):
    os.makedirs(plotdir_out)

# Seaborn boxplots with swarmplot overlay for each feature - saved to file
plt.ioff()   
for f, feature in enumerate(features2plot):
    # NB: Currently n=6 L4 first snippet replicates for each bacterial food (1 imaging date per food) to date
    print("Plotting feature: '%s'" % feature)
    
    # Seaborn boxplots for each feature
    fig = plt.figure(figsize=[10,7])
    sns.boxplot(x="food_type", y=feature, data=L4_1st_snippets_df, showfliers=False, showmeans=True,\
                     meanprops={"marker":"x", "markersize":5, "markeredgecolor":"k"},\
                     flierprops={"marker":"x", "markersize":15, "markeredgecolor":"r"})
    sns.swarmplot(x="food_type", y=feature, data=L4_1st_snippets_df, s=10, marker=".", color='k')
    
    # Save boxplots
    plotpath_out = os.path.join(plotdir_out, feature + '_1st_snippet_preconditioned_from_L4.eps')
    savefig(plotpath_out, saveFormat='eps')
    
    # Close plot
    plt.close(fig)

#%%

# Look for differences between L4-prefed vs. Naive adults - time dependence? does it take time for food-behavioural effects to emerge?

#naive_1st_snippets_df = first_snippets_df[first_snippets_df['preconditioned_from_L4'].str.lower() == "no"]

#ttest_stats_L4vsNaive, ttest_pvalues_L4vsNaive = ttest_ind(naive_1st_snippets_df.drop(columns=non_data_columns),\
#                                                           OP50_control_df.drop(columns=non_data_columns), axis=0)

#%% PRINCIPAL COMPONENTS ANALYSIS
            
# =============================================================================
# # Investigate feature summaries for just the 1st video snippets to begin with
# filepaths = [os.path.join(file.replace('/MaskedVideos/', '/Results/'), '000000_featuresN.hdf5')\
#              for file in metadata[metadata.date_yyyymmdd==int(date)]['filename']]
# #    feats_df[feats_df['file_id'] == x]
# 
# fileIDs = [files_df[files_df['file_name']==file]['file_id'].values[:] for file in filepaths]
# feat_indices = [feats_df[feats_df['file_id']==fileID].index.values[:] for fileID in fileIDs]
# 
# meta1stfeatsums_df = feats_df.iloc[feat_indices]
# 
# ##### Prepare features summary data for PCA #####
# # Drop non-data column(s)
# data = feats_df.drop(columns='file_id')
# 
# # Drop columns that are all zeroes
# data.drop(columns=data.columns[(data==0).all()], inplace=True)
# 
# # Drop columns with too many nans
# nan_threshold = 0.75
# data.dropna(axis='columns', thresh=nan_threshold, inplace=True)
# 
# # Impute data to fill in remaining nans with mean value
# data = data.apply(lambda x : x.fillna(x.mean(axis=0)))
# 
# # Normalise the data before performing principal components analysis
# zscores = data.apply(zscore)
# 
# ##### Perform PCA on extracted features #####
# pca = PCA()
# pca.fit(zscores)
# 
# # Plot summary data from PCA: explained variance (most important feats)
# important_feats, fig = pcainfo(pca, zscores)
# 
# # Save plot of PCA explained variance
# plotpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'PCA_explained_' + date + '.eps')
# directory = os.path.dirname(plotpath)
# if not os.path.exists(directory):
#     os.makedirs(directory)
# savefig(plotpath, tight_layout=True, tellme=True, saveFormat='eps')
# 
# # Project zscores onto pc
# projected = pca.transform(zscores)  # produces a matrix
# 
# toc = time.time()
# print("Time taken: %.1f seconds" % (toc - tic))    
# =============================================================================


#%% MANUAL LABELLING OF FOOD REGIONS (To check time spent feeding throughout video)
    
# TODO: Automate food labelling using (U-Net) ML algorithm -- pre-trained already by Luigi
# =============================================================================
# # Return list of pathnames for masked videos in the data directory under given imaging dates
# maskedfilelist = []
# date_total = []
# for i, expDate in enumerate(IMAGING_DATES):
#     tmplist = lookforfiles(os.path.join(DATA_DIR, "MaskedVideos", expDate), ".*.hdf5$")
#     date_total.append(len(tmplist))
#     maskedfilelist.extend(tmplist)
# print("%d masked videos found for imaging dates provided:\n%s" % (len(maskedfilelist), [*zip(IMAGING_DATES, date_total)]))
#
# print("\nManual labelling:\nTotal masked videos found: %d\n" % len(maskedfilelist))
# 
# # Interactive plotting (for user input when labelling plots)
# tic = time.time()
# for i in range(len(maskedfilelist)):    
#     maskedfilepath = maskedfilelist[i]
#     # Manually outline + assign labels to food regions + save coordinates + trajectory overlay to file           
#     manuallabelling(maskedfilepath, n_poly=1, out_dir='MicrobiomeAssay', save=True, skip=True)
# print("Manual labelling complete!\n(Time taken: %d seconds.)\n" % (time.time() - tic))
# =============================================================================


#%% PLOT WORM TRAJECTORIES


