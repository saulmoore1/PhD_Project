#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULE: CLEAN

@author: sm5911
@date: 09/08/2019

"""

# IMPORTS
import time
import pandas as pd


#%% FUNCTIONS
def fillNaNgroupby(df, group_by, non_data_cols=None, method='mean', axis=0):
    """ A function to impute missing values in given dataframe by replacing
        with mean value by given grouping variable. """
    
    def groupMeanValue(group, axis=axis):
        group = group.fillna(group.mean(axis=axis))
        return group

    tic = time.time()

    original_cols = list(df.columns)
    if isinstance(non_data_cols, pd.Index):
        non_data_cols = list(non_data_cols)
    if non_data_cols:
        if not isinstance(non_data_cols, list):
            print("Please provide non data columns as a list or pandas.Index object")
        else:    
            if group_by in non_data_cols:
                #print("'%s' found in non-data columns" % group_by)
                non_data_cols_nogroupby = [col for col in non_data_cols if col != group_by]
                datacols = [col for col in df.columns if col not in non_data_cols_nogroupby]
            elif group_by not in non_data_cols:
                datacols = [col for col in df.columns if col not in non_data_cols]

            nondata = df[non_data_cols]
            data = df[datacols]
            
            n_nans = data.isna().sum(axis=axis).sum()
            if n_nans > 0:
                print("Imputing %d missing values using %s value for each '%s'" % (n_nans, method, group_by))
    
                data = data.groupby(group_by).transform(groupMeanValue)
                df = pd.concat([nondata, data], axis=1, sort=False)
            else:
                print("Woohoo! No NaN values found in data!")
    else:
        n_nans = df.isna().sum(axis=axis).sum()
        print("Imputing %d missing values using %s value for each '%s'" % (n_nans, method, group_by))
        df = df.groupby(group_by).transform(groupMeanValue)
    
    toc = time.time()
    print("Done.\n(Time taken: %.1f seconds)" % (toc - tic))
    return df[original_cols]

#%%
def cleanSummaryResults(full_results_df, impute_NaNs_by_group=False, preconditioned_from_L4=True,\
                        nondatacols=None, snippet=0, nan_threshold=0.75):
    """ 
    
    """
    if preconditioned_from_L4:
        L4_yesno = 'yes'
    else:
        L4_yesno = 'no'
    # Filter feature summary results to look at 1st video snippets only
    first_snippets_df = full_results_df[full_results_df['filename'].str.contains('00000{0}_featuresN.hdf5'.format(snippet))]
    
    # Filter feature summary results to look at L4-prefed worms (long food exposure) only
    L4_1st_snippets_df = first_snippets_df[first_snippets_df['preconditioned_from_L4'].str.lower() == L4_yesno]
    
    # Divide dataframe into 2 dataframes: data (feature summaries) and non-data (metadata)
    colnames_all = list(L4_1st_snippets_df.columns)
    if nondatacols:
        colnames_nondata = nondatacols
        colnames_data = [col for col in colnames_all if col not in colnames_nondata]
    else:
        # Use defaults
        colnames_nondata = colnames_all[:25]
        colnames_data = colnames_all[25:]
        
    L4_1st_snippets_data = L4_1st_snippets_df[colnames_data]
    L4_1st_snippets_nondata = L4_1st_snippets_df[colnames_nondata]
    
    # Drop data columns with too many nan values
    colnamesBefore = L4_1st_snippets_data.columns
    L4_1st_snippets_data = L4_1st_snippets_data.dropna(axis=1, thresh=nan_threshold)
    colnamesAfter = L4_1st_snippets_data.columns
    nan_cols = len(colnamesBefore) - len(colnamesAfter)
    print("Dropped %d features with too many NaNs" % nan_cols)
    
    # All dropped features here have to do with the 'food_edge' (which is undefined, so NaNs are expected)
    droppedFeatsList_NaN = [col for col in colnamesBefore if col not in colnamesAfter]
    
    # Drop data columns that contain only zeros
    colnamesBefore = L4_1st_snippets_data.columns
    L4_1st_snippets_data = L4_1st_snippets_data.drop(columns=L4_1st_snippets_data.columns[(L4_1st_snippets_data == 0).all()])
    colnamesAfter = L4_1st_snippets_data.columns
    zero_cols = len(colnamesBefore) - len(colnamesAfter)
    print("Dropped %d features with all zeros" % zero_cols)
    
    # All dropped features here have to do with 'd_blob' (blob derivative calculations)
    droppedFeatsList_allZero = [col for col in colnamesBefore if col not in colnamesAfter]
    
    if not impute_NaNs_by_group:
        # Impute remaining NaN values (using global mean feature value for each food)
        n_nans = L4_1st_snippets_data.isna().sum(axis=1).sum()
        if n_nans > 0:
            print("Imputing %d missing values using global mean value for each feature" % n_nans)  
            L4_1st_snippets_data = L4_1st_snippets_data.fillna(L4_1st_snippets_data.mean(axis=1))
        else:
            print("No need to impute! No remaining NaN values found in feature summary results.")
    
    # Re-combine into full results dataframe
    L4_1st_snippets_df = pd.concat([L4_1st_snippets_nondata, L4_1st_snippets_data], axis=1, sort=False)
    
    if impute_NaNs_by_group:
        # Impute remaining NaN values (using mean feature value for each food)
        n_nans = L4_1st_snippets_data.isna().sum(axis=1).sum()
        if n_nans > 0:    
            L4_1st_snippets_df = fillNaNgroupby(df=L4_1st_snippets_df, group_by='food_type',\
                                                non_data_cols=colnames_nondata)
        else:
            print("No need to impute! No remaining NaN values found in feature summary results.") 
            
    return L4_1st_snippets_df, droppedFeatsList_NaN, droppedFeatsList_allZero
