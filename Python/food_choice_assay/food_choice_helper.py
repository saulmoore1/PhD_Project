#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Food Choice

Module of helper functions for analysing C. elegans food choice

@author: sm5911
@date: 19/02/2019

"""

#%% IMPORTS

import numpy as np
import pandas as pd

#%% FUNCTIONS

def onfood(poly_dict, df, returnNone=True):
    """ A function for evaluating whether a set of coordinates fall inside/outside 
        of polygon regions/shapes. 
        
        Parameters
        ----------            
        poly_dict : dict
            Dictionary of polygon (eg. food region) x,y coords, 
        df : pd.DataFrame
            Dataframe of worm centroid x,y coords by frame_number from Tierpsy featuresN data. 
        returnNone : bool
            Return proportion of worms off-food as well
            
        Returns
        -------
        df : pd.DataFrame
            Dataframe with presence-absence truth matrix appended (on food/not on food) 
    """
    
    import numpy as np
    from matplotlib import path as mpath

    for key, values in poly_dict.items():
        polygon = mpath.Path(values, closed=True)
        df[key] = polygon.contains_points(df[['x','y']])
        
    # Infer 'None' column
    if returnNone:
        df['None'] = np.logical_not(np.logical_or(*[df[key] for key in poly_dict.keys()]))
        
    return(df)


def foodchoice(df, mean=True, std=False, tellme=False):
    """ A function to calculate and return a dataframe of the mean proportion 
        of worms present in each food region (columns) in each frame (rows). 
        It takes as input: a dataframe (truth matrix) of ON/OFF food for each 
        tracked entity in each frame. If mean=False, returns counts on/off food. """
        
    colnames = list(df.columns[4:])
    # Super-vectorized pandas operations using 'groupby'
    if not mean:
        out_df = df.groupby(['frame_number'])[colnames].sum()
    else:
        if std:
            fundict = {x:['mean','std'] for x in colnames}
            out_df = df.groupby('frame_number').agg(fundict)
        else:
            out_df = df.groupby(['frame_number'])[colnames].mean()
    if tellme:
        if not mean:
            for i, food in enumerate(colnames):
                print("Mean number of worms feeding on %s: %.2f"\
                      % (food, sum(out_df[food])/len(out_df[food])))            
        else:
            if std: 
                for i, food in enumerate(colnames):
                    print("Mean percentage feeding on %s: %.2f%%"\
                          % (food, sum(out_df[food]['mean'])/len(out_df[food]['mean'])*100))
            else:
                for i, food in enumerate(colnames):
                    print("Mean percentage feeding on %s: %.2f%%"\
                          % (food, sum(out_df[food])/len(out_df[food])*100))
    return(out_df)

  
def summarystats(df, NoneColumn=True):
    """ A function to compute summary statistics for food choice presence/absence 
        data. Returns: the following statistics for each food region: """
    
    from scipy import stats

    summary_stats = ['mean', 'median', 'std', 'sem', 'conf_min', 'conf_max', 'max', 'min', 'IQR']
    
    if not NoneColumn:
        df['None'] = 1 - df[list(df.columns)].sum(axis=1)
    
    # Compute summary statistics
    out_array = np.zeros((len(summary_stats), df.shape[1]), dtype=float)
    for i, food in enumerate(df.columns):
        out_array[0,i] = df[food].mean()
        out_array[1,i] = df[food].median()
        out_array[2,i] = df[food].std()
        out_array[3,i] = stats.sem(df[food])
        out_array[4,i] = stats.norm.interval(0.95, loc=df[food].mean(), scale=stats.sem(df[food]))[0]
        out_array[5,i] = stats.norm.interval(0.95, loc=df[food].mean(), scale=stats.sem(df[food]))[1]
        out_array[6,i] = df[food].max() 
        out_array[7,i] = df[food].min()
        out_array[8,i] = stats.iqr(df[food])
    
    summary_df = pd.DataFrame(out_array, columns=[df.columns], index=summary_stats)
    
    return(summary_df)

    
def leavingeventsroll(df, nfood=2, window=50, removeNone=True):
    """ A function for inferring worm leaving events on food patches. It accepts 
        as input, a dataframe comprising a truth matrix of (on food/
        not on food) (rows=len(trajectory_data),columns=food), and returns a 
        dataframe of leaving rates for each worm ID tracked by Tierpsy. """
        
    colnames = df.columns.values.tolist()
    if removeNone:
        colnames.remove('None')
    
    # Pre-allocate growing dataframe to store leaving event data
    out_df = pd.DataFrame(columns=colnames) # dtype=int for speed?
    df_group_worm = df.groupby(['worm_id'])
    unique_worm_ids = np.unique(df['worm_id'])
    
    for worm in unique_worm_ids:
        df_worm = df_group_worm.get_group(worm)
        for fc, food in enumerate(colnames[-nfood:]):
            food_roll = pd.Series(df_worm[food],index=df_worm.index).rolling(window=window, center=True).mean()
            
            # Crop to remove NaNs (false positives in diff computation when converted to type int)
            food_roll = food_roll[window//2:-window//2+1]
            
            # Determine 'true' leaving events
            true_leaving = (food_roll < 0.5).astype(int).diff() == 1
            
            if any(true_leaving):
                leaving_info = df_worm.iloc[np.where(true_leaving == True)]
                
                for i in range(leaving_info.shape[0]):
                    leaving_event = df_worm.iloc[np.where(true_leaving == True)[0][i]]
                    out_df = out_df.append(leaving_event, ignore_index=True)
    if removeNone:
        out_df = out_df.drop("None", axis=1)
    return(out_df)

        
def movingaverage(x, N):
    """ A function for calculating a moving average along given vector x, 
        by a sliding window size of N. """
        
    cumsum = np.cumsum(np.insert(x.values, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def movingbins(x, binsize=1000):
    x = x.values
    bin_means = (np.histogram(x, bins=int(np.round(len(x)/binsize)), weights=x)[0] /
                 np.histogram(x, bins=int(np.round(len(x)/binsize)))[0])
    
    return bin_means
