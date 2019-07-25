#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULE: CALCULATE

@author: sm5911
@date: 19/02/2019

"""

# IMPORTS
import os, copy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import path as mpath
from scipy import stats

# CUSTOM IMPORTS
from Save import savefig

#%% FUNCTIONS
def onfood(poly_dict, df, returnNone=True):
    """ A function for evaluating whether a set of coordinates fall inside/outside 
        of polygon regions/shapes. 
        
        INPUTS:
            
        (1) poly_dict - a dictionary of polygon (eg. food region) x,y coords, 
        (2) df - a dataframe of worm centroid x,y coords by frame_number from Tierpsy
            generated featuresN data. 
        (Optional)
        (3) returnNone - True*/False; return proportion off-food as well? 
            
        The dataframe is returned with a presence-absence truth matrix appended 
        (ie. on food/not on food). """
        
    for key, values in poly_dict.items():
        polygon = mpath.Path(values, closed=True)
        df[key] = polygon.contains_points(df[['x','y']])
        
    # Infer 'None' column
    if returnNone:
        df['None'] = np.logical_not(np.logical_or(*[df[key] for key in poly_dict.keys()]))
    return(df)

#%%
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

#%%    
def summarystats(df, NoneColumn=True):
    """ A function to compute summary statistics for food choice presence/absence 
        data. Returns: the following statistics for each food region: """
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

#%%    
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

#%%
def leavingevents(df, window=50, removeNone=True, plot=True, savePath=None):
    """ A function to investigate how long worms leave the food for. It accepts
        as input a truth matrix of ON/OFF food returns a dataframe of leaving event information.
        - savePath --plots a time-series plot of leaving events and saves to file path provided. """
        
    colnames = list(df.select_dtypes(include=['bool']).columns.copy())
    if removeNone:
        colnames.remove('None')
    foods = copy.deepcopy(colnames) # TODO: FIX
    colnames.append('leaving_duration_nframes')
    df_group_worm = df.groupby(['worm_id'])
    worm_ids = np.unique(df['worm_id'])
    # Pre-allocate growing dataframe for storing how long each worm leaves each food for
    out_df = pd.DataFrame(columns=colnames)
    if plot:
        plt.close('all'); plt.figure(figsize=(15,3))
    for worm in worm_ids:
        df_worm = df_group_worm.get_group(worm)
        # TODO: Do not treat foods independently! What about worms that leave to enter the other food? Ok for leaving event selection stage, but not perfect
        for fc, food in enumerate(foods):
            # For each worm on each food patch, find indices of onfood_df where leaving/entering events occur
            leaving = np.array(df_worm.iloc[np.where(df_worm[food].astype(int).diff() == -1)[0]].index)
            entering = np.array(df_worm.iloc[np.where(df_worm[food].astype(int).diff() == 1)[0]].index)
            if len(leaving) > 0: # If there is a leaving event at all, then...
                if len(entering) > 0:
                    if len(entering) == len(leaving): # If the number of leaving & entering events are the SAME, then...
                        if leaving[0] < entering[0]: # If the FIRST event is a LEAVING event, then...GOOD!
                            pass
                        elif entering[0] < leaving[0]: # Else if the FIRST event is an ENTERING event, then...BAD!
                            # Delete first entering event
                            entering = entering[1:]
                            # AND compare leaving duration to end of trajectory by adding end of trajectory index as last 'entering event'
                            entering = np.insert(entering, len(entering), df_worm.index[-1])
                    elif not len(entering) == len(leaving): # Else, if the number of leaving & entering events are DIFFERENT, then...
                        if leaving[0] < entering[0]: # If the FIRST event is a LEAVING event, then...
                            # Add end of trajectory index as last 'entering event' (ie. compare to end of trajectory)
                            entering = np.insert(entering, len(entering), df_worm.index[-1])
                        elif entering[0] < leaving[0]: # Else, if the first event is an entering event, then...
                            # Delete first entry event
                            entering = entering[1:]
                else: # If one leaving but no entering events, then...
                    entering = np.array([df_worm.index[-1]]) # Compare to end of trajectory
                leaving_duration = entering - leaving
                leaving_df = df_worm.loc[leaving - 1] # -1 to index the frame just before the leaving event, so we know which food it left FROM
                # Append columns for leaving duration to leaving frame data for worm on food
                if removeNone:
                    leaving_df = leaving_df.drop("None", axis=1)
                leaving_df['leaving_duration_nframes'] = pd.Series(leaving_duration, index=leaving_df.index)
                # Append as rows to out-dataframe
                out_df = out_df.append(leaving_df, sort=True)
                
                # Plot leaving events through time
                if plot:
                    plt.plot(df_worm['frame_number'], df_worm[food])
                    plt.xlim(0,np.max(df.frame_number))
                    plt.xlabel("Frame Number", fontsize=15, labelpad=10)
                    plt.ylabel("Proportion Feeding", fontsize=15, labelpad=10)
                    for frame in leaving_df['frame_number']:
                        plt.axvline(frame, ls='--', lw=2, color='r')
    if plot:
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        plt.show(); plt.pause(1)
        if savePath:
            directory = os.path.dirname(savePath)
            if not os.path.exists(directory):
                os.makedirs(directory)
            savefig(savePath, tight_layout=True, tellme=True, saveFormat='png')
    return(out_df)

#%%    
#def leavingstats(df, tellme=False):
#    """ A function to compute summary statistics for food choice assay leaving 
#        event data. Returns: mean, median, std, stderr, conf_min & conf_max of
#        the rate of leaving events from each food source. """
#    summary_df = df
#    return(summary_df)

#%%        
def movingaverage(x, N):
    """ A function for calculating a moving average along given vector x, 
        by a sliding window size of N. """
        
    cumsum = np.cumsum(np.insert(x.values, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

#%%
def movingbins(x, binsize=1000):
    x = x.values
    bin_means = (np.histogram(x, bins=int(np.round(len(x)/binsize)), weights=x)[0] /
                 np.histogram(x, bins=int(np.round(len(x)/binsize)))[0])
    return bin_means

#%%    
def pcainfo(pca, zdf):
    """ A function to plot PCA explained variance, and print the top 10 most 
        important features in the first principle component (P.C.) """
    
    cum_expl_var_frac = np.cumsum(pca.explained_variance_ratio_)

    # Plot explained variance
    fig, ax = plt.subplots()
    plt.plot(range(1,len(cum_expl_var_frac)+1),
             cum_expl_var_frac,
             marker='o')
    ax.set_xlabel('P.C. #')
    ax.set_ylabel('explained $\sigma^2$')
    ax.set_ylim((0,1.05))
    fig.tight_layout()
    
    # print important features
    important_feats = zdf.columns[np.argsort(pca.components_[0]**2)[-10:][::-1]]
    
    print("Top features in first principle component:")
    for feat in important_feats:
        print(feat)

    return important_feats, fig








