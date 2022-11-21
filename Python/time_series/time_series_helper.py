#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-series Analysis

@author: sm5911
@date: 01/03/2021

"""

#%% Imports 

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path 
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches

from write_data.write import write_list_to_file
from tierpsytools.read_data.get_timeseries import read_timeseries

#%% Functions

def get_strain_timeseries(metadata, 
                          project_dir, 
                          strain='BW', 
                          group_by='bacteria_strain',
                          feature_list=['motion_mode'],
                          save_dir=None,
                          n_wells=96,
                          verbose=True,
                          return_error_log=False):
    """ Load saved timeseries reults for strain, or compile from featuresN timeseries data """

    strain_timeseries = None
    
    if save_dir is not None:
        save_path = Path(save_dir) / '{0}_timeseries.csv'.format(strain)
        if save_path.exists():
            if verbose:
                print("Loading timeseries data for %s..." % strain)
            strain_timeseries = pd.read_csv(save_path)
            assert all(f in strain_timeseries.columns for f in feature_list)

    if strain_timeseries is None: 
        print("Compiling timeseries for %s..." % strain)
        strain_meta = metadata.groupby(group_by).get_group(strain)
                    
        # make dict of video imgstore names and wells we need to extract for strain data
        video_list = sorted(strain_meta['featuresN_filename'].unique())
        grouped_video = strain_meta.groupby('featuresN_filename')
        video_dict = {vid : sorted(grouped_video.get_group(vid)['well_name'].unique()) 
                      for vid in video_list}
          
        feature_list = [feature_list] if isinstance(feature_list, str) else feature_list
        assert isinstance(feature_list, list)
        colnames = ['worm_index','timestamp','well_name']
        colnames.extend(feature_list)
        
        error_log = []
        strain_timeseries_list = []
        for featuresN_file, wells in tqdm(video_dict.items()):
            
            filename = Path(featuresN_file)

            try:
                df = read_timeseries(filename, 
                                     names=colnames,
                                     only_wells=wells if n_wells != 6 else None)
                
                df['filename'] = filename
                if len(wells) == 1:
                    df['well_name'] = wells[0]
                                    
                strain_timeseries_list.append(df)
                
            except Exception as E:
                if verbose:
                    print("ERROR reading file! %s" % filename)
                    print(E)
                error_log.append(filename)
                
        # compile timeseries data for strain 
        strain_timeseries = pd.concat(strain_timeseries_list, axis=0, ignore_index=True)
        
        # save timeseries dataframe to file
        if save_dir is not None:
            if verbose:
                print("Saving timeseries data for %s..." % strain)
            save_dir.mkdir(exist_ok=True, parents=True)
            strain_timeseries.to_csv(save_path, index=False)
                 
            if len(error_log) > 0:
                write_list_to_file(error_log, Path(save_dir) / 'error_log.txt')
                
    return strain_timeseries

def plot_timeseries_phenix(df, colour_dict, window=1000, acclimtime=0, annotate=True,\
                           legend=True, ax=None, count=False, show=True, **kwargs):
    """ Function to plot time series of mean proportion of worms on food, given 
        an input dataframe containing mean and std for each food, and a dictionary
        of plot colours.
        Arguments: 
        - window (default = 1000) Number frames for moving average smoothing)
        - orderby (default = None) If provided, first groups df by variable and 
          calculates either mean/sum
        - count (default = False) Return counts (number of worms), not mean proportion of worms
    """
    
    # List of food labels + dictionary keys for plot colours
    food_labels = list(df.columns.levels[0])
    colour_keys = [food.split('_')[0] for food in food_labels]
    
    # Initialise plot
    if not ax:
        fig, ax = plt.subplots(figsize=(12,6))
    
    # Calculate moving window + plot time series
    for i, food in enumerate(food_labels):
        if window:
            moving_mean = df[food]['mean'].rolling(window=window, center=True).mean()
            moving_std = df[food]['std'].rolling(window=window, center=True).mean()
            
            # Plot time series
            ax.plot(moving_mean, color=colour_dict[colour_keys[i]], **kwargs) # OPTIONAL: yerr=moving_std
            ax.fill_between(moving_mean.index, moving_mean-moving_std, moving_mean+moving_std,\
                                  color=colour_dict[colour_keys[i]], alpha=0.25, edgecolor=None)
        else:
            # Plot un-smoothed time series
            ax.plot(df[food]['mean'], color=colour_dict[colour_keys[i]], **kwargs)
            ax.fill_between(df[food]['mean'].index, df[food]['mean']-df[food]['std'], df[food]['mean']+df[food]['std'],
                            color=colour_dict[colour_keys[i]], alpha=0.5, edgecolor=None)
    if annotate:
        if count:
            plt.ylim(-0.05, df.max(axis=1).max() + 0.25)
            plt.ylabel("Number of Worms", fontsize=15, labelpad=10)
            #plt.axhline(y=10, color='k', linestyle='--') # plot line at n_worms (y) = 10
        else:
            plt.ylim(-0.05, 1.15)
            plt.ylabel("Proportion Feeding", fontsize=15, labelpad=10)
        xticks = np.linspace(0, np.round(max(df.index),-5), num=10, endpoint=True).astype(int)
        ax.set_xticks(xticks)
        xticklabels = np.ceil(np.linspace(0, np.round(max(df.index), -5), num=10, endpoint=True)/25/900)/4
        xticklabels = [str(int(lab*60)) for lab in xticklabels]
        ax.set_xticklabels(xticklabels)
        plt.xlim(0, max(df.index))
        plt.xlabel("Time (minutes)", fontsize=15, labelpad=10)
    else:
        # Turn off tick/axes labels
        x_axis = ax.axes.get_xaxis()
        x_axis.set_label_text("")
        
    # Account for acclimation time
    if acclimtime > 0:
        x = np.arange(0, acclimtime)
        y = acclimtime
        ax.fill_between(x, y, -0.05, color='grey', alpha='0.5', interpolate=True)
        ax.axvline(0, ls='-', lw=1, color='k')
        ax.axvline(acclimtime, ls='-', lw=1, color='k')
        if annotate:
            plt.text(acclimtime/max(df.index)+0.01, 0.97, "Acclimation: {0} mins".format(int(acclimtime/25/60)),\
                     ha='left', va='center', transform=ax.transAxes, rotation=0, color='k')
            # ax.axvline(acclimtime + window/2, ls='-', lw=2, color='r') # Gap due to window smoothing
            
    # Add plot legend
    if legend:
        patches = []
        legend_keys = list(np.unique(colour_keys))
        for key in legend_keys:
            patch = mpatches.Patch(color=colour_dict[key], label=key)
            patches.append(patch)
        plt.tight_layout(rect=[0, 0, 0.88, 0.98])
        plt.legend(handles=patches, labels=legend_keys, loc=(1.02, 0.8),\
                   borderaxespad=0.4, frameon=False, fontsize=15)
    if show:
        plt.show(); plt.pause(0.0001)
    return(plt)
            