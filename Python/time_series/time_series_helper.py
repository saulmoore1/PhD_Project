#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-series Analysis

@author: sm5911
@date: 01/03/2021

"""

#%% Imports

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

from preprocessing.compile_hydra_data import process_feature_summaries

# TODO: Update timeseries plots for plotting windowed summaries
# Luigi: making more summaries and handling windows is a pain, no tierpsytools functions exist yet

#%% Globals

EXAMPLE_METADATA_PATH = "/Volumes/hermes$/KeioScreen_96WP/AuxiliaryFiles/metadata_annotated.csv"
EXAMPLE_RESULTS_DIR = "/Volumes/hermes$/KeioScreen_96WP/Results"
EXAMPLE_FEATURE_LIST = ['speed_abs_50th']

#%% Functions

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
    
    from matplotlib import patches as mpatches

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


def plot_timeseries_hydra(metadata_df, features_df, feature):
    """ """
    
    print(metadata_df, features_df, feature)
    #FIXME - read the metadata.csv and the results (which are not aligned in the files) and not use the align_bluelight option
    # whole summary for prestim, the whole summary for bluelight, the whole poststim summary = 3 timepoints
    # use tierpsytools, so your metadata will have a column named ‘bluelight’ that can be either 
    # ‘prestim’, ‘bluelight’, ‘poststim’. Map these values to 0, 1, 2. Do NOT use the ‘align bluelight’ 
    # options. then use seaborn with bluelight as your x value
    # keep bluelight not aligned - that leaves the df in a “long format” that seaborn likes a lot
    # Luigi wrote something for Ida but is not ready for tierpsytools. 
    # "Unless Andre is specifically telling you to look at windowed summaries, you should not go down the rabbit hole" - Luigi
    pass
    

#%% Main

if __name__ == '__main__':
    # Accept feature list from command line
    parser = argparse.ArgumentParser(description='Time-series analysis of selected features')
    parser.add_argument("--compiled_metadata_path", help="Path to compiled metadata file",
                        default=EXAMPLE_METADATA_PATH)
    parser.add_argument("--results_dir", help="Path to 'Results' directory containing full features\
                        and filenames summary files", default=EXAMPLE_RESULTS_DIR)
    parser.add_argument('--feature_list', help="List of selected features for time-series analysis", 
                        nargs='+', default=EXAMPLE_FEATURE_LIST)
    args = parser.parse_args()

    assert Path(args.compiled_metadata_path).exists()
    assert Path(args.results_dir).is_dir()
    assert type(args.feature_list) == list
    
    # Read metadata
    metadata = pd.read_csv(args.compiled_metadata_path, dtype={"comments":str})
    
    # Ensure align bluelight is False
    # NB: leaves the df in a “long format” that seaborn likes
    metadata, features = process_feature_summaries(metadata, args.results_dir, align_bluelight=False)
    
    # Find masked HDF5 video files
    print("%d selected features loaded." % len(args.feature_list))

    # Perform time-series analysis for selected features
    for f in args.feature_list:
        plot_timeseries_hydra(metadata, features, f)
            
        