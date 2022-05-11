#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-series Analysis

@author: sm5911
@date: 01/03/2021

"""

#%% Functions

def get_strain_timeseries(metadata, 
                          project_dir, 
                          strain='BW', 
                          group_by='bacteria_strain', 
                          save_dir=None,
                          only_wells=None,
                          verbose=True):
    """ Load saved timeseries reults for strain, or compile from featuresN timeseries data """

    import pandas as pd
    from tqdm import tqdm
    from pathlib import Path 
    from tierpsytools.read_data.get_timeseries import read_timeseries

    strain_timeseries = None
    
    if save_dir is not None:
        save_path = Path(save_dir) / '{0}_timeseries.csv'.format(strain)
        if save_path.exists():
            if verbose:
                print("Loading timeseries data for %s.." % strain)
            strain_timeseries = pd.read_csv(save_path)

    if strain_timeseries is None:        
        strain_meta = metadata.groupby(group_by).get_group(strain)
                    
        # make dict of video imgstore names and wells we need to extract for strain data
        video_list = sorted(strain_meta['imgstore_name'].unique())
        grouped_video = strain_meta.groupby('imgstore_name')
        video_dict = {vid : sorted(grouped_video.get_group(vid)['well_name'].unique()) 
                      for vid in video_list}     
          
        strain_timeseries_list = []
        for imgstore, wells in tqdm(video_dict.items()):
            
            filename = Path(project_dir) / 'Results' / imgstore / 'metadata_featuresN.hdf5'
            df = read_timeseries(filename, 
                                 names=['worm_index','timestamp','motion_mode','well_name'],
                                 only_wells=only_wells)
            df['filename'] = filename
            if len(wells) == 1:
                df['well_name'] = wells[0]
    
            strain_timeseries_list.append(df)
                
        # compile timeseries data for strain 
        strain_timeseries = pd.concat(strain_timeseries_list, axis=0, ignore_index=True)
        
        # save timeseries dataframe to file
        if save_dir is not None:
            if verbose:
                print("Saving timeseries data for %s" % strain)
            save_dir.mkdir(exist_ok=True, parents=True)
            strain_timeseries.to_csv(save_path, index=False)
                 
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
    
    import numpy as np
    from matplotlib import pyplot as plt
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
            