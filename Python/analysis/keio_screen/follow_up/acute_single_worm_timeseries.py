#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse results of acute single worm experiments, where a single worms were picked onto 35mm plates,
seeded with either E. coli BW25113 or BW25113ΔfepD bacteria, and tracked as soon as the worm
approached the bacterial lawn 

(30-minute videos with 10 seconds bluelight every 5 minutes)

@author: sm5911
@date: 26/02/2022

"""

#%% Imports

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
from preprocessing.compile_hydra_data import compile_metadata
from time_series.time_series_helper import get_strain_timeseries
from time_series.plot_timeseries import plot_timeseries_motion_mode, add_bluelight_to_plot

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Keio_Acute_Single_Worm"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Acute_Single_Worm"

IMAGING_DATES = ['20220206','20220209','20220212']
N_WELLS = 6

CONTROL_STRAIN = 'BW'
FEATURE = 'motion_mode_paused_fraction'

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
MIN_NSKEL_PER_VIDEO = None
MIN_NSKEL_SUM = 50
PVAL_THRESH = 0.05

FPS = 25
VIDEO_LENGTH_SECONDS = 30*60
BIN_SIZE_SECONDS = 5
SMOOTH_WINDOW_SECONDS = 5
THRESHOLD_N_SECONDS = 10
BLUELIGHT_TIMEPOINTS_MINUTES = [5,10,15,20,25]
N_WELLS = 6

#%% Function 

def plot_timeseries_motion_mode_single_worm(df, window=None, mode=None, max_n_frames=None,
                                            title=None, figsize=(12,6), ax=None, saveAs=None,
                                            sns_colour_palette='pastel', colour=None, 
                                            bluelight_frames=None,
                                            cols = ['motion_mode','filename','well_name','timestamp'], 
                                            alpha=0.5):
    """ Plot motion mode timeseries from 'timeseries_data' for a given treatment (eg. strain) 
    
        Inputs
        ------
        df : pd.DataFrame
            Compiled dataframe of 'timeseries_data' from all featuresN HDF5 files for a given 
            treatment (eg. strain) 
        window : int
            Moving average window of n frames
        error : bool
            Add error to timeseries plots
        mode : str
            The motion mode you would like to plot (choose from: ['stationary','forwards','backwards'])
        max_n_frames : int
            Maximum number of frames in video (x axis limit)
        title : str
            Title of figure (optional, ax is returned so title and other plot params can be added later)
        figsize : tuple
            Size of figure to be passed to plt.subplots figsize param
        ax : matplotlib AxesSubplot, None
            Axis of figure subplot
        saveAs : str
            Path to save directory
        sns_colour_palette : str
            Name of seaborn colour palette
        colour : str, None
            Plot single colour for plot (if plotting a single strain or a single motion mode)
        bluelight_frames : list
            List of tuples for (start, end) frame numbers of each bluelight stimulus (optional)
        cols : list
            List of cols to group_by
            
        Returns
        -------
        fig : matplotlib Figure 
            If ax is None, so the figure may be saved
            
        ax : matplotlib AxesSubplot
            For iterative plotting   
    """
 
    # discrete data mapping
    motion_modes = ['stationary','forwards','backwards']
    motion_dict = dict(zip([0,1,-1], motion_modes))

    if mode is not None:
        if type(mode) == int or type(mode) == float:
            mode = motion_dict[mode]     
        else:
            assert type(mode) == str 
            mode = 'stationary' if mode == 'paused' else mode
            assert mode in motion_modes
    
    assert all(c in df.columns for c in cols)

    # drop NaN data
    df = df.loc[~df['motion_mode'].isna(), cols]
     
    # map whether forwards, backwards or stationary motion in each frame
    df['motion_name'] = df['motion_mode'].map(motion_dict)
    assert not df['motion_name'].isna().any()
        
    # total number of worms recorded at each timestamp (across all videos)
    total_timestamp_count = df.groupby(['timestamp'])['filename'].count()
    
    # total number of worms in each motion mode at each timestamp
    motion_mode_count = df.groupby(['timestamp','motion_name'])['filename'].count().reset_index()
    motion_mode_count = motion_mode_count.rename(columns={'filename':'count'})
        
    frac_mode = pd.merge(motion_mode_count, total_timestamp_count, 
                          left_on='timestamp', right_on=total_timestamp_count.index, 
                          how='left')
        
    # divide by total filename count
    frac_mode['fraction'] = frac_mode['count'] / frac_mode['filename']
    
    # subset for motion mode
    plot_df = frac_mode[frac_mode['motion_name']==mode][['timestamp','fraction']]
     
    # crop timeseries data to standard video length (optional)
    if max_n_frames:
        plot_df = plot_df[plot_df['timestamp'] <= max_n_frames]
    
    # moving average (optional)
    if window:
        plot_df = plot_df.set_index('timestamp').rolling(window=window, 
                                                         center=True).mean().reset_index()

    if ax is None:
        fig, ax = plt.subplots(figsize=(15,6))

    # motion_ls_dict = dict(zip(motion_modes, ['-','--','-.']))                
    sns.lineplot(data=plot_df, 
                 x='timestamp', 
                 y='fraction', 
                 ax=ax, 
                 ls='-', # motion_ls_dict[mode] if len(mode_list) > 1 else '-',
                 hue=None, #'motion_name' if colour is None else None, 
                 palette=None, #palette if colour is None else None,
                 color=colour)

    # add decorations
    if bluelight_frames is not None:
        ax = add_bluelight_to_plot(ax, bluelight_frames=bluelight_frames, alpha=alpha)

    if title:
        plt.title(title, pad=10)

    if saveAs is not None:
        Path(saveAs).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(saveAs)
    
    if ax is None:
        return fig, ax
    else:
        return ax
    
    
#%% Main

if __name__ == '__main__':

    AUX_DIR = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    RES_DIR = Path(PROJECT_DIR) / 'Results'

    META_PATH = Path(SAVE_DIR) / 'metadata.csv'
    FEAT_PATH = Path(SAVE_DIR) / 'features.csv'
    
    # load/compile metadata
    metadata, metadata_path = compile_metadata(aux_dir=AUX_DIR,
                                               imaging_dates=IMAGING_DATES,
                                               add_well_annotations=N_WELLS==96,
                                               n_wells=N_WELLS,
                                               from_source_plate=False)            
    
    save_dir = Path(SAVE_DIR) / 'timeseries'
    
    # TODO: omit data for Hydra05 to see if this fixes the bug due to timestamps lagging on some LoopBio videos
    #metadata = metadata[metadata['instrument_name'] != 'Hydra05']
       
    # create bins for frame of first food encounter
    bins = [int(b) for b in np.linspace(0, VIDEO_LENGTH_SECONDS*FPS, 
                                        int(VIDEO_LENGTH_SECONDS/BIN_SIZE_SECONDS+1))]
    metadata['first_food_binned_freq'] = pd.cut(x=metadata['first_food_frame'], bins=bins)
    first_food_freq = metadata.groupby('first_food_binned_freq', as_index=False).count()

    # plot histogram of binned frequency of first food encounter 
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12,6))    
    sns.barplot(x=first_food_freq['first_food_binned_freq'].astype(str), 
                y=first_food_freq['first_food_frame'], alpha=0.9, palette='rainbow')        
    ax.set_xticks([x - 0.5 for x in ax.get_xticks()])
    ax.set_xticklabels([str(int(b / FPS)) for b in bins], rotation=45)
    ax.set_xlim(0, np.where(bins > metadata['first_food_frame'].max())[0][0])
    ax.set_xlabel("Time until first food encounter (seconds)", fontsize=15, labelpad=10)
    ax.set_ylabel("Number of videos", fontsize=15, labelpad=10)
    ax.set_title("N = {} videos".format(metadata.shape[0]), loc='right')
    plt.tight_layout()
    
    # save histogram
    save_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_dir / "first_food_encounter.pdf")
    plt.close()
    
    # Subset to remove all videos where the worm took >10 seconds (250 frames) to reach the food
    # from the start of the video recording
    # NB: inculding the 'hump' up to around <75 seconds makes no visible difference to the plot
    metadata = metadata[metadata['first_food_frame'] < THRESHOLD_N_SECONDS*FPS]
    
    sample_sizes = metadata.groupby('bacteria_strain').count()['first_food_frame']
    print("\nThreshold time until food encounter: {} seconds".format(THRESHOLD_N_SECONDS))
    for s in sample_sizes.index:
        print('{0}: n={1}'.format(s, sample_sizes.loc[s]))
        
    mean_delay_seconds = int(metadata['first_food_frame'].mean()) / FPS
    print("Worms took %.1f seconds on average to reach food" % mean_delay_seconds)
    
    # Timeseries plots
    
    strain_list = sorted(list(metadata['bacteria_strain'].unique()))
    colours = sns.color_palette(palette="tab10", n_colors=len(strain_list))
    bluelight_frames = [(i*60*FPS, i*60*FPS+10*FPS) for i in BLUELIGHT_TIMEPOINTS_MINUTES]

    plot_dir = save_dir / 'motion_mode_plots_max_delay={}s'.format(THRESHOLD_N_SECONDS)
    plot_dir.mkdir(exist_ok=True)

    # both strains together, for each motion mode
    for mode in ['forwards','backwards','stationary']:

        plt.close('all')
        fig, ax = plt.subplots(figsize=(15,5))

        for s, strain in enumerate(strain_list):
            print("Plotting motion mode %s timeseries for %s..." % (mode, strain))

            strain_timeseries = get_strain_timeseries(metadata,
                                                      project_dir=PROJECT_DIR, 
                                                      strain=strain,
                                                      group_by='bacteria_strain',
                                                      save_dir=save_dir / 'data')
            
            ax = plot_timeseries_motion_mode_single_worm(df=strain_timeseries,
                                                         window=SMOOTH_WINDOW_SECONDS*FPS,
                                                         mode=mode,
                                                         max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
                                                         title=None,
                                                         saveAs=None,
                                                         ax=ax,
                                                         bluelight_frames=bluelight_frames,
                                                         colour=colours[s],
                                                         alpha=0.75)
            
        ax.axvspan(mean_delay_seconds*FPS-FPS, mean_delay_seconds*FPS+FPS, facecolor='k', alpha=1)
        ax.axvspan(THRESHOLD_N_SECONDS*FPS-FPS, THRESHOLD_N_SECONDS*FPS+FPS, facecolor='r', alpha=1)
        xticks = np.linspace(0,VIDEO_LENGTH_SECONDS*FPS, 31)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
        ax.set_xlabel('Time (minutes)', fontsize=15, labelpad=10)
        ax.set_ylabel('Fraction {}'.format(mode), fontsize=15, labelpad=10)
        ax.legend(strain_list, fontsize=12, frameon=False, loc='best')
        ax.set_title("motion mode fraction '%s' (total n=%d worms)" % (mode, metadata.shape[0]),
                     fontsize=15, pad=10)
        # save plot
        plt.savefig(plot_dir / '{}.png'.format(mode), dpi=300)  
        plt.close()
        
        
