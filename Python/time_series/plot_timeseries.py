#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 18:41:27 2021

@author: sm5911
"""

#%% Imports

import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from read_data.read import get_skeleton_data
from matplotlib import pyplot as plt

#%% Globals

FILENAMES_SUMMARIES_PATH = '/Volumes/hermes$/KeioScreen2_96WP/Results/20210928/filenames_summary_tierpsy_plate_20211004_140945.csv'

METADATA_PATH = '/Volumes/hermes$/KeioScreen2_96WP/AuxiliaryFiles/metadata.csv'

BLUELIGHT_FRAMES = [(1500,1751),(4000,4251),(6500,6751)]

#%% Function

def add_bluelight_to_plot(fig, ax, bluelight_frames=BLUELIGHT_FRAMES):
    """ Add lines to plot to indicate video frames where bluelight stimulus was delivered 
    
        Inputs
        ------
        fig, ax : figure and axes from plt.subplots()
        bluelight_frames : list of tuples (start, end) 
    """
    assert type(bluelight_frames) == list or type(bluelight_frames) == tuple
    if not type(bluelight_frames) == list:
        bluelight_frames = [bluelight_frames]
    
    for bt in bluelight_frames:
        (start, stop) = bt       
        ax.axvspan(start, stop, facecolor='blue', alpha=0.75)
     
    return fig, ax

def plot_timeseries_hydra(filename, x='timestamp', y='motion_mode', saveDir=None, window=100, **kwargs):
    """ Timeseries plot of feature (y) throughout video """
    
    timeseries_data = get_skeleton_data(filename, rig='Hydra', dataset='timeseries_data')
        
    wells_list = list(timeseries_data['well_name'].unique())

    if not len(wells_list) == 16:
        stem = Path(filename).parent.name
        print("Missing results for %d well(s): '%s'" % (16 - len(wells_list), stem))
     
    # get data for each well in turn
    grouped_well = timeseries_data.groupby('well_name')
    for well in wells_list:
        well_data = grouped_well.get_group(well)

        xmax = max(9000, well_data[x].max())

        # frame average
        grouped_frame = well_data.groupby(x)
        well_mean = grouped_frame[y].mean()
        well_std = grouped_frame[y].std()
        
        # moving average (optional)
        if window:
            well_mean = well_mean.rolling(window=window, center=True).mean()
            well_std = well_std.rolling(window=window, center=True).std()

        colours = []
        for mm in np.array(well_mean):
            if np.isnan(mm):
                #colours.append('white')
                colours.append([255,255,255]) # white
            elif int(mm) == 1:
                #colours.append('blue')
                colours.append([0,0,255]) # blue
            elif int(mm) == -1:
                #colours.append('red')
                colours.append([255,0,0]) # red
            else:
                #colours.append('grey')
                colours.append([128,128,128]) # gray
        colours = np.array(colours) / 255.0
                
        # cmap = plt.get_cmap('Greys', 3)
        # cmap.set_under(color='red', alpha=0)
        # cmap.set_over(color='blue', alpha=0)
        
        # Plot time series                
        plt.close('all')
        fig, ax = plt.subplots(figsize=(12,6))

        #sns.scatterplot(x=well_mean.index, y=well_mean.values, ax=ax)#hue=well_mean.index, palette=colours, ax=ax)
        ax.scatter(x=well_mean.index, y=well_mean.values, c=colours, ls='-', marker='.', **kwargs)
        ax.set_xlim(0, xmax)
        ax.axhline(0, 0, xmax, ls='--', marker='o') 
        ax.fill_between(well_mean.index, well_mean-well_std, well_mean+well_std, 
                        where=(well_mean > 0), facecolor='blue', alpha=0.5) # egdecolor=None
        ax.fill_between(well_mean.index, well_mean-well_std, well_mean+well_std, 
                        where=(well_mean < 0), facecolor='red', alpha=0.5) # egdecolor=None        
        # TODO: hue='worm_index'

        fig, ax = add_bluelight_to_plot(fig, ax)
        
        # sns.scatterplot(x=x, y=y, data=timeseries_data, **kwargs)
        if saveDir:
            Path(saveDir).mkdir(exist_ok=True, parents=True)
            plt.savefig(Path(saveDir) / 'roaming_state_{}.png'.format(well))
            plt.close()
        else:
            plt.show()
        
    return

def main(args):
    assert Path(args.filenames_summaries_path).exists()
    filenames_summaries = pd.read_csv(args.filenames_summaries_path, comment="#")
    filenames_list = filenames_summaries.loc[filenames_summaries['is_good'],'filename'].to_list()

    for filename in tqdm(filenames_list):
        stem = Path(filename).parent.name
        print("Plotting motion mode timeseries for '%s'" % stem)

        plot_timeseries_hydra(filename, saveDir=args.save_dir / stem, window=None)
        
        break
    
#%% Main
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot worm motion mode timeseries for videos in \
                                     filenames summaries file")
    parser.add_argument('-f', '--filenames_summaries_path', help="Path to tierpsy filenames \
                        summaries file", default=FILENAMES_SUMMARIES_PATH, type=str)
    parser.add_argument('--save_dir', help="Path to save timeseries plots", default=None, type=str)
    args = parser.parse_args()
    
    if args.save_dir is None:
        args.save_dir = Path(args.filenames_summaries_path).parent / 'motion_mode_timeseries'
    
    main(args)
