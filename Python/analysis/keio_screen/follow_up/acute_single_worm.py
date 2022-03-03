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

import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
from tierpsytools.read_data.get_timeseries import read_timeseries
from preprocessing.compile_hydra_data import process_metadata #process_feature_summaries
from time_series.plot_timeseries import plot_timeseries_motion_mode

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Keio_Acute_Single_Worm"
IMAGING_DATES = ['20220206', '20220209', '20220212']

FPS = 25
VIDEO_LENGTH_SECONDS = 30*60
BIN_SIZE_SECONDS = 5
SMOOTH_WINDOW_SECONDS = 5
THRESHOLD_N_SECONDS = 10
BLUELIGHT_TIMEPOINTS_MINUTES = [5,10,15,20,25]

# 5:35, 290:300, 305:315, 315:325, 590:600, 605:615, 615:625, 890:900, 905:915, 915:925, 1190:1200, 
# 1205:1215, 1215:1225, 1490:1500, 1505:1515, 1515:1525, 1790:1800, 1805:1815, 1815:1825

WINDOW_DICT_SECONDS = {0:(5,35), 1:(290,300), 2:(305,315), 
                       3:(315,325), 4:(590,600), 5:(605,615), 
                       6:(615,625), 7:(890,900), 8:(905,915), 
                       9:(915,925), 10:(1190,1200), 11:(1205,1215), 
                       12:(1215,1225), 13:(1490,1500), 14:(1505.1515), 
                       15:(1515,1525), 16:(1790,1800), 17:(1805,1815), 18:(1815,1825)}

#%% Functions
    
#%% Main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyse single worm data")
    parser.add_argument('-r', "--project_dir", help="Path to project root directory", 
                        default=PROJECT_DIR, type=str)
    parser.add_argument('-d', "--imaging_dates", help="List of imaging dates to analyse", 
                        default=IMAGING_DATES, nargs='+', type=str)
    parser.add_argument('-n', "--n_wells", help="Number of wells in plate under each Hydra rig \
    (only 6-well and 96-well are currently supported)", default=6, type=int)
    parser.add_argument('-s', "--save_dir", help="Path to save directory", default=None, type=str)
    args = parser.parse_args()
    
    aux_dir = Path(args.project_dir) / "AuxiliaryFiles"
    args.save_dir = Path(args.project_dir) / "Results" if args.save_dir is None else args.save_dir
        
    # process metadata
    metadata, metadata_path = process_metadata(aux_dir, 
                                               imaging_dates=args.imaging_dates, 
                                               add_well_annotations=False,
                                               n_wells=6)
       
    # create bins
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
    save_path = Path(args.save_dir) / "first_food_encounter.pdf"
    save_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=300)

    # Subset to remove all videos where the worm took >10 seconds (250 frames)
    # to reach the food from the start of the video recording
    metadata = metadata[metadata['first_food_frame'] < THRESHOLD_N_SECONDS*FPS]
    sample_sizes = metadata.groupby('gene_name').count()['first_food_frame']
    print("\nThreshold time until food encounter: {} seconds".format(THRESHOLD_N_SECONDS))
    for s in sample_sizes.index:
        print('{0}: n={1}'.format(s, sample_sizes.loc[s]))
        
    mean_delay_seconds = int(metadata['first_food_frame'].mean()) / 25
    print("Worms took %.1f seconds on average to reach food" % mean_delay_seconds)
    
    # Timeseries plots for worms that took <10 seconds to reach food
    # (then try with inculding 'hump' <75 seconds, see if it makes a difference?)
    
    grouped_strain = metadata.groupby('gene_name')

    colours = sns.color_palette(palette="tab10", n_colors=len(metadata['gene_name'].unique()))
    bluelight_frames = [(i*60*FPS, i*60*FPS+10*FPS) for i in BLUELIGHT_TIMEPOINTS_MINUTES]

    # both strains together, for each motion mode
    for mode in ['forwards','backwards','stationary']:
    
        plt.close('all')
        fig, ax = plt.subplots(figsize=(15,5))
        save_path = Path(args.save_dir) / 'timeseries_plots' / 'motion_mode_{}.pdf'.format(mode)
        
        for s, strain in enumerate(metadata['gene_name'].unique()):
            print("Plotting motion mode %s timeseries for %s..." % (mode, strain))
            
            strain_meta = grouped_strain.get_group(strain)
            
            strain_timeseries_list = []
            for i in strain_meta.index:
                imgstore = strain_meta.loc[i, 'imgstore_name']
                
                df = read_timeseries(Path(args.project_dir) / "Results" / imgstore /\
                                     'metadata_featuresN.hdf5',
                                     names=['worm_index','timestamp','motion_mode'],
                                     only_wells=None)
                strain_timeseries_list.append(df)
                
            # compile timeseries data for strain 
            strain_timeseries = pd.concat(strain_timeseries_list, axis=0)
                    
            ax = plot_timeseries_motion_mode(df=strain_timeseries,
                                             window=SMOOTH_WINDOW_SECONDS*FPS,
                                             error=True,
                                             mode=mode,
                                             max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
                                             title="Timeseries motion mode %s (total n=%d worms)" %\
                                                 (mode, metadata.shape[0]),
                                             #figsize=(15,5), saveAs=save_path,
                                             ax=ax, #ax=None
                                             bluelight_frames=bluelight_frames,
                                             cols=['timestamp', 'motion_mode'],
                                             colour=colours[s],
                                             alpha=0.75)
            
        ax.axvspan(mean_delay_seconds*FPS-FPS, mean_delay_seconds*FPS+FPS, facecolor='red', alpha=1)
        xticks = np.linspace(0,VIDEO_LENGTH_SECONDS*FPS, 31)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
        ax.set_xlabel('Time (minutes)', fontsize=15, labelpad=10)
        ax.set_ylabel('Motion mode {}'.format(mode), fontsize=15, labelpad=10)
        ax.legend(metadata['gene_name'].unique(), fontsize=12, frameon=False, loc='best')
        
        plt.savefig(save_path, dpi=300)
        plt.close('all')
        
    
    # TODO: process_feature_summaries
        
