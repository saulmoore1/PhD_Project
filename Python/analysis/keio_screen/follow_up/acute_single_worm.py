#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse results of acute single worm experiments, where a single worms were picked onto 35mm plates,
seeded with either E. coli BW25113 or BW25113ΔfepD bacteria, and tracked as soon as the worm
approached the bacterial lawn 

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

from preprocessing.compile_hydra_data import process_metadata, process_feature_summaries

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Keio_Acute_Single_Worm"
IMAGING_DATES = ['20220206', '20220209']

FPS = 25
VIDEO_LENGTH_FRAMES = 30*60*FPS

THRESHOLD_N_SECONDS = 10

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
    bins = [int(b) for b in np.linspace(0, VIDEO_LENGTH_FRAMES, int(30*60/10+1))]
    first_food_frame = metadata[['first_food_frame']].copy()
    first_food_frame['first_food_binned_freq'] = pd.cut(x=first_food_frame['first_food_frame'], bins=bins)
    first_food_freq = first_food_frame.groupby(first_food_frame['first_food_binned_freq'], 
                                               as_index=False).count()

    # plot histogram of binned frequency of first food encounter 
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12,6))    
    sns.barplot(x=first_food_freq['first_food_binned_freq'].astype(str), 
                y=first_food_freq['first_food_frame'], alpha=0.8)        
    ax.set_xticks([x - 0.5 for x in ax.get_xticks()])
    ax.set_xticklabels([str(int(b / FPS)) for b in bins], rotation=45)
    ax.set_xlim(0, np.where(bins > first_food_frame['first_food_frame'].max())[0][0])
    ax.set_xlabel("Time until first food encounter (seconds)", fontsize=15, labelpad=10)
    ax.set_ylabel("Number of videos", fontsize=15, labelpad=10)
    plt.tight_layout()
    
    # save histogram
    save_path = Path(args.save_dir) / "first_food_encounter.pdf"
    save_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=300)

    # Threshold to remove all video entries where the worm took longer than 1 second (25 frames)
    # to reach the food from the start of the video recording
    (first_food_frame['first_food_frame'] < THRESHOLD_N_SECONDS*FPS).sum()
    
