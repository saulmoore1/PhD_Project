#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checking for systematic loss of tracked worm trajectories throughout Keio Screen videos
(particularly towards the end of the videos), as an indicator of lawn-leaving phenotype. 
I want to confirm that fepD bacteria do not elicit increased lawn-leaving phenotype and that this
is not the explanation for decreased motion mode paused on this mutant food.

Steps (for each screen):
    1. load metadata and get features filepath from imgstore_name column
    2. groupby(gene_name) => yields metadata for the videos/wells of each gene
    3. read those videos timeseries for that well, and collate timeseries df
    4. run n_worms_per_frame on each well's timeseries, and collate results df

@author: sm5911
@date: 21/02/2022

"""

#%% Imports

import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from time import time
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt

from tierpsytools.analysis.count_worms import n_worms_per_frame, _fraction_of_time_with_n_worms
from tierpsytools.read_data.get_timeseries import get_timeseries

#%% Globals
PROJECT_DIR = "/Volumes/hermes$/KeioScreen_96WP"
METADATA_PATH = "/Users/sm5911/Documents/Keio_Screen/metadata.csv"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Screen"

# PROJECT_DIR = "/Volumes/hermes$/KeioScreen2_96WP"
# METADATA_PATH = "/Users/sm5911/Documents/Keio_Conf_Screen/metadata.csv"
# SAVE_DIR = "/Users/sm5911/Documents/Keio_Conf_Screen"

# PROJECT_DIR = "/Volumes/hermes$/Keio_Rescue_96WP"
# METADATA_PATH = "/Users/sm5911/Documents/Keio_Rescue/metadata.csv"
# SAVE_DIR = "/Users/sm5911/Documents/Keio_Rescue"

STRAIN_LIST = ['wild_type', 'fepD']

MAX_N_VIDEOS = 10 # cap at first 10 videos (for speed, especially for control data)

SMOOTHING = 100 # None; window size for moving average to smooth n worms timeseries scatterplot

#%% Functions

#%% Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provide project root directory and directory to \
    save plots")
    parser.add_argument('--project_dir', help="Path to root directory of project to analyse", 
                        type=str, default=PROJECT_DIR)
    parser.add_argument('--metadata_path', help="Path to compiled project metadata", 
                        type=str, default=METADATA_PATH)
    parser.add_argument('--save_dir', help="Path to save plots", type=str, default=SAVE_DIR)    
    args = parser.parse_args()
    
    tic = time()
    
    if not args.metadata_path:
        args.metadata_path = Path(args.project_dir) / "AuxiliaryFiles" / "metadata.csv"
    
    # load metadata + store featuresN filepath info
    metadata = pd.read_csv(args.metadata_path, dtype={"comments":str, "source_plate_id":str})

    assert 'featuresN_filename' in metadata.columns # filename is for 'prestim' only by default (when BL aligned)
    
    grouped = metadata.groupby('gene_name')
    
    # get metadata info for each strain in turn
    for strain in STRAIN_LIST:
        strain_meta = grouped.get_group(strain)
        print("\nN=%d prestim featuresN files found for %s" % (strain_meta.shape[0], strain))
        
        # compile timeseries data for strain
        n_worms_list = []
        frac_time_list = []
        for i, idx in enumerate(tqdm(strain_meta.index[:MAX_N_VIDEOS])):
            featfile = strain_meta.loc[idx, 'featuresN_filename']
            well_name = strain_meta.loc[idx, 'well_name']
            
            # read video time-series
            ts = get_timeseries(root_dir=Path(featfile).parent,
                                names=None,
                                only_wells=[well_name])[1][0]
            
            # compute number of worms per frame
            n_worms = pd.DataFrame(n_worms_per_frame(ts['timestamp']))
            n_worms['videoID'] = i          
            n_worms_list.append(n_worms)
            
            frac_worms = pd.DataFrame(_fraction_of_time_with_n_worms(
                n_worms_per_frame(ts['timestamp']), max_n=3))
            frac_worms['videoID'] = i
            frac_time_list.append(frac_worms)
            
        n_worms_strain = pd.concat(n_worms_list).reset_index(drop=False)        
        n_worms_strain_avg = n_worms_strain.groupby('timestamp').mean()['n_worms']
        
        # TODO: Plot fraction of time with n worms tracked
        _frac_worms_strain = pd.concat(frac_time_list).reset_index(drop=False)

        # smooth timeseries scatter plot by averaging across a moving time window (optional)
        if SMOOTHING:
            n_worms_strain_avg = n_worms_strain_avg.rolling(window=SMOOTHING).mean()
        
        # plot average number of tracked worms in each frame
        plt.close('all')
        fig, ax = plt.subplots(figsize=(12,6))
        sns.lineplot(x=n_worms_strain_avg.index, y=n_worms_strain_avg.values, ax=ax)
        ax.set_xlim(0, max(n_worms_strain_avg.index))
        ax.set_ylim(0, 4)
        fig.canvas.draw()
        xticks = [t.get_text() for t in ax.get_xticklabels()]
        ax.set_xticks(np.array([0,1,2,3,4,5])*25*60) # values
        ax.set_yticks([0,1,2,3,4])
        ax.set_xticklabels([int(int(t)/25/60) for t in np.array([0,1,2,3,4,5])*25*60])
        ax.set_xlabel("Time (minutes)", fontsize=15, labelpad=10)
        ax.set_ylabel("Mean number of worms", fontsize=15, labelpad=10)
        title = strain + (" (N videos=%d" % min(strain_meta.shape[0], MAX_N_VIDEOS) +
                          ", smoothing window=%d seconds)" % int(SMOOTHING/25) if SMOOTHING else ")")
        ax.set_title(title, loc='right', fontsize=15, pad=10)
        
        # sava timeseries plot
        save_path = Path(args.save_dir) / 'worm_tracking_checks' / '{}_n_worms_timeseries.pdf'.format(strain)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300)
        
    # TODO: investigate worm trajectory length - barplot of frequency of trajectories of varying lengths (time, s)
        
    toc = time()      
    print("Done in %.2f seconds" % (toc - tic))
        
    
