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

# PROJECT_DIR = "/Volumes/hermes$/KeioScreen_96WP"
# METADATA_PATH = "/Users/sm5911/Documents/Keio_Screen/metadata.csv"
# SAVE_DIR = "/Users/sm5911/Documents/Keio_Screen"

PROJECT_DIR = "/Volumes/hermes$/KeioScreen2_96WP"
METADATA_PATH = "/Users/sm5911/Documents/Keio_Conf_Screen/metadata.csv"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Conf_Screen"

STRAIN_LIST = None #['wild_type', 'fepD', 'fes', 'fepB', 'fepG', 'sdhD', 'nuoC', 'entA', 'atpB', 'sdhA']
STIM_TYPE = 'poststim'

MAX_N_WORMS_TRACKED = 5 # maximum number of worms to calculate fraction of time tracked for
MAX_N_VIDEOS = 10 # cap at first 10 videos (for speed, especially for control data)
FPS = 25 # video frame rate (frames per second)
SMOOTHING = 100 # None; window size for moving average to smooth n worms timeseries scatterplot

FRAC_TIME_THRESH = 0.5 # threshold fraction of time n worms tracked for (for finding aversive strains)

#%% Functions

def compile_n_worms_frac(dirpath, strain_list, mode=STIM_TYPE):
    """ Load and compile n_frac_worms_tracked CSV data for each strain in strain list provided """
    
    n_worms_frac_list = []
    for strain in tqdm(strain_list):
        save_dir = Path(dirpath) / mode / strain
        n_worms_frac_save_path = save_dir / '{0}_n_worms_frac_time_tracked_{1}.csv'.format(strain, 
                                                                                           mode)
        n_worms_frac = pd.read_csv(n_worms_frac_save_path, header=0, index_col=None)
        n_worms_frac['gene_name'] = strain
        n_worms_frac_list.append(n_worms_frac)
        
    # compile n_worms_frac dataframe
    frac_n_worms_tracked = pd.concat(n_worms_frac_list, axis=0)
    
    return frac_n_worms_tracked

#%% Main

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Provide project root directory and directory to \
    save plots")
    parser.add_argument('--project_dir', help="Path to root directory of project to analyse", 
                        type=str, default=PROJECT_DIR)
    parser.add_argument('--metadata_path', help="Path to compiled project metadata", 
                        type=str, default=METADATA_PATH)
    parser.add_argument('--save_dir', help="Path to save plots", type=str, default=SAVE_DIR)   
    parser.add_argument('--strain_list', help="List of Keio gene names to analyse", 
                        default=STRAIN_LIST, nargs='+', type=str)
    args = parser.parse_args()
    
    tic = time()
    
    if not args.metadata_path:
        args.metadata_path = Path(args.project_dir) / "AuxiliaryFiles" / "metadata.csv"
    
    # load metadata + store featuresN filepath info
    metadata = pd.read_csv(args.metadata_path, dtype={"comments":str, "source_plate_id":str})

    assert 'featuresN_filename' in metadata.columns # default filename == 'prestim' when BL aligned
    
    rawvideo_dir = Path(args.project_dir) / "RawVideos"
        
    if args.strain_list is None:
        args.strain_list = list(sorted(metadata['gene_name'].unique()))
    
    grouped = metadata.groupby('gene_name')
        
    # group by each strain in turn: compile metadata and timeseries for strain, then plot 
    # trajectory length, number and fraction of worms tracked
    for strain in tqdm(args.strain_list):
        
        strain_save_dir = Path(args.save_dir) / 'worm_tracking_checks' / STIM_TYPE / strain
        n_worms_frac_save_path = strain_save_dir /\
            '{0}_n_worms_frac_time_tracked_{1}.csv'.format(strain, STIM_TYPE)
            
        # if already processed...
        if not n_worms_frac_save_path.exists():
        
            strain_meta = grouped.get_group(strain)
            
            # drop NaN filenames
            strain_meta = strain_meta.dropna(axis=0, 
                                             how='any', 
                                             subset=['imgstore_name_{}'.format(STIM_TYPE)])
            print("\nN=%d %s featuresN files found for %s" % (strain_meta.shape[0], STIM_TYPE, strain))

            # compile timeseries data for strain
            n_worms_list = []
            frac_time_list = []
            traj_duration_list = []
            error_files = []
            for i, idx in enumerate(strain_meta.index[:MAX_N_VIDEOS]):
                # change featuresN filepath to find poststim or bluelight file for this video
                
                featfile = Path(args.project_dir) / 'Results' /\
                    strain_meta.loc[idx, 'imgstore_name_{}'.format(STIM_TYPE)] /\
                        'metadata_featuresN.hdf5'

                well_name = strain_meta.loc[idx, 'well_name']
                
                try:
                    # read video time-series
                    ts = get_timeseries(root_dir=Path(featfile).parent,
                                        names=None,
                                        only_wells=[well_name])[1][0]
                    
                    # compute number of worms per frame
                    n_worms = pd.DataFrame(n_worms_per_frame(ts['timestamp']))
                    n_worms_list.append(n_worms)
                    
                    # compute fraction of time tracking how many worms
                    frac_worms = pd.DataFrame(_fraction_of_time_with_n_worms(
                        n_worms_per_frame(ts['timestamp']), max_n=MAX_N_WORMS_TRACKED))
                    frac_time_list.append(frac_worms)
                    
                    # compute trajectory duration (seconds)
                    worm_traj_duration = ts.groupby('worm_index').count()['timestamp']
                    traj_duration_list.append(worm_traj_duration)  
                    
                except Exception as EE:
                    print("\nWARNING! Could not read file:\n%s\n%s" % (featfile, EE))
                    error_files.append(featfile)
                    
                
            ##### Trajectory duration - barplot of frequency of trajectories ranked by n frames       
            
            traj_duration = pd.concat(traj_duration_list).reset_index(drop=False)
            traj_duration = traj_duration.sort_values(by='timestamp', ascending=True)
            traj_duration = traj_duration[['timestamp']].reset_index(drop=True)
            
            # create bins
            bins = [int(b) for b in np.linspace(0, 5*60*FPS, 31)] # 5-minute videos
            #bins = np.linspace(0, np.round(traj_duration['timestamp'].max(), -2), 16)
            traj_duration['traj_binned_freq'] = pd.cut(x=traj_duration['timestamp'], bins=bins)
            traj_duration = traj_duration.dropna(axis=0, how='any') # drop NaN value timestamps > 7500
            traj_freq = traj_duration.groupby(traj_duration['traj_binned_freq'], as_index=False).count()
    
            plt.close('all')
            fig, ax = plt.subplots(figsize=(15,6))
            sns.barplot(x=traj_freq['traj_binned_freq'].astype(str), 
                        y=traj_freq['timestamp'], alpha=0.8)        
            ax.set_xticks([x - 0.5 for x in ax.get_xticks()])
            ax.set_xticklabels([str(int(b / FPS)) for b in bins], rotation=45)
            
            if max(bins) > traj_duration['timestamp'].max():
                ax.set_xlim(0, np.where(bins > traj_duration['timestamp'].max())[0][0])
                
            ax.set_xlabel("Trajectory duration (seconds)", fontsize=15, labelpad=10)
            ax.set_ylabel("Number of worm trajectories", fontsize=15, labelpad=10)
            title = strain + (" (N videos=%d)" % min(strain_meta.shape[0], MAX_N_VIDEOS))
            ax.set_title(title, loc='right', fontsize=15, pad=10)
            plt.tight_layout()
            
            # save trajectory duration histogram
            save_path = strain_save_dir / '{}_trajectory_duration.pdf'.format(strain)
            save_path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(save_path, dpi=300)
            
            
            ##### Number of worms tracked time-series
            
            n_worms_strain = pd.concat(n_worms_list).reset_index(drop=False)        
            n_worms_strain_avg = n_worms_strain.groupby('timestamp').mean()['n_worms']
    
            # smooth timeseries scatter plot by averaging across a moving time window (optional)
            if SMOOTHING:
                n_worms_strain_avg = n_worms_strain_avg.rolling(window=SMOOTHING).mean()
                title = title.replace(")", ", smoothing window=%d seconds)" % int(SMOOTHING/25))
            
            # plot average number of tracked worms in each frame
            plt.close('all')
            fig, ax = plt.subplots(figsize=(12,6))
            sns.lineplot(x=n_worms_strain_avg.index, y=n_worms_strain_avg.values, ax=ax)
            ax.set_xlim(0, max(n_worms_strain_avg.index))
            ax.set_ylim(0, 4)
            fig.canvas.draw()
            xticks = [t.get_text() for t in ax.get_xticklabels()]
            ax.set_xticks(np.array([0,1,2,3,4,5])*FPS*60) # values
            ax.set_yticks([0,1,2,3,4])
            ax.set_xticklabels([int(int(t)/FPS/60) for t in np.array([0,1,2,3,4,5])*FPS*60])
            ax.set_xlabel("Time (minutes)", fontsize=15, labelpad=10)
            ax.set_ylabel("Mean number of worms", fontsize=15, labelpad=10)
            ax.set_title(title, loc='right', fontsize=15, pad=10)
            
            # sava timeseries plot
            plt.savefig(strain_save_dir / '{}_n_worms_timeseries.pdf'.format(strain), dpi=300)
        
     
            ##### Fraction of time where n worms are tracked
            
            frac_worms_strain = pd.concat(frac_time_list).reset_index(drop=False)
            frac_worms_strain_avg = frac_worms_strain.groupby('n_worms').mean()['time_fraction']
            
            # plot fraction of time n worms tracked
            plt.close('all')
            fig, ax = plt.subplots(figsize=(8,8))
            sns.barplot(x=frac_worms_strain_avg.index, y=frac_worms_strain_avg.values, 
                        palette='rainbow')
            ax.set_xlabel("Number of worms tracked", fontsize=15, labelpad=10)
            ax.set_ylabel("Fraction of time tracked for", fontsize=15, labelpad=10)
            ax.set_title(strain + (" (N videos=%d)" % min(strain_meta.shape[0], MAX_N_VIDEOS)), 
                         loc='right', fontsize=15, pad=10)
            
            # save plot
            plt.savefig(strain_save_dir / '{}_n_worms_frac_time_tracked.pdf'.format(strain), dpi=300)
    
            # save fraction of time n worms tracked
            frac_worms_strain_avg.to_csv(n_worms_frac_save_path, header=True, index=True)    

            if len(error_files) > 0:
                with (strain_save_dir / 'error_file_log.txt').open(mode='w') as f:
                    f.write('\n'.join([str(e) for e in error_files]) + '\n')
                
                    
    # Check all Keio results for strains which cause lawn-leaving phenotypes
    # ie. the highest fraction of time with n=0 worms tracked, or the lowest fraction of time with 
    # 3 worms tracked, and save a list of aversive strains 

    frac_n_worms_tracked = compile_n_worms_frac(dirpath=Path(args.save_dir) / 'worm_tracking_checks', 
                                                strain_list=args.strain_list,
                                                mode=STIM_TYPE).reset_index(drop=True)
    
    # strains with highest ranked fraction of time with n worms tracked - aversive foods
    
    ### AVERSIVE STRAINS - 0 worms tracked

    print("\nPlotting aversive strains (0 worms)...")

    # filter strains for n worms tracked
    frac_n_worms = frac_n_worms_tracked.query("n_worms==0")
    frac_n_worms = frac_n_worms.sort_values(by='time_fraction', ascending=False)

    # save sorted list for n worms
    save_path = Path(args.save_dir) / 'worm_tracking_checks' /\
        'fraction_0_worms_tracked_{}.csv'.format(STIM_TYPE)
    frac_n_worms.to_csv(save_path, header=True, index=False)
    
    # plot strains ranked by fraction of time 0 worms were tracked for
    plt.close('all')
    fig, ax = plt.subplots(figsize=(80,4), dpi=100)
    sns.lineplot(x='gene_name', y='time_fraction', data=frac_n_worms, ax=ax)
    plt.setp(ax.get_xticklabels(), ha="center", rotation=90, fontsize=1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim(min(frac_n_worms['time_fraction'])-0.1, max(frac_n_worms['time_fraction']+0.1))
    ax.set_xlabel('')
    plt.tight_layout()
    plt.savefig(Path(args.save_dir) / 'worm_tracking_checks' /\
                'ranked fraction 0 worms tracked {}.pdf'.format(STIM_TYPE))

    # plot top100 most aversive strains (close-up view)
    top_frac = frac_n_worms[:100]
    
    plt.close('all')
    fig, ax = plt.subplots(figsize=(30,3), dpi=100)
    sns.lineplot(x='gene_name', y='time_fraction', data=top_frac, ax=ax)       
    plt.setp(ax.get_xticklabels(), ha="center", rotation=90, fontsize=8)
    plt.yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim(min(top_frac['time_fraction'])-0.1, max(top_frac['time_fraction']+0.1))
    ax.set_xlabel('')
    plt.tight_layout()
    plt.savefig(Path(args.save_dir) / 'worm_tracking_checks' /\
                'aversive strains {} (0 worms tracked top100 strains).pdf'.format(STIM_TYPE))
    

    ### AVERSIVE STRAINS - 3 worms tracked

    print("\nPlotting aversive strains (3 worms)...")

    # filter strains for n worms tracked
    frac_n_worms = frac_n_worms_tracked.query("n_worms==3")
    frac_n_worms = frac_n_worms.sort_values(by='time_fraction', ascending=True)

    # save sorted list for n worms
    save_path = Path(args.save_dir) / 'worm_tracking_checks' /\
        'fraction_3_worms_tracked_{}.csv'.format(STIM_TYPE)
    frac_n_worms.to_csv(save_path, header=True, index=False)
    
    # plot strains ranked by fraction of time 3 worms were NOT tracked for
    plt.close('all')
    fig, ax = plt.subplots(figsize=(80,4), dpi=100)
    sns.lineplot(x='gene_name', y='time_fraction', data=frac_n_worms, ax=ax)
    plt.setp(ax.get_xticklabels(), ha="center", rotation=90, fontsize=1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim(min(frac_n_worms['time_fraction'])-0.1, max(frac_n_worms['time_fraction']+0.1))
    ax.set_xlabel('')
    plt.tight_layout()
    plt.savefig(Path(args.save_dir) / 'worm_tracking_checks' /\
                'ranked fraction 3 worms tracked {}.pdf'.format(STIM_TYPE))

    # plot top100 most aversive strains (close-up view)
    top_frac = frac_n_worms[frac_n_worms['time_fraction'] < 0.1]
    
    plt.close('all')
    fig, ax = plt.subplots(figsize=(30,3), dpi=100)
    sns.lineplot(x='gene_name', y='time_fraction', data=top_frac, ax=ax)       
    plt.setp(ax.get_xticklabels(), ha="center", rotation=90, fontsize=1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim(min(top_frac['time_fraction'])-0.1, max(top_frac['time_fraction']+0.1))
    ax.set_xlabel('')
    plt.tight_layout()
    plt.savefig(Path(args.save_dir) / 'worm_tracking_checks' /\
                'aversive strains {} (3 worms tracked <10% of the time).pdf'.format(STIM_TYPE))


    # TODO: check to see if the 5 avoidance genes show up in keio tracking data for zero worms tracked
    
    toc = time()      
    print("Done in %.2f seconds" % (toc - tic))
            
    