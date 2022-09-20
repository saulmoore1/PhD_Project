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
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from preprocessing.compile_hydra_data import compile_metadata
from time_series.time_series_helper import get_strain_timeseries
from time_series.plot_timeseries import plot_timeseries_motion_mode

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

motion_modes = ['forwards','stationary','backwards']

#%% Functions

def acute_single_worm_timeseries(metadata, 
                                 project_dir, 
                                 save_dir, 
                                 group_by='bacteria_strain',
                                 control='BW',
                                 bluelight_windows_separately=False,
                                 n_wells=N_WELLS,
                                 smoothing=120):
    """ Timeseries plots of repeated bluelight stimulation of BW and fepD
        (10 seconds BL delivered every 30 minutes, for 5 hours total)
    """
        
    # get timeseries for BW
    BW_ts = get_strain_timeseries(metadata[metadata[group_by]==control], 
                                  project_dir=project_dir, 
                                  strain=control,
                                  group_by=group_by,
                                  n_wells=n_wells,
                                  save_dir=save_dir / 'data',
                                  verbose=False)
    
    # get timeseries for fepD
    fepD_ts = get_strain_timeseries(metadata[metadata[group_by]=='fepD'], 
                                    project_dir=project_dir, 
                                    strain='fepD',
                                    group_by=group_by,
                                    n_wells=n_wells,
                                    save_dir=save_dir / 'data',
                                    verbose=False)
 
    colour_dict = dict(zip([control, 'fepD'], sns.color_palette("pastel", 2)))
    bluelight_frames = [(i*60*FPS, i*60*FPS+10*FPS) for i in BLUELIGHT_TIMEPOINTS_MINUTES]

    for mode in motion_modes:
        print("Plotting timeseries '%s' fraction for BW vs fepD..." % mode)

        if bluelight_windows_separately:
            
            for pulse, timepoint in enumerate(tqdm(BLUELIGHT_TIMEPOINTS_MINUTES), start=1):

                plt.close('all')
                fig, ax = plt.subplots(figsize=(15,5), dpi=150)
        
                ax = plot_timeseries_motion_mode(df=BW_ts,
                                                 window=smoothing*FPS,
                                                 error=True,
                                                 mode=mode,
                                                 max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
                                                 title=None,
                                                 saveAs=None,
                                                 ax=ax,
                                                 bluelight_frames=bluelight_frames,
                                                 colour=colour_dict[control],
                                                 alpha=0.25)
                
                ax = plot_timeseries_motion_mode(df=fepD_ts,
                                                 window=smoothing*FPS,
                                                 error=True,
                                                 mode=mode,
                                                 max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
                                                 title=None,
                                                 saveAs=None,
                                                 ax=ax,
                                                 bluelight_frames=bluelight_frames,
                                                 colour=colour_dict['fepD'],
                                                 alpha=0.25)
            
                xticks = np.linspace(0, VIDEO_LENGTH_SECONDS*FPS, int(VIDEO_LENGTH_SECONDS/60)+1)
                ax.set_xticks(xticks)
                ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
                
                # -30secs before to +2mins after each pulse
                xlim_range = (timepoint*60-30, timepoint*60+120)
                ax.set_xlim([xlim_range[0]*FPS, xlim_range[1]*FPS])

                ax.set_xlabel('Time (minutes)', fontsize=12, labelpad=10)
                ax.set_ylabel('Fraction {}'.format(mode), fontsize=12, labelpad=10)
                ax.set_title('BW vs fepD (bluelight pulse {0} = {1} min)'.format(pulse,
                    timepoint), fontsize=12, pad=10)
                ax.legend(['BW', 'fepD'], fontsize=12, frameon=False, loc='upper right')
        
                # save plot
                save_path = save_dir /\
                    'motion_mode_{0}_bluelight_pulse{1}_{2}min.pdf'.format(mode, pulse, timepoint)
                print("Saving to: %s" % save_path)
                plt.savefig(save_path)  
                
        else:    
            plt.close('all')
            fig, ax = plt.subplots(figsize=(30,5), dpi=150)
    
            ax = plot_timeseries_motion_mode(df=BW_ts,
                                             window=smoothing*FPS,
                                             error=True,
                                             mode=mode,
                                             max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
                                             title=None,
                                             saveAs=None,
                                             ax=ax,
                                             bluelight_frames=bluelight_frames,
                                             colour=colour_dict[control],
                                             alpha=0.25)
            
            ax = plot_timeseries_motion_mode(df=fepD_ts,
                                             window=smoothing*FPS,
                                             error=True,
                                             mode=mode,
                                             max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
                                             title=None,
                                             saveAs=None,
                                             ax=ax,
                                             bluelight_frames=bluelight_frames,
                                             colour=colour_dict['fepD'],
                                             alpha=0.25)
        
            xticks = np.linspace(0, VIDEO_LENGTH_SECONDS*FPS, int(VIDEO_LENGTH_SECONDS/60)+1)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
            ax.set_xlabel('Time (minutes)', fontsize=12, labelpad=10)
            ax.set_ylabel('Fraction {}'.format(mode), fontsize=12, labelpad=10)
            ax.set_title('BW vs fepD', fontsize=12, pad=10)
            ax.legend(['BW', 'fepD'], fontsize=12, frameon=False, loc='upper right')
    
            # save plot
            save_path = save_dir / 'motion_mode_{}.pdf'.format(mode)
            print("Saving to: %s" % save_path)
            plt.savefig(save_path)  

    return

    
#%% Main

if __name__ == '__main__':

    AUX_DIR = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    RES_DIR = Path(PROJECT_DIR) / 'Results'

    META_PATH = Path(SAVE_DIR) / 'metadata.csv'
    
    if Path(META_PATH).exists():    
        # load metadata
        metadata = pd.read_csv(META_PATH, index_col=None)
    else:
        # compile metadata
        metadata, _ = compile_metadata(aux_dir=AUX_DIR,
                                       imaging_dates=IMAGING_DATES,
                                       add_well_annotations=N_WELLS==96,
                                       n_wells=N_WELLS,
                                       from_source_plate=False)            
    
    save_dir = Path(SAVE_DIR) / 'timeseries'
    
    # TODO: omit data for Hydra05 to see if this fixes the bug due to timestamps lagging on some 
    #       LoopBio videos: metadata = metadata[metadata['instrument_name'] != 'Hydra05']
       
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
        
    strain_list = sorted(list(metadata['bacteria_strain'].unique()))
    colours = sns.color_palette(palette="tab10", n_colors=len(strain_list))
    bluelight_frames = [(i*60*FPS, i*60*FPS+10*FPS) for i in BLUELIGHT_TIMEPOINTS_MINUTES]

    # full timeseries plots - BW vs fepD for each motion mode
    acute_single_worm_timeseries(metadata, 
                                 project_dir=None, 
                                 save_dir=save_dir,
                                 n_wells=N_WELLS,
                                 control='BW',
                                 group_by='bacteria_strain',
                                 bluelight_windows_separately=False,
                                 smoothing=10)

    # timeseries plots BW vs fepD around each blue light window
    acute_single_worm_timeseries(metadata, 
                                 project_dir=None, 
                                 save_dir=save_dir,
                                 n_wells=N_WELLS,
                                 control='BW',
                                 group_by='bacteria_strain',
                                 bluelight_windows_separately=True,
                                 smoothing=5)
        
# =============================================================================
#     plot_dir = save_dir / 'motion_mode_plots_max_delay={}s'.format(THRESHOLD_N_SECONDS)
#     plot_dir.mkdir(exist_ok=True)
# 
#     # Full timeseries plots - both strains together, for each motion mode
#     for mode in motion_modes:
#         print("Plotting motion mode %s fraction timeseries" % mode)
# 
#         plt.close('all')
#         fig, ax = plt.subplots(figsize=(15,5))
#         
#         for s, strain in enumerate(strain_list):
# 
#             strain_timeseries = get_strain_timeseries(metadata,
#                                                       project_dir=PROJECT_DIR, 
#                                                       strain=strain,
#                                                       n_wells=N_WELLS,
#                                                       group_by='bacteria_strain',
#                                                       save_dir=save_dir / 'data')
#             
#             ax = plot_timeseries_motion_mode(df=strain_timeseries,
#                                              window=SMOOTH_WINDOW_SECONDS*FPS, 
#                                              error=True, 
#                                              mode=mode, 
#                                              max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
#                                              title=None, 
#                                              figsize=(12,6), 
#                                              ax=ax, 
#                                              saveAs=None,
#                                              sns_colour_palette='pastel', 
#                                              colour=colours[s], 
#                                              bluelight_frames=bluelight_frames,
#                                              cols = ['motion_mode','filename','well_name','timestamp'], 
#                                              alpha=0.5)
#             
#         ax.axvspan(mean_delay_seconds*FPS-FPS, mean_delay_seconds*FPS+FPS, facecolor='k', alpha=1)
#         ax.axvspan(THRESHOLD_N_SECONDS*FPS-FPS, THRESHOLD_N_SECONDS*FPS+FPS, facecolor='r', alpha=1)
#         xticks = np.linspace(0,VIDEO_LENGTH_SECONDS*FPS, 31)
#         ax.set_xticks(xticks)
#         ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
#         ax.set_xlabel('Time (minutes)', fontsize=15, labelpad=10)
#         ax.set_ylabel('Fraction {}'.format(mode), fontsize=15, labelpad=10)
#         ax.legend(strain_list, fontsize=12, frameon=False, loc='best')
#         ax.set_title("motion mode fraction '%s' (total n=%d worms)" % (mode, metadata.shape[0]),
#                      fontsize=15, pad=10)
#         # save plot
#         plt.savefig(plot_dir / '{}.png'.format(mode), dpi=300)  
#         plt.close()
# =============================================================================
    

