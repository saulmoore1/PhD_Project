#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2b - Timeseries of BW / fepD / BW+paraquat

@author: sm5911
@date: 10/06/2023

"""

#%% Imports 

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt

from time_series.plot_timeseries import plot_timeseries, get_strain_timeseries

#%% Globals

PROJECT_DIR = "/Users/sm5911/Documents/Keio_UV_Paraquat_Antioxidant"
SAVE_DIR = "/Users/sm5911/OneDrive - Imperial College London/Publications/Keio_Paper/Figures/Fig2"

FEATURE = 'speed'

BLUELIGHT_TIMEPOINTS_SECONDS = [(60, 70),(160, 170),(260, 270)]
WINDOW_NAME_DICT = {5:"20-30 seconds after blue light 3"}
WINDOW_DICT = {5:(290,300)}

N_WELLS = 6
FPS = 25
DPI = 300

#%% Functions

def main():
    
    # load clean metadata
    metadata_path = Path(PROJECT_DIR) / 'metadata.csv'
    
    metadata = pd.read_csv(metadata_path, header=0, index_col=None, dtype={'comments':str})
    
    # subset metadata results for bluelight videos only 
    bluelight_videos = [i for i in metadata['imgstore_name'] if 'bluelight' in i]
    metadata = metadata[metadata['imgstore_name'].isin(bluelight_videos)]
    
    # subset metadata for window 5 (20-30 seconds after blue light pulse 3)
    metadata = metadata[metadata['window']==5]
    
    # subset to remove antioxidant results
    metadata = metadata[~metadata['drug_type'].isin(['NAC','Vitamin C'])]

    # subset for results for live cultures only
    metadata = metadata.query("is_dead=='N'")
    
    # subset for 1mM paraquat results only
    metadata = metadata[metadata['drug_imaging_plate_conc']!=0.5]
    
    treatment_cols = ['food_type','drug_type']
    metadata['treatment'] = metadata[treatment_cols].astype(str).agg('-'.join, axis=1)
    metadata['treatment'] = [i.replace('-nan','') for i in metadata['treatment']]

    # drop fepD+paraquat results
    metadata = metadata[metadata['treatment']!='fepD-Paraquat']
    
    treatment_list = list(metadata['treatment'].unique())

    # timeseries speed: BW, fepD, BW+paraquat
         
    bluelight_frames = [(i*FPS, j*FPS) for (i, j) in BLUELIGHT_TIMEPOINTS_SECONDS]        
    col_dict = dict(zip(treatment_list, sns.color_palette('tab10', n_colors=len(treatment_list))))

    plt.close('all')
    fig, ax = plt.subplots(figsize=(15,6), dpi=DPI)

    for group in tqdm(treatment_list):
        
        # load timeseries data
        group_ts = get_strain_timeseries(metadata,
                                         project_dir=Path(PROJECT_DIR),
                                         strain=group,
                                         group_by='treatment',
                                         feature_list=[FEATURE],
                                         save_dir=Path(PROJECT_DIR) / 'timeseries-speed',
                                         n_wells=N_WELLS,
                                         verbose=True)
        
        # plot timeseries for each treatment (group)
        ax = plot_timeseries(df=group_ts,
                             feature=FEATURE,
                             error=True,
                             max_n_frames=360*FPS, 
                             smoothing=10*FPS, 
                             ax=ax,
                             bluelight_frames=bluelight_frames,
                             colour=col_dict[group])

    plt.ylim(-20, 300)
    xticks = np.linspace(0, 360*FPS, int(360/60)+1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])
    ax.set_xlabel('Time (minutes)', fontsize=25, labelpad=10)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.set_ylabel("Speed (µm s$^{-1}$)", fontsize=25, labelpad=15)
    ax.yaxis.set_tick_params(labelsize=20)
    leg = ax.legend(treatment_list, fontsize=20, frameon=False, 
                    loc='upper right', handletextpad=0.75)
    # change the line width for the legend
    for l in leg.get_lines():
        l.set_linewidth(3)
    plt.subplots_adjust(left=0.1, top=0.98, bottom=0.15, right=0.98)

    # save plot
    save_path = Path(SAVE_DIR) / 'Fig2b.pdf'
    print("Saving to: %s" % save_path)
    plt.savefig(save_path)

    return

#%% Main

if __name__ == '__main__':
    main()
    
