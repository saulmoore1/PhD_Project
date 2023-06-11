#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1g - timeseries of fepD vs BW

@author: sm5911
@date: 21/05/2023

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
SAVE_DIR = "/Users/sm5911/OneDrive - Imperial College London/Publications/Keio_Paper/Figures/Fig1"

BLUELIGHT_TIMEPOINTS_SECONDS = [(60, 70),(160, 170),(260, 270)]
N_WELLS = 6
FPS = 25

#%% Functions

def main():
    
    metadata_path = Path(PROJECT_DIR) / 'metadata.csv'    
    metadata = pd.read_csv(metadata_path, header=0, index_col=None, dtype={'comments':str})
    
    # subset metadata results for bluelight videos only 
    bluelight_videos = [i for i in metadata['imgstore_name'] if 'bluelight' in i]
    metadata = metadata[metadata['imgstore_name'].isin(bluelight_videos)]
    
    # subset metadata for window 5 (20-30 seconds after blue light pulse 3)
    metadata = metadata[metadata['window']==5]
    
    # subset for live results only
    metadata = metadata.query("is_dead=='N'")
    metadata = metadata[~np.logical_and(metadata['drug_type']=='Paraquat',
                                        metadata['drug_imaging_plate_conc']==0.5)]
    metadata = metadata[~np.logical_and(metadata['drug_type']!='Paraquat',
                                        metadata['drug_imaging_plate_conc']==1)]
   
    treatment_cols = ['food_type','drug_type']
    treatments_to_drop = ['BW-NAC','BW-Vitamin C','fepD-Paraquat']
    metadata['treatment'] = metadata[treatment_cols].astype(str).agg('-'.join, axis=1)
    metadata['treatment'] = [i.replace('-nan','') for i in metadata['treatment']]
    metadata = metadata[~metadata['treatment'].isin(treatments_to_drop)]
    
    # timeseries   
    
    bluelight_frames = [(i*FPS, j*FPS) for (i, j) in (BLUELIGHT_TIMEPOINTS_SECONDS)]
    feature = 'speed'
    ts_dir = Path(PROJECT_DIR) / 'timeseries-speed'
    
    # timeseries fepD vs BW

    plt.close('all')
    fig, ax = plt.subplots(figsize=(15,6), dpi=300)
    col_dict = dict(zip(['BW','fepD'], sns.color_palette('tab10', 2)))

    for ts_file in ['BW-nan-nan-N_timeseries.csv','fepD-nan-nan-N_timeseries.csv']:

        ts = pd.read_csv(ts_dir / ts_file)
    
        # crop timeseries from -40 to +100 seconds (around start of final bluelight pulse)
        xmin = BLUELIGHT_TIMEPOINTS_SECONDS[-1][0] - 40
        xmax = BLUELIGHT_TIMEPOINTS_SECONDS[-1][0] + 100
        ts = ts[np.logical_and(ts['timestamp']>=xmin, ts['timestamp']<=xmax)]
        
        ax = plot_timeseries(df=ts,
                             feature=feature,
                             error=True,
                             max_n_frames=None, 
                             smoothing=10*FPS, 
                             ax=ax,
                             bluelight_frames=[bluelight_frames[-1]],
                             colour=col_dict[ts_file.split('-')[0]])

    plt.ylim(-20, 300)
    xticks = np.linspace(xmin*FPS, xmax*FPS, int((xmax-xmin)/60)+1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
    ax.set_xlabel('Time (minutes)', fontsize=20, labelpad=10)
    ylab = feature.replace('_50th'," (µm s$^{-1}$)")
    ax.set_ylabel(ylab, fontsize=20, labelpad=10)
    ax.legend(['BW','fepD'], fontsize=12, frameon=False, loc='best', handletextpad=1)
    plt.subplots_adjust(left=0.1, top=0.98, bottom=0.15, right=0.98)
    
    # save plot
    save_path = Path(SAVE_DIR) / 'speed_final_bluelight_fepD_vs_BW.pdf'
    print("Saving to: %s" % save_path)
    plt.savefig(save_path)


    for treatment in ['fepD','BW-Paraquat']:
        treatment_list = ['BW', treatment]
        
        for group in tqdm(treatment_list):
            
            # get control timeseries
            group_ts = get_strain_timeseries(metadata,
                                             project_dir=Path(PROJECT_DIR),
                                             strain=group,
                                             group_by='treatment',
                                             feature_list=[feature],
                                             save_dir=save_dir,
                                             n_wells=N_WELLS,
                                             verbose=True)
            
            ax = plot_timeseries(df=group_ts,
                                 feature=feature,
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
        ax.set_xlabel('Time (minutes)', fontsize=20, labelpad=10)
        ylab = feature.replace('_50th'," (µm s$^{-1}$)")
        ax.set_ylabel(ylab, fontsize=20, labelpad=10)
        ax.legend(treatment_list, fontsize=12, frameon=False, loc='best', handletextpad=1)
        plt.subplots_adjust(left=0.1, top=0.98, bottom=0.15, right=0.98)
        
        # save plot
        print("Saving to: %s" % save_path)
        plt.savefig(save_path)

    # timeseries vs fepD -- ['fepD-NAC', 'fepD-Vitamin C']

    for treatment in ['fepD-NAC', 'fepD-Vitamin C']:
        treatment_list = ['fepD', treatment]

    return

#%% Main

if __name__ == '__main__':
    main()
    
