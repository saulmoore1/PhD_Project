#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: sm5911
"""

#%% Imports

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
from time_series.plot_timeseries import plot_timeseries
from time_series.time_series_helper import get_strain_timeseries

#%% Globals

SAVE_DIR = "/Users/sm5911/Documents/PhD_DLBG/Fig1d"
RENAME_DICT = {"BW" : "wild_type"}
BLUELIGHT_TIMEPOINTS_SECONDS = [(60,70),(160,170),(260,270)]

#%% Functions

def plot_timeseries_feature(metadata,
                            save_dir,
                            feature='speed',
                            group_by='gene_name',
                            control='wild_type',
                            groups_list=None,
                            n_wells=96,
                            bluelight_stim_type='bluelight',
                            bluelight_timepoints_seconds=[(60,70),(160,170),(260,270)],
                            video_length_seconds=360,
                            smoothing=10,
                            fps=25,
                            ylim_minmax=None,
                            palette='tab10',
                            col_dict=None):
        
    if groups_list is not None:
        assert isinstance(groups_list, list) 
        assert all(g in metadata[group_by].unique() for g in groups_list)
    else:
        groups_list = sorted(metadata[group_by].unique())
    groups_list = [g for g in groups_list if g != control]
    assert control in metadata[group_by].unique()
    
    if bluelight_stim_type is not None and 'window' not in metadata.columns:
        metadata['imgstore_name'] = metadata['imgstore_name_{}'.format(bluelight_stim_type)]
        
    if 'window' in metadata.columns:
        assert bluelight_stim_type is not None
        stimtype_videos = [i for i in metadata['imgstore_name'] if bluelight_stim_type in i]
        metadata = metadata[metadata['imgstore_name'].isin(stimtype_videos)]
    
    if bluelight_timepoints_seconds is not None:
        bluelight_frames = [(i*fps, j*fps) for (i, j) in bluelight_timepoints_seconds]
    
    # get control timeseries
    control_ts = get_strain_timeseries(metadata,
                                       strain=control,
                                       group_by=group_by,
                                       feature_list=[feature],
                                       save_dir=save_dir,
                                       n_wells=n_wells,
                                       verbose=True)

    for group in groups_list:
        save_path = save_dir / 'Fig1d_{0}_{1}_{2}.pdf'.format(group, feature, 
                                                              bluelight_stim_type)
        
        if not save_path.exists():
            group_ts = get_strain_timeseries(metadata,
                                             strain=group,
                                             group_by=group_by,
                                             feature_list=[feature],
                                             save_dir=save_dir,
                                             n_wells=n_wells,
                                             verbose=True)
            
            print("Plotting '%s' timeseries for %s vs %s" % (feature, group, control))
            if col_dict is None:
                col_dict = dict(zip([control, group], sns.color_palette(palette, 2)))
            
            plt.close('all')
            fig, ax = plt.subplots(figsize=(15,6), dpi=300)
            ax = plot_timeseries(df=control_ts,
                                 feature=feature,
                                 error=True, 
                                 max_n_frames=video_length_seconds*fps, 
                                 smoothing=smoothing*fps, 
                                 ax=ax,
                                 bluelight_frames=(bluelight_frames if 
                                                   bluelight_stim_type == 'bluelight' else None),
                                 colour=col_dict[control])
            
            ax = plot_timeseries(df=group_ts,
                                 feature=feature,
                                 error=True, 
                                 max_n_frames=video_length_seconds*fps, 
                                 smoothing=smoothing*fps, 
                                 ax=ax,
                                 bluelight_frames=(bluelight_frames if 
                                                   bluelight_stim_type == 'bluelight' else None),
                                 colour=col_dict[group])
            
            if ylim_minmax is not None:
                assert isinstance(ylim_minmax, tuple)
                plt.ylim(ylim_minmax[0], ylim_minmax[1])
                    
            xticks = np.linspace(0, video_length_seconds*fps, int(video_length_seconds/60)+1)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(x/fps/60)) for x in xticks])   
            ax.set_xlabel('Time (minutes)', fontsize=12, labelpad=10)
            ylab = feature.replace("_", " (µm s$^{-1}$)") if feature == 'speed' else feature
            ax.set_ylabel(ylab, fontsize=12, labelpad=10)
            ax.legend([control, group], fontsize=12, frameon=False, loc='best', handletextpad=1)
            plt.subplots_adjust(left=0.1, top=0.95, bottom=0.1, right=0.95)
    
            # save plot
            print("Saving to: %s" % save_path)
            plt.savefig(save_path)

    return

def main():
    
    metadata_path_local = Path(SAVE_DIR) / 'metadata.csv'
    
    metadata = pd.read_csv(metadata_path_local, header=0, index_col=None, 
                           dtype={'comments':str})
    
    # remove entries for results with missing gene name metadata
    n_rows = metadata.shape[0]
    metadata = metadata[~metadata['gene_name'].isna()]
    if metadata.shape[0] < n_rows:
        print("Removed %d row entries with missing gene name (or empty wells)" % (
            n_rows - metadata.shape[0]))
    
    # subset metadata results for bluelight videos only 
    if not 'bluelight' in metadata.columns:
        metadata['bluelight'] = [i.split('_run')[-1].split('_')[1] for i in 
                                 metadata['imgstore_name']]
    metadata = metadata[metadata['bluelight']=='bluelight']
        
    # rename gene names in metadata
    for k, v in RENAME_DICT.items():
        metadata.loc[metadata['gene_name'] == k, 'gene_name'] = v
        
    # subset metadata for wild type and fepD
    metadata = metadata[metadata['gene_name'].isin(['wild_type','fepD'])]

    metadata['window'] = metadata['window'].astype(int)    
    metadata = metadata[metadata['window']==0]
    
    plot_timeseries_feature(metadata,
                            save_dir=Path(SAVE_DIR),
                            group_by='gene_name',
                            control='wild_type',
                            groups_list=['wild_type','fepD'],
                            feature='speed',
                            n_wells=96,
                            bluelight_stim_type='bluelight',
                            video_length_seconds=360,
                            bluelight_timepoints_seconds=BLUELIGHT_TIMEPOINTS_SECONDS,
                            smoothing=10,
                            fps=25,
                            ylim_minmax=(-20,300)) # ylim_minmax for speed feature only
   
    return

#%% Main

if __name__ == "__main__":
    main()

