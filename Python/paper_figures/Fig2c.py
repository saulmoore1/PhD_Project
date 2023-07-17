#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2d - Fast effect - Arousal-like behaviour on fepD occurs quickly, within 30 minutes on food

@author: sm5911
@date: 12/06/2023

"""

#%% Imports

import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import transforms

from visualisation.plotting_helper import sig_asterix
from tierpsytools.analysis.statistical_tests import _multitest_correct
from analysis.keio_screen.follow_up.keio_acute_effect import (
    WINDOW_LIST,
    WINDOW_NAME_DICT, 
    WINDOW_DICT, 
    BLUELIGHT_TIMEPOINTS_MINUTES,
    fast_effect_stats)

#%% Globals

PROJECT_DIR = "/Users/sm5911/Documents/Keio_Acute_Effect"
SAVE_DIR = "/Users/sm5911/OneDrive - Imperial College London/Publications/Keio_Paper/Figures/Fig2"

FEATURE = 'speed_50th'
FDR_METHOD = 'fdr_bh'
P_VALUE_THRESHOLD = 0.05
DPI = 600

STRAINS_LIST = ['BW','fepD']

#%% Functions

def main():
    
    # load clean metadata and feature summaries results
    metadata_path = Path(PROJECT_DIR) / 'metadata.csv'
    features_path = Path(PROJECT_DIR) / 'features.csv'
    
    metadata = pd.read_csv(metadata_path, header=0, index_col=None, dtype={'comments':str})
    features = pd.read_csv(features_path, header=0, index_col=None)

    # subset metadata for bluelight videos only
    bluelight_videos = [i for i in metadata['imgstore_name'] if 'bluelight' in i]
    metadata = metadata[metadata['imgstore_name'].isin(bluelight_videos)]

    # subset metadata for 'arousal' windows only (20-30s after blue light)
    metadata = metadata[metadata['window'].isin(WINDOW_LIST)]
        
    # subset metadata for BW and fepD only
    metadata = metadata[metadata['gene_name'].isin(STRAINS_LIST)]
    
    # reindex features for new metadata subset
    features = features[[FEATURE]].reindex(metadata.index)
            
    # stats - applied separately for each window, then corrected for multiple testing
        
    pvals_dict = {}
    for window in WINDOW_LIST:
        meta_window = metadata[metadata['window']==window]
        feat_window = features.reindex(meta_window.index)
        
        ttest_df = fast_effect_stats(meta_window, 
                                     feat_window, 
                                     group_by='gene_name',
                                     control='BW',
                                     save_dir=None,
                                     feature_set=None,
                                     pvalue_threshold=P_VALUE_THRESHOLD,
                                     fdr_method=FDR_METHOD)
        
        pvals_dict[window] = ttest_df.loc[FEATURE, 'pvals_fepD']
    
    # apply correction for multiple testing
    reject, corrected_pvals = _multitest_correct(pd.Series(list(pvals_dict.values())), 
                                                 multitest_method=FDR_METHOD, fdr=P_VALUE_THRESHOLD)
        
    pvals_dict = dict(zip(WINDOW_LIST, corrected_pvals))
    pvals = pd.DataFrame.from_dict(pvals_dict, orient='index', columns=['pvals'])

    # remove one outlier datapoint for the plot
    outlier_idx = list(features[features[FEATURE]<-20].index)
    assert len(outlier_idx)==1
    features = features.drop(axis=0, index=outlier_idx)
    metadata = metadata.reindex(features.index)
    
    # boxplots
    plot_df = metadata.join(features)
    
    plt.close('all')
    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=(18,9))
    sns.boxplot(data=plot_df,
                x='window',
                y='speed_50th',
                hue='gene_name',
                order=WINDOW_LIST,
                hue_order=['BW','fepD'],
                dodge=True,
                # colour=colours,
                # palette=colour_dict,
                showfliers=False, 
                showmeans=False,
                meanprops={"marker":"x", 
                           "markersize":5,
                           "markeredgecolor":"k"},
                flierprops={"marker":"x", 
                            "markersize":15, 
                            "markeredgecolor":"r"},
                width=0.8)
    sns.stripplot(data=plot_df,
                  x='window',
                  y='speed_50th',
                  hue='gene_name',
                  order=WINDOW_LIST,
                  hue_order=['BW','fepD'],
                  dodge=True,
                  s=10,
                  palette=[sns.color_palette('Greys',2)[1]],
                  color=None,
                  marker=".",
                  edgecolor='k',
                  linewidth=0.2)

    # add pvalues to plot
    # y_offset_label = 35
    fontsize_label = 25
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled

    # metadata_grouped = metadata.groupby('window')
    for i, window in enumerate(WINDOW_LIST, start=0):
        text = ax.get_xticklabels()[i]
        assert text.get_text() == str(window)
        
        # meta_window = metadata_grouped.get_group(window)
        # meta_fep_window = meta_window[meta_window['gene_name']=='fepD']
        # feat_window = features.reindex(meta_fep_window.index)
        # y_pos = feat_window[FEATURE].max() + y_offset_label

        p = pvals.loc[window,'pvals']
        p_text = sig_asterix([p],ns=True)[0]
        ax.text(i+0.19, 1, p_text, fontsize=fontsize_label, ha='center', va='bottom', 
                transform=trans)
            
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[:2], labels[:2], fontsize=fontsize_label, loc='upper right',
    #           frameon=False, handletextpad=0.75)
    ax.get_legend().remove()

    ax.set_xlabel('Time (minutes)', fontsize=30, labelpad=30)
    
    # rename x-axis tick labels with time in minutes
    time_dict = dict(zip(WINDOW_LIST, BLUELIGHT_TIMEPOINTS_MINUTES))
    ax.axes.set_xticklabels([time_dict[int(l.get_text())] for l in ax.axes.get_xticklabels()], 
                            fontsize=25, rotation=0, ha='center')
    ax.tick_params(axis='x', which='major', pad=10)
    ax.axes.set_ylabel('Speed (µm s$^{-1}$)', fontsize=30, labelpad=30)                             
    plt.yticks(fontsize=25)
    plt.ylim(-20, 300)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    linewidth = 2
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.xaxis.set_tick_params(width=linewidth, length=linewidth*3)
    ax.yaxis.set_tick_params(width=linewidth, length=linewidth*2)
        
    # scale x axis for annotations    
    # trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled
        
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.15, top=0.9)
    plt.savefig(Path(SAVE_DIR) / 'Fig2c.png', dpi=DPI)  

    return

 
#%% Main

if __name__ == '__main__':
    main()

