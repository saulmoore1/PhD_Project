#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4e - Investigating overlap between Walhout developmental delay hits and our arousal hits

@author: sm5911
@date: 16/06/2023

"""

#%% Imports

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.stats import sem

from analysis.keio_screen.follow_up.walhout_arousal import load_walhout_244

#%% Globals

PROJECT_DIR = "/Users/sm5911/Documents/Keio_Screen_Initial"
WALHOUT_SI_PATH = "/Users/sm5911/Documents/Keio_Screen_Initial/Walhout_2019_arousal_crossref/Walhout_2019_SI_Table1.xlsx"
SAVE_DIR = "/Users/sm5911/OneDrive - Imperial College London/Publications/Keio_Paper/Figures/Fig4"

FEATURE = 'speed_50th_bluelight'
DPI = 900

#%% Functions

def errorbar_sigfeats(features, metadata, group_by, fset, control=None, rank_by='median', 
                      highlight_subset=None, max_feats2plt=10, figsize=[30,6], fontsize=4, 
                      tight_layout=None, color='dimgray', saveDir=None, saveName=None, 
                      highlight_colour='red', **kwargs):
    """ Plot mean feature value with errorbars (+/- 1.98 * std) for all groups in 
        metadata['group_by'] for each feature in feature set provided 
    """
        
    if highlight_subset is not None:
        assert all(s in metadata[group_by].unique() for s in highlight_subset)
    
    grouped = metadata[[group_by]].join(features).groupby(group_by)
    mean_strain, median_strain = grouped.mean(), grouped.median()
            
    max_feats2plt = len(fset) if max_feats2plt is None else max_feats2plt
    
    # Plot all strains (ranked by median) for top n significant features (ranked by ANOVA p-value)
    for f, feat in enumerate(tqdm(fset[:max_feats2plt])):
        # Errorbar plot
        if rank_by == 'median':
            order = median_strain[feat].sort_values(ascending=True).index.to_list()
            df_ordered = median_strain.reindex(order).reset_index(drop=False)[[group_by, feat]]  
        elif rank_by == 'mean':
            order = mean_strain[feat].sort_values(ascending=True).index.to_list()
            df_ordered = mean_strain.reindex(order).reset_index(drop=False)[[group_by, feat]]  

        df_ordered['error'] = [sem(features.loc[metadata[group_by]==strain, feat]) for strain in order]
        #error = [1.98 * features.loc[metadata[group_by]==strain, feat].std() for strain in order]

        if control is not None:
            df_ordered['colour'] = ['blue' if s == control else 'grey' for s in df_ordered[group_by]]
        else:
            df_ordered['colour'] = ['grey' for s in df_ordered[group_by]]

        if highlight_subset is not None:
            df_ordered.loc[np.where(df_ordered[group_by].isin(highlight_subset))[0],'colour'] = highlight_colour
                    
        plt.close('all')
        fig, ax = plt.subplots(figsize=figsize)
        ax.errorbar(x=group_by,
                    y=feat, 
                    yerr='error',
                    color='grey',
                    data=df_ordered, 
                    **kwargs)
        
        idxs = np.where(df_ordered['colour']!='grey')[0]
        values = df_ordered.loc[idxs,feat].values
        errors = df_ordered.loc[idxs,'error'].values
        colours = df_ordered.loc[idxs,'colour'].values
            
        for pos, y, err, colour in zip(idxs, values, errors, colours):
            ax.errorbar(pos, y, err, color=colour)
                
        _ = plt.xticks(rotation=90, ha='center', fontsize=fontsize, color=df_ordered['colour'])
        _ = [t.set_color(i) for (i,t) in zip(df_ordered['colour'], ax.xaxis.get_ticklabels())]
                
        if rank_by == 'median':
            plt.axhline(median_strain.loc[control, feat], c='dimgray', ls='--')
        elif rank_by == 'mean':
            plt.axhline(mean_strain.loc[control, feat], c='dimgray', ls='--')
            
        # ax.axes.get_xaxis().get_label().set_visible(False) # remove x axis label
        linewidth = 2
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(linewidth)
        ax.spines['bottom'].set_linewidth(linewidth)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, pad=5)
        ax.yaxis.set_tick_params(width=linewidth, length=linewidth*2)
        ax.axes.set_xticklabels([''])
        ax.axes.set_xlabel('Bacteria strains (ranked)', fontsize=30, labelpad=30)  
        ax.axes.set_ylabel('Speed (µm s$^{-1}$)', fontsize=30, labelpad=30)                             
        plt.yticks(fontsize=30)
        plt.ylim(-40, 300)
        
        if tight_layout is not None:
            plt.tight_layout(rect=tight_layout)
             
        if saveDir is not None:
            Path(saveDir).mkdir(exist_ok=True, parents=True)
            plt.savefig(Path(saveDir) / saveName, dpi=DPI, format='pdf')
        
    return


def main():
    
    # load my keio screen results
    metadata = pd.read_csv(Path(PROJECT_DIR) / 'metadata.csv', header=0, index_col=None, 
                           dtype={'comments':str})
    features = pd.read_csv(Path(PROJECT_DIR) / 'features.csv', header=0, index_col=None)
    
    # load Walhout (2019) gene list
    walhout_244 = load_walhout_244()
    walhout_244.insert(0, 'wild_type')

    # errorbar plots for all genes (highlighting Walhout genes)
    errorbar_sigfeats(features, 
                      metadata, 
                      group_by='gene_name',
                      fset=[FEATURE],
                      control='wild_type',
                      highlight_subset=[s for s in walhout_244 if s != 'wild_type'],
                      rank_by='mean',
                      figsize=(25, 8),
                      fontsize=3,
                      ms=2,
                      elinewidth=1.5,
                      fmt='.',
                      tight_layout=[0.01,0.01,0.99,0.99],
                      saveDir=SAVE_DIR,
                      saveName='Fig4e.pdf')

    return 

#%% Main

if __name__ == '__main__':
    main()