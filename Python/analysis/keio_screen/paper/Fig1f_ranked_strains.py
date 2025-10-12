#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: sm5911
"""

#%% Imports

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.stats import sem

#%% Globals

SAVE_DIR = "/Users/sm5911/Documents/PhD_DLBG/Fig1f"
RENAME_DICT = {"FECE" : "fecE",
               "BW" : "wild_type"}

#%% Functions

def errorbar_sigfeats(features, metadata, group_by, fset, control=None, rank_by='median', 
                      highlight_subset=None, max_feats2plt=None, figsize=[130,6], fontsize=4, 
                      tight_layout=None, color='dimgray', saveDir=None, saveName=None, 
                      highlight_colour='red', **kwargs):
    """ Plot mean feature value with errorbars (+/- 1.98 * std) for all groups in 
        metadata['group_by'] for each feature in feature set provided 
    """
    
    plt.ioff() if saveDir is not None else plt.ion()
    
    if highlight_subset is not None:
        assert all(s in metadata[group_by].unique() for s in highlight_subset)
        
    assert all(f in features.columns for f in fset)
    
    # Boxplots of significant features by ANOVA/LMM (all groups)
    grouped = metadata[[group_by]].join(features[fset]).groupby(group_by)
    
    mean_strain = grouped.mean()
    median_strain = grouped.median()
            
    max_feats2plt = len(fset) if max_feats2plt is None else max_feats2plt
    
    # Plot all strains (ranked by median) for top n significant features 
    # (ranked by ANOVA p-value)
    for f, feat in enumerate(tqdm(fset[:max_feats2plt])):
        # Errorbar plot
        if rank_by == 'median':
            order = median_strain[feat].sort_values(ascending=True).index.to_list()
            df_ordered = median_strain.reindex(order).reset_index(
                drop=False)[[group_by, feat]]  
        elif rank_by == 'mean':
            order = mean_strain[feat].sort_values(ascending=True).index.to_list()
            df_ordered = mean_strain.reindex(order).reset_index(
                drop=False)[[group_by, feat]]  

        df_ordered['error'] = [sem(features.loc[metadata[group_by]==strain, feat]) 
                               for strain in order]
        #error = [1.98 * features.loc[metadata[group_by]==strain, feat].std() for strain in order]

        if saveDir is not None:
            data_save_path = Path(saveDir) / "Fig1f_ranked_strains_{}_data.csv".format(feat)
            df_ordered.to_csv(data_save_path, header=True, index=False)

        if control is not None:
            df_ordered['colour'] = ['blue' if s == control else 'grey' for s in 
                                    df_ordered[group_by]]
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
                
        _ = plt.xticks(rotation=90, ha='center', fontsize=fontsize)#, color=df_ordered['colour'].tolist())
        _ = [t.set_color(i) for (i,t) in zip(df_ordered['colour'].tolist(), ax.xaxis.get_ticklabels())]
        #ax.tick_params(axis="x", color=colour)
                
        if rank_by == 'median':
            plt.axhline(median_strain.loc[control, feat], c='dimgray', ls='--')
            #med_of_med = median_ordered.median() # FOR PLOTTING MEDIAN OF MEDIANS
            #plt.axhline(med_of_med, c='', ls='--')
        elif rank_by == 'mean':
            plt.axhline(mean_strain.loc[control, feat], c='dimgray', ls='--')
            
        # plt.tick_params(
        #     axis='x',          # changes apply to the x-axis
        #     which='both',      # both major and minor ticks are affected
        #     bottom=False,      # ticks along the bottom edge are off
        #     top=False,         # ticks along the top edge are off
        #     labelbottom=True)  # labels along the bottom edge are off
        plt.title(feat, pad=10, fontsize=20)
        
        if tight_layout is not None:
            plt.tight_layout(rect=tight_layout)
             
        if saveDir is not None:
            Path(saveDir).mkdir(exist_ok=True, parents=True)
            plt.savefig(Path(saveDir) / 'Fig1f_strains_ranked_{0}.pdf'.format(feat))
        
    return

def main():
    
    metadata_path_local = Path(SAVE_DIR) / 'metadata.csv'
    features_path_local = Path(SAVE_DIR) / 'features.csv'
    
    metadata = pd.read_csv(metadata_path_local, header=0, index_col=None, 
                           dtype={'comments':str})
    features = pd.read_csv(features_path_local, header=0, index_col=None)
    
    assert not features.isna().sum(axis=1).any()
    assert not (features.std(axis=1) == 0).any()
    assert set(features.index) == set(metadata.index)    
    assert (len(metadata['gene_name'].unique()) ==
            len(metadata['gene_name'].str.upper().unique()))

    # rename gene names in metadata
    for k, v in RENAME_DICT.items():
        metadata.loc[metadata['gene_name'] == k, 'gene_name'] = v

    # remove entries for results with missing gene name metadata
    n_rows = metadata.shape[0]
    metadata = metadata[~metadata['gene_name'].isna()]
    if metadata.shape[0] < n_rows:
        print("Removed %d row entries with missing gene name (or empty wells)" % (
            n_rows - metadata.shape[0]))
        
    features = features.reindex(metadata.index)
           
    # errorbar plots for all genes (highlighting fep genes) for speed 50th bluelight
    highlight_list = sorted([s for s in metadata['gene_name'].unique() if 
                             s.startswith('atp') or s.startswith('fep') or 
                             s.startswith('sdh') or s.startswith('nuo') or 
                             s == 'fes'])
    
    errorbar_sigfeats(features, 
                      metadata, 
                      group_by='gene_name',
                      fset=['speed_50th_bluelight',
                            'motion_mode_paused_frequency_bluelight'],
                      control='wild_type',
                      highlight_subset=highlight_list,
                      rank_by='mean',
                      figsize=(30,10),
                      fontsize=10,
                      ms=10,
                      elinewidth=4,
                      fmt='.',
                      tight_layout=[0.01,0.01,0.99,0.99],
                      saveDir=Path(SAVE_DIR))
    
    return

#%% Main

if __name__ == "__main__":
    main()