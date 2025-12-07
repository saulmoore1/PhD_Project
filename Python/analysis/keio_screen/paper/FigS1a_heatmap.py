#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: sm5911
"""

#%% Imports

import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy.stats import zscore
from matplotlib import pyplot as plt
from matplotlib import patches
from tierpsytools.preprocessing.filter_data import select_feat_set

#%% Globals

SAVE_DIR = "/Users/sm5911/Documents/PhD_DLBG/FigS1a heatmap"

RENAME_DICT = {"FECE" : "fecE",
               "BW" : "wild_type"}

#%% Functions

def plot_clustermap(featZ, 
                    meta, 
                    group_by,
                    colour_by=None,
                    row_colours=True,
                    col_linkage=None,
                    method='complete',
                    metric='euclidean',
                    saveto=None,
                    figsize=[10,8],
                    sns_colour_palette="Pastel1",
                    sub_adj={'bottom':0,'left':0,'top':1,'right':1},
                    label_size=5,
                    show_xlabels=True,
                    bluelight_col_colours=True):
    """ Seaborn clustermap (hierarchical clustering heatmap)
    
        Inputs
        ------
        featZ - pd.DatFrame, dataframe of normalised feature results
    """                
    
    assert (featZ.index == meta.index).all()
    
    if type(group_by) != list:
        group_by = [group_by]
    n = len(group_by)
    # if not (n == 1 or n == 2):
    #     raise IOError("Must provide either 1 or 2 'group_by' parameters")        
        
    # Store feature names
    fset = featZ.columns
        
    # Compute average value for strain for each feature (not each well)
    featZ_grouped = featZ.join(meta).groupby(group_by, dropna=False).mean().reset_index()
    
    if colour_by is None:
        colour_by = group_by[0]
    assert colour_by in meta.columns
        
    var_list = list(featZ_grouped[colour_by].unique())

    # Row colors
    if row_colours is False:
        row_colours = None
    if row_colours is not None:
        row_colours = []
        if len(var_list) > 1 or n == 1:
            var_colour_dict = dict(zip(var_list, sns.color_palette("tab10", len(var_list))))
            row_cols_var = featZ_grouped[group_by[0]].map(var_colour_dict)
            row_colours.append(row_cols_var)
        if n == 2:
            date_list = list(featZ_grouped[group_by[1]].unique())
            date_colour_dict = dict(zip(date_list, sns.color_palette("Blues", len(date_list))))
            #date_colour_dict=dict(zip(set(date_list),sns.hls_palette(len(set(date_list)),l=0.5,s=0.8)))
            row_cols_date = featZ_grouped[group_by[1]].map(date_colour_dict)
            row_cols_date.name = None
            row_colours.append(row_cols_date)  

    # Column colors
    if bluelight_col_colours:
        bluelight_colour_dict = dict(zip(['prestim','bluelight','poststim'], 
                                         sns.color_palette(sns_colour_palette, 3)))
        feat_colour_dict = {f:bluelight_colour_dict[f.split('_')[-1]] for f in fset}
    
    if type(label_size) == tuple:
        x_label_size, y_label_size = label_size
    else:
        x_label_size = label_size
        y_label_size = label_size
        
    # Plot clustermap
    plt.close('all')
    sns.set(font_scale=0.8)
    cg = sns.clustermap(data=featZ_grouped[fset], 
                        row_colors=row_colours,
                        col_colors=fset.map(feat_colour_dict) if bluelight_col_colours else None,
                        #standard_scale=1, z_score=1,
                        col_linkage=col_linkage,
                        metric=metric, 
                        method=method,
                        vmin=-2, vmax=2,
                        figsize=figsize,
                        xticklabels=fset if show_xlabels else False,
                        yticklabels=featZ_grouped[group_by].astype(str).agg(' - '.join, axis=1),
                        #cbar_pos=(0.5, 0.01, 0.1, 0.01), # (left, bottom, width, height)
                        cbar_kws={'orientation': 'horizontal',
                                  'label': None, #'Z-value'
                                  #'shrink': 1,
                                  'ticks': [-2, -1, 0, 1, 2],
                                  'drawedges': False},
                        linewidths=0)  
    #col_linkage = cg.dendrogram_col.calculated_linkage
    
    if show_xlabels:
        labels = cg.ax_heatmap.xaxis.get_majorticklabels()
        plt.setp(labels, rotation=90, fontsize=x_label_size)
        
    cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_yticklabels(), rotation=0, 
                                  fontsize=y_label_size, ha='left', va='center') 
    #plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), fontsize=15)
    #cg.ax_heatmap.axes.set_xticklabels([]); cg.ax_heatmap.axes.set_yticklabels([])
    
    if bluelight_col_colours:
        patch_list = []
        for l, key in enumerate(bluelight_colour_dict.keys()):
            patch = patches.Patch(color=bluelight_colour_dict[key], label=key)
            patch_list.append(patch)
        lg = plt.legend(handles=patch_list, 
                        labels=bluelight_colour_dict.keys(), 
                        title="Stimulus",
                        frameon=True,
                        loc='upper right',
                        bbox_to_anchor=(0.99, 0.99), 
                        bbox_transform=plt.gcf().transFigure,
                        fontsize=12, handletextpad=0.2)
        lg.get_title().set_fontsize(15)
    
    plt.subplots_adjust(top=sub_adj['top'], bottom=sub_adj['bottom'], 
                        left=sub_adj['left'], right=sub_adj['right'], 
                        hspace=0.01, wspace=0.01)
    #plt.tight_layout(rect=[0, 0, 1, 1], w_pad=0.5)
        
    # Add custom colorbar to right hand side
    # from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    # from mpl_toolkits.axes_grid1.colorbar import colorbar
    # # split axes of heatmap to put colorbar
    # ax_divider = make_axes_locatable(cg.ax_heatmap)
    # # define size and padding of axes for colorbar
    # cax = ax_divider.append_axes('right', size = '5%', pad = '2%')
    # # Heatmap returns an axes obj but you need to get a mappable obj (get_children)
    # colorbar(cg.ax_heatmap.get_children()[0], 
    #          cax = cax, 
    #          orientation = 'vertical', 
    #          ticks=[-2, -1, 0, 1, 2])
    # # locate colorbar ticks
    # cax.yaxis.set_ticks_position('right')
    
    # Save clustermap
    if saveto:
        saveto.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(saveto, dpi=300)
    else:
        plt.show()
        
    data = featZ_grouped.iloc[featZ_grouped.index[cg.dendrogram_row.reordered_ind]
                              ].set_index('gene_name')
    data = data[data.columns[cg.dendrogram_col.reordered_ind]]
    
    return data

def heatmap(metadata, features):
    
    assert not features.isna().sum(axis=0).any()
    assert not (features.std(axis=0) == 0).any()
    
    featZ_df = features.apply(zscore, axis=0)
    
    # Clustermap of full data       
    full_clustermap_path = Path(SAVE_DIR) / 'FigS1a_heatmap.pdf'
    data = plot_clustermap(featZ=featZ_df, 
                           meta=metadata[['gene_name']], 
                           group_by='gene_name',
                           col_linkage=None,
                           row_colours=False,
                           method='complete',
                           metric='euclidean',
                           figsize=[20, 35],
                           saveto=full_clustermap_path,
                           sub_adj={'bottom':0.01,'left':0.01,'top':0.99,'right':0.97},
                           label_size=(10,7),
                           show_xlabels=False)
    
    # save data for heatmap    
    save_path = Path(SAVE_DIR) / 'FigS1a_heatmap_data.csv'
    data.to_csv(save_path, header=True, index=True)
    
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
    
    # subset for top16 tierpsy features
    features = select_feat_set(features, tierpsy_set_name='tierpsy_256', 
                               append_bluelight=True)
 
    strain_list = sorted(list(metadata['gene_name'].unique()))
    print("Number of strains:", len(strain_list))
    
    heatmap(metadata, features)

    return

#%% Main

if __name__ == "__main__":
    main()