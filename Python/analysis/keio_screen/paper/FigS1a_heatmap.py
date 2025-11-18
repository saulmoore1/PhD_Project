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
from scipy.stats import zscore
from matplotlib import pyplot as plt
from matplotlib import patches
from tierpsytools.preprocessing.filter_data import select_feat_set
from tierpsytools.analysis.statistical_tests import univariate_tests
from tierpsytools.analysis.statistical_tests import get_effect_sizes

#%% Globals

SAVE_DIR = "/Users/sm5911/Documents/PhD_DLBG/FigS1a"

#%% Functions

def average_plate_control_data(metadata, features, 
                               control='wild_type', 
                               grouping_var='gene_name', 
                               plate_var='imaging_plate_id'):
    
    """ Average data for control plate on each experiment day to yield a single 
        mean datapoint for the control. This reduces the control sample size to 
        equal the test strain sample size, for t-test comparison. Information 
        for the first well in the control sample on each day is used as the 
        accompanying metadata for mean feature results. 
        
        Input
        -----
        features, metadata : pd.DataFrame
            Feature summary results and metadata dataframe with multiple 
            entries per day
            
        Returns
        -------
        features, metadata : pd.DataFrame
            Feature summary results and metadata with control data averaged 
            (single sample per day)
    """
        
    # Subset results for control data
    control_metadata = metadata[metadata[grouping_var]==control]
    control_features = features.reindex(control_metadata.index)

    # calculate mean of control for each plate (collapses data to a single 
    # datapoint for strain comparison)
    mean_control = control_metadata[[grouping_var, plate_var]].join(
        control_features).groupby(by=[grouping_var, plate_var]).mean().reset_index()
    
    # Append remaining control metadata column info (with first well data for each date)
    remaining_cols = [c for c in control_metadata.columns.to_list() if 
                      c not in [grouping_var, plate_var]]
    
    mean_control_row_data = []
    for i in mean_control.index:
        # for each control plate:
        plate = mean_control.loc[i, plate_var]
        # get the metadata for the first well of the control plate
        control_plate_meta = control_metadata.loc[control_metadata[plate_var] == plate]
        first_well_meta = control_plate_meta.loc[control_plate_meta.index[0], remaining_cols]
        # get the mean feature values for the control plate
        plate_mean = mean_control.loc[mean_control[plate_var] == plate].squeeze(axis=0)
        # concatenate metadata and mean feature values for control plate + append to list
        plate_mean_and_meta = pd.concat([plate_mean,first_well_meta])
        mean_control_row_data.append(plate_mean_and_meta)
    
    # create dataframe of all control plate mean data + split into metadata and features
    control_mean = pd.DataFrame.from_records(mean_control_row_data)
    control_metadata = control_mean[control_metadata.columns.to_list()]
    control_features = control_mean[control_features.columns.to_list()]

    # replace control data with control plate means
    features = pd.concat([features.loc[metadata[grouping_var] != control, :], 
                          control_features], axis=0).reset_index(drop=True)        
    metadata = pd.concat([metadata.loc[metadata[grouping_var] != control, :], 
                          control_metadata.loc[:, metadata.columns.to_list()]], 
                          axis=0).reset_index(drop=True)
    
    assert all(metadata.index == features.index)

    return metadata, features

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
    
    # Save clustermap
    if saveto:
        saveto.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(saveto, dpi=300)
    else:
        plt.show()
    
    return cg

def heatmap(metadata, features):
    
    assert not features.isna().sum(axis=0).any()
    assert not (features.std(axis=0) == 0).any()
    
    featZ_df = features.apply(zscore, axis=0)
    
    # Clustermap of full data       
    full_clustermap_path = Path(SAVE_DIR) / 'FigS1a_heatmap.pdf'
    fg = plot_clustermap(featZ=featZ_df, 
                         meta=metadata[['gene_name']], 
                         group_by='gene_name',
                         col_linkage=None,
                         row_colours=False,
                         method='complete',
                         metric='euclidean',
                         figsize=[20, 35],
                         saveto=full_clustermap_path,
                         sub_adj={'bottom':0.13,'left':0.01,'top':0.99,'right':0.97},
                         label_size=(3,2))
    
    # save data for heatmap    
    row_names = [r.get_text() for r in fg.ax_heatmap.yaxis.get_majorticklabels()]
    row_names = pd.DataFrame(data=row_names, index=fg.data.index, 
                             columns=['gene_name'], dtype=str)
    heatmap_data = row_names.join(fg.data)
    save_path = Path(SAVE_DIR) / 'Fig1b_heatmap_data.csv'
    heatmap_data.to_csv(save_path, header=True, index=False)
    
    return

def nearest_neighbour_clustering(metadata, features):
    
    from clustering.nearest_neighbours import cluster_linkage_pdist
    from scipy.cluster.hierarchy import cophenet #linkage
    from scipy.spatial.distance import pdist #squareform
    from clustering.nearest_neighbours import nearest_neighbours 

    # Compute cluster linkage array
    Z, X = cluster_linkage_pdist(features, 
                                 metadata, 
                                 groupby='gene_name',
                                 saveDir=None, 
                                 method='average', 
                                 metric='euclidean')
    
    # Compare pairwise distances between all samples to hierarchical clustering distances 
    # The closer the value to 1 the better the clustering preserves original distances
    c, coph_dists = cophenet(Z, pdist(X)); print("Cophenet: %.3f" % c)

    # Find nearest neighbours by ranking the computed sqaureform distance matrix between all strains
    names_df, distances_df = nearest_neighbours(X=X,
                                                strain_list=None,
                                                distance_metric='euclidean', 
                                                saveDir=None)
    
    return names_df, distances_df

def main():
    
    metadata_path_local = Path(SAVE_DIR) / 'metadata.csv'
    features_path_local = Path(SAVE_DIR) / 'features.csv'

    metadata = pd.read_csv(metadata_path_local, header=0, index_col=None, dtype={'comments':str})
    features = pd.read_csv(features_path_local, header=0, index_col=None)

    assert not features.isna().sum(axis=1).any()
    assert not (features.std(axis=1) == 0).any()
    assert set(features.index) == set(metadata.index)    
    assert (len(metadata['gene_name'].unique()) ==
            len(metadata['gene_name'].str.upper().unique()))
    
    # subset for top16 tierpsy features
    features = select_feat_set(features, tierpsy_set_name='tierpsy_256', append_bluelight=True)
     
    # average control data for each experiment day
    metadata, features = average_plate_control_data(metadata,
                                                    features,
                                                    control='wild_type',
                                                    grouping_var='gene_name', 
                                                    plate_var='imaging_plate_id')

    # t-tests comparing each bacterial strain to wild-type control
    strain_list = sorted(list(metadata['gene_name'].unique()))
    print("Performing t-tests comparing %d strains to BW25113 wild-type control (%d features)" %\
          (len(strain_list), features.shape[1]))
        
    stats_t, pvals_t, reject_t = univariate_tests(X=features,
                                                  y=metadata['gene_name'],
                                                  control='wild_type',
                                                  test='t-test',
                                                  comparison_type='binary_each_group',
                                                  multitest_correction=None, # uncorrected
                                                  alpha=0.05)

    effect_sizes_t = get_effect_sizes(X=features,
                                      y=metadata['gene_name'],
                                      control='wild_type',
                                      linked_test='t-test')
    
    # compile t-test results
    stats_t.columns = ['stats_' + str(c) for c in stats_t.columns]
    pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
    reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
    effect_sizes_t.columns = ['effect_size_' + str(c) for c in effect_sizes_t.columns]
    ttest_results = pd.concat([stats_t, pvals_t, reject_t, effect_sizes_t], axis=1)
        
    # Find strains that elicit a behavioural change in at least one feature 
    # relative to BW wild-type by t-test (p<0.05, FDR controlled at 5% using 
    # Benjamini-Hochberg correction, 753 features)
    pvals = ttest_results[[c for c in ttest_results.columns if 'pvals_' in c]]
    pvals.columns = [c.split('pvals_')[-1] for c in pvals.columns]
    mask = (pvals < 0.05).any(axis=0)    
    hits = mask[mask].index.tolist()
    n_hits = len(hits)

    # TODO: save to file

    # t-tests
    # av, all, None   => 3873
    #     all, None   => 3873
    # av, all, fdr_bh => 2690
    #     all, fdr_bh => 3854
    # av, all, fdr_by => 399
    #     all, fdr_by => 3570
    # av, 256, None   => 3872
    #     256, None   => 3872
    # av, 256, fdr_bh => 86    * Not 1860!
    #     256, fdr_bh => 3331
    # av, 256, fdr_by => 0
    #     256, fdr_by => 2361
    # av, 16, None    => 3080
    #     16, None    => 3516
    # av, 16, fdr_bh  => 0
    #     16, fdr_bh  => 1016
    # av, 16, fdr_by  => 0
    #     16, fdr_by  => 419
    
    # pvals_list = []
    # for strain in strain_list:
    #     if strain == 'wild_type':
    #         continue
    #     strain_meta = metadata[metadata['gene_name'].isin(['wild_type',strain])]
    #     strain_feat = features.reindex(strain_meta.index)
        
    #     stats, pvals, reject = univariate_tests(X=strain_feat,
    #                                             y=strain_meta['gene_name'],
    #                                             control='wild_type',
    #                                             test='t-test',
    #                                             comparison_type='binary_each_group',
    #                                             multitest_correction='fdr_bh',
    #                                             alpha=0.05) 
        
    #     pvals_list.append(pvals)
        
    # pvals_df = pd.concat(pvals_list, axis=1)    
    # mask = (pvals_df < 0.05).any(axis=0)    
    # hits = mask[mask].index.tolist()
    # n_hits = len(hits)
    
    # t-test corrected in loop (which is wrong!)
    # av, all, None   => 3873
    #     all, None   => 3873
    # av, all, fdr_bh => 1625
    #     all, fdr_bh => 3424
    # av, all, fdr_by => 748
    #     all, fdr_by => 3149
    # av, 256, None   => 3872
    #     256, None   => 3872
    # av, 256, fdr_bh => 933
    #     256, fdr_bh => 2582
    # av, 256, fdr_by => 266
    #     256, fdr_by => 2119
    # av, 16, None    => 3080
    #     16, None    => 3516
    # av, 16, fdr_bh  => 211
    #     16, fdr_bh  => 1275
    # av, 16, fdr_by  => 58
    #     16, fdr_by  => 821  
    
    # metadata = metadata[metadata['gene_name'].isin(hits)]
    # features = features.reindex(metadata.index)

    heatmap(metadata, features)
    
    # rank strains by lowest p-value for any feature (Tierpsy 16, t-tests uncorrected)
    ranked_pval = pvals.min(axis=0).sort_values(ascending=True)
    lowest_100_pval = ranked_pval[:100].index.tolist()
    # 59 chosen hit strains were manually curated from this 'lowest_100_pval' list
    
    # read curated list of 59 hit strains from file
    selected_hits_path = Path(SAVE_DIR) /\
        '59_selected_strains_from_initial_keio_top100_lowest_pval_tierpsy16.txt'
    selected_hits = []
    with open(selected_hits_path, 'r') as fid:
        for line in fid:
            selected_hits.append(line.strip('\n'))
            
    manually_dropped_hits = set(lowest_100_pval).difference(set(selected_hits))
    manually_added_hits = set(selected_hits).difference(lowest_100_pval)
    
    # find 3 nearest neighbours to each selected hit strain (Tierpsy 256)
    metadata = pd.read_csv(metadata_path_local, header=0, index_col=None, dtype={'comments':str})
    features = pd.read_csv(features_path_local, header=0, index_col=None)
    features = select_feat_set(features, tierpsy_set_name='tierpsy_256', append_bluelight=True)    
    names_df, distances_df = nearest_neighbour_clustering(metadata, features)
    neighbour_names_df = names_df.loc[selected_hits, 1:3]
    neighbour_list = np.unique(np.asmatrix(neighbour_names_df).flatten().tolist())
    neighbour_list = sorted(set(neighbour_list) - set(selected_hits))

    return

#%% Main

if __name__ == "__main__":
    main()
    
#%% Other observations

# NB: umuC - nearest neighbour to fepD for initial screen strains (all strains not 1860 hits, Tierpsy16 not Tierpsy 256)
# NB: ndh - lowest motion mode paused frequency for confirmation screen strains (optimal BL window features, not full video)

