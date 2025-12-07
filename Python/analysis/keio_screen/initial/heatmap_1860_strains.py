#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse Keio Screen results

Please run the following scripts beforehand:
1. preprocessing/compile_keio_results.py
2. statistical_testing/perform_keio_stats.py

THRESHOLD MAX DISTANCE FOR CLUSTERING: 8 (Tierpsy16, fdr_bh, all strains)

@author: sm5911
@date: 19/04/2021
"""

#%% IMPORTS

import pandas as pd
from pathlib import Path
#import seaborn as sns
#from matplotlib import pyplot as plt
#from matplotlib import patches
from scipy.stats import zscore
from tierpsytools.preprocessing.filter_data import select_feat_set
from clustering.hierarchical_clustering import plot_clustermap
#from clustering.hierarchical_clustering import plot_barcode_heatmap

#%% GLOBALS

SAVE_DIR = "/Users/sm5911/Documents/PhD_DLBG/Other/Keio_Initial_Screen/heatmap"

FEATURE_SET = "tierpsy_256"
COLLAPSE_CONTROL = True
FDR_METHOD = "fdr_bh"

#JSON = "20210406_parameters_keio_screen.json"

# cleaning parameters
DATES = ["20210406", "20210413", "20210420", "20210427", "20210504", "20210511"]
ALIGN_BLUELIGHT = True     # append stimulus type to feature names
OMIT_STRAINS = None        # do not remove any particular strains from the analysis
PERCENTILE_TO_USE = None   # include all tierpsy percentile features
DROP_SIZE_FEATURES = False # include all features related to worm size
NORM_FEATURES_ONLY = False # include features that are not normalised
REMOVE_OUTLIERS = False    # do not remove outliers
IMPUTE_NANS = True         # fill missing feature data with global mean for that feature
MAX_VALUE_CAP = 1e15       # maximum cap on feature value (drop extreme erroneus values)
NAN_THRESHOLD_ROW = 0.8    # drop samples with missing values for >80% of features
NAN_THRESHOLD_COL = 0.05   # drop features where >5% samples have missing values for that feature
MIN_NSKEL_PER_VIDEO = None # do not drop samples with less than n skeletons per video
MIN_NSKEL_SUM = 6000       # drop samples with <6000 skeletons in total across video frames

# import clean_feature_summaries_old

#%% FUNCTIONS
        
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
    
    features = select_feat_set(features, 
                               tierpsy_set_name=FEATURE_SET, 
                               append_bluelight=True)
    
    if COLLAPSE_CONTROL:
        print("\nCollapsing control data (mean of each day)")
        metadata, features = average_plate_control_data(metadata,
                                                        features,
                                                        control='wild_type', 
                                                        grouping_var='gene_name', 
                                                        plate_var='imaging_plate_id')
        
    # 571 significant features by ANOVA (tierpsy256, fdr_bh)
    anova_path = Path(SAVE_DIR) / "ANOVA_results.csv"
    anova_table = pd.read_csv(anova_path, index_col=0)            
    pvals = anova_table.sort_values(by='pvals', ascending=True)['pvals']
    fset_anova = pvals[pvals < 0.05].index.to_list()
    print("\n%d significant features found by ANOVA (P<0.05, %s)" % (len(fset_anova), 
                                                                     FDR_METHOD))

    # 456 significant features by t-test (tierpsy256, fdr_bh)
    ttest_path = Path(SAVE_DIR) / "t-test_results.csv"
    ttest_table = pd.read_csv(ttest_path, index_col=0)
    pvals_t = ttest_table[[c for c in ttest_table if "pvals_" in c]] 
    pvals_t.columns = [c.split('pvals_')[-1] for c in pvals_t.columns]       
    fset_ttest = pvals_t[(pvals_t < 0.05).sum(axis=1) > 0].index.to_list()
    print("%d significant features found by t-test (P<0.05, %s)" % (len(fset_ttest), 
                                                                    FDR_METHOD))

    # Rank strains by number of sigfeats by t-test 
    ranked_nsig = (pvals_t < 0.05).sum(axis=0).sort_values(ascending=False)
    ranked_nsig_df = pd.DataFrame(ranked_nsig, 
                                  index=ranked_nsig.index, 
                                  columns=["n_sigfeats"])
    
    # Save ranked strains to file
    ranked_nsig_df.to_csv(Path(SAVE_DIR) / "1860_hit_strains_ttest_tierpsy256_fdr_bh.csv",
                          header=True, index=True)
    
    # 1860 hit strains with 1 or more significant features
    hit_strains_nsig = ranked_nsig[ranked_nsig > 0].index.to_list()
    print("%d significant strains (with 1 or more significant features)" %\
          len(hit_strains_nsig))
    
    # Save hit strains list to file
    hit_strains_save_path = Path(SAVE_DIR) / "1860_hit_strains_ttest_tierpsy256_fdr_bh.txt"
    with open(hit_strains_save_path, 'a') as fid:
        for strain in hit_strains_nsig:
            fid.write(strain + '\n')
            
    # # plot ranked strains by number of significant features
    # ranked_nsig_plot_path = Path(SAVE_DIR) / 'ranked_n_significant_features.svg'
    # plt.close('all')
    # fig, ax = plt.subplots(figsize=(20,6))
    # ax.plot(ranked_nsig)
    # ax.set_xticklabels(ranked_nsig.index.to_list(), rotation=90, fontsize=5)
    # plt.xlabel("Strains (ranked)", fontsize=12, labelpad=10)
    # plt.ylabel("Number of significant features", fontsize=12, labelpad=10)
    # plt.subplots_adjust(left=0.08, right=0.98, bottom=0.15)
    # plt.savefig(ranked_nsig_plot_path)
    
    # # Rank strains by p-value + select top 100 strains with lowest p-value for any feature
    # ranked_pval = pvals_t.min(axis=0).sort_values(ascending=True)
    # sig_strains_pval = ranked_pval.index[:100].to_list()
    
    # # plot ranked strains by lowest p-value
    # ranked_lowest_pval_plot_path = Path(SAVE_DIR) / 'ranked_lowest_pvalue.svg'
    # plt.close('all')
    # fig, ax = plt.subplots(figsize=(20,6))
    # ax.plot(ranked_pval)
    # plt.axhline(y=0.05, c='dimgray', ls='--')
    # ax.set_xticklabels(ranked_nsig.index.to_list(), rotation=90, fontsize=5)
    # plt.xlabel("Strains (ranked)", fontsize=12, labelpad=10)
    # plt.ylabel("Lowest p-value by t-test", fontsize=12, labelpad=10)
    # plt.subplots_adjust(left=0.08, right=0.98, bottom=0.15)
    # plt.savefig(ranked_lowest_pval_plot_path)
    # plt.close()
            
    # subset for hit strains (1+ significant features)   
    metadata = metadata[metadata['gene_name'].isin(['wild_type'] + hit_strains_nsig)]
    features = features.reindex(metadata.index)

    # hit strains heatmap (n=1860 strains with 1+ significant features)   
    featZ = features.apply(zscore, axis=0)

    hits_clustermap_path = Path(SAVE_DIR) / 'heatmap.pdf'
    data = plot_clustermap(featZ, metadata[['gene_name']], 
                           group_by='gene_name',
                           row_colours=None,
                           method='complete', 
                           metric='euclidean',
                           figsize=[20,30],
                           sub_adj={'bottom':0.01,'left':0,'top':1,'right':0.95},
                           saveto=hits_clustermap_path,
                           label_size=8,
                           show_xlabels=False)
    
    # clustered feature order for 1860 hit strains
    _ = data.columns

    # save data for heatmap    
    save_path = Path(SAVE_DIR) / 'initial_screen_1860_hits_heatmap_data.csv'
    data.to_csv(save_path, header=True, index=True)

    # # control heatmap - cluster control data and store feature order to apply to full data
    # control_metadata = pd.read_csv(metadata_path_local, header=0, index_col=None, 
    #                                dtype={'comments':str})
    # control_features = pd.read_csv(features_path_local, header=0, index_col=None)
    # control_metadata = control_metadata[control_metadata['gene_name']=='wild_type']
    # control_features = control_features.reindex(control_metadata.index)
    
    # control_features = select_feat_set(control_features, 
    #                                    tierpsy_set_name=feature_set, 
    #                                    append_bluelight=True)

    # control_featZ = control_features.apply(zscore, axis=0)
    # control_metadata['date_yyyymmdd'] = control_metadata['date_yyyymmdd'].astype(str)

    # control_clustermap_path = Path(SAVE_DIR) / 'control_heatmap.pdf'
    # data = plot_clustermap(control_featZ, control_metadata[['date_yyyymmdd']],
    #                        group_by='date_yyyymmdd',
    #                        method='complete', 
    #                        metric='euclidean',
    #                        figsize=[20,6],
    #                        sub_adj={'bottom':0.05,'left':0,'top':1,'right':0.85},
    #                        saveto=control_clustermap_path,
    #                        label_size=15,
    #                        show_xlabels=False)
    
    # control_clustered_features = data.columns

    # pvals_heatmap = anova_table.loc[control_clustered_features, 'pvals']
    # pvals_heatmap.name = 'P < 0.05'
    # assert all(f in featZ.columns for f in pvals_heatmap.index)

    # heatmap_path = Path(SAVE_DIR) / 'barcode_heatmap.pdf'
    # plot_barcode_heatmap(featZ=featZ[control_clustered_features], 
    #                      meta=metadata[['gene_name']], 
    #                      group_by=['gene_name'], 
    #                      pvalues_series=pvals_heatmap,
    #                      p_value_threshold=0.05,
    #                      selected_feats=None,
    #                      saveto=heatmap_path,
    #                      figsize=[20,30],
    #                      sns_colour_palette="Pastel1",
    #                      label_size=10)        

    return
    
#%% MAIN
if __name__ == "__main__":
    main()

