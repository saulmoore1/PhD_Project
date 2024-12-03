#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse Initial Keio Screen 

- Compiles and cleans metadata and feature summaries
- Wild-type BW25113 control data is averaged for each experiment day 
  (similar sample sizes for statistical testing)
- Tierpsy16 feature set is used to perform initial screen analysis (ANOVA/t-tests)
- No significant features are found by t-test for any strain when corrected for multiple testing 
  (fdr_by/fdr_bh), so the top 100 strains with lowest p-value for any feature are selected as hits
- A list of 59 hits is manually curated from top 100 strains with lowest p-value for any feature
- Nearest neighbour cluster analysis is performed to expand the set of hit strains to include other
  strains that elicit similar behaviour to selected hits (3 nearest neighbour strains to each of
  the 59 selected hits, using Tierpsy256 feature set)
- Yeilds 229 hits for confirmation screen
  
- To run from command line:
    - cd /Users/sm5911/Documents/GitHub/PhD_Project/Python
    - conda activate tierpsytools
    - python -m analysis.keio_screen.initial.keio_screen_initial_20241129

@author: sm5911
@date: 28/11/2024

"""

#%% Imports

import numpy as np
import pandas as pd
import seaborn as sns
from time import time
from pathlib import Path
from matplotlib import pyplot as plt
#from scipy.stats import zscore
from scipy.spatial.distance import pdist #squareform
from scipy.cluster.hierarchy import cophenet #linkage

from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from preprocessing.append_supplementary_info import load_supplementary_7, append_supplementary_7
from filter_data.clean_feature_summaries import clean_summary_results
from tierpsytools.preprocessing.filter_data import select_feat_set
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from tierpsytools.analysis.statistical_tests import _multitest_correct
from visualisation.plotting_helper import sig_asterix
from write_data.write import write_list_to_file
from read_data.read import read_list_from_file
from clustering.nearest_neighbours import nearest_neighbours 
from clustering.nearest_neighbours import cluster_linkage_seaborn, cluster_linkage_pdist

#%% Globals

# paths
PROJECT_DIR = "/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Screen_Initial"
SAVE_DIR = "/Users/sm5911/Documents/PhD_DLBG/3_Keio_Screen_Initial"
SUPPLEMENTARY_INFO_PATH = Path(PROJECT_DIR) /\
    "AuxiliaryFiles/Baba_et_al_2006/Supporting_Information/Supplementary_Table_7.xls"
CURATED_HIT_STRAINS_PATH = Path(SAVE_DIR) /\
    "59_selected_strains_from_initial_keio_top100_lowest_pval_tierpsy16.txt"

# preprocessing parameters
EXPERIMENT_DATES = ["20210406", "20210413", "20210420", "20210427", "20210504", "20210511"]
N_WELLS = 96 # number of wells per plate
NAN_THRESHOLD_ROW = 0.8  # threshold proportion of NaN values across features to drop sample
NAN_THRESHOLD_COL = 0.05 # threshold proportion of NaN values across samples to drop feature
MIN_NSKEL_SUM = 6000 # minimum number of skeletons across prestim/bluelight/poststim videos to keep sample
RENAME_DICT = {"FECE" : "fecE",
               "AroP" : "aroP",
               "TnaB" : "tnaB"}

# statistics parameters (ANOVA / t-test)
N_TIERPSY_FEATS = 16

# no significant features found for any strain when multiple test correction was applied, 
# so strains were ranked by lowest p-value (uncorrected for multiple comparisons)
FDR_METHOD = 'fdr_bh'
P_VALUE_THRESHOLD = 0.05 # p-value threshold

# nearest neighbour analysis parameters
N_NEIGHBOURS = 3 # number of nearest neighbours to each hit strain to include/expand strain list
LINKAGE_METHOD = 'average' # clustering linkage method (scipy.cluster.hierarchy.linkage)
DISTANCE_METRIC = 'euclidean' # clustering distance metric (scipy.spatial.distance.pdist)
    
#%% Functions

def average_plate_control_data(metadata, features, control='wild_type', grouping_var='gene_name', 
                               plate_var='imaging_plate_id'):
    """ Average data for control plate on each experiment day to yield a single mean datapoint for 
        the control. This reduces the control sample size to equal the test strain sample size, for 
        t-test comparison. Information for the first well in the control sample on each day is used 
        as the accompanying metadata for mean feature results. 
        
        Input
        -----
        features, metadata : pd.DataFrame
            Feature summary results and metadata dataframe with multiple entries per day
            
        Returns
        -------
        features, metadata : pd.DataFrame
            Feature summary results and metadata with control data averaged (single sample per day)
    """
        
    # Subset results for control data
    control_metadata = metadata[metadata[grouping_var]==control]
    control_features = features.reindex(control_metadata.index)

    # Take mean of control for each plate = collapse to single datapoint for strain comparison
    mean_control = control_metadata[[grouping_var, plate_var]].join(control_features).groupby(
                                    by=[grouping_var, plate_var]).mean().reset_index()
    
    # Append remaining control metadata column info (with first well data for each date)
    remaining_cols = [c for c in control_metadata.columns.to_list() if c not in [grouping_var, 
                                                                                 plate_var]]
    mean_control_row_data = []
    for i in mean_control.index: # use .agg() here?
        plate = mean_control.loc[i, plate_var]
        control_plate_meta = control_metadata.loc[control_metadata[plate_var] == plate]
        first_well = control_plate_meta.loc[control_plate_meta.index[0], remaining_cols]
        first_well_mean = first_well.append(mean_control.loc[mean_control[plate_var] == plate
                                                             ].squeeze(axis=0))
        mean_control_row_data.append(first_well_mean)
    
    control_mean = pd.DataFrame.from_records(mean_control_row_data)
    control_metadata = control_mean[control_metadata.columns.to_list()]
    control_features = control_mean[control_features.columns.to_list()]

    features = pd.concat([features.loc[metadata[grouping_var] != control, :], 
                          control_features], axis=0).reset_index(drop=True)        
    metadata = pd.concat([metadata.loc[metadata[grouping_var] != control, :], 
                          control_metadata.loc[:, metadata.columns.to_list()]], 
                          axis=0).reset_index(drop=True)
    
    assert all(metadata.index == features.index)

    return metadata, features

def stats(metadata,
          features,
          group_by='gene_name',
          control='wild_type',
          feature_list=None,
          save_dir=None,
          p_value_threshold=0.05,
          fdr_method='fdr_bh',
          n_tierpsy_feats=N_TIERPSY_FEATS):
    
    """ Perform ANOVA tests to compare worms on each bacterial food vs BW25113 control for each 
        Tierpsy feature in feature_list """

    assert all(metadata.index == features.index)
    
    if feature_list is None:
        feature_list = list(features.columns)
    else:
        assert type(feature_list) == list
        assert all(f in features.columns for f in feature_list)
        features = features[feature_list]
        
    n_strains = metadata[group_by].nunique()
    print("Performing ANOVA tests for variation among %d %ss for %d features" %\
          (n_strains, group_by, len(feature_list)))
    
    # perform ANOVA + record results before & after correcting for multiple comparisons               
    stats, pvals, reject = univariate_tests(X=features, 
                                            y=metadata[group_by], 
                                            control=control, 
                                            test='ANOVA',
                                            comparison_type='multiclass',
                                            multitest_correction=None, # uncorrected
                                            alpha=p_value_threshold,
                                            n_permutation_test=None)

    # get effect sizes
    effect_sizes = get_effect_sizes(X=features, 
                                    y=metadata[group_by],
                                    control=control,
                                    effect_type=None,
                                    linked_test='ANOVA')

    # compile ANOVA results (uncorrected)
    anova_results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
    anova_results.columns = ['stats','effect_size','pvals','reject']     
    anova_results['significance'] = sig_asterix(anova_results['pvals'])
    anova_results = anova_results.sort_values(by=['pvals'], ascending=True) # rank pvals
    
    # save ANOVA results (uncorrected)
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        results_path = Path(save_dir) /\
            'Tierpsy{}_ANOVA_results_uncorrected.csv'.format(n_tierpsy_feats)
        anova_results.to_csv(results_path, header=True, index=True)

    # correct for multiple comparisons
    if fdr_method is not None:
        reject, pvals = _multitest_correct(pvals, 
                                           multitest_method=fdr_method,
                                           fdr=p_value_threshold)
                                                
        # compile ANOVA results (corrected)
        anova_results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
        anova_results.columns = ['stats','effect_size','pvals','reject']     
        anova_results['significance'] = sig_asterix(anova_results['pvals'])
        anova_results = anova_results.sort_values(by=['pvals'], ascending=True) # rank pvals
        
        # save ANOVA results (corrected)
        if save_dir is not None:
            results_corrected_path = Path(save_dir) /\
                'Tierpsy{}_ANOVA_results_corrected.csv'.format(n_tierpsy_feats)
            anova_results.to_csv(results_corrected_path, header=True, index=True)
            
    # t-tests comparing each bacterial strain to wild-type control
    print("Performing t-tests comparing %d %ss to BW25113 wild-type control (%d features)" %\
          (n_strains, group_by, len(feature_list)))
    stats_t, pvals_t, reject_t = univariate_tests(X=features,
                                                  y=metadata[group_by],
                                                  control=control,
                                                  test='t-test',
                                                  comparison_type='binary_each_group',
                                                  multitest_correction=None, # uncorrected
                                                  alpha=p_value_threshold)

    effect_sizes_t = get_effect_sizes(X=features,
                                      y=metadata[group_by],
                                      control=control,
                                      linked_test='t-test')
    
    # compile t-test results (uncorrected)
    stats_t.columns = ['stats_' + str(c) for c in stats_t.columns]
    pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
    reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
    effect_sizes_t.columns = ['effect_size_' + str(c) for c in effect_sizes_t.columns]
    ttest_results = pd.concat([stats_t, pvals_t, reject_t, effect_sizes_t], axis=1)
    
    # save t-test results (uncorrected)
    if save_dir is not None:
        ttest_results_path = Path(save_dir) /\
            'Tierpsy{}_t-test_results_uncorrected.csv'.format(n_tierpsy_feats)
        ttest_results.to_csv(ttest_results_path, header=True, index=True)
    
    # correct for multiple comparisons
    if fdr_method is not None:
        pvals_t.columns = [c.split("_")[-1] for c in pvals_t.columns]
        reject_t, pvals_t = _multitest_correct(pvals_t, 
                                               multitest_method=fdr_method,
                                               fdr=p_value_threshold)
        
        # compile t-test results (corrected)
        pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
        reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
        ttest_results = pd.concat([stats_t, effect_sizes_t, pvals_t, reject_t], axis=1)
        
        # save t-test results (corrected)
        if save_dir is not None:
            ttest_results_corrected_path = Path(save_dir) /\
                'Tierpsy{}_t-test_results_corrected.csv'.format(n_tierpsy_feats)
            ttest_results.to_csv(ttest_results_corrected_path, header=True, index=True)            
        
    return anova_results, ttest_results

def clean_data():
    
    tic = time()
        
    metadata_path_local = Path(SAVE_DIR) / 'metadata.csv'
    features_path_local = Path(SAVE_DIR) / 'features.csv'
    
    aux_dir = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    res_dir = Path(PROJECT_DIR) / 'Results'
    
    if not metadata_path_local.exists() or not features_path_local.exists():
        print("Cleaning metadata and feature summaries")

        # compile metadata and feature summaries        
        metadata, metadata_path = compile_metadata(aux_dir, 
                                                   imaging_dates=EXPERIMENT_DATES,
                                                   n_wells=N_WELLS, 
                                                   add_well_annotations=True,
                                                   from_source_plate=False)
        
        features, metadata = process_feature_summaries(metadata_path, 
                                                       results_dir=res_dir, 
                                                       compile_day_summaries=True, 
                                                       imaging_dates=EXPERIMENT_DATES, 
                                                       align_bluelight=True, 
                                                       window_summaries=False,
                                                       n_wells=N_WELLS)

        # fill in 'gene_name' for control as 'wild_type' (for control plates where gene name is missing)
        metadata.loc[metadata['source_plate_id'] == "BW", 'gene_name'] = "wild_type"

        # rename gene names in metadata
        for k, v in RENAME_DICT.items():
            metadata.loc[metadata['gene_name'] == k, 'gene_name'] = v

        # clean results - Remove bad well data + features with too many NaNs/zero std + impute NaNs
        features, metadata = clean_summary_results(features,
                                                   metadata,
                                                   feature_columns=None,
                                                   nan_threshold_row=NAN_THRESHOLD_ROW,
                                                   nan_threshold_col=NAN_THRESHOLD_COL,
                                                   max_value_cap=1e15,
                                                   imputeNaN=True,
                                                   min_nskel_per_video=None,
                                                   min_nskel_sum=MIN_NSKEL_SUM,
                                                   drop_size_related_feats=False,
                                                   norm_feats_only=False,
                                                   percentile_to_use=None,
                                                   no_nan_cols=['worm_strain','gene_name'])
        
        assert not any(metadata['worm_strain'].isna()) and not any(metadata['gene_name'].isna())
        
        # assert no duplicate genes due to case-sensitivity
        assert len(metadata['gene_name'].unique()) == len(metadata['gene_name'].str.upper().unique())
        
        # add COG category info from Baba et al. (2006) supplementary info to metadata
        if not 'COG_category' in metadata.columns:
            supplementary_7 = load_supplementary_7(SUPPLEMENTARY_INFO_PATH)
            metadata = append_supplementary_7(metadata, supplementary_7, column_name='gene_name')
            assert set(metadata.index) == set(features.index)    
        COG_families = {'Information storage and processing' : ['J', 'K', 'L', 'D', 'O'], 
                        'Cellular processes' : ['M', 'N', 'P', 'T', 'C', 'G', 'E'], 
                        'Metabolism' : ['F', 'H', 'I', 'Q', 'R'], 
                        'Poorly characterised' : ['S', 'U', 'V']}
        COG_mapping_dict = {i : k for (k, v) in COG_families.items() for i in v}        
        COG_info = []
        for i in metadata['COG_category']:
            try:
                COG_info.append(COG_mapping_dict[i])
            except:
                COG_info.append('Unknown') # np.nan
        metadata['COG_info'] = COG_info
        
        # save clean metadata and features
        metadata.to_csv(metadata_path_local, index=False)
        features.to_csv(features_path_local, index=False)
        
    else:
        print("Found existing clean metadata and feature summaries")
        metadata = pd.read_csv(metadata_path_local, header=0, index_col=None, dtype={'comments':str})
        features = pd.read_csv(features_path_local, header=0, index_col=None)
        
    print("Done in %.1f seconds" % (time()-tic))    
        
    return metadata, features

def find_hits(metadata, features, n_tierpsy_feats, fdr_method, average_control_per_day=True):

    tic = time()
    print("Finding hit strains (Tierpsy{0}, {1}, p<{2})".format(
        n_tierpsy_feats, fdr_method, P_VALUE_THRESHOLD))

    assert not features.isna().sum(axis=1).any() and not (features.std(axis=1) == 0).any()        
    assert not any(metadata['worm_strain'].isna()) and not any(metadata['gene_name'].isna())
        
    # subset for Tierpsy feature set
    features = select_feat_set(features,
                               tierpsy_set_name='tierpsy_{}'.format(n_tierpsy_feats), 
                               append_bluelight=True)        
    
    # average control data for each experiment day
    if average_control_per_day:
        metadata, features = average_plate_control_data(metadata,
                                                        features,
                                                        control='wild_type',
                                                        grouping_var='gene_name', 
                                                        plate_var='imaging_plate_id')
    
    stats_dir = Path(SAVE_DIR) / 'Stats' / 'Tierpsy{0}_{1}'.format(n_tierpsy_feats, fdr_method)
    plots_dir = Path(SAVE_DIR) / 'Plots' / 'Tierpsy{0}_{1}'.format(n_tierpsy_feats, fdr_method)
    
    strain_list = sorted(metadata['gene_name'].unique())
    print("%d bacterial strains in total will be analysed" % len(strain_list))
    
    # perform ANOVA for each feature and correct for multiple comparisons
    anova_results, ttest_results = stats(metadata,
                                         features,
                                         group_by='gene_name',
                                         control='wild_type',
                                         feature_list=list(features.columns),
                                         save_dir=stats_dir,
                                         p_value_threshold=P_VALUE_THRESHOLD,
                                         fdr_method=fdr_method,
                                         n_tierpsy_feats=n_tierpsy_feats)
    
    # use reject mask to find significant feature list (ANOVA)
    # 33/48 significant features found by ANOVA (p<0.05, fdr_bh)
    fset = anova_results['pvals'].loc[anova_results['reject']
                                      ].sort_values(ascending=True).index.to_list()
    print("%d/%d significant features found by ANOVA (P<%.2f, %s)" %\
          (len(fset), features.shape[1], P_VALUE_THRESHOLD, fdr_method))
    Path(stats_dir).mkdir(parents=True, exist_ok=True)
    sigfeats_path = stats_dir / 'Tierpsy{}_ANOVA_sigfeats.txt'.format(n_tierpsy_feats)
    write_list_to_file(fset, sigfeats_path)
    
    # extract p-values for t-tests
    pvals_t = ttest_results[[c for c in ttest_results.columns if 'pvals_' in c]]
    pvals_t.columns = [c.split('pvals_')[-1] for c in pvals_t.columns]  

    # count number of features that are significant for at least one strain
    # 0/48 significant features found by t-test (p<0.05, fdr_bh)
    fset_t = pvals_t[(pvals_t < P_VALUE_THRESHOLD).sum(axis=1) > 0].index.to_list()
    print("%d/%d significant features found by t-test (P<%.2f, %s)" %\
          (len(fset_t), features.shape[1], P_VALUE_THRESHOLD, fdr_method))
    sigfeats_t_path = stats_dir / 'Tierpsy{}_t-test_sigfeats.txt'.format(n_tierpsy_feats)
    write_list_to_file(fset_t, sigfeats_t_path)
    
# =============================================================================
#     # rank strains by number of sigfeats (t-test)
#     reject_t = ttest_results[[c for c in ttest_results.columns if 'reject_' in c]]
#     reject_t.columns = [c.split('reject_')[-1] for c in reject_t.columns]
#     ranked_nsig = reject_t.sum(axis=0).sort_values(ascending=False)
#     ranked_nsig_path = stats_dir / 'Tierpsy{}_ranked_strains_nsig.csv'.format(n_tierpsy_feats)
#     ranked_nsig.to_csv(ranked_nsig_path, header=True, index=True)
#     hit_strains_nsig = ranked_nsig[ranked_nsig > 0].index.to_list()
#     print("%d strains with 1 or more significant features" % len(hit_strains_nsig))
#     hit_strains_nsig_path = stats_dir /\
#        'Tierpsy{}_top100_hits_ranked_by_most_sigfeats.txt'.format(n_tierpsy_feats)
#     write_list_to_file(hit_strains_nsig[:100], hit_strains_nsig_path)
# =============================================================================

    # rank strains by lowest p-value for any feature (t-test, uncorrected p-values)
    ttest_results_uncorrected_path = Path(stats_dir) /\
        'Tierpsy{}_t-test_results_uncorrected.csv'.format(n_tierpsy_feats)
    ttest_results_uncorrected = pd.read_csv(ttest_results_uncorrected_path, header=0, index_col=0)
    pvals_t_uncorrected = ttest_results_uncorrected[[c for c in ttest_results_uncorrected.columns
                                                    if 'pvals_' in c]]
    pvals_t_uncorrected.columns = [c.split('pvals_')[-1] for c in pvals_t_uncorrected.columns]    
    ranked_pval = pvals_t_uncorrected.min(axis=0).sort_values(ascending=True)
    ranked_pval_path = stats_dir / 'Tierpsy{}_ranked_strains_pval.csv'.format(n_tierpsy_feats)
    ranked_pval.to_csv(ranked_pval_path, header=True, index=True)
            
    # plot strains ranked by: (1) number of significant features, and (2) lowest p-value of any feature
    plt.close('all')
    sns.set_style('ticks')
# =============================================================================
#     fig, (ax1, ax2) = plt.subplots(2, 1, sharey=False, figsize=[150,10])
#     sns.lineplot(x=ranked_nsig.index,
#                  y=ranked_nsig.values,
#                  ax=ax1, linewidth=0.7)
#     ax1.set_xticklabels(ranked_nsig.index.to_list(), rotation=90, fontsize=2)
#     ax1.xaxis.set_tick_params(width=0.3, pad=0.5)
#     ax1.set_ylabel('Number of significant features', fontsize=12, labelpad=10)
#     ax1.set_xlim(-5, len(strain_list)+5)
# =============================================================================
    fig, ax2 = plt.subplots(1, 1, figsize=[150,10])
    sns.lineplot(x=ranked_pval.index,
                 y=-np.log10(ranked_pval.values),
                 ax=ax2, linewidth=0.7)
    ax2.set_xticklabels(ranked_pval.index.to_list(), rotation=90, fontsize=2)
    ax2.xaxis.set_tick_params(width=0.3, pad=0.5)
    #ax2.axhline(y=-np.log10(P_VALUE_THRESHOLD), c='dimgray', ls='--', lw=0.7)
    ax2.set_xlabel('Strains (ranked)', fontsize=12, labelpad=10)
    ax2.set_ylabel('-log10 p-value (uncorrected) of most important feature', fontsize=12, labelpad=10)
    ax2.set_xlim(-5, len(strain_list)+5)
    fig.align_xlabels()

    # save figure
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig_save_path = plots_dir /\
        "Tierpsy{}_ranked_hits_lowest_p-value_t-test_uncorrected.svg".format(n_tierpsy_feats)
    plt.savefig(fig_save_path, bbox_inches='tight', transparent=True)
    
    # select hits - top 100 strains with lowest p-value for any feature (Tierpsy16, uncorrected)
    hit_strains_pval = ranked_pval[:100].index.to_list()
    hit_strains_pval_path = stats_dir /\
        'Tierpsy{}_top100_hits_lowest_p-value_t-test_uncorrected.txt'.format(n_tierpsy_feats)
    write_list_to_file(hit_strains_pval, hit_strains_pval_path)
    
    print("Done in %.1f seconds" % (time()-tic))    
    
    return hit_strains_pval

def nearest_neighbour_cluster_analysis():
    
    """ Nearest neighbour analysis of 59 hit strains manually curated from top 100 strains ranked 
        by lowest p-value (uncorrected) of any feature by t-test, to expand the list of candidate
        strains to inclide the 3 nearest neightbour strains to each hit strain that elicit similar
        worm behaviour. The list of strains which will be used to find the nearest neighbours to 
        these 59 hits is the 1860 strains found to be significant by t-test using Tierpsy256 feature
        set (p<0.05, fdr_bh). The Euclidean distance between these 1860 strains in phenotype space 
        is used to find the 3 nearest neigbour strains to each of the 59 hits, thus expanding the 
        list of hit strains to yeild a total of 229 candidate strains for the confirmation screen.
    """
    tic = time()

    metadata_path_local = Path(SAVE_DIR) / 'metadata.csv'
    features_path_local = Path(SAVE_DIR) / 'features.csv'
    
    # load clean metadata and features summaries
    metadata = pd.read_csv(metadata_path_local, header=0, index_col=None, dtype={'comments':str})
    features = pd.read_csv(features_path_local, header=0, index_col=None)

    stats_dir = Path(SAVE_DIR) / 'Stats'

    # load set of 59 hit strains curated from top 100 strains with lowest p-value for any feature 
    # (Tierpsy16, fdr_bh)
    manually_curated_hits_path_local = stats_dir /\
        '59_manually_curated_hits_from_top100_ranked_lowest_uncorrected_p-value_for_any_Tierpsy16_feature.txt'
    selected_strain_list = read_list_from_file(manually_curated_hits_path_local)

    # before performing nearest neighbour cluster analysis, first identify the list of significant 
    # strains which will be used to select nearest neighbours from (Tierpsy256, p<0.05, fdr_bh)
    # (n=1860 strains)
    n_tierpsy_feats_nn = 256
    fdr_method_nn = 'fdr_bh'
    _ = find_hits(metadata, 
                  features, 
                  n_tierpsy_feats_nn, # Tierpsy256 used to obtain list of strains to search for nearest neighbours
                  fdr_method_nn, # Benjamini-Hochberg correction used to correct for multiple comparisons
                  average_control_per_day=True)
        
    # TODO: Work out why this does not yield 1860 strains that are significant for at least 1 feature (Tierpsy256, fdr_bh)
    
    # load list of strains from which to find nearest neighbours 
    ttest_results_path = stats_dir / "Tierpsy256_fdr_bh" / "Tierpsy256_t-test_results_corrected.csv"
    ttest_results = pd.read_csv(ttest_results_path, header=0, index_col=0)
    pvals_t = ttest_results[[c for c in ttest_results.columns if 'pvals_' in c]]
    pvals_t.columns = [c.split('pvals_')[-1] for c in pvals_t.columns]  
    
    # number of strains that are significant for at least one feature
    nsig_nn = ((pvals_t < 0.05).sum(axis=0).sort_values(ascending=False)>0).sum()
    print("%d strains significant for at least 1 feature (t-test, Tierpsy%d, %s, p<0.05)" %\
          (nsig_nn, n_tierpsy_feats_nn, fdr_method_nn))
    
    # print("%d/%d significant features found by t-test (P<%.2f, %s)" %\
    #       (len(fset_t), features.shape[1], P_VALUE_THRESHOLD, fdr_method))
    # sigfeats_t_path = stats_dir / 'Tierpsy{}_t-test_sigfeats.txt'.format(n_tierpsy_feats)
        
    # subset for Tierpsy256 feature set
    features = select_feat_set(features,
                               tierpsy_set_name='tierpsy_256', 
                               append_bluelight=True) 
            
    n_strains, n_feats = metadata['gene_name'].nunique(), len(features.columns)
    
    cluster_dir = stats_dir / 'nearest_neighbour_cluster_analysis'
    save_dir = cluster_dir / ("%d_strains_%d_features" % (n_strains, n_feats))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    ##### Hierarchical clustering #####

    # Cluster linkage array
    print("Computing cluster linkage array...")
    Z, X = cluster_linkage_seaborn(features, 
                                   metadata, 
                                   groupby='gene_name', 
                                   saveDir=save_dir, 
                                   method=LINKAGE_METHOD, 
                                   metric=DISTANCE_METRIC)
    _Z, _X = cluster_linkage_pdist(features, 
                                   metadata, 
                                   groupby='gene_name',
                                   saveDir=save_dir, 
                                   method=LINKAGE_METHOD, 
                                   metric=DISTANCE_METRIC)
    
    # Assert that the two methods are identical
    assert np.allclose(Z, _Z)

    # Compare pairwise distances between all samples to hierarchical clustering distances 
    # The closer the value to 1 the better the clustering preserves original distances
    c, coph_dists = cophenet(Z, pdist(X)); print("Cophenet: %.3f" % c)

    # Find nearest neighbours by ranking the computed sqaureform distance matrix between all strains
    names_df, distances_df = nearest_neighbours(X=X,
                                                strain_list=None,
                                                distance_metric=DISTANCE_METRIC, 
                                                saveDir=save_dir)
        
    # Record confirmation screen hit strains in/not in initial hit strain list
    assert all(s in names_df.index for s in selected_strain_list)
    neighbour_names_df = names_df.loc[selected_strain_list, 1:N_NEIGHBOURS]
    neighbour_names_df.to_excel(Path(save_dir) / "nearest_{}_neighbours.xlsx".format(N_NEIGHBOURS))
    
    # Save N nearest neighbours list
    neighbour_list = np.unique(np.asmatrix(neighbour_names_df).flatten().tolist())
    neighbour_list = sorted(set(neighbour_list) - set(selected_strain_list))
    write_list_to_file(neighbour_list, Path(save_dir) / "neighbour_strains.txt")

    new_strain_list = sorted(set(neighbour_list).union(set(selected_strain_list))) # n=205
    atp_genes = [s for s in names_df.index if s.startswith('atp') and s not in new_strain_list]
    nuo_genes = [s for s in names_df.index if s.startswith('nuo') and s not in new_strain_list]
    extra_strains = ['fiu','fhuE','fhuA','tonB','exbD','exbB','entA','entB','entC','entE','entF',
                     'fes','cirA']
    extra_strain_list = atp_genes + nuo_genes + extra_strains
    assert not any(s in new_strain_list for s in extra_strain_list)
    write_list_to_file(extra_strain_list, Path(save_dir) / "extra_strains_added.txt")

    new_strain_list.extend(extra_strain_list)

    # Save expanded list of hit strains for confirmational screen, including the N closest strains 
    # to each hit strain selected from the initial screen
    print("Saving new hit strain list of %d genes to file" % len(new_strain_list))
    write_list_to_file(sorted(new_strain_list), Path(save_dir) /\
                       "{}_selected_strains_for_confirmation_screen.txt".format(len(new_strain_list)))
       
    print("Done in %.1f seconds" % (time()-tic))

    return sorted(new_strain_list)

def main():
    
    # clean metadata and feature summaries
    metadata, features = clean_data()
    
    # find hit strains (top 100 strains with lowest p-value of any feature)
    top100_hits_ranked_lowest_pval_uncorrected = find_hits(metadata, 
                                                           features, 
                                                           N_TIERPSY_FEATS, 
                                                           FDR_METHOD,
                                                           average_control_per_day=True)
    
    # no significant features found for any strain when multiple test correction was applied, 
    # so strains were ranked by lowest p-value (uncorrected for multiple comparisons)
    
    # manually added strains (n=18)
    manually_added_strains_list = ['aroP','fepA','fepB','fepC','nuoE','nuoG','nuoH','nuoI','nuoJ',
                                   'nuoL','nuoM','nuoN','tnaB','trpA','trpB','trpC','trpD','trpE']
    
    # manually dropped strains (n=59)
    manually_dropped_strains_list = ['aceK','argC','artQ','bcr','clpX','cpxA','cysC','dps','eamA',
                                     'gadX','galU','glvG','gmr','hisC','hypB','kefA','lpd','mhpF',
                                     'mntR','mtlA','mtlR','mutH','narK','npr','nusB','oppF','paaE',
                                     'pabC','potD','proV','rbfA','rffD','rseA','rzpR','sapF','smpA',
                                     'thiI','tufA','ybdZ','ybhE','ybjE','ydcE','ydfX','yfeG','yfeK',
                                     'yfiB','yfiP','ygaD','ygbK','ygfF','yhdP','yhhX','yjbO','ykfH',
                                     'yncN','yoeA','yphH','yqjG','ytfB']
    
    # 59 hit strains after manual curation (59 strains dropped, 18 strains added)    
    manually_curated_59_initial_hits = sorted(
        set(top100_hits_ranked_lowest_pval_uncorrected).union(
            set(manually_added_strains_list)).difference(
                set(manually_dropped_strains_list)))

    # reconcile hit strains with original curated list of hits (c. 2019) - exact match
    # curated_hits = read_list_from_file(CURATED_HIT_STRAINS_PATH)
    # manually_added_strains = set(curated_hits) - set(top100_hits_ranked_lowest_pval_uncorrected)
    # manually_dropped_strains = set(top100_hits_ranked_lowest_pval_uncorrected) - set(curated_hits)
    # assert set(manually_curated_59_initial_hits) == set(curated_hits)
    # assert (len(top100_hits_ranked_lowest_pval_uncorrected) # n=100
    #         - len(manually_dropped_strains_list) # n=59
    #         + len(manually_added_strains_list) # n=18
    #         == len(curated_hits)) # n=59         

    # save the manually curated list of initial hit strains to find nearest neighbours to
    manually_curated_hits_path_local = Path(SAVE_DIR) / 'Stats' /\
        '59_manually_curated_hits_from_top100_ranked_lowest_uncorrected_p-value_for_any_Tierpsy16_feature.txt'
    write_list_to_file(manually_curated_59_initial_hits, manually_curated_hits_path_local)
    
    # find nearest 3 neighbours to each curated hit strain to expand list of hit genes (Tierpsy256)
    candidate_hits_for_confirmation_screen = nearest_neighbour_cluster_analysis()
    print("%d candidate strains identified for confirmation screen." %\
          len(candidate_hits_for_confirmation_screen))

    return

#%% Main

if __name__ == '__main__':
    main()
