#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse Initial Keio Screen 

- compile metadata and feature summaries
- clean metadata and feature summaries
- run analysis

@author: sm5911
@date: 28/11/2024

"""

#%% Imports

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt

from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from preprocessing.append_supplementary_info import load_supplementary_7, append_supplementary_7
from filter_data.clean_feature_summaries import clean_summary_results
from tierpsytools.preprocessing.filter_data import select_feat_set
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes, _multitest_correct
from visualisation.plotting_helper import sig_asterix
from write_data.write import write_list_to_file

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Screen_Initial"
SAVE_DIR = "/Users/sm5911/Documents/PhD_DLBG/3_Keio_Screen_Initial"

PATH_SUP_INFO = Path(PROJECT_DIR) / "AuxiliaryFiles/Baba_et_al_2006/Supporting_Information/Supplementary_Table_7.xls"

EXPERIMENT_DATES = ["20210406", "20210413", "20210420", "20210427", "20210504", "20210511"]

N_WELLS = 96
NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
MIN_NSKEL_SUM = 6000

N_TIERPSY_FEATS = 16 #256, None
TEST = 'ANOVA' # 'Kruskal-Wallis'
FDR_METHOD = 'fdr_bh' # fdr_by
P_VALUE_THRESHOLD = 0.05
TOP_N_HITS = 100

RENAME_DICT = {"FECE" : "fecE",
               "AroP" : "aroP",
               "TnaB" : "tnaB"}

#%% Functions

def stats(metadata,
          features,
          group_by='gene_name',
          control='wild_type',
          test='ANOVA',
          feature_list=None,
          save_dir=None,
          p_value_threshold=0.05,
          fdr_method='fdr_bh',
          n_tierpsy_feats=None):
    
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
    print("Performing %s tests for variation among %d %ss for %d features" %\
          (test, n_strains, group_by, len(feature_list)))
    
    # perform ANOVA + record results before & after correcting for multiple comparisons               
    stats, pvals, reject = univariate_tests(X=features, 
                                            y=metadata[group_by], 
                                            control=control, 
                                            test=test,
                                            comparison_type='multiclass',
                                            multitest_correction=None, # uncorrected
                                            alpha=p_value_threshold,
                                            n_permutation_test=None)

    # get effect sizes
    effect_sizes = get_effect_sizes(X=features, 
                                    y=metadata[group_by],
                                    control=control,
                                    effect_type=None,
                                    linked_test=test)

    # compile ANOVA results (uncorrected)
    anova_results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
    anova_results.columns = ['stats','effect_size','pvals','reject']     
    anova_results['significance'] = sig_asterix(anova_results['pvals'])
    anova_results = anova_results.sort_values(by=['pvals'], ascending=True) # rank pvals
    
    # save ANOVA results (uncorrected)
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        results_path = Path(save_dir) / 'Tierpsy{0}_{1}_results_uncorrected.csv'.format(n_tierpsy_feats, test)
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
            results_corrected_path = Path(save_dir) / 'Tierpsy{0}_{1}_results_corrected.csv'.format(n_tierpsy_feats, test)
            anova_results.to_csv(results_corrected_path, header=True, index=True)
            
    # t-tests comparing each bacterial strain to wild-type control
    test_t = 't-test' if test == 'ANOVA' else 'Mann-Whitney test'
    stats_t, pvals_t, reject_t = univariate_tests(X=features,
                                                  y=metadata[group_by],
                                                  control=control,
                                                  test=test_t,
                                                  comparison_type='binary_each_group',
                                                  multitest_correction=None, # uncorrected
                                                  alpha=p_value_threshold)

    effect_sizes_t = get_effect_sizes(X=features,
                                      y=metadata[group_by],
                                      control=control,
                                      linked_test=test_t)
    
    # compile t-test results (uncorrected)
    stats_t.columns = ['stats_' + str(c) for c in stats_t.columns]
    pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
    reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
    effect_sizes_t.columns = ['effect_size_' + str(c) for c in effect_sizes_t.columns]
    ttest_results = pd.concat([stats_t, pvals_t, reject_t, effect_sizes_t], axis=1)
    
    # save t-test results (uncorrected)
    if save_dir is not None:
        ttest_results_path = Path(save_dir) / 'Tierpsy{0}_{1}_results_uncorrected.csv'.format(n_tierpsy_feats, test_t)
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
            ttest_results_corrected_path = Path(save_dir) / 'Tierpsy{0}_{1}_results_corrected.csv'.format(n_tierpsy_feats, test_t)
            ttest_results.to_csv(ttest_results_corrected_path, header=True, index=True)            
        
    return anova_results, ttest_results

def main():
    
    metadata_path_local = Path(SAVE_DIR) / 'metadata.csv'
    features_path_local = Path(SAVE_DIR) / 'features.csv'
    
    aux_dir = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    res_dir = Path(PROJECT_DIR) / 'Results'
    
    if not metadata_path_local.exists() or not features_path_local.exists():

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
            supplementary_7 = load_supplementary_7(PATH_SUP_INFO)
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
        metadata = pd.read_csv(metadata_path_local, header=0, index_col=None, dtype={'comments':str})
        features = pd.read_csv(features_path_local, header=0, index_col=None)
        
    # subset for Tierpsy feature set
    n_tierpsy_feats = 'All' if N_TIERPSY_FEATS is None else N_TIERPSY_FEATS
    features = select_feat_set(features,
                               tierpsy_set_name='tierpsy_{}'.format(n_tierpsy_feats), 
                               append_bluelight=True)        
    
    stats_dir = Path(SAVE_DIR) / 'Stats'
    plots_dir = Path(SAVE_DIR) / 'Plots'
    
    strain_list = sorted(metadata['gene_name'].unique())
    print("%d bacterial strains in total will be analysed" % len(strain_list))
    
    # perform ANOVA for each feature and correct for multiple comparisons
    anova_results, ttest_results = stats(metadata,
                                         features,
                                         group_by='gene_name',
                                         control='wild_type',
                                         test=TEST,
                                         feature_list=list(features.columns),
                                         save_dir=stats_dir,
                                         p_value_threshold=P_VALUE_THRESHOLD,
                                         fdr_method=FDR_METHOD,
                                         n_tierpsy_feats=n_tierpsy_feats)
    
    # use reject mask to find significant feature list (ANOVA)
    fset = anova_results['pvals'].loc[anova_results['reject']].sort_values(ascending=True).index.to_list()
    print("%d significant features found by %s (P<%.2f, %s)" % (len(fset), TEST, P_VALUE_THRESHOLD, FDR_METHOD))
    Path(stats_dir).mkdir(parents=True, exist_ok=True)
    sigfeats_path = stats_dir / 'Tierpsy{0}_{1}_sigfeats.txt'.format(n_tierpsy_feats, TEST)
    write_list_to_file(fset, sigfeats_path)
    
    # rank strains by number of sigfeats (t-test)
    reject_t = ttest_results[[c for c in ttest_results.columns if 'reject_' in c]]
    reject_t.columns = [c.split('reject_')[-1] for c in reject_t.columns]
    ranked_nsig = reject_t.sum(axis=0).sort_values(ascending=False)
    hit_strains_nsig = ranked_nsig[ranked_nsig > 0].index.to_list()
    print("%d strains with 1 or more significant features" % len(hit_strains_nsig))
    hit_strains_nsig_path = stats_dir / 'Tierpsy{0}_top{1}_hits_ranked_by_most_sigfeats.txt'.format(n_tierpsy_feats, TOP_N_HITS)
    write_list_to_file(hit_strains_nsig[:TOP_N_HITS], hit_strains_nsig_path)
    
    # rank strains by lowest p-value for any feature (t-test)
    pvals_t = ttest_results[[c for c in ttest_results.columns if 'pvals_' in c]]
    pvals_t.columns = [c.split('pvals_')[-1] for c in pvals_t.columns]
    ranked_pval = pvals_t.min(axis=0).sort_values(ascending=True)
    hit_strains_pval = ranked_pval[ranked_pval < P_VALUE_THRESHOLD].index.to_list()
    hit_strains_pval_path = stats_dir / 'Tierpsy{0}_top{1}_hits_ranked_by_lowest_pval.txt'.format(n_tierpsy_feats, TOP_N_HITS)
    write_list_to_file(hit_strains_pval[:TOP_N_HITS], hit_strains_pval_path)
    
    assert all(s in hit_strains_pval for s in hit_strains_nsig)
    
    # plot strains ranked by: (1) number of significant features, and (2) lowest p-value of any feature
    plt.close('all')
    sns.set_style('ticks')
    fig, (ax1, ax2) = plt.subplots(2, 1, sharey=False, figsize=[150,10])
    sns.lineplot(x=ranked_nsig.index,
                 y=ranked_nsig.values,
                 ax=ax1, linewidth=0.7)
    ax1.set_xticklabels(ranked_nsig.index.to_list(), rotation=90, fontsize=2)
    ax1.xaxis.set_tick_params(width=0.3, pad=0.5)
    ax1.set_ylabel('Number of significant features', fontsize=12, labelpad=10)
    ax1.set_xlim(-5, len(strain_list)+5)
    sns.lineplot(x=ranked_pval.index,
                 y=-np.log10(ranked_pval.values),
                 ax=ax2, linewidth=0.7)
    ax2.set_xticklabels(ranked_pval.index.to_list(), rotation=90, fontsize=2)
    ax2.xaxis.set_tick_params(width=0.3, pad=0.5)
    ax2.axhline(y=P_VALUE_THRESHOLD, c='dimgray', ls='--', lw=0.7)
    ax2.set_xlabel('Strains (ranked)', fontsize=12, labelpad=10)
    ax2.set_ylabel('-log10 p-value of most significant feature', fontsize=12, labelpad=10)
    ax2.set_xlim(-5, len(strain_list)+5)
    fig.align_xlabels()

    # save figure
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig_save_path = plots_dir / "Tierpsy{}_ranked_hit_strains_t-test.svg".format(N_TIERPSY_FEATS)
    plt.savefig(fig_save_path, bbox_inches='tight', transparent=True)

    
    
    
    
    return

#%% Main

if __name__ == '__main__':
    main()
