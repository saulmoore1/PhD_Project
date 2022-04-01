#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keio Mutant Worm Screen - Response to BW vs fepD bacteria


@author: sm5911
@date: 30/03/2022

"""

#%% Imports

import pandas as pd
from pathlib import Path
from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from statistical_testing.stats_helper import single_feature_window_stats
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Keio_Mutant_Worm_6WP"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Mutant_Worm"

IMAGING_DATES = ['20220305','20220314','20220321']
N_WELLS = 6

FEATURE = 'motion_mode_paused_fraction'

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
MIN_NSKEL_PER_VIDEO = None
MIN_NSKEL_SUM = 500
PVAL_THRESH = 0.05

WINDOW_DICT_SECONDS = {0:(1790,1800), 1:(1805,1815), 2:(1815,1825),
                       3:(1850,1860), 4:(1865,1875), 5:(1875,1885),
                       6:(1910,1920), 7:(1925,1935), 8:(1935,1945)}

WINDOW2TEST = 2

#%% Functions


#%% Main

if __name__ == '__main__':
    
    AUX_DIR = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    RES_DIR = Path(PROJECT_DIR) / 'Results'

    META_PATH = Path(SAVE_DIR) / 'metadata.csv'
    FEAT_PATH = Path(SAVE_DIR) / 'features.csv'
    
    if not META_PATH.exists() and not FEAT_PATH.exists():
    
        # compile metadata
        metadata, metadata_path = compile_metadata(aux_dir=AUX_DIR, 
                                                   imaging_dates=IMAGING_DATES, 
                                                   n_wells=N_WELLS,
                                                   add_well_annotations=N_WELLS==96,
                                                   from_source_plate=False)
        
        # compile window summaries
        features, metadata = process_feature_summaries(metadata_path=metadata_path, 
                                                       results_dir=RES_DIR, 
                                                       compile_day_summaries=True,
                                                       imaging_dates=IMAGING_DATES, 
                                                       align_bluelight=False,
                                                       window_summaries=True,
                                                       n_wells=N_WELLS)
        
        # clean results
        features, metadata = clean_summary_results(features, 
                                                   metadata,
                                                   feature_columns=None,
                                                   nan_threshold_row=NAN_THRESHOLD_ROW,
                                                   nan_threshold_col=NAN_THRESHOLD_COL,
                                                   max_value_cap=1e15,
                                                   imputeNaN=True,
                                                   min_nskel_per_video=MIN_NSKEL_PER_VIDEO,
                                                   min_nskel_sum=MIN_NSKEL_SUM,
                                                   drop_size_related_feats=False,
                                                   norm_feats_only=False,
                                                   percentile_to_use=None)
    
        assert not features.isna().sum(axis=1).any()
        assert not (features.std(axis=1) == 0).any()
    
        # save features
        metadata.to_csv(META_PATH, index=False)
        features.to_csv(FEAT_PATH, index=False) 
    
    else:
        # load clean metadata and features
        metadata = pd.read_csv(META_PATH, dtype={'comments':str, 'source_plate_id':str})
        features = pd.read_csv(FEAT_PATH, index_col=None)

    # statistics: perform ANOVA and pairwise t-tests comparing mutant worms vs N2 at each window 
    single_feature_window_stats(metadata,
                                features,
                                group_by='worm_strain',
                                control='N2',
                                feat=FEATURE,
                                save_dir=SAVE_DIR,
                                windows=sorted(WINDOW_DICT_SECONDS.keys()),
                                pvalue_threshold=PVAL_THRESH)
    
    
    save_dir = Path(SAVE_DIR) / 'window_{}'.format(WINDOW2TEST)
    
    # def func for statistical tests
    
    # 7 worm strains:       N2 vs 'cat-2', 'eat-4', 'osm-5', 'pdfr-1', 'tax-2', 'unc-25'
    # 2 bacteria strains:   BW vs fepD
    # 1 feature:            'motion_mode_paused_fraction'
    # 1 window:             2 (corresponding to 30 minutes on food, just after first BL stimulus)

    # focus on just one window = 30min just after BL (window=2)
    window_metadata = metadata[metadata['window']==WINDOW2TEST]
    window_features = features[[FEATURE]].reindex(window_metadata.index)

    # statistics: perform t-tests comparing fepD vs BW for each worm strain
    
    stats_dir =  save_dir / "Stats"
    worm_strain_list = list(window_metadata['worm_strain'].unique())
    
    ttest_list = []
    for worm in worm_strain_list:
        worm_window_meta = window_metadata[window_metadata['worm_strain']==worm]
        worm_window_feat = features[[FEATURE]].reindex(worm_window_meta.index)
        
        stats, pvals, reject = univariate_tests(X=worm_window_feat,
                                                y=worm_window_meta['bacteria_strain'],
                                                control='BW',
                                                test='t-test',
                                                comparison_type='binary_each_group',
                                                multitest_correction='fdr_by',
                                                alpha=PVAL_THRESH,
                                                n_permutation_test=None)

        # get effect sizes
        effect_sizes = get_effect_sizes(X=worm_window_feat, 
                                        y=worm_window_meta['bacteria_strain'],
                                        control='BW',
                                        effect_type=None,
                                        linked_test='t-test')
        
        # compile t-test results
        stats.columns = ['stats_' + str(c) for c in stats.columns]
        pvals.columns = ['pvals_' + str(c) for c in pvals.columns]
        reject.columns = ['reject_' + str(c) for c in reject.columns]
        effect_sizes.columns = ['effect_size_' + str(c) for c in effect_sizes.columns]
        ttest_df = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
    
        # record the worm strain as the index instead of the feature
        ttest_df = ttest_df.rename(index={FEATURE:worm})
        ttest_list.append(ttest_df)

    ttest_path = stats_dir / 'ttest_mutant_worm_fepD_vs_BW_results.csv'
    ttest_results = pd.concat(ttest_list, axis=0)
    ttest_results.to_csv(ttest_path, header=True, index=True)
    
    # plotting: paired boxplots of BW vs fepD for each worm strain
    plot_dir = save_dir / "Plots"
    
    # from visualisation.plotting_helper import errorbar_sigfeats, boxplots_sigfeats
    # boxplots_sigfeats(window_features,
    #                   y_class=window_metadata['worm_strain'],
    #                   control='N2',
    #                   pvals=ttest_results[['pvals_fepD']].T.rename(index={'pvals_fepD':FEATURE}), 
    #                   z_class=window_metadata['bacteria_strain'],
    #                   feature_set=None,
    #                   saveDir=plot_dir / 'paired_boxplots',
    #                   p_value_threshold=PVAL_THRESH,
    #                   drop_insignificant=False,
    #                   max_sig_feats=None,
    #                   max_strains=None,
    #                   sns_colour_palette="tab10",
    #                   verbose=False)
    
    
    
