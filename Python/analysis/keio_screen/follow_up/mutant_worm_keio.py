#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keio Mutant Worm Screen - Response to BW vs fepD bacteria


@author: sm5911
@date: 30/03/2022

"""

#%% Imports

import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import transforms
from matplotlib import pyplot as plt
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

WINDOW_DICT_STIM_TYPE = {0:'prestim\n(30min)',1:'bluelight\n(30min)',2:'poststim\n(30min)',
                         3:'prestim\n(31min)',4:'bluelight\n(31min)',5:'poststim\n(31min)',
                         6:'prestim\n(32min)',7:'bluelight\n(32min)',8:'poststim\n(32min)'}

#%% Functions

def single_feature_window_mutant_worm_stats(metadata,
                                            features,
                                            save_dir,
                                            window=2,
                                            feature='motion_mode_paused_fraction',
                                            pvalue_threshold=0.05,
                                            fdr_method='fdr_by'):
    """ T-tests comparing BW vs fepD for each mutant worm """
    
    # 7 worm strains:       N2 vs 'cat-2', 'eat-4', 'osm-5', 'pdfr-1', 'tax-2', 'unc-25'
    # 2 bacteria strains:   BW vs fepD
    # 1 feature:            'motion_mode_paused_fraction'
    # 1 window:             2 (corresponding to 30 minutes on food, just after first BL stimulus)

    # focus on just one window = 30min just after BL (window=2)
    window_metadata = metadata[metadata['window']==window]

    # statistics: perform t-tests comparing fepD vs BW for each worm strain
    worm_strain_list = list(window_metadata['worm_strain'].unique())

    ttest_list = []
    for worm in worm_strain_list:
        worm_window_meta = window_metadata[window_metadata['worm_strain']==worm]
        worm_window_feat = features[[feature]].reindex(worm_window_meta.index)
        
        stats, pvals, reject = univariate_tests(X=worm_window_feat,
                                                y=worm_window_meta['bacteria_strain'],
                                                control='BW',
                                                test='t-test',
                                                comparison_type='binary_each_group',
                                                multitest_correction=fdr_method,
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
        ttest_df = ttest_df.rename(index={feature:worm})
        ttest_list.append(ttest_df)

    ttest_path = Path(save_dir) / 'pairwise_ttests' /\
        'ttest_mutant_worm_fepD_vs_BW_window_{}_results.csv'.format(window)
    ttest_path.parent.mkdir(exist_ok=True, parents=True)
    ttest_results = pd.concat(ttest_list, axis=0)
    ttest_results.to_csv(ttest_path, header=True, index=True)
        
    return

def plot_1window_fepD_vs_BW(metadata, 
                            features, 
                            feat='motion_mode_paused_fraction', 
                            window=2,
                            save_dir=None):
    """ Plot paired boxplots of BW vs fepD for each worm strain side-by-side """
    
    # subset for window only 
    window_metadata = metadata[metadata['window']==window]
    window_features = features.reindex(window_metadata.index)
    
    plot_df = window_metadata[['worm_strain','bacteria_strain','date_yyyymmdd']
                              ].join(window_features[[feat]])
    
    worm_strain_list = list(plot_df['worm_strain'].unique())
    bacteria_strain_list = list(plot_df['bacteria_strain'].unique())
    
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='worm_strain', 
                y=feat, 
                order=worm_strain_list, 
                hue='bacteria_strain', 
                hue_order=bacteria_strain_list, 
                dodge=True, 
                ax=ax, 
                data=plot_df,
                palette='tab10', 
                showfliers=False)
    dates = list(plot_df['date_yyyymmdd'].unique())
    date_col_dict = dict(zip(dates, sns.color_palette('Greys', n_colors=len(dates))))
    for date in dates:
        sns.stripplot(x='worm_strain', 
                      y=feat, 
                      order=worm_strain_list,
                      hue='bacteria_strain', 
                      hue_order=bacteria_strain_list, 
                      dodge=True, 
                      ax=ax, 
                      data=plot_df[plot_df['date_yyyymmdd']==date], 
                      s=3, 
                      marker='D',
                      color=sns.set_palette(palette=[date_col_dict[date]], 
                                            n_colors=len(bacteria_strain_list)))
    
    # scale plot y-axis
    scale_outliers = False
    if scale_outliers:
        grouped_strain = plot_df.groupby('worm_strain')
        y_bar = grouped_strain[feat].median() # median is less skewed by outliers
        Q1, Q3 = grouped_strain[feat].quantile(0.25), grouped_strain[feat].quantile(0.75)
        IQR = Q3 - Q1
        plt.ylim(min(y_bar) - 2.5 * max(IQR), max(y_bar) + 2.5 * max(IQR))

    # load t-test results: fepD vs BW (for each worm strain)
    t_test_path = stats_dir / 'bacteria_strain' / 'pairwise_ttests' /\
        'ttest_mutant_worm_fepD_vs_BW_window_{}_results.csv'.format(window)
    pvals = pd.read_csv(t_test_path, index_col=0)['pvals_fepD']
        
    # annotate p-values
    for ii, strain in enumerate(worm_strain_list):
        p = pvals.loc[strain]
        text = ax.get_xticklabels()[ii]
        assert text.get_text() == strain
        p_text = 'P < 0.001' if p < 0.001 else 'P = %.3f' % p
        #y = (y_bar[antiox] + 2 * IQR[antiox]) if scale_outliers_box else plot_df[feature].max()
        #h = (max(IQR) / 10) if scale_outliers_box else (y - plot_df[feature].min()) / 50
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        plt.plot([ii-.2, ii-.2, ii+.2, ii+.2], 
                 [0.98, 0.99, 0.99, 0.98], lw=1.5, c='k', transform=trans)
        ax.text(ii, 1.01, p_text, fontsize=9, ha='center', va='bottom', transform=trans)

    # legend and labels
    n_labs = len(bacteria_strain_list)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:n_labs], labels[:n_labs], fontsize=15, frameon=False, loc=(1.01, 0.9),
              handletextpad=0.5)
    ax.set_xlabel('')
    ax.set_ylabel(feat.replace('_',' '), fontsize=12, labelpad=10)
    ax.set_title('Window {0}: {1}'.format(window, WINDOW_DICT_STIM_TYPE[window].replace('\n',' ')),
                 loc='left', pad=30, fontsize=18)

    if save_dir is not None:
        save_path = Path(save_dir) / '{}.png'.format(feat)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    return 
    
def plot_1worm_fepD_vs_BW(metadata, 
                          features, 
                          feat='motion_mode_paused_fraction',
                          worm_strain='N2',
                          save_dir=None):
    
    # subset for worm strain to plot
    worm_metadata = metadata[metadata['worm_strain']==worm_strain]
    worm_features = features.reindex(worm_metadata.index)
    
    plot_df = worm_metadata[['bacteria_strain','window','date_yyyymmdd']
                            ].join(worm_features[[feat]])
    
    window_list = list(plot_df['window'].unique())
    bacteria_strain_list = list(plot_df['bacteria_strain'].unique())
    
    plt.close('all')
    fig, ax = plt.subplots(figsize=(15,6))
    sns.boxplot(x='window', 
                y=feat, 
                order=window_list, 
                hue='bacteria_strain', 
                hue_order=bacteria_strain_list, 
                dodge=True, 
                ax=ax, 
                data=plot_df,
                palette='tab10', 
                showfliers=False)
    dates = list(plot_df['date_yyyymmdd'].unique())
    date_col_dict = dict(zip(dates, sns.color_palette('Greys', n_colors=len(dates))))
    for date in dates:
        sns.stripplot(x='window', 
                      y=feat, 
                      order=window_list,
                      hue='bacteria_strain', 
                      hue_order=bacteria_strain_list, 
                      dodge=True, 
                      ax=ax, 
                      data=plot_df[plot_df['date_yyyymmdd']==date], 
                      s=3, 
                      marker='D',
                      color=sns.set_palette(palette=[date_col_dict[date]], 
                                            n_colors=len(bacteria_strain_list)))
    
    # scale plot y-axis
    scale_outliers = False
    if scale_outliers:
        grouped_strain = plot_df.groupby('window')
        y_bar = grouped_strain[feat].median() # median is less skewed by outliers
        Q1, Q3 = grouped_strain[feat].quantile(0.25), grouped_strain[feat].quantile(0.75)
        IQR = Q3 - Q1
        plt.ylim(min(y_bar) - 2.5 * max(IQR), max(y_bar) + 2.5 * max(IQR))
        
    # annotate p-values
    for ii, window in enumerate(window_list):
        # load t-test results for window and subset for fepD vs BW results for worm strain
        t_test_path = stats_dir / 'bacteria_strain' / 'pairwise_ttests' /\
            'ttest_mutant_worm_fepD_vs_BW_window_{}_results.csv'.format(window)
        pvals = pd.read_csv(t_test_path, index_col=0)['pvals_fepD']
        p = pvals.loc[worm_strain]
        text = ax.get_xticklabels()[ii]
        assert text.get_text() == str(window)
        p_text = 'P < 0.001' if p < 0.001 else 'P = %.3f' % p
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        plt.plot([ii-.2, ii-.2, ii+.2, ii+.2], 
                 [0.98, 0.99, 0.99, 0.98], lw=1.5, c='k', transform=trans)
        ax.text(ii, 1.01, p_text, fontsize=9, ha='center', va='bottom', transform=trans)

    # legend and labels
    n_labs = len(bacteria_strain_list)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:n_labs], labels[:n_labs], fontsize=15, frameon=False, loc=(1.01, 0.9),
              handletextpad=0.5)
    ax.set_xlabel('')
    ax.set_xticklabels([WINDOW_DICT_STIM_TYPE[w] for w in window_list])
    ax.set_ylabel(feat.replace('_',' '), fontsize=12, labelpad=10)
    ax.set_title(worm_strain, loc='left', pad=30, fontsize=18)

    if save_dir is not None:
        save_path = Path(save_dir) / '{}.png'.format(feat)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    return 

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

    stats_dir =  Path(SAVE_DIR) / "Stats"
    window_list = sorted(WINDOW_DICT_SECONDS.keys())

    # statistics: perform ANOVA and pairwise t-tests comparing mutant worms vs N2 at each window 
    single_feature_window_stats(metadata,
                                features,
                                group_by='worm_strain',
                                control='N2',
                                feat=FEATURE,
                                save_dir=stats_dir / 'worm_strain',
                                windows=window_list,
                                pvalue_threshold=PVAL_THRESH)
    
    # statistics: pairwise t-test of BW vs fepB for each mutant worm (at each window)
    for window in window_list: 
        single_feature_window_mutant_worm_stats(metadata,
                                                features,
                                                save_dir=stats_dir / 'bacteria_strain',
                                                window=window,
                                                feature=FEATURE,
                                                pvalue_threshold=0.05,
                                                fdr_method='fdr_by')
    
    # plotting: paired boxplots of BW vs fepD for each worm strain
    plot_dir = Path(SAVE_DIR) / "Plots"
    
    # plot for each window
    for window in window_list:
        plot_1window_fepD_vs_BW(metadata, 
                                features, 
                                feat='motion_mode_paused_fraction', 
                                window=window,
                                save_dir=plot_dir / 'boxplots_BW_vs_fepD' / 'window_{}'.format(window))
        
    # plot for each worm
    worm_strain_list = list(metadata['worm_strain'].unique())
    for worm in worm_strain_list:
        plot_1worm_fepD_vs_BW(metadata, 
                              features, 
                              feat=FEATURE, 
                              worm_strain=worm,
                              save_dir=plot_dir / 'boxplots_BW_vs_fepD' / '{}'.format(worm))    
    
