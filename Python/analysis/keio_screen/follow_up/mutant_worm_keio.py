#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keio Mutant Worm Screen - Response to BW vs fepD bacteria


@author: sm5911
@date: 30/03/2022

"""

#%% Imports

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from matplotlib import transforms
from matplotlib import pyplot as plt
from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from statistical_testing.stats_helper import window_stats
from time_series.time_series_helper import get_strain_timeseries
from time_series.plot_timeseries import plot_timeseries_motion_mode
# from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Keio_Mutant_Worm_6WP"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Mutant_Worm"

IMAGING_DATES = ['20220305','20220314','20220321']
N_WELLS = 6

FEATURE = 'motion_mode_forward_fraction'

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
MIN_NSKEL_PER_VIDEO = None
MIN_NSKEL_SUM = 500
PVAL_THRESH = 0.05
FPS = 25
VIDEO_LENGTH_SECONDS = 38*60
BLUELIGHT_TIMEPOINTS_MINUTES = [30,31,32]
BLUELIGHT_WINDOWS_ONLY_TS = True

BIN_SIZE_SECONDS = 5
SMOOTH_WINDOW_SECONDS = 5

WINDOW_DICT_SECONDS = {0:(1790,1800), 1:(1805,1815), 2:(1815,1825),
                       3:(1850,1860), 4:(1865,1875), 5:(1875,1885),
                       6:(1910,1920), 7:(1925,1935), 8:(1935,1945)}

WINDOW_DICT_STIM_TYPE = {0:'prestim\n(30min)',1:'bluelight\n(30min)',2:'poststim\n(30min)',
                         3:'prestim\n(31min)',4:'bluelight\n(31min)',5:'poststim\n(31min)',
                         6:'prestim\n(32min)',7:'bluelight\n(32min)',8:'poststim\n(32min)'}

scale_outliers_box = True

#%% Functions

# =============================================================================
# def single_feature_window_mutant_worm_stats(metadata,
#                                             features,
#                                             save_dir,
#                                             window=2,
#                                             feature='motion_mode_forward_fraction',
#                                             pvalue_threshold=0.05,
#                                             fdr_method='fdr_by'):
#     """ T-tests comparing BW vs fepD for each mutant worm """
#     
#     # 7 worm strains:       N2 vs 'cat-2', 'eat-4', 'osm-5', 'pdfr-1', 'tax-2', 'unc-25'
#     # 2 bacteria strains:   BW vs fepD
#     # 1 feature:            'motion_mode_paused_fraction'
#     # 1 window:             2 (corresponding to 30 minutes on food, just after first BL stimulus)
# 
#     # focus on just one window = 30min just after BL (window=2)
#     window_metadata = metadata[metadata['window']==window]
# 
#     # statistics: perform t-tests comparing fepD vs BW for each worm strain
#     worm_strain_list = list(window_metadata['worm_strain'].unique())
# 
#     ttest_list = []
#     for worm in worm_strain_list:
#         worm_window_meta = window_metadata[window_metadata['worm_strain']==worm]
#         worm_window_feat = features[[feature]].reindex(worm_window_meta.index)
#         
#         stats, pvals, reject = univariate_tests(X=worm_window_feat,
#                                                 y=worm_window_meta['bacteria_strain'],
#                                                 control='BW',
#                                                 test='t-test',
#                                                 comparison_type='binary_each_group',
#                                                 multitest_correction=fdr_method,
#                                                 alpha=PVAL_THRESH,
#                                                 n_permutation_test=None)
# 
#         # get effect sizes
#         effect_sizes = get_effect_sizes(X=worm_window_feat, 
#                                         y=worm_window_meta['bacteria_strain'],
#                                         control='BW',
#                                         effect_type=None,
#                                         linked_test='t-test')
#         
#         # compile t-test results
#         stats.columns = ['stats_' + str(c) for c in stats.columns]
#         pvals.columns = ['pvals_' + str(c) for c in pvals.columns]
#         reject.columns = ['reject_' + str(c) for c in reject.columns]
#         effect_sizes.columns = ['effect_size_' + str(c) for c in effect_sizes.columns]
#         ttest_df = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
#     
#         # record the worm strain as the index instead of the feature
#         ttest_df = ttest_df.rename(index={feature:worm})
#         ttest_list.append(ttest_df)
# 
#     ttest_path = Path(save_dir) / 'pairwise_ttests' /\
#         'ttest_mutant_worm_fepD_vs_BW_window_{}_results.csv'.format(window)
#     ttest_path.parent.mkdir(exist_ok=True, parents=True)
#     ttest_results = pd.concat(ttest_list, axis=0)
#     ttest_results.to_csv(ttest_path, header=True, index=True)
#         
#     return
# 
# def plot_1window_fepD_vs_BW(metadata,
#                             features, 
#                             feat='motion_mode_forward_fraction', 
#                             window=2,
#                             save_dir=None):
#     """ Plot paired boxplots of BW vs fepD for each worm strain side-by-side """
#     
#     # subset for window only 
#     window_metadata = metadata[metadata['window']==window]
#     window_features = features.reindex(window_metadata.index)
#     
#     plot_df = window_metadata[['worm_strain','bacteria_strain','date_yyyymmdd']
#                               ].join(window_features[[feat]])
#     
#     worm_strain_list = list(plot_df['worm_strain'].unique())
#     bacteria_strain_list = list(plot_df['bacteria_strain'].unique())
#     
#     plt.close('all')
#     fig, ax = plt.subplots(figsize=(12,6))
#     sns.boxplot(x='worm_strain', 
#                 y=feat, 
#                 order=worm_strain_list, 
#                 hue='bacteria_strain', 
#                 hue_order=bacteria_strain_list, 
#                 dodge=True, 
#                 ax=ax, 
#                 data=plot_df,
#                 palette='tab10', 
#                 showfliers=False)
#     dates = list(plot_df['date_yyyymmdd'].unique())
#     date_col_dict = dict(zip(dates, sns.color_palette('Greys', n_colors=len(dates))))
#     for date in dates:
#         sns.stripplot(x='worm_strain', 
#                       y=feat, 
#                       order=worm_strain_list,
#                       hue='bacteria_strain', 
#                       hue_order=bacteria_strain_list, 
#                       dodge=True, 
#                       ax=ax, 
#                       data=plot_df[plot_df['date_yyyymmdd']==date], 
#                       s=3, 
#                       marker='D',
#                       color=sns.set_palette(palette=[date_col_dict[date]], 
#                                             n_colors=len(bacteria_strain_list)))
#     
#     # scale plot y-axis
#     scale_outliers = False
#     if scale_outliers:
#         grouped_strain = plot_df.groupby('worm_strain')
#         y_bar = grouped_strain[feat].median() # median is less skewed by outliers
#         Q1, Q3 = grouped_strain[feat].quantile(0.25), grouped_strain[feat].quantile(0.75)
#         IQR = Q3 - Q1
#         plt.ylim(min(y_bar) - 2.5 * max(IQR), max(y_bar) + 2.5 * max(IQR))
# 
#     # load t-test results: fepD vs BW (for each worm strain)
#     t_test_path = stats_dir / 'bacteria_strain' / 'pairwise_ttests' /\
#         'ttest_mutant_worm_fepD_vs_BW_window_{}_results.csv'.format(window)
#     pvals = pd.read_csv(t_test_path, index_col=0)['pvals_fepD']
#         
#     # annotate p-values
#     for ii, strain in enumerate(worm_strain_list):
#         p = pvals.loc[strain]
#         text = ax.get_xticklabels()[ii]
#         assert text.get_text() == strain
#         p_text = 'P < 0.001' if p < 0.001 else 'P = %.3f' % p
#         #y = (y_bar[antiox] + 2 * IQR[antiox]) if scale_outliers_box else plot_df[feature].max()
#         #h = (max(IQR) / 10) if scale_outliers_box else (y - plot_df[feature].min()) / 50
#         trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
#         plt.plot([ii-.2, ii-.2, ii+.2, ii+.2], 
#                  [0.98, 0.99, 0.99, 0.98], lw=1.5, c='k', transform=trans)
#         ax.text(ii, 1.01, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
# 
#     # legend and labels
#     n_labs = len(bacteria_strain_list)
#     handles, labels = ax.get_legend_handles_labels()
#     ax.legend(handles[:n_labs], labels[:n_labs], fontsize=15, frameon=False, loc=(1.01, 0.9),
#               handletextpad=0.5)
#     ax.set_xlabel('')
#     ax.set_ylabel(feat.replace('_',' '), fontsize=12, labelpad=10)
#     ax.set_title('Window {0}: {1}'.format(window, WINDOW_DICT_STIM_TYPE[window].replace('\n',' ')),
#                  loc='left', pad=30, fontsize=18)
# 
#     if save_dir is not None:
#         save_path = Path(save_dir) / '{}.pdf'.format(feat)
#         save_path.parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(save_path, dpi=300)
#     else:
#         plt.show()
# 
#     return 
#     
# def plot_1worm_fepD_vs_BW(metadata, 
#                           features, 
#                           feat='motion_mode_forward_fraction',
#                           worm_strain='N2',
#                           save_dir=None):
#     
#     # subset for worm strain to plot
#     worm_metadata = metadata[metadata['worm_strain']==worm_strain]
#     worm_features = features.reindex(worm_metadata.index)
#     
#     plot_df = worm_metadata[['bacteria_strain','window','date_yyyymmdd']
#                             ].join(worm_features[[feat]])
#     
#     window_list = list(plot_df['window'].unique())
#     bacteria_strain_list = list(plot_df['bacteria_strain'].unique())
#     
#     plt.close('all')
#     fig, ax = plt.subplots(figsize=(15,6))
#     sns.boxplot(x='window', 
#                 y=feat, 
#                 order=window_list, 
#                 hue='bacteria_strain', 
#                 hue_order=bacteria_strain_list, 
#                 dodge=True, 
#                 ax=ax, 
#                 data=plot_df,
#                 palette='tab10', 
#                 showfliers=False)
#     dates = list(plot_df['date_yyyymmdd'].unique())
#     date_col_dict = dict(zip(dates, sns.color_palette('Greys', n_colors=len(dates))))
#     for date in dates:
#         sns.stripplot(x='window', 
#                       y=feat, 
#                       order=window_list,
#                       hue='bacteria_strain', 
#                       hue_order=bacteria_strain_list, 
#                       dodge=True, 
#                       ax=ax, 
#                       data=plot_df[plot_df['date_yyyymmdd']==date], 
#                       s=3, 
#                       marker='D',
#                       color=sns.set_palette(palette=[date_col_dict[date]], 
#                                             n_colors=len(bacteria_strain_list)))
#     
#     # scale plot y-axis
#     scale_outliers = False
#     if scale_outliers:
#         grouped_strain = plot_df.groupby('window')
#         y_bar = grouped_strain[feat].median() # median is less skewed by outliers
#         Q1, Q3 = grouped_strain[feat].quantile(0.25), grouped_strain[feat].quantile(0.75)
#         IQR = Q3 - Q1
#         plt.ylim(min(y_bar) - 2.5 * max(IQR), max(y_bar) + 2.5 * max(IQR))
#         
#     # annotate p-values
#     for ii, window in enumerate(window_list):
#         # load t-test results for window and subset for fepD vs BW results for worm strain
#         t_test_path = stats_dir / 'bacteria_strain' / 'pairwise_ttests' /\
#             'ttest_mutant_worm_fepD_vs_BW_window_{}_results.csv'.format(window)
#         pvals = pd.read_csv(t_test_path, index_col=0)['pvals_fepD']
#         p = pvals.loc[worm_strain]
#         text = ax.get_xticklabels()[ii]
#         assert text.get_text() == str(window)
#         p_text = 'P < 0.001' if p < 0.001 else 'P = %.3f' % p
#         trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
#         plt.plot([ii-.2, ii-.2, ii+.2, ii+.2], 
#                  [0.98, 0.99, 0.99, 0.98], lw=1.5, c='k', transform=trans)
#         ax.text(ii, 1.01, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
# 
#     # legend and labels
#     n_labs = len(bacteria_strain_list)
#     handles, labels = ax.get_legend_handles_labels()
#     ax.legend(handles[:n_labs], labels[:n_labs], fontsize=15, frameon=False, loc=(1.01, 0.9),
#               handletextpad=0.5)
#     ax.set_xlabel('')
#     ax.set_xticklabels([WINDOW_DICT_STIM_TYPE[w] for w in window_list])
#     ax.set_ylabel(feat.replace('_',' '), fontsize=12, labelpad=10)
#     ax.set_title(worm_strain, loc='left', pad=30, fontsize=18)
# 
#     if save_dir is not None:
#         save_path = Path(save_dir) / '{}.pdf'.format(feat)
#         save_path.parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(save_path, dpi=300)
#     else:
#         plt.show()
# 
#     return 
# =============================================================================

def boxplots(metadata, 
             features,
             save_dir=None,
             stats_dir=None,
             feature_list=None,
             window_list=None):
    
    if feature_list is not None:
        assert isinstance(feature_list, list) and all(f in features.columns 
                                                      for f in feature_list)
    else:
        feature_list = features.columns.tolist()
    
    if window_list is not None:
        assert isinstance(window_list, list) and all(f in metadata['window'].unique() 
                                                     for f in window_list)
    else:
        window_list = sorted(metadata['window'].unique())
        
    for window in tqdm(window_list):
        # for food in metadata['bacteria_strain'].unique():
        #     print("Plotting boxplots comparing mutant worms vs N2 on %s" % food)
            
        window_meta = metadata.query("window == @window").copy()
        window_feat = features.reindex(window_meta.index).copy()
        plot_df = window_meta.join(window_feat)

        dates = plot_df['date_yyyymmdd'].unique()
        date_lut = dict(zip(dates, sns.color_palette('Greys', n_colors=len(dates))))
        
        worm_strain_list = ['N2'] + [w for w in sorted(plot_df['worm_strain'].unique()) 
                                     if w != 'N2']
        bacteria_strain_list = ['BW', 'fepD']
        
        # load t-test results
        ttest_path = Path(stats_dir) / 't-tests' / 't-test_results_window_{}.csv'.format(window)
        ttest_df = pd.read_csv(ttest_path, index_col=0)
        pvals = ttest_df[[c for c in ttest_df.columns if 'pvals_' in c]]
        pvals.columns = [c.split('pvals_')[-1] for c in pvals.columns]

        for feat in tqdm(feature_list):
            
            plt.close('all')
            fig, ax = plt.subplots(figsize=(12,8), dpi=300)
            sns.boxplot(x='worm_strain', 
                        y=feat, 
                        order=worm_strain_list, 
                        hue='bacteria_strain',
                        hue_order=bacteria_strain_list, 
                        dodge=True,
                        data=plot_df, 
                        palette='tab10',
                        ax=ax, 
                        showfliers=False)
            for date in date_lut.keys():
                date_df = plot_df.query("date_yyyymmdd == @date")
                ax = sns.stripplot(x='worm_strain', 
                                   y=feat, 
                                   order=worm_strain_list, 
                                   hue='bacteria_strain', 
                                   hue_order=bacteria_strain_list,
                                   dodge=True,
                                   data=date_df, 
                                   ax=ax, 
                                   color=sns.set_palette(palette=[date_lut[date]], 
                                                         n_colors=len(worm_strain_list)),
                                   marker='D',
                                   alpha=0.7,
                                   size=4)
            bacteria_lut = dict(zip(bacteria_strain_list, 
                                    sns.color_palette('tab10', len(bacteria_strain_list))))
            markers = [plt.Line2D([0,0],[0,0], color=color, marker='o', linestyle='') 
                       for color in bacteria_lut.values()]
            plt.legend(markers, bacteria_lut.keys(), numpoints=1, frameon=False, 
                       loc='best', markerscale=0.75, fontsize=8, handletextpad=0.2)
            
            # annotate p-values on plot
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

            for ii, worm in enumerate(worm_strain_list):                        
                text = ax.get_xticklabels()[ii]
                assert text.get_text() == worm
                if worm == 'N2':
                    p = pvals.loc[feat, 'N2-fepD']
                    p_text = 'P<0.001' if p < 0.001 else 'P=%.3f' % p
                    ax.text(ii+0.2, 1.02, p_text, fontsize=6, ha='center', va='bottom', transform=trans)
                    continue
                else:
                    p1 = pvals.loc[feat, str(worm) + '-' + bacteria_strain_list[0]]
                    p2 = pvals.loc[feat, str(worm) + '-' + bacteria_strain_list[1]]
                    p1_text = 'P<0.001' if p1 < 0.001 else 'P=%.3f' % p1
                    p2_text = 'P<0.001' if p2 < 0.001 else 'P=%.3f' % p2
                    ax.text(ii+0.2, 1.02, p1_text, fontsize=6, ha='center', va='bottom', transform=trans)
                    ax.text(ii-0.2, 1.02, p2_text, fontsize=6, ha='center', va='bottom', transform=trans)

            ax.set_xlabel('Worm - Bacteria', fontsize=15, labelpad=10)
            ax.set_ylabel(feat.replace('_',' '), fontsize=15, labelpad=10)
            
            save_path = Path(save_dir) / 'window_{}'.format(window) / '{}.pdf'.format(feat)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)

    return

def timeseries_motion_mode(metadata):
    
    worm_strain_list = ['N2'] + [w for w in sorted(metadata['worm_strain'].unique()) 
                                 if w != 'N2']
    bacteria_strain_list = ['BW', 'fepD']
    
    # timeseries: motion mode paused fraction over time
    for worm in worm_strain_list:
        
        # both bacteria together, for each worm/motion mode
        for mode in ['forwards','backwards','stationary']:
            print("Plotting timeseries '%s' fraction for %s..." % (mode, worm)) 

            plt.close('all')
            fig, ax = plt.subplots(figsize=(15,5))

            colours = sns.color_palette(palette="tab10", n_colors=len(bacteria_strain_list))
            bluelight_frames = [(i*60*FPS, i*60*FPS+10*FPS) for i in BLUELIGHT_TIMEPOINTS_MINUTES]
                    
            for b, bacteria in enumerate(bacteria_strain_list):
                
                strain_metadata = metadata[np.logical_and(metadata['worm_strain']==worm,
                                                          metadata['bacteria_strain']==bacteria)]
            
                # get timeseries data for worm strain
                strain_timeseries = get_strain_timeseries(strain_metadata, 
                                                          project_dir=PROJECT_DIR, 
                                                          strain=bacteria,
                                                          group_by='bacteria_strain',
                                                          n_wells=N_WELLS,
                                                          save_dir=Path(SAVE_DIR) / 'Data' / worm,
                                                          verbose=False)
    
                ax = plot_timeseries_motion_mode(df=strain_timeseries,
                                                 window=SMOOTH_WINDOW_SECONDS*FPS,
                                                 error=True,
                                                 mode=mode,
                                                 max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
                                                 title=None,
                                                 #figsize=(15,5), 
                                                 saveAs=None, #saveAs=save_path,
                                                 ax=ax, #ax=None,
                                                 bluelight_frames=bluelight_frames,
                                                 colour=colours[b],
                                                 alpha=0.25)
            
            xticks = np.linspace(0, VIDEO_LENGTH_SECONDS*FPS, int(VIDEO_LENGTH_SECONDS/60)+1)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
            ax.set_xlabel('Time (minutes)', fontsize=15, labelpad=10)
            ax.set_ylabel('Fraction {}'.format(mode), fontsize=15, labelpad=10)
            ax.legend(bacteria_strain_list, fontsize=12, frameon=False, loc='best')
            ax.set_title('motion mode fraction {}'.format(mode), fontsize=15, pad=10)

            if BLUELIGHT_WINDOWS_ONLY_TS:
                ts_plot_dir = plot_dir / 'timeseries_bluelight' / worm
                ax.set_xlim([min(BLUELIGHT_TIMEPOINTS_MINUTES)*60*FPS-60*FPS, 
                             max(BLUELIGHT_TIMEPOINTS_MINUTES)*60*FPS+70*FPS])
            else:
                ts_plot_dir = plot_dir / 'timeseries' / worm
    
            # save plot
            ts_plot_dir.mkdir(exist_ok=True, parents=True)
            plt.savefig(ts_plot_dir / '{0}_{1}.pdf'.format(worm, mode), dpi=300)  
            plt.close()
    
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
    plot_dir = Path(SAVE_DIR) / "Plots"
    
    window_list = sorted(WINDOW_DICT_SECONDS.keys())
    worm_strain_list = list(metadata['worm_strain'].unique())
    bacteria_strain_list = sorted(metadata['bacteria_strain'].unique())

    # ANOVA and t-tests comparing mutant worms on fepD/BW vs N2 on BW (for each window)
    metadata['treatment'] = metadata[['worm_strain','bacteria_strain']].agg('-'.join, axis=1)
    window_stats(metadata,
                 features,
                 group_by='treatment',
                 control='N2-BW',
                 feature_list=[FEATURE],
                 save_dir=stats_dir,
                 windows=window_list,
                 pvalue_threshold=PVAL_THRESH,
                 fdr_method='fdr_by')
    
    # boxplots for each bacteria strain comparing mutants with control at each window
    boxplots(metadata, 
             features,
             save_dir=plot_dir,
             stats_dir=stats_dir,
             feature_list=[FEATURE],
             window_list=window_list)
    
    # plot timeseries comparing forwards fraction of each mutant worms on BW vs fepD bacteria
    timeseries_motion_mode(metadata)
    
    # # statistics: pairwise t-test of BW vs fepB for each mutant worm (at each window)
    # for window in window_list: 
    #     single_feature_window_mutant_worm_stats(metadata,
    #                                             features,
    #                                             save_dir=stats_dir / 'bacteria_strain',
    #                                             window=window,
    #                                             feature=FEATURE,
    #                                             pvalue_threshold=0.05,
    #                                             fdr_method='fdr_by')
    
    # plotting: paired boxplots of BW vs fepD for each worm strain
    
    # # plot for each window
    # for window in window_list:
    #     plot_1window_fepD_vs_BW(metadata, 
    #                             features, 
    #                             feat=FEATURE, 
    #                             window=window,
    #                             save_dir=plot_dir / 'boxplots_BW_vs_fepD' / 'window_{}'.format(window))
        
    # # plot for each worm
    # for worm in worm_strain_list:
    #     plot_1worm_fepD_vs_BW(metadata, 
    #                           features, 
    #                           feat=FEATURE, 
    #                           worm_strain=worm,
    #                           save_dir=plot_dir / 'boxplots_BW_vs_fepD' / '{}'.format(worm))    
        
