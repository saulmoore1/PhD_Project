#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compile acute single worm metadata and feature summaries

@author: sm5911
@date: 30/03/2022

"""

#%% Imports

import pandas as pd
from pathlib import Path
from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from statistical_testing.stats_helper import single_feature_window_stats

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Keio_Acute_Single_Worm"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Acute_Single_Worm"

IMAGING_DATES = ['20220206','20220209','20220212']
N_WELLS = 6

CONTROL_STRAIN = 'BW'
FEATURE = 'motion_mode_paused_fraction'

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
MIN_NSKEL_PER_VIDEO = None
MIN_NSKEL_SUM = 50
PVAL_THRESH = 0.05

THRESHOLD_N_SECONDS = 75
FPS = 25

WINDOW_DICT_SECONDS = {0:(290,300), 1:(305,315), 2:(315,325), 
                       3:(590,600), 4:(605,615), 5:(615,625), 
                       6:(890,900), 7:(905,915), 8:(915,925), 
                       9:(1190,1200), 10:(1205,1215), 11:(1215,1225), 
                       12:(1490,1500), 13:(1505.1515), 14:(1515,1525)}

WINDOW_NUMBERS = [9,10,11]#[12,13,14] #sorted(WINDOW_DICT_SECONDS.keys())

WINDOW_DICT_STIM_TYPE = {0:'prestim\n(5min)',1:'bluelight\n(5min)',2:'poststim\n(5min)',
                         3:'prestim\n(10min)',4:'bluelight\n(10min)',5:'poststim\n(10min)',
                         6:'prestim\n(15min)',7:'bluelight\n(15min)',8:'poststim\n(15min)',
                         9:'prestim\n(20min)',10:'bluelight\n(20min)',11:'poststim\n(20min)',
                         12:'prestim\n(25min)',13:'bluelight\n(25min)',14:'poststim\n(25min)'}

#%% Functions

def window_boxplot_fepD_vs_BW(metadata, 
                              features, 
                              feat='motion_mode_paused_fraction',
                              windows=None,
                              save_dir=None):    
    
    import seaborn as sns
    from matplotlib import transforms
    from matplotlib import pyplot as plt

    plot_df = metadata[['bacteria_strain','window','date_yyyymmdd']].join(features[[feat]])
    
    if windows is not None:
        assert all(w in sorted(plot_df['window'].unique()) for w in windows)
        plot_df = plot_df[plot_df['window'].isin(windows)]
    else:
        windows = sorted(plot_df['window'].unique())
    
    bacteria_strain_list = ['BW', 'fepD']
    
    plt.close('all')
    fig, ax = plt.subplots(figsize=(max(8,len(windows)),8))
    sns.boxplot(x='window', 
                y=feat, 
                order=windows,
                hue='bacteria_strain', 
                hue_order=bacteria_strain_list, 
                dodge=True,
                ax=ax, 
                palette='tab10', 
                showfliers=False,
                data=plot_df)
    dates = list(plot_df['date_yyyymmdd'].unique())
    date_col_dict = dict(zip(dates, sns.color_palette('Greys', n_colors=len(dates))))
    for date in dates:
        sns.stripplot(x='window',
                      y=feat,
                      order=windows,
                      hue='bacteria_strain',
                      hue_order=bacteria_strain_list,
                      dodge=True,
                      ax=ax,
                      s=3, marker='D',
                      color=sns.set_palette(palette=[date_col_dict[date]], 
                                            n_colors=len(bacteria_strain_list)),
                      data=plot_df[plot_df['date_yyyymmdd']==date])
    
    # scale plot y-axis
    scale_outliers = False
    if scale_outliers:
        grouped_strain = plot_df.groupby('window')
        y_bar = grouped_strain[feat].median() # median is less skewed by outliers
        Q1, Q3 = grouped_strain[feat].quantile(0.25), grouped_strain[feat].quantile(0.75)
        IQR = Q3 - Q1
        plt.ylim(min(y_bar) - 2.5 * max(IQR), max(y_bar) + 2.5 * max(IQR))
 
    # load t-test results for fepD vs BW at each window
    t_test_path = stats_dir / 'pairwise_ttests' / 'fepD_window_results.csv'
    ttest_df = pd.read_csv(t_test_path, index_col=0)
    pvals = ttest_df[[c for c in ttest_df if 'pvals_' in c]]

    # annotate p-values
    for ii, window in enumerate(windows):
        p = pvals.loc[feat, 'pvals_{}'.format(window)]
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
    ax.legend(handles[:n_labs], labels[:n_labs], fontsize=12, frameon=False, loc=(1.01, 0.9),
              handletextpad=0.2)
    ax.set_xlabel('')
    ax.set_xticklabels([WINDOW_DICT_STIM_TYPE[w] for w in windows])
    ax.set_ylabel(feat.replace('_',' '), fontsize=12, labelpad=10)

    plt.subplots_adjust(right=0.85)
    
    if save_dir is not None:
        save_path = Path(save_dir) / '{}_windows'.format(len(windows)) / '{}.png'.format(feat)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()    
      
    return
        
#%% Main

if __name__ == "__main__":
    
    AUX_DIR = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    RES_DIR = Path(PROJECT_DIR) / 'Results'

    META_PATH = Path(SAVE_DIR) / 'metadata.csv'
    FEAT_PATH = Path(SAVE_DIR) / 'features.csv'
    
    if not META_PATH.exists() and not FEAT_PATH.exists():
    
        # compile metadata
        metadata, metadata_path = compile_metadata(aux_dir=AUX_DIR,
                                                   imaging_dates=IMAGING_DATES,
                                                   add_well_annotations=N_WELLS==96,
                                                   n_wells=N_WELLS,
                                                   from_source_plate=False)
            
        # compile window summaries
        features, metadata = process_feature_summaries(metadata_path,
                                                       results_dir=RES_DIR,
                                                       compile_day_summaries=True,
                                                       imaging_dates=IMAGING_DATES,
                                                       align_bluelight=False,
                                                       window_summaries=True)
        
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
        
        # save clean metadata and features
        metadata.to_csv(META_PATH, index=False)
        features.to_csv(FEAT_PATH, index=False) 
    
    else:
        # load clean metadata and features
        metadata = pd.read_csv(META_PATH, dtype={'comments':str, 'source_plate_id':str})
        features = pd.read_csv(FEAT_PATH, index_col=None)
            
    # subset to remove entries for worms that took > 10 seconds to encounter food
    metadata = metadata[metadata['first_food_frame'] < THRESHOLD_N_SECONDS*FPS]
    features = features.reindex(metadata.index)

    # statistics: perform pairwise t-tests comparing fepD vs BW at each window 
    stats_dir = Path(SAVE_DIR) / "Stats"    
    single_feature_window_stats(metadata,
                                features,
                                group_by='bacteria_strain',
                                control=CONTROL_STRAIN,
                                feat=FEATURE,
                                windows=sorted(WINDOW_NUMBERS), #sorted(WINDOW_DICT_SECONDS.keys()),
                                save_dir=stats_dir,
                                pvalue_threshold=PVAL_THRESH,
                                fdr_method='fdr_by')
    
    # plotting: pairwise box plots comparing fepD vs BW at each timepoint
    plot_dir = Path(SAVE_DIR) / "Plots" 
    window_boxplot_fepD_vs_BW(metadata, 
                              features, 
                              feat=FEATURE, 
                              windows=sorted(WINDOW_NUMBERS), 
                              save_dir=plot_dir)
