#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compile acute single worm metadata and feature summaries

@author: sm5911
@date: 30/03/2022

"""

#%% Imports

import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from matplotlib import transforms
from matplotlib import pyplot as plt

from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from statistical_testing.stats_helper import window_stats

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Keio_Acute_Single_Worm"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Acute_Single_Worm"

IMAGING_DATES = ['20220206','20220209','20220212']
N_WELLS = 6

CONTROL_STRAIN = 'BW'
feature_list = ['motion_mode_forward_fraction']

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
MIN_NSKEL_PER_VIDEO = None
MIN_NSKEL_SUM = 50
PVAL_THRESH = 0.05

THRESHOLD_N_SECONDS = 75
FPS = 25

WINDOW_DICT_SECONDS = {0:(330,360), 1:(630,660), 2:(930,960), 3:(1230,1260), 4:(1530,1560)}
WINDOW_NUMBERS = sorted(WINDOW_DICT_SECONDS.keys())

#%% Functions

def window_boxplots(metadata, 
                    features, 
                    group_by='bacteria_strain',
                    control='BW',
                    feature_list=['motion_mode_paused_fraction'],
                    windows=None,
                    save_dir=None,
                    stats_dir=None,
                    scale_outliers=False):
    
        
    plot_df = metadata[['bacteria_strain','window','date_yyyymmdd']].join(features[feature_list])
    
    if windows is not None:
        assert isinstance(windows, list) and all(w in metadata['window'].unique() for w in windows)
        metadata = metadata[metadata['window'].isin(windows)]
    else:
        windows = sorted(plot_df['window'].unique())

    for window in tqdm(windows):
        window_meta = metadata.query("window==@window")
        
        plot_df = window_meta[[group_by, 'date_yyyymmdd']].join(features.reindex(window_meta.index))

        strain_list = sorted(window_meta[group_by].unique())
        strain_list = [s for s in strain_list if s != control]
        
        if stats_dir is not None:
            # load t-test pvalues
            t_test_path = stats_dir / 't-tests' / 't-test_results_window_{}.csv'.format(window)
            ttest_df = pd.read_csv(t_test_path, index_col=0)
            pvals = ttest_df[[c for c in ttest_df if 'pvals_' in c]]

        for feat in feature_list:            
            plt.close('all')
            fig, ax = plt.subplots(figsize=(max(8, len(strain_list)),8), dpi=300)
            sns.boxplot(x=group_by,
                        y=feat, 
                        order=[control] + strain_list,
                        hue=None, 
                        hue_order=None, 
                        dodge=False,
                        ax=ax, 
                        palette='tab10', 
                        showfliers=False,
                        data=plot_df)
            dates = list(plot_df['date_yyyymmdd'].unique())
            date_col_dict = dict(zip(dates, sns.color_palette('Greys', n_colors=len(dates))))
            for date in dates:
                sns.stripplot(x=group_by,
                              y=feat,
                              order=[control] + strain_list,
                              hue=None,
                              hue_order=None,
                              dodge=False,
                              ax=ax,
                              s=5, marker='D',
                              color=date_col_dict[date],
                              data=plot_df[plot_df['date_yyyymmdd']==date])
            
            # scale plot y-axis
            if scale_outliers:
                grouped_strain = plot_df.groupby(group_by)
                y_bar = grouped_strain[feat].median() # median is less skewed by outliers
                Q1, Q3 = grouped_strain[feat].quantile(0.25), grouped_strain[feat].quantile(0.75)
                IQR = Q3 - Q1
                plt.ylim(min(y_bar) - 2.5 * max(IQR), max(y_bar) + 2.5 * max(IQR))
             
            # annotate p-values
            for ii, strain in enumerate(strain_list, start=1):
                p = pvals.loc[feat, 'pvals_{}'.format(strain)]
                text = ax.get_xticklabels()[ii]
                assert text.get_text() == str(strain)
                p_text = 'P < 0.001' if p < 0.001 else 'P = %.3f' % p
                trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                # plt.plot([ii-.2, ii-.2, ii+.2, ii+.2], 
                #          [0.98, 0.99, 0.99, 0.98], lw=1.5, c='k', transform=trans)
                ax.text(ii, 1.01, p_text, fontsize=9, ha='center', va='bottom', transform=trans)

            ax.set_xlabel(group_by.replace('_',' '), fontsize=12, labelpad=10)
            ax.set_ylabel(feat.replace('_',' '), fontsize=12, labelpad=10)
            # plt.subplots_adjust(right=0.85)
    
            save_path = Path(save_dir) / 'window_{}'.format(window) / '{}.pdf'.format(feat)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
      
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
                                                       imaging_dates=None,
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
    window_stats(metadata,
                 features,
                 group_by='bacteria_strain',
                 control=CONTROL_STRAIN,
                 feature_list=feature_list,
                 windows=sorted(WINDOW_NUMBERS), #sorted(WINDOW_DICT_SECONDS.keys()),
                 save_dir=stats_dir,
                 pvalue_threshold=PVAL_THRESH,
                 fdr_method='fdr_by')

    plot_dir = Path(SAVE_DIR) / "Plots" 
    
    # box plots (for each window) comparing fepD vs BW
    window_boxplots(metadata, 
                    features, 
                    group_by='bacteria_strain',
                    control='BW',
                    feature_list=feature_list, 
                    windows=sorted(WINDOW_NUMBERS), 
                    save_dir=plot_dir,
                    stats_dir=stats_dir)
