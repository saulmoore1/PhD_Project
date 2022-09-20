#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compile acute single worm metadata and feature summaries

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
from visualisation.plotting_helper import sig_asterix, all_in_one_boxplots
from write_data.write import write_list_to_file
from time_series.plot_timeseries import plot_timeseries_feature

from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from tierpsytools.preprocessing.filter_data import select_feat_set

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Keio_Acute_Single_Worm"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Acute_Single_Worm"

# IMAGING_DATES = ['20220206','20220209','20220212']
N_WELLS = 6

CONTROL_STRAIN = 'BW'
FEATURE_SET = ['speed_50th']

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
MIN_NSKEL_PER_VIDEO = None
MIN_NSKEL_SUM = 50
PVAL_THRESH = 0.05

THRESHOLD_N_SECONDS = 75
FPS = 25

VIDEO_LENGTH_SECONDS = 30*60
BIN_SIZE_SECONDS = 5
SMOOTH_WINDOW_SECONDS = 5
BLUELIGHT_TIMEPOINTS_SECONDS = [(300,310), (600,610), (900,910), (1200,1210), (1500,1510)]


# Featsums 10 seconds centred on end of each BL pulse and also 20-30 seconds after end of each BL pulse
WINDOW_DICT = {0:(305,315),1:(330,340),
               2:(605,615),3:(630,640),
               4:(905,915),5:(930,940),
               6:(1205,1215),7:(1230,1240),
               8:(1505,1515),9:(1530,1540),
               10:(1805,1815),11:(1830,1840)}

WINDOW_NAME_DICT = {0:"blue light 1", 1: "20-30 seconds after blue light 1",
                    2:"blue light 2", 3: "20-30 seconds after blue light 2",
                    4:"blue light 3", 5: "20-30 seconds after blue light 3",
                    6:"blue light 4", 7: "20-30 seconds after blue light 4",
                    8:"blue light 5", 9: "20-30 seconds after blue light 5",
                    10:"blue light 6", 11: "20-30 seconds after blue light 6"}

# 305:315,330:340,605:615,630:640,905:915,930:940,1205:1215,1230:1240,1505:1515,1530:1540,
#1805:1815,1830:1840

#%% Functions

def single_worm_stats(metadata,
                      features,
                      group_by='treatment',
                      control='BW',
                      save_dir=None,
                      feature_set=None,
                      pvalue_threshold=0.05,
                      fdr_method='fdr_bh'):
    
    # check case-sensitivity
    assert len(metadata[group_by].unique()) == len(metadata[group_by].str.upper().unique())
    
    if feature_set is not None:
        feature_set = [feature_set] if isinstance(feature_set, str) else feature_set
        assert isinstance(feature_set, list)
        assert(all(f in features.columns for f in feature_set))
    else:
        feature_set = features.columns.tolist()
        
    features = features[feature_set].reindex(metadata.index)

    # print mean sample size
    sample_size = metadata.groupby(group_by).count()
    print("Mean sample size of %s: %d" % (group_by, int(sample_size[sample_size.columns[-1]].mean())))

    n = len(metadata[group_by].unique())
        
    fset = []
    if n > 2:
   
        # Perform ANOVA - is there variation among strains at each window?
        anova_path = Path(save_dir) / 'ANOVA' / 'ANOVA_results.csv'
        anova_path.parent.mkdir(parents=True, exist_ok=True)

        stats, pvals, reject = univariate_tests(X=features, 
                                                y=metadata[group_by], 
                                                control=control, 
                                                test='ANOVA',
                                                comparison_type='multiclass',
                                                multitest_correction=fdr_method,
                                                alpha=pvalue_threshold,
                                                n_permutation_test=None)

        # get effect sizes
        effect_sizes = get_effect_sizes(X=features,
                                        y=metadata[group_by],
                                        control=control,
                                        effect_type=None,
                                        linked_test='ANOVA')

        # compile + save results
        test_results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
        test_results.columns = ['stats','effect_size','pvals','reject']     
        test_results['significance'] = sig_asterix(test_results['pvals'])
        test_results = test_results.sort_values(by=['pvals'], ascending=True) # rank by p-value
        test_results.to_csv(anova_path, header=True, index=True)

        # use reject mask to find significant feature set
        fset = pvals.loc[reject['ANOVA']].sort_values(by='ANOVA', ascending=True).index.to_list()

        if len(fset) > 0:
            print("%d significant features found by ANOVA by '%s' (P<%.2f, %s)" %\
                  (len(fset), group_by, pvalue_threshold, fdr_method))
            anova_sigfeats_path = anova_path.parent / 'ANOVA_sigfeats.txt'
            write_list_to_file(fset, anova_sigfeats_path)
             
    # Perform t-tests
    stats_t, pvals_t, reject_t = univariate_tests(X=features,
                                                  y=metadata[group_by],
                                                  control=control,
                                                  test='t-test',
                                                  comparison_type='binary_each_group',
                                                  multitest_correction=fdr_method,
                                                  alpha=pvalue_threshold)
    
    effect_sizes_t = get_effect_sizes(X=features,
                                      y=metadata[group_by],
                                      control=control,
                                      linked_test='t-test')
    
    stats_t.columns = ['stats_' + str(c) for c in stats_t.columns]
    pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
    reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
    effect_sizes_t.columns = ['effect_size_' + str(c) for c in effect_sizes_t.columns]
    ttest_results = pd.concat([stats_t, pvals_t, reject_t, effect_sizes_t], axis=1)
    
    # save results
    ttest_path = Path(save_dir) / 't-test' / 't-test_results.csv'
    ttest_path.parent.mkdir(exist_ok=True, parents=True)
    ttest_results.to_csv(ttest_path, header=True, index=True)
    
    nsig = sum(reject_t.sum(axis=1) > 0)
    print("%d significant features between any %s vs %s (t-test, P<%.2f, %s)" %\
          (nsig, group_by, control, pvalue_threshold, fdr_method))

    return

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
                                                   imaging_dates=None,
                                                   add_well_annotations=False,
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
          
    # load feature set
    if FEATURE_SET is not None:
        # subset for selected feature set (and remove path curvature features)
        if isinstance(FEATURE_SET, int) and FEATURE_SET in [8,16,256]:
            features = select_feat_set(features, 'tierpsy_{}'.format(FEATURE_SET), append_bluelight=True)
            features = features[[f for f in features.columns if 'path_curvature' not in f]]
        elif isinstance(FEATURE_SET, list) or isinstance(FEATURE_SET, set):
            assert all(f in features.columns for f in FEATURE_SET)
            features = features[FEATURE_SET].copy()
    feature_list = features.columns.tolist()

    # subset metadata results for bluelight videos only 
    bluelight_videos = [i for i in metadata['imgstore_name'] if 'bluelight' in i]
    metadata = metadata[metadata['imgstore_name'].isin(bluelight_videos)]
    
    metadata['window'] = metadata['window'].astype(int)
    window_list = list(metadata['window'].unique())

    # create bins for frame of first food encounter
    bins = [int(b) for b in np.linspace(0, VIDEO_LENGTH_SECONDS*FPS, 
                                        int(VIDEO_LENGTH_SECONDS/BIN_SIZE_SECONDS+1))]
    metadata['first_food_binned_freq'] = pd.cut(x=metadata['first_food_frame'], bins=bins)
    first_food_freq = metadata.groupby('first_food_binned_freq', as_index=False).count()

    # plot histogram of binned frequency of first food encounter 
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12,6))    
    sns.barplot(x=first_food_freq['first_food_binned_freq'].astype(str), 
                y=first_food_freq['first_food_frame'], alpha=0.9, palette='rainbow')        
    ax.set_xticks([x - 0.5 for x in ax.get_xticks()])
    ax.set_xticklabels([str(int(b / FPS)) for b in bins], rotation=45)
    ax.set_xlim(0, np.where(bins > metadata['first_food_frame'].max())[0][0])
    ax.set_xlabel("Time until first food encounter (seconds)", fontsize=15, labelpad=10)
    ax.set_ylabel("Number of videos", fontsize=15, labelpad=10)
    ax.set_title("N = {} videos".format(metadata.shape[0]), loc='right')
    plt.tight_layout()
    
    # save histogram
    save_dir = Path(SAVE_DIR) / 'Plots'
    save_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_dir / "first_food_encounter.pdf")
    plt.close()
    
    # Subset to remove all videos where the worm took >10 seconds (250 frames) to reach the food
    # from the start of the video recording
    # NB: inculding the 'hump' up to around <75 seconds makes no visible difference to the plot
    metadata = metadata[metadata['first_food_frame'] < THRESHOLD_N_SECONDS*FPS]
    features = features.reindex(metadata.index)

    for window in tqdm(window_list):
        meta_window = metadata[metadata['window']==window]
        feat_window = features.reindex(meta_window.index)

        stats_dir = Path(SAVE_DIR) / 'Stats' / WINDOW_NAME_DICT[window]
        plot_dir = Path(SAVE_DIR) / 'Plots' / WINDOW_NAME_DICT[window]
    
        # statistics: perform pairwise t-tests comparing fepD vs BW at each window
        single_worm_stats(meta_window,
                          feat_window,
                          group_by='bacteria_strain',
                          control='BW',
                          save_dir=stats_dir,
                          feature_set=FEATURE_SET,
                          pvalue_threshold=0.05,
                          fdr_method='fdr_bh')
        colour_dict = dict(zip(['BW','fepD'], sns.color_palette('tab10', 2)))
        all_in_one_boxplots(meta_window,
                            feat_window,
                            group_by='bacteria_strain',
                            control='BW',
                            save_dir=plot_dir,
                            ttest_path=stats_dir / 't-test' / 't-test_results.csv',
                            pvalue_threshold=0.05,
                            order=['BW','fepD'],
                            colour_dict=colour_dict,
                            figsize=(8,6),
                            fontsize=20,
                            ylim_minmax=(-120,350),
                            subplots_adjust={'bottom':0.15, 'top':0.9, 'left':0.15, 'right':0.95})

    metadata = metadata[metadata['window']==0]

    # plot speed timeseries
    plot_timeseries_feature(metadata,
                            project_dir=Path(PROJECT_DIR),
                            save_dir=Path(SAVE_DIR) / 'timeseries-speed',
                            group_by='bacteria_strain',
                            control='BW',
                            groups_list=None,
                            feature='speed',
                            n_wells=6,
                            bluelight_stim_type='bluelight',
                            video_length_seconds=VIDEO_LENGTH_SECONDS,
                            bluelight_timepoints_seconds=BLUELIGHT_TIMEPOINTS_SECONDS,
                            smoothing=10,
                            fps=FPS,
                            ylim_minmax=(-50,300))  
    
    