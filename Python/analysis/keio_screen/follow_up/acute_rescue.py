#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse Keio Follow-up Acute Effect Antioxidant Rescue experiment
- window feature summaries for Ziwei's optimal windows around each bluelight stimulus
- Bluelight delivered for 10 seconds every 5 minutes, for a total of 45 minutes

When do we start to see an effect on worm behaviour? At which timepoint/window? 
Do we still see arousal of worms on siderophore mutants, even after a short period of time?

@author: sm5911
@date: 20/01/2022

"""

#%% Imports

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
# from scipy.stats import zscore

from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from visualisation.plotting_helper import sig_asterix, boxplots_sigfeats, all_in_one_boxplots
from write_data.write import write_list_to_file
from time_series.plot_timeseries import plot_window_timeseries_feature #plot_timeseries_motion_mode

from tierpsytools.preprocessing.filter_data import select_feat_set
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes

#%% Globals

PROJECT_DIR = '/Volumes/hermes$/Keio_Acute_Rescue'
SAVE_DIR = '/Users/sm5911/Documents/Keio_Acute_Rescue'

N_WELLS = 6

FEATURE_SET = ['speed_50th']

nan_threshold_row = 0.8
nan_threshold_col = 0.05

scale_outliers_box = False

ALL_WINDOWS = False
WINDOW_LIST = None # if ALL_WINDOWS is False

# mapping dictionary - windows summary window number to corresponding timestamp (seconds)
# WINDOW_FRAME_DICT = {0:(290,300), 1:(305,315), 2:(315,325), 
#                      3:(590,600), 4:(605,615), 5:(615,625), 
#                      6:(890,900), 7:(905,915), 8:(915,925), 
#                      9:(1190,1200), 10:(1205,1215), 11:(1215,1225), 
#                      12:(1490,1500), 13:(1505,1515), 14:(1515,1525), 
#                      15:(1790,1800), 16:(1805,1815), 17:(1815,1825), 
#                      18:(2090,2100), 19:(2105,2115), 20:(2115,2125), 
#                      21:(2390,2400), 22:(2405,2415), 23:(2415,2425)}

# Featsums 10 seconds centred on end of each BL pulse and also 20-30 seconds after end of each BL pulse
WINDOW_DICT = {0:(305,315),1:(330,340),
               2:(605,615),3:(630,640),
               4:(905,915),5:(930,940),
               6:(1205,1215),7:(1230,1240),
               8:(1505,1515),9:(1530,1540),
               10:(1805,1815),11:(1830,1840),
               12:(2105,2115),13:(2130,2140),
               14:(2405,2415),15:(2430,2440)}

WINDOW_NAME_DICT = {0:"blue light 1", 1: "20-30 seconds after blue light 1",
                    2:"blue light 2", 3: "20-30 seconds after blue light 2",
                    4:"blue light 3", 5: "20-30 seconds after blue light 3",
                    6:"blue light 4", 7: "20-30 seconds after blue light 4",
                    8:"blue light 5", 9: "20-30 seconds after blue light 5",
                    10:"blue light 6", 11: "20-30 seconds after blue light 6",
                    12:"blue light 7", 13: "20-30 seconds after blue light 7",
                    14:"blue light 8", 15: "20-30 seconds after blue light 8"}

#305:315,330:340,605:615,630:640,905:915,930:940,1205:1215,1230:1240,1505:1515,1530:1540,
#1805:1815,1830:1840,2105:2115,2130:2140,2405:2415,2430:2440

BLUELIGHT_TIMEPOINTS_MINUTES = [5,10,15,20,25,30,35,40]
FPS = 25
VIDEO_LENGTH_SECONDS = 45*60

#%% Functions

def acute_rescue_stats(metadata,
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

# def acute_rescue_stats(features, 
#                        metadata, 
#                        save_dir, 
#                        control_strain, 
#                        control_antioxidant, 
#                        control_window,
#                        feature_set=FEATURE_SET,
#                        fdr_method='fdr_by',
#                        pval_threshold=0.05):
#     """ Pairwise t-tests for each window comparing worm 'motion mode paused fraction' on 
#         Keio mutants vs BW control 
        
#         # One could fit a multiple linear regression model: to account for strain*antioxidant in a 
#         # single model: Y (motion_mode) = b0 + b1*X1 (strain) + b2*X2 (antiox) + e (error)
#         # But this is a different type of question: we care about difference in means between 
#         # fepD vs BW (albeit under different antioxidant treatments), and not about modelling their 
#         # relationship, therefore individual t-tests (multiple-test-corrected) should suffice
        
#         1. For each treatment condition, t-tests comparing fepD vs BW for motion_mode
        
#         2. For fepD and BW separately, f-tests for equal variance among antioxidant treatment groups,
#         then ANOVA tests for significant differences between antioxidants, then individual t-tests
#         comparing each treatment to control
        
#         Inputs
#         ------
#         features, metadata : pandas.DataFrame
        
#         window_list : list
#             List of windows (int) to perform statistics (separately for each window provided, 
#             p-values are adjusted for multiple test correction)
        
#         save_dir : str
#             Directory to save statistics results
            
#         control_strain
#         control_antioxidant
#         fdr_method
        
#     """

#     stats_dir =  Path(save_dir) / "Stats" / 'fdr_bh'
#     stats_dir.mkdir(parents=True, exist_ok=True)

#     strain_list = [control_strain] + [s for s in set(metadata['gene_name'].unique()) if s != control_strain]  
#     antiox_list = [control_antioxidant] + [a for a in set(metadata['antioxidant'].unique()) if 
#                                            a != control_antioxidant]
#     window_list = [control_window] + [w for w in set(metadata['window'].unique()) if w != control_window]

#     # categorical variables to investigate: 'gene_name', 'antioxidant' and 'window'
#     print("\nInvestigating difference in fraction of worms paused between hit strain and control " +
#           "(for each window), in the presence/absence of antioxidants:\n")    

#     # For each strain separately...
#     for strain in strain_list:
#         strain_meta = metadata[metadata['gene_name']==strain]
#         strain_feat = features.reindex(strain_meta.index)

#         # 1. Is there any variation in fraction paused wtr antioxidant treatment?
#         #    - ANOVA on pooled window data, then pairwise t-tests for each antioxidant
        
#         print("Performing ANOVA on pooled window data for significant variation in fraction " +
#               "of worms paused among different antioxidant treatments for %s..." % strain)
        
#         # perform ANOVA (correct for multiple comparisons)             
#         stats, pvals, reject = univariate_tests(X=strain_feat[feature_set], 
#                                                 y=strain_meta['antioxidant'], 
#                                                 test='ANOVA',
#                                                 control=control_antioxidant,
#                                                 comparison_type='multiclass',
#                                                 multitest_correction=fdr_method,
#                                                 alpha=pval_threshold,
#                                                 n_permutation_test=None) # 'all'
    
#         # get effect sizes
#         effect_sizes = get_effect_sizes(X=strain_feat[feature_set], 
#                                         y=strain_meta['antioxidant'],
#                                         control=control_antioxidant,
#                                         effect_type=None,
#                                         linked_test='ANOVA')
    
#         # compile
#         test_results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
#         test_results.columns = ['stats','effect_size','pvals','reject']     
#         test_results['significance'] = sig_asterix(test_results['pvals'])
#         test_results = test_results.sort_values(by=['pvals'], ascending=True) # rank pvals
        
#         # save results
#         anova_path = Path(stats_dir) / 'ANOVA_{}_variation_across_antioxidants.csv'.format(strain)
#         test_results.to_csv(anova_path, header=True, index=True)
              
#         print("Performing t-tests comparing each antioxidant treatment to None (pooled window data)")
        
#         stats_t, pvals_t, reject_t = univariate_tests(X=strain_feat[feature_set],
#                                                       y=strain_meta['antioxidant'],
#                                                       test='t-test',
#                                                       control=control_antioxidant,
#                                                       comparison_type='binary_each_group',
#                                                       multitest_correction=fdr_method,
#                                                       alpha=pval_threshold)
#         effect_sizes_t =  get_effect_sizes(X=strain_feat[feature_set], 
#                                            y=strain_meta['antioxidant'], 
#                                            control=control_antioxidant,
#                                            effect_type=None,
#                                            linked_test='t-test')
            
#         # compile + save t-test results
#         stats_t.columns = ['stats_' + str(c) for c in stats_t.columns]
#         pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
#         reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
#         effect_sizes_t.columns = ['effect_size_' + str(c) for c in effect_sizes_t.columns]
#         ttest_results = pd.concat([stats_t, effect_sizes_t, pvals_t, reject_t], axis=1)
#         ttest_save_path = stats_dir / 't-test_{}_antioxidant_results.csv'.format(strain)
#         ttest_save_path.parent.mkdir(exist_ok=True, parents=True)
#         ttest_results.to_csv(ttest_save_path, header=True, index=True)
    
#         # 2. Is there any variation in fraction paused wrt window (time) across the videos?
#         #    - ANOVA on pooled antioxidant data, then pairwise for each window
        
#         print("Performing ANOVA on pooled antioxidant data for significant variation in fraction " +
#               "of worms paused across (bluelight) window summaries for %s..." % strain)
        
#         # perform ANOVA (correct for multiple comparisons)
#         stats, pvals, reject = univariate_tests(X=strain_feat[feature_set],
#                                                 y=strain_meta['window'],
#                                                 test='ANOVA',
#                                                 control=control_window,
#                                                 comparison_type='multiclass',
#                                                 multitest_correction=fdr_method,
#                                                 alpha=pval_threshold,
#                                                 n_permutation_test=None)
        
#         # get effect sizes
#         effect_sizes = get_effect_sizes(X=strain_feat[feature_set],
#                                         y=strain_meta['window'],
#                                         control=control_window,
#                                         effect_type=None,
#                                         linked_test='ANOVA')

#         # compile
#         test_results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
#         test_results.columns = ['stats','effect_size','pvals','reject']     
#         test_results['significance'] = sig_asterix(test_results['pvals'])
#         test_results = test_results.sort_values(by=['pvals'], ascending=True) # rank pvals
        
#         # save results
#         anova_path = Path(stats_dir) / 'ANOVA_{}_variation_across_windows.csv'.format(strain)
#         test_results.to_csv(anova_path, header=True, index=True)

#         print("Performing t-tests comparing each window with the first (pooled antioxidant data)")
        
#         stats_t, pvals_t, reject_t = univariate_tests(X=strain_feat[feature_set],
#                                                       y=strain_meta['window'],
#                                                       test='t-test',
#                                                       control=control_window,
#                                                       comparison_type='binary_each_group',
#                                                       multitest_correction=fdr_method,
#                                                       alpha=pval_threshold)
#         effect_sizes_t =  get_effect_sizes(X=strain_feat[feature_set], 
#                                            y=strain_meta['window'], 
#                                            control=control_window,
#                                            effect_type=None,
#                                            linked_test='t-test')
            
#         # compile + save t-test results
#         stats_t.columns = ['stats_' + str(c) for c in stats_t.columns]
#         pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
#         reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
#         effect_sizes_t.columns = ['effect_size_' + str(c) for c in effect_sizes_t.columns]
#         ttest_results = pd.concat([stats_t, effect_sizes_t, pvals_t, reject_t], axis=1)
#         ttest_save_path = stats_dir / 't-test_{}_window_results.csv'.format(strain)
#         ttest_save_path.parent.mkdir(exist_ok=True, parents=True)
#         ttest_results.to_csv(ttest_save_path, header=True, index=True)   
         
#     # Pairwise t-tests - is there a difference between strain vs control?

#     control_meta = metadata[metadata['gene_name']==control_strain]
#     control_feat = features.reindex(control_meta.index)
#     control_df = control_meta.join(control_feat[feature_set])

#     for strain in strain_list[1:]: # skip control_strain at first index postion         
#         strain_meta = metadata[metadata['gene_name']==strain]
#         strain_feat = features.reindex(strain_meta.index)
#         strain_df = strain_meta.join(strain_feat[feature_set])

#         # 3. Is there a difference between strain vs control at any window?
        
#         print("\nPairwise t-tests for each window (pooled antioxidants) comparing fraction of " + 
#               "worms paused on %s vs control:" % strain)

#         stats, pvals, reject = pairwise_ttest(control_df, 
#                                               strain_df, 
#                                               feature_list=feature_set, 
#                                               group_by='window', 
#                                               fdr_method=fdr_method,
#                                               fdr=0.05)
 
#         # compile table of results
#         stats.columns = ['stats_' + str(c) for c in stats.columns]
#         pvals.columns = ['pvals_' + str(c) for c in pvals.columns]
#         reject.columns = ['reject_' + str(c) for c in reject.columns]
#         test_results = pd.concat([stats, pvals, reject], axis=1)
        
#         # save results
#         ttest_strain_path = stats_dir / 'pairwise_ttests' / 'window' /\
#                             '{}_window_results.csv'.format(strain)
#         ttest_strain_path.parent.mkdir(parents=True, exist_ok=True)
#         test_results.to_csv(ttest_strain_path, header=True, index=True)
                             
#         # for each antioxidant treatment condition...
#         for antiox in antiox_list:
#             print("Pairwise t-tests for each window comparing fraction of " + 
#                   "worms paused on %s vs control with '%s'" % (strain, antiox))

#             antiox_control_df = control_df[control_df['antioxidant']==antiox]
#             antiox_strain_df = strain_df[strain_df['antioxidant']==antiox]
            
#             stats, pvals, reject = pairwise_ttest(antiox_control_df,
#                                                   antiox_strain_df,
#                                                   feature_list=feature_set,
#                                                   group_by='window',
#                                                   fdr_method=fdr_method,
#                                                   fdr=0.05)
        
#             # compile table of results
#             stats.columns = ['stats_' + str(c) for c in stats.columns]
#             pvals.columns = ['pvals_' + str(c) for c in pvals.columns]
#             reject.columns = ['reject_' + str(c) for c in reject.columns]
#             test_results = pd.concat([stats, pvals, reject], axis=1)
            
#             # save results
#             ttest_strain_path = stats_dir / 'pairwise_ttests' / 'window' /\
#                                 '{0}_{1}_window_results.csv'.format(strain, antiox)
#             ttest_strain_path.parent.mkdir(parents=True, exist_ok=True)
#             test_results.to_csv(ttest_strain_path, header=True, index=True)

#         # 4. Is there a difference between strain vs control for any antioxidant?

#         print("\nPairwise t-tests for each antioxidant (pooled windows) comparing fraction of " + 
#               "worms paused on %s vs control:" % strain)

#         stats, pvals, reject = pairwise_ttest(control_df, 
#                                               strain_df, 
#                                               feature_list=feature_set, 
#                                               group_by='antioxidant', 
#                                               fdr_method=fdr_method,
#                                               fdr=0.05)
 
#         # compile table of results
#         stats.columns = ['stats_' + str(c) for c in stats.columns]
#         pvals.columns = ['pvals_' + str(c) for c in pvals.columns]
#         reject.columns = ['reject_' + str(c) for c in reject.columns]
#         test_results = pd.concat([stats, pvals, reject], axis=1)
        
#         # save results
#         ttest_strain_path = stats_dir / 'pairwise_ttests' / 'antioxidant' /\
#                             '{}_antioxidant_results.csv'.format(strain)
#         ttest_strain_path.parent.mkdir(parents=True, exist_ok=True)
#         test_results.to_csv(ttest_strain_path, header=True, index=True)
                             
#         # For each window...
#         for window in window_list:
#             print("Pairwise t-tests for each antioxidant comparing fraction of " + 
#                   "worms paused on %s vs control at window %d" % (strain, window))

#             window_control_df = control_df[control_df['window']==window]
#             window_strain_df = strain_df[strain_df['window']==window]
            
#             stats, pvals, reject = pairwise_ttest(window_control_df,
#                                                   window_strain_df,
#                                                   feature_list=feature_set,
#                                                   group_by='antioxidant',
#                                                   fdr_method=fdr_method,
#                                                   fdr=0.05)
        
#             # compile table of results
#             stats.columns = ['stats_' + str(c) for c in stats.columns]
#             pvals.columns = ['pvals_' + str(c) for c in pvals.columns]
#             reject.columns = ['reject_' + str(c) for c in reject.columns]
#             test_results = pd.concat([stats, pvals, reject], axis=1)
            
#             # save results
#             ttest_strain_path = stats_dir / 'pairwise_ttests' / 'antioxidant' /\
#                                 '{0}_window{1}_antioxidant_results.csv'.format(strain, window)
#             ttest_strain_path.parent.mkdir(parents=True, exist_ok=True)
#             test_results.to_csv(ttest_strain_path, header=True, index=True)
               
#     return

# =============================================================================
# def analyse_acute_rescue(features, 
#                          metadata,
#                          save_dir,
#                          control_strain, 
#                          control_antioxidant, 
#                          control_window,
#                          fdr_method='fdr_by',
#                          pval_threshold=0.05,
#                          remove_outliers=False):
#  
#     stats_dir =  Path(save_dir) / "Stats" / fdr_method
#     plot_dir = Path(save_dir) / "Plots" / fdr_method
# 
#     strain_list = [control_strain] + [s for s in metadata['gene_name'].unique() if s != control_strain]  
#     antiox_list = [control_antioxidant] + [a for a in metadata['antioxidant'].unique() if 
#                                            a != control_antioxidant]
#     window_list = [control_window] + [w for w in metadata['window'].unique() if w != control_window]
# 
#     # categorical variables to investigate: 'gene_name', 'antioxidant' and 'window'
#     print("\nInvestigating difference in fraction of worms paused between hit strain and control " +
#           "(for each window), in the presence/absence of antioxidants:\n")    
#             
#     # plot dates as different colours (in loop)
#     date_lut = dict(zip(list(metadata['date_yyyymmdd'].unique()), 
#                         sns.color_palette('Greys', n_colors=len(metadata['date_yyyymmdd'].unique()))))
#         
#     for strain in strain_list[1:]: # skip control_strain
#         plot_meta = metadata[np.logical_or(metadata['gene_name']==strain, 
#                                            metadata['gene_name']==control_strain)]
#         plot_feat = features.reindex(plot_meta.index)
#         plot_df = plot_meta.join(plot_feat[[FEATURE]])
#         
#         # Is there a difference between strain vs control at any window? (pooled antioxidant data)
#         print("Plotting windows for %s vs control" % strain)
#         plt.close('all')
#         fig, ax = plt.subplots(figsize=((len(window_list) if len(window_list) >= 20 else 12),8))
#         ax = sns.boxplot(x='window', y=FEATURE, hue='gene_name', hue_order=strain_list, order=window_list,
#                          data=plot_df, palette='tab10', dodge=True, ax=ax)
#         for date in date_lut.keys():
#             date_df = plot_df[plot_df['date_yyyymmdd']==date]   
#             ax = sns.stripplot(x='window', y=FEATURE, hue='gene_name', order=window_list,
#                                hue_order=strain_list, data=date_df, 
#                                palette={control_strain:date_lut[date], strain:date_lut[date]}, 
#                                alpha=0.7, size=4, dodge=True, ax=ax)
#         n_labs = len(plot_df['gene_name'].unique())
#         handles, labels = ax.get_legend_handles_labels()
#         ax.legend(handles[:n_labs], labels[:n_labs], fontsize=15, frameon=False, loc='upper right')
#                 
#         # scale plot to omit outliers (>2.5*IQR from mean)
#         if scale_outliers_box:
#             grouped_strain = plot_df.groupby('window')
#             y_bar = grouped_strain[FEATURE].median() # median is less skewed by outliers
#             # Computing IQR
#             Q1 = grouped_strain[FEATURE].quantile(0.25)
#             Q3 = grouped_strain[FEATURE].quantile(0.75)
#             IQR = Q3 - Q1
#             plt.ylim(-0.02, max(y_bar) + 3 * max(IQR))
#             
#         # load t-test results + annotate p-values on plot
#         for ii, window in enumerate(window_list):
#             ttest_strain_path = stats_dir / 'pairwise_ttests' / 'window' /\
#                                 '{}_window_results.csv'.format(strain)
#             ttest_strain_table = pd.read_csv(ttest_strain_path, index_col=0, header=0)
#             strain_pvals_t = ttest_strain_table[[c for c in ttest_strain_table if "pvals_" in c]] 
#             strain_pvals_t.columns = [c.split('pvals_')[-1] for c in strain_pvals_t.columns] 
#             p = strain_pvals_t.loc[FEATURE, str(window)]
#             text = ax.get_xticklabels()[ii]
#             assert text.get_text() == str(window)
#             p_text = 'P<0.001' if p < 0.001 else 'P=%.3f' % p
#             #y = (y_bar[antiox] + 2 * IQR[antiox]) if scale_outliers_box else plot_df[feature].max()
#             #h = (max(IQR) / 10) if scale_outliers_box else (y - plot_df[feature].min()) / 50
#             trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
#             plt.plot([ii-.3, ii-.3, ii+.3, ii+.3], 
#                      [0.98, 0.99, 0.99, 0.98], #[y+h, y+2*h, y+2*h, y+h], 
#                      lw=1.5, c='k', transform=trans)
#             ax.text(ii, 1.01, p_text, fontsize=9, ha='center', va='bottom', transform=trans,
#                     rotation=(0 if len(window_list) <= 20 else 90))
#             
#         ax.set_xticks(range(len(window_list)+1))
#         xlabels = [str(int(WINDOW_FRAME_DICT[w][0]/60)) for w in window_list]
#         ax.set_xticklabels(xlabels)
#         x_text = 'Time (minutes)' if ALL_WINDOWS else 'Time of bluelight 10-second burst (minutes)'
#         ax.set_xlabel(x_text, fontsize=15, labelpad=10)
#         ax.set_ylabel(FEATURE.replace('_',' '), fontsize=15, labelpad=10)
#         
#         fig_savepath = plot_dir / 'window_boxplots' / strain / (FEATURE + '.png')
#         fig_savepath.parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(fig_savepath)
#     
#     
#         # Is there a difference between strain vs control for any antioxidant? (pooled window data)
#         plt.close('all')
#         fig, ax = plt.subplots(figsize=(10,8))
#         ax = sns.boxplot(x='antioxidant', y=FEATURE, hue='gene_name', hue_order=strain_list, data=plot_df,
#                           palette='tab10', dodge=True, order=antiox_list)
#         ax = sns.swarmplot(x='antioxidant', y=FEATURE, hue='gene_name', hue_order=strain_list, data=plot_df,
#                           color='k', alpha=0.7, size=4, dodge=True, order=antiox_list)
#         n_labs = len(plot_df['gene_name'].unique())
#         handles, labels = ax.get_legend_handles_labels()
#         ax.legend(handles[:n_labs], labels[:n_labs], fontsize=15, frameon=False, loc='upper right')
#         ax.set_xlabel('antioxidant', fontsize=15, labelpad=10)
#         ax.set_ylabel(FEATURE.replace('_',' '), fontsize=15, labelpad=10)
#         
#         # scale plot to omit outliers (>2.5*IQR from mean)
#         if scale_outliers_box:
#             grouped_strain = plot_df.groupby('antioxidant')
#             y_bar = grouped_strain[FEATURE].median() # median is less skewed by outliers
#             # Computing IQR
#             Q1 = grouped_strain[FEATURE].quantile(0.25)
#             Q3 = grouped_strain[FEATURE].quantile(0.75)
#             IQR = Q3 - Q1
#             plt.ylim(min(y_bar) - 2 * max(IQR), max(y_bar) + 2 * max(IQR))
#             
#         # annotate p-values
#         for ii, antiox in enumerate(antiox_list):
#             ttest_strain_path = stats_dir / 'pairwise_ttests' / 'antioxidant' /\
#                                 '{}_antioxidant_results.csv'.format(strain)
#             ttest_strain_table = pd.read_csv(ttest_strain_path, index_col=0, header=0)
#             strain_pvals_t = ttest_strain_table[[c for c in ttest_strain_table if "pvals_" in c]] 
#             strain_pvals_t.columns = [c.split('pvals_')[-1] for c in strain_pvals_t.columns] 
#             p = strain_pvals_t.loc[FEATURE, antiox]
#             text = ax.get_xticklabels()[ii]
#             assert text.get_text() == antiox
#             p_text = 'P < 0.001' if p < 0.001 else 'P = %.3f' % p
#             #y = (y_bar[antiox] + 2 * IQR[antiox]) if scale_outliers_box else plot_df[feature].max()
#             #h = (max(IQR) / 10) if scale_outliers_box else (y - plot_df[feature].min()) / 50
#             trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
#             plt.plot([ii-.2, ii-.2, ii+.2, ii+.2], 
#                       [0.8, 0.81, 0.81, 0.8], #[y+h, y+2*h, y+2*h, y+h], 
#                       lw=1.5, c='k', transform=trans)
#             ax.text(ii, 0.82, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
#                 
#         fig_savepath = plot_dir / 'antioxidant_boxplots' / strain / (FEATURE + '.png')
#         fig_savepath.parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(fig_savepath)
#         
#     # Plot for each strain separately to see whether antioxidants had an effect at all
#     for strain in strain_list:
#             
#         plt.close('all')
#         fig, ax = plt.subplots(figsize=(10,8))
#         ax = sns.boxplot(x='antioxidant', y=FEATURE, order=antiox_list, 
#                          dodge=True, data=plot_df[plot_df['gene_name']==strain])
#         ax = sns.swarmplot(x='antioxidant', y=FEATURE, order=antiox_list, 
#                            dodge=True, data=plot_df[plot_df['gene_name']==strain],
#                            alpha=0.7, size=4, color='k')        
#         n_labs = len(plot_df['antioxidant'].unique())
#         handles, labels = ax.get_legend_handles_labels()
#         ax.legend(handles[:n_labs], labels[:n_labs], fontsize=15, frameon=False, loc='upper right')
#         ax.set_xlabel('antioxidant', fontsize=15, labelpad=10)
#         ax.set_ylabel(FEATURE.replace('_',' '), fontsize=15, labelpad=10)
#         
#         # scale plot to omit outliers (>2.5*IQR from mean)
#         if scale_outliers_box:
#             grouped_strain = plot_df.groupby('antioxidant')
#             y_bar = grouped_strain[FEATURE].median() # median is less skewed by outliers
#             # Computing IQR
#             Q1 = grouped_strain[FEATURE].quantile(0.25)
#             Q3 = grouped_strain[FEATURE].quantile(0.75)
#             IQR = Q3 - Q1
#             plt.ylim(min(y_bar) - 2 * max(IQR), max(y_bar) + 2 * max(IQR))
#             
#         # annotate p-values
#         for ii, antiox in enumerate(antiox_list):
#             if antiox == control_antioxidant:
#                 continue
#             # load antioxidant results for strain
#             ttest_strain_path = stats_dir / 't-test_{}_antioxidant_results.csv'.format(strain)
#             ttest_strain_table = pd.read_csv(ttest_strain_path, index_col=0, header=0)
#             strain_pvals_t = ttest_strain_table[[c for c in ttest_strain_table if "pvals_" in c]] 
#             strain_pvals_t.columns = [c.split('pvals_')[-1] for c in strain_pvals_t.columns] 
#             p = strain_pvals_t.loc[FEATURE, antiox]
#             text = ax.get_xticklabels()[ii]
#             assert text.get_text() == antiox
#             p_text = 'P < 0.001' if p < 0.001 else 'P = %.3f' % p
#             trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
#             #plt.plot([ii-.2, ii-.2, ii+.2, ii+.2], [0.98, 0.99, 0.98, 0.99], lw=1.5, c='k', transform=trans)
#             ax.text(ii, 1.01, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
#                 
#         plt.title(strain, fontsize=18, pad=30)
#         fig_savepath = plot_dir / 'antioxidant_boxplots' / strain / (FEATURE + '.png')
#         fig_savepath.parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(fig_savepath)
#         
#         
#     # Hierarchical Clustering Analysis
#     #   - Clustermap of features by strain, to see if data cluster into groups
#     #   - Control data is clustered first, feature order is stored and ordering applied to 
#     #     full data for comparison
#     
#     # subset for Tierpsy top16 features only
#     features = select_feat_set(features, tierpsy_set_name='tierpsy_16', append_bluelight=False)
#     
#     # Ensure no NaNs or features with zero standard deviation before normalisation
#     assert not features.isna().sum(axis=0).any()
#     assert not (features.std(axis=0) == 0).any()
#        
#     # Extract data for control
#     control_feat_df = features[metadata['gene_name']==control_strain]
#     control_meta_df = metadata.reindex(control_feat_df.index)
#     
#     control_feat_df, control_meta_df = clean_summary_results(features=control_feat_df,
#                                                              metadata=control_meta_df,
#                                                              imputeNaN=False)
#     
# 
#     #zscores = (df-df.mean())/df.std() # minus mean, divide by std
#     controlZ_feat_df = control_feat_df.apply(zscore, axis=0)
# 
#     # plot clustermap for control        
#     control_clustermap_path = plot_dir / 'heatmaps' / '{}_clustermap.pdf'.format(control_strain)
#     cg = plot_clustermap(featZ=controlZ_feat_df,
#                          meta=control_meta_df,
#                          row_colours=True,
#                          group_by=['gene_name','antioxidant'],
#                          col_linkage=None,
#                          method='complete',#[linkage, complete, average, weighted, centroid]
#                          figsize=(20,10),
#                          show_xlabels=True,
#                          label_size=15,
#                          sub_adj={'bottom':0.6,'left':0,'top':1,'right':0.85},
#                          saveto=control_clustermap_path,
#                          bluelight_col_colours=False)
# 
#     # extract clustered feature order
#     clustered_features = np.array(controlZ_feat_df.columns)[cg.dendrogram_col.reordered_ind]
#      
#     featZ_df = features.apply(zscore, axis=0)
#     
#     # Save stats table to CSV   
#     # if not stats_path.exists():
#     #     # Add z-normalised values
#     #     z_stats = featZ_df.join(meta_df[GROUPING_VAR]).groupby(by=GROUPING_VAR).mean().T
#     #     z_mean_cols = ['z-mean ' + v for v in z_stats.columns.to_list()]
#     #     z_stats.columns = z_mean_cols
#     #     stats_table = stats_table.join(z_stats)
#     #     first_cols = [m for m in stats_table.columns if 'mean' in m]
#     #     last_cols = [c for c in stats_table.columns if c not in first_cols]
#     #     first_cols.extend(last_cols)
#     #     stats_table = stats_table[first_cols].reset_index()
#     #     first_cols.insert(0, 'feature')
#     #     stats_table.columns = first_cols
#     #     stats_table['feature'] = [' '.join(f.split('_')) for f in stats_table['feature']]
#     #     stats_table = stats_table.sort_values(by='{} p-value'.format((T_TEST_NAME if 
#     #                                  len(run_strain_list) == 2 else TEST_NAME)), ascending=True)
#     #     stats_table_path = stats_dir / 'stats_summary_table.csv'
#     #     stats_table.to_csv(stats_table_path, header=True, index=None)
#     
#     # Clustermap of full data - antioxidants  
#     full_clustermap_path = plot_dir / 'heatmaps' / '{}_clustermap.pdf'.format('gene_antioxidant')
#     _ = plot_clustermap(featZ=featZ_df,
#                         meta=metadata, 
#                         group_by=['gene_name','antioxidant'],
#                         col_linkage=None,
#                         method='complete',
#                         figsize=(20,10),
#                         show_xlabels=True,
#                         label_size=15,
#                         sub_adj={'bottom':0.6,'left':0,'top':1,'right':0.85},
#                         saveto=full_clustermap_path,
#                         bluelight_col_colours=False)
# 
#     # Heatmap of strain/antioxidant treatment, ordered by control clustered feature order
#     heatmap_date_path = plot_dir / 'heatmaps' / 'gene_antioxidant_heatmap.pdf'
#     plot_barcode_heatmap(featZ=featZ_df[clustered_features], 
#                          meta=metadata, 
#                          group_by=['gene_name','antioxidant'], 
#                          pvalues_series=None,
#                          saveto=heatmap_date_path,
#                          figsize=(20,6),
#                          sns_colour_palette="Pastel1")    
#       
#     # Clustermap of full data - windows  
#     full_clustermap_path = plot_dir / 'heatmaps' / '{}_clustermap.pdf'.format('gene_window')
#     _ = plot_clustermap(featZ=featZ_df,
#                         meta=metadata, 
#                         group_by=['gene_name','window'],
#                         col_linkage=None,
#                         method='complete',
#                         figsize=(20,10),
#                         show_xlabels=True,
#                         label_size=15,
#                         sub_adj={'bottom':0.6,'left':0,'top':1,'right':0.85},
#                         saveto=full_clustermap_path,
#                         bluelight_col_colours=False)
#                   
#     # Principal Components Analysis (PCA)
# 
#     if remove_outliers:
#         outlier_path = plot_dir / 'mahalanobis_outliers.pdf'
#         features, inds = remove_outliers_pca(df=features, 
#                                             features_to_analyse=None, 
#                                             saveto=outlier_path)
#         metadata = metadata.reindex(features.index)
#         featZ_df = features.apply(zscore, axis=0)
#   
#     # project data + plot PCA
#     #from tierpsytools.analysis.decomposition import plot_pca
#     pca_dir = plot_dir / 'PCA'
#     _ = plot_pca(featZ=featZ_df, 
#                  meta=metadata, 
#                  group_by='gene_name', 
#                  n_dims=2,
#                  control=control_strain,
#                  var_subset=None, 
#                  saveDir=pca_dir,
#                  PCs_to_keep=10,
#                  n_feats2print=10,
#                  sns_colour_palette="Set1",
#                  figsize=(12,8),
#                  sub_adj={'bottom':0.1,'left':0.1,'top':0.95,'right':0.7},
#                  legend_loc=[1.02,0.6],
#                  hypercolor=False) 
#          
#     # t-distributed Stochastic Neighbour Embedding (tSNE)
# 
#     tsne_dir = plot_dir / 'tSNE'
#     perplexities = [5,15,30] # NB: perplexity parameter should be roughly equal to group size
#     
#     _ = plot_tSNE(featZ=featZ_df,
#                   meta=metadata,
#                   group_by='gene_name',
#                   var_subset=None,
#                   saveDir=tsne_dir,
#                   perplexities=perplexities,
#                   figsize=(8,8),
#                   label_size=15,
#                   size=20,
#                   sns_colour_palette="Set1")
#    
#     # Uniform Manifold Projection (UMAP)
# 
#     umap_dir = plot_dir / 'UMAP'
#     n_neighbours = [5,15,30] # NB: n_neighbours parameter should be roughly equal to group size
#     min_dist = 0.1 # Minimum distance parameter
#     
#     _ = plot_umap(featZ=featZ_df,
#                   meta=metadata,
#                   group_by='gene_name',
#                   var_subset=None,
#                   saveDir=umap_dir,
#                   n_neighbours=n_neighbours,
#                   min_dist=min_dist,
#                   figsize=(8,8),
#                   label_size=15,
#                   size=20,
#                   sns_colour_palette="Set1")
#     
#     _ = plot_pca_2var(featZ=featZ_df, 
#                       meta=metadata, 
#                       var1='gene_name',
#                       var2='antioxidant',
#                       saveDir=pca_dir,
#                       PCs_to_keep=10,
#                       n_feats2print=10,
#                       sns_colour_palette="Set1",
#                       label_size=15,
#                       figsize=[9,8],
#                       sub_adj={'bottom':0,'left':0,'top':1,'right':1})
# 
#     return
# =============================================================================

def acute_rescue_boxplots(metadata,
                          features,
                          control,
                          group_by='treatment',
                          feature_set=None,
                          save_dir=None,
                          stats_dir=None,
                          pvalue_threshold=0.05,
                          fdr_method='fdr_by'):
    
    feature_set = features.columns.tolist() if feature_set is None else feature_set
    assert isinstance(feature_set, list) and all(f in features.columns for f in feature_set)
                    
    # load t-test results for window
    if stats_dir is not None:
        ttest_path = Path(stats_dir) / 't-test' / 't-test_results.csv'
        ttest_df = pd.read_csv(ttest_path, header=0, index_col=0)
        pvals = ttest_df[[c for c in ttest_df.columns if 'pval' in c]]
        pvals.columns = [c.replace('pvals_','') for c in pvals.columns]
    
    boxplots_sigfeats(features,
                      y_class=metadata[group_by],
                      control=control,
                      pvals=pvals if stats_dir is not None else None,
                      z_class=None,
                      feature_set=feature_set,
                      saveDir=Path(save_dir),
                      drop_insignificant=True if feature_set is None else False,
                      p_value_threshold=pvalue_threshold,
                      scale_outliers=True)
    
    return

# def acute_rescue_timeseries(metadata, 
#                             project_dir, 
#                             save_dir, 
#                             group_by='treatment',
#                             control='wild_type_None',
#                             bluelight_windows_separately=False,
#                             n_wells=N_WELLS,
#                             smoothing=10):
#     """ Timeseries plots of repeated bluelight stimulation of BW and fepD
#         (10 seconds BL delivered every 30 minutes, for 5 hours total)
#     """
        
#     # get timeseries for BW
#     control_ts = get_strain_timeseries(metadata[metadata[group_by]==control], 
#                                        project_dir=project_dir, 
#                                        strain=control,
#                                        group_by=group_by,
#                                        n_wells=n_wells,
#                                        save_dir=save_dir,
#                                        verbose=True)
    
#     treatment_list = list(t for t in metadata['treatment'].unique() if t != control)

#     bluelight_frames = [(i*60*FPS, i*60*FPS+10*FPS) for i in BLUELIGHT_TIMEPOINTS_MINUTES]

#     for treatment in tqdm(treatment_list):
        
#         colour_dict = dict(zip([control, treatment], sns.color_palette("pastel", 2)))

#         # get timeseries for treatment
#         treatment_ts = get_strain_timeseries(metadata[metadata[group_by]==treatment], 
#                                              project_dir=project_dir, 
#                                              strain=treatment,
#                                              group_by=group_by,
#                                              n_wells=n_wells,
#                                              save_dir=save_dir,
#                                              verbose=True)

#         for mode in motion_modes:
#             print("Plotting timeseries %s fraction for %s vs %s..." % (mode, control, treatment))
    
#             if bluelight_windows_separately:
                
#                 for pulse, timepoint in enumerate(tqdm(BLUELIGHT_TIMEPOINTS_MINUTES), start=1):
    
#                     plt.close('all')
#                     fig, ax = plt.subplots(figsize=(15,5), dpi=150)
            
#                     ax = plot_timeseries_motion_mode(df=control_ts,
#                                                      window=smoothing*FPS,
#                                                      error=True,
#                                                      mode=mode,
#                                                      max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
#                                                      title=None,
#                                                      saveAs=None,
#                                                      ax=ax,
#                                                      bluelight_frames=bluelight_frames,
#                                                      colour=colour_dict[control],
#                                                      alpha=0.25)
                    
#                     ax = plot_timeseries_motion_mode(df=treatment_ts,
#                                                      window=smoothing*FPS,
#                                                      error=True,
#                                                      mode=mode,
#                                                      max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
#                                                      title=None,
#                                                      saveAs=None,
#                                                      ax=ax,
#                                                      bluelight_frames=bluelight_frames,
#                                                      colour=colour_dict[treatment],
#                                                      alpha=0.25)
                
#                     xticks = np.linspace(0, VIDEO_LENGTH_SECONDS*FPS, int(VIDEO_LENGTH_SECONDS/60)+1)
#                     ax.set_xticks(xticks)
#                     ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
                    
#                     # -30secs before to +2mins after each pulse
#                     xlim_range = (timepoint*60-30, timepoint*60+120)
#                     ax.set_xlim([xlim_range[0]*FPS, xlim_range[1]*FPS])
    
#                     ax.set_xlabel('Time (minutes)', fontsize=12, labelpad=10)
#                     ax.set_ylabel('Fraction {}'.format(mode), fontsize=12, labelpad=10)
#                     ax.set_title('{0} vs {1} (bluelight pulse {2} = {3} min)'.format(
#                         control, treatment, pulse, timepoint), fontsize=12, pad=10)
#                     ax.legend([control, treatment], fontsize=12, frameon=False, loc='upper right')
            
#                     # save plot
#                     save_path = save_dir / treatment /\
#                         'motion_mode_{0}_bluelight_pulse{1}_{2}min.pdf'.format(mode,pulse,timepoint)
#                     print("Saving to: %s" % save_path)
#                     plt.savefig(save_path)  
                    
#             else:    
#                 plt.close('all')
#                 fig, ax = plt.subplots(figsize=(30,5), dpi=150)
        
#                 ax = plot_timeseries_motion_mode(df=control_ts,
#                                                  window=smoothing*FPS,
#                                                  error=True,
#                                                  mode=mode,
#                                                  max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
#                                                  title=None,
#                                                  saveAs=None,
#                                                  ax=ax,
#                                                  bluelight_frames=bluelight_frames,
#                                                  colour=colour_dict[control],
#                                                  alpha=0.25)
                
#                 ax = plot_timeseries_motion_mode(df=treatment_ts,
#                                                  window=smoothing*FPS,
#                                                  error=True,
#                                                  mode=mode,
#                                                  max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
#                                                  title=None,
#                                                  saveAs=None,
#                                                  ax=ax,
#                                                  bluelight_frames=bluelight_frames,
#                                                  colour=colour_dict[treatment],
#                                                  alpha=0.25)
            
#                 xticks = np.linspace(0, VIDEO_LENGTH_SECONDS*FPS, int(VIDEO_LENGTH_SECONDS/60)+1)
#                 ax.set_xticks(xticks)
#                 ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
#                 ax.set_xlabel('Time (minutes)', fontsize=12, labelpad=10)
#                 ax.set_ylabel('Fraction {}'.format(mode), fontsize=12, labelpad=10)
#                 ax.set_title('{0} vs {1}'.format(control, treatment), fontsize=12, pad=10)
#                 ax.legend([control, treatment], fontsize=12, frameon=False, loc='upper right')
        
#                 # save plot
#                 save_path = save_dir / treatment / 'motion_mode_{}.pdf'.format(mode)
#                 save_path.parent.mkdir(exist_ok=True, parents=True)
#                 print("Saving to: %s" % save_path)
#                 plt.savefig(save_path)  

#     return
    
#%% Main

if __name__ == '__main__':

    aux_dir = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    res_dir = Path(PROJECT_DIR) / 'Results'
    
    metadata_local_path = Path(SAVE_DIR) / 'metadata.csv'
    features_local_path = Path(SAVE_DIR) / 'features.csv'
    
    if not metadata_local_path.exists() or not features_local_path.exists():

        # load metadata    
        metadata, metadata_path = compile_metadata(aux_dir, 
                                                   imaging_dates=None, 
                                                   add_well_annotations=False, 
                                                   n_wells=N_WELLS)
        
        features, metadata = process_feature_summaries(metadata_path,
                                                       res_dir, 
                                                       compile_day_summaries=True, 
                                                       imaging_dates=None, 
                                                       align_bluelight=False, 
                                                       window_summaries=True,
                                                       n_wells=N_WELLS)
     
        # Clean results - Remove bad well data + features with too many NaNs/zero std + impute
        features, metadata = clean_summary_results(features, 
                                                   metadata,
                                                   feature_columns=None,
                                                   nan_threshold_row=nan_threshold_row,
                                                   nan_threshold_col=nan_threshold_col,
                                                   max_value_cap=1e15,
                                                   imputeNaN=True,
                                                   min_nskel_per_video=None,
                                                   min_nskel_sum=None,
                                                   drop_size_related_feats=False,
                                                   norm_feats_only=False,
                                                   percentile_to_use=None)
    
        assert not features.isna().sum(axis=1).any()
        assert not (features.std(axis=1) == 0).any()
                
        # save clean metadata and features
        metadata.to_csv(metadata_local_path, index=False)
        features.to_csv(features_local_path, index=False) 
    
    else:
        # load clean metadata and features
        metadata = pd.read_csv(metadata_local_path, dtype={'comments':str, 'source_plate_id':str})
        features = pd.read_csv(features_local_path, index_col=None)

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
    
    # Drop results missing antioxidant name
    metadata = metadata[~metadata['antioxidant'].isna()]
    
    treatment_cols = ['gene_name','antioxidant']
    metadata['treatment'] = metadata[treatment_cols].astype(str).agg('-'.join, axis=1)
    control = 'BW-None'

    metadata['window'] = metadata['window'].astype(int)
    window_list = list(metadata['window'].unique())

    # drop results for trolox and resveratrol
    metadata = metadata.query("antioxidant=='None' or antioxidant=='vitC' or antioxidant=='NAC'")

    for window in tqdm(window_list):
        window_meta = metadata[metadata['window']==window]
        window_feat = features.reindex(window_meta.index)
        
        stats_dir = Path(SAVE_DIR) / 'Stats' / WINDOW_NAME_DICT[window]
        plot_dir = Path(SAVE_DIR) / 'Plots' / WINDOW_NAME_DICT[window]
        
        # statistics save path
        acute_rescue_stats(window_meta, 
                           window_feat,
                           group_by='treatment',
                           control=control,
                           save_dir=stats_dir,
                           feature_set=FEATURE_SET,
                           pvalue_threshold=0.05,
                           fdr_method='fdr_bh')
        order = ['BW-None','BW-vitC','BW-NAC','fepD-None','fepD-vitC','fepD-NAC']
        colour_labels = sns.color_palette('tab10', 2)
        colours = [colour_labels[0] if 'BW' in s else colour_labels[1] for s in order]
        colour_dict = {key:col for (key,col) in zip(order, colours)}
        all_in_one_boxplots(window_meta,
                            window_feat,
                            group_by='treatment',
                            control=control,
                            save_dir=plot_dir,
                            ttest_path=stats_dir / 't-test' / 't-test_results.csv',
                            feature_set=FEATURE_SET,
                            pvalue_threshold=0.05,
                            order=order,
                            colour_dict=colour_dict,
                            figsize=(20, 10),
                            ylim_minmax=None,
                            vline_boxpos=[2],
                            fontsize=15,
                            subplots_adjust={'bottom': 0.2, 'top': 0.95, 'left': 0.05, 'right': 0.98})
        
    
    # timeseries plots of speed for fepD vs BW control for each window
    
    BLUELIGHT_TIMEPOINTS_SECONDS = [(i*60,i*60+10) for i in BLUELIGHT_TIMEPOINTS_MINUTES]
    
    plot_window_timeseries_feature(metadata=metadata,
                                   project_dir=Path(PROJECT_DIR),
                                   save_dir=Path(SAVE_DIR) / 'timeseries-speed',
                                   group_by='treatment',
                                   control=control,
                                   groups_list=None,
                                   feature='speed',
                                   n_wells=6,
                                   bluelight_timepoints_seconds=BLUELIGHT_TIMEPOINTS_SECONDS,
                                   bluelight_windows_separately=True,
                                   smoothing=10,
                                   figsize=(15,5),
                                   fps=FPS,
                                   ylim_minmax=(-20,280),
                                   video_length_seconds=VIDEO_LENGTH_SECONDS)
    
    