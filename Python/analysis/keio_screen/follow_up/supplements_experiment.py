#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keio fepD vs BW: supplementation with enterobactin/iron/paraquat

Adding supplements BW and fepD lawns
- 

@author: sm5911
@date: 11/05/2022

"""

#%% Imports

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from visualisation.plotting_helper import sig_asterix, all_in_one_boxplots
from write_data.write import write_list_to_file
from time_series.plot_timeseries import plot_window_timeseries_feature # plot_timeseries_motion_mode
# from analysis.keio_screen.check_keio_screen_worm_trajectories import check_tracked_objects

from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from tierpsytools.preprocessing.filter_data import select_feat_set

#%% Globals

PROJECT_DIR = '/Volumes/hermes$/Keio_Supplements_6WP'
SAVE_DIR = '/Users/sm5911/Documents/Keio_Supplements'
N_WELLS = 6

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
MIN_NSKEL_PER_VIDEO = None
MIN_NSKEL_SUM = 500

FEATURE_SET = ['speed_50th']

# Featsums 10 seconds centred on end of each BL pulse and also 20-30 seconds after end of each BL pulse
WINDOW_DICT_SECONDS = {0:(1805,1815), 1:(1830,1840), 2:(1865,1875),
                       3:(1890,1900), 4:(1925,1935), 5:(1950,1960)}

WINDOW_NAME_DICT = {0:"blue light 1", 1: "20-30 seconds after blue light 1",
                    2:"blue light 2", 3: "20-30 seconds after blue light 2",
                    4:"blue light 3", 5: "20-30 seconds after blue light 3"}

food_type_list = ['BW', 'fepD']
drug_type_list = ['none', 'enterobactin', 'feCl3', 'fe2O12S3', 'paraquat']
paraquat_conc_mM = [0.5, 1, 2, 4]
iron_conc_mM = [1, 4]
iron_treatment_list = ['BW-none-nan', 'BW-feCl3-1.0', 'BW-feCl3-4.0',
                       'BW-fe2O12S3-1.0', 'BW-fe2O12S3-4.0', 
                       'fepD-none-nan', 'fepD-feCl3-1.0', 'fepD-feCl3-4.0', 
                       'fepD-fe2O12S3-1.0', 'fepD-fe2O12S3-4.0']
paraquat_treatment_list = ['BW-none-nan', 'BW-paraquat-0.5', 'BW-paraquat-1.0', 
                           'BW-paraquat-2.0', 'BW-paraquat-4.0', 
                           'fepD-none-nan', 'fepD-paraquat-0.5', 'fepD-paraquat-1.0',
                           'fepD-paraquat-2.0', 'fepD-paraquat-4.0']
    
BLUELIGHT_TIMEPOINTS_MINUTES = [30,31,32]
FPS = 25
VIDEO_LENGTH_SECONDS = 38*60
SMOOTH_WINDOW_SECONDS = 5
BLUELIGHT_WINDOWS_ONLY_TS = True

#%% Functions

def supplements_stats(metadata,
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

# def supplements_stats(metadata, 
#                       features, 
#                       save_dir,
#                       window=WINDOW_NUMBER,
#                       feature=FEATURE,
#                       pvalue_threshold=0.05,
#                       fdr_method='fdr_by'):
#     """ T-tests comparing supplementation of BW and fepD lawns with enterobactin, iron and paraquat
#         against BW and fepD withoout the drugs added. 
#         - BW vs BW + enterobactin
#         - BW vs BW + iron
#         - BW vs BW + paraquat
#         - fepD vs fepD + enterobactin
#         - fepD vs fepD + iron
#         - fepD vs fepD + paraquat
#     """
    
#     # subset for window of interest
#     window_meta = metadata.query("window==@window")
#     save_dir = Path(save_dir) / 'window_{}'.format(window)

    
#     # testing difference in motion mode forwards on fepD vs BW
#     no_drug_meta = window_meta.query("drug_type=='none' and solvent!='DMSO'")
#     do_stats(metadata=no_drug_meta,
#              features=features.reindex(no_drug_meta.index),
#              group_by='food_type',
#              control='BW',
#              save_dir=save_dir / 'fepD_vs_BW',
#              feat=feature,
#              pvalue_threshold=pvalue_threshold,
#              fdr_method=fdr_method)
    
#     # testing addition of enterobactin to BW and fepD lawns - all compared to BW control
#     enterobactin_meta = window_meta.query("drug_type=='enterobactin' or drug_type=='none'")
#     enterobactin_meta['treatment'] = enterobactin_meta[['food_type','drug_type']].agg('-'.join, axis=1)
#     do_stats(metadata=enterobactin_meta, 
#              features=features.reindex(enterobactin_meta.index), 
#              group_by='treatment',
#              control='BW-none',
#              save_dir=save_dir / 'enterobactin',
#              feat=feature,
#              pvalue_threshold=pvalue_threshold,
#              fdr_method=fdr_method)
    
#     # testing addition of enterobactin to fepD lawns - compared to fepD
#     fepD_meta = window_meta.query("food_type=='fepD'")
#     fepD_enterobactin_meta = fepD_meta.query("drug_type=='enterobactin' or drug_type=='none'")
#     fepD_enterobactin_meta['treatment'] = fepD_enterobactin_meta[['food_type','drug_type']
#                                                                  ].agg('-'.join, axis=1)
#     do_stats(metadata=fepD_enterobactin_meta, 
#              features=features.reindex(fepD_enterobactin_meta.index), 
#              group_by='treatment',
#              control='fepD-none',
#              save_dir=save_dir / 'enterobactin' / 'fepD',
#              feat=feature,
#              pvalue_threshold=pvalue_threshold,
#              fdr_method=fdr_method)
    
#     # testing addition of iron to BW and fepD lawns at different conc's
#     iron_meta = window_meta.query("drug_type=='feCl3' or drug_type=='fe2O12S3' or drug_type=='none'")
#     iron_meta['treatment'] = iron_meta[['food_type','drug_type','imaging_plate_drug_conc']
#                                         ].agg('-'.join, axis=1)
#     do_stats(metadata=iron_meta,
#              features=features.reindex(iron_meta.index),
#              group_by='treatment',
#              control='BW-none-nan',
#              save_dir=save_dir / 'iron',
#              feat=feature,
#              pvalue_threshold=pvalue_threshold,
#              fdr_method=fdr_method)
    
#     # testing addition of paraquat at different conc's
#     paraquat_meta = window_meta.query("drug_type=='paraquat' or drug_type=='none'")
#     paraquat_meta['treatment'] = paraquat_meta[['food_type','drug_type','imaging_plate_drug_conc']
#                                                ].agg('-'.join, axis=1)
#     do_stats(metadata=paraquat_meta,
#              features=features.reindex(paraquat_meta.index),
#              group_by='treatment',
#              control='BW-none-nan',
#              save_dir=save_dir / 'paraquat',
#              feat=feature,
#              pvalue_threshold=pvalue_threshold,
#              fdr_method=fdr_method,
#              ttest_if_nonsig=True)
    
#     return

# def supplements_plots(metadata,
#                       features,
#                       plot_dir,
#                       stats_dir,
#                       window=WINDOW_NUMBER,
#                       feature=FEATURE):
#     """ Boxplots showing results of supplementation experiments using enterobactin, iron and 
#         paraquat
#     """
    
#     assert metadata.shape[0] == features.shape[0]
        
#     window_meta = metadata.query("window==@window")
#     stats_dir = Path(stats_dir) / 'window_{}'.format(window)
#     plot_dir = Path(plot_dir) / 'window_{}'.format(window)
#     plot_dir.mkdir(parents=True, exist_ok=True)
    
#     # difference in motion mode forwards on fepD vs BW
#     no_drug_meta = window_meta.query("drug_type=='none' and solvent!='DMSO'")
#     plot_df = no_drug_meta.join(features.reindex(no_drug_meta.index))

#     # boxplots for BW vs fepD
#     plt.close('all')
#     fig, ax = plt.subplots(figsize=(12,8))
#     sns.boxplot(x='food_type', y=feature, order=food_type_list, ax=ax, data=plot_df,
#                 palette='tab10', showfliers=False) 
#                 #hue='is_dead', hue_order=is_dead_list, dodge=True
#     sns.stripplot(x='food_type', y=feature, order=food_type_list, ax=ax, data=plot_df,
#                   s=5, marker='D', color='k')
#     ax.set_xlabel('')
#     ax.set_ylabel(feature.replace('_',' '), fontsize=12, labelpad=10)
#     ax.set_title('fepD vs BW (supplements experiment control)', pad=30, fontsize=18)
#     # annotate p-values - load t-test results for each treatment vs BW control
#     ttest_path = stats_dir / 'fepD_vs_BW' / 'food_type_ttest_results.csv'
#     ttest_df = pd.read_csv(ttest_path, index_col=0)
#     p = ttest_df.loc[feature, 'pvals_fepD']
#     assert ax.get_xticklabels()[1].get_text() == 'fepD'
#     p_text = '***\nP < 0.001' if p < 0.001 else sig_asterix([p])[0] + '\nP = %.3f' % p
#     trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
#     ax.text(1, 1.01, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
#     plt.savefig(plot_dir / 'fepD_vs_BW.pdf', dpi=300)


#     # addition of enterobactin to BW and fepD
#     enterobactin_meta = window_meta.query("drug_type=='enterobactin' or drug_type=='none'")
#     enterobactin_meta['treatment'] = enterobactin_meta[['food_type','drug_type']].agg('-'.join, axis=1)
#     plot_df = enterobactin_meta.join(features.reindex(enterobactin_meta.index))
    
#     # boxplots comparing addition of enterobactin to BW and fepD
#     plt.close('all')
#     fig, ax = plt.subplots(figsize=(12,8))
#     col_dict = dict(zip(drug_type_list[:2], sns.color_palette('Paired', len(drug_type_list[:2]))))
#     sns.boxplot(x='food_type', y=feature, order=food_type_list, ax=ax, data=plot_df,
#                 hue='drug_type', hue_order=drug_type_list[:2], dodge=True,
#                 palette=col_dict, showfliers=False) 
#                 #hue='is_dead', hue_order=is_dead_list, dodge=True
#     sns.stripplot(x='food_type', y=feature, order=food_type_list, ax=ax, data=plot_df,
#                   hue='drug_type', hue_order=drug_type_list[:2], dodge=True,
#                   s=5, marker='D', color='k')
#     handles = []
#     for label in col_dict.keys():
#         handles.append(mpatches.Patch(color=col_dict[label]))
#     ax.legend(handles, col_dict.keys(), loc='best', frameon=False)

#     ax.set_xlabel('')
#     ax.set_ylabel(feature.replace('_',' '), fontsize=12, labelpad=10)
#     # scale plot to exclude outliers > 2.5 * IQR
#     grouped_strain = plot_df.groupby(['food_type','drug_type'])
#     y_bar = grouped_strain[feature].median() # median is less skewed by outliers
#     Q1, Q3 = grouped_strain[feature].quantile(0.25), grouped_strain[feature].quantile(0.75)
#     IQR = Q3 - Q1
#     plt.ylim(min(y_bar) - 2.5 * max(IQR), max(y_bar) + 2.5 * max(IQR))
#     ax.set_title('Addition of enterobactin to BW and fepD', pad=30, fontsize=18)
#     # annotate p-values - load t-test results for each treatment vs BW control
#     ttest_path = stats_dir / 'enterobactin' / 'treatment_ttest_results.csv'
#     ttest_df = pd.read_csv(ttest_path, index_col=0)
#     for i, food in enumerate(food_type_list):
#         assert ax.get_xticklabels()[i].get_text() == food
#         for ii, drug in enumerate(drug_type_list[:2]):
#             if food=='BW' and drug=='none':
#                 continue
#             p = ttest_df.loc[feature, 'pvals_' + food + '-' + drug]
#             p_text = '***\nP < 0.001' if p < 0.001 else sig_asterix([p])[0] + '\nP = %.3f' % p
#             trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
#             offset = -.2 if ii==0 else .2
#             ax.text(i + offset, 1.01, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
#     ttest_path = stats_dir / 'enterobactin' / 'fepD' / 'treatment_ttest_results.csv'
#     ttest_df = pd.read_csv(ttest_path, index_col=0)
#     p = ttest_df.loc[feature, 'pvals_fepD-enterobactin']
#     p_text = '***\nP < 0.001' if p < 0.001 else sig_asterix([p])[0] + '\nP = %.3f' % p
#     ax.text(1, 0.94, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
#     plt.plot([0.8,0.8,1.2,1.2],[0.92,0.93,0.93,0.92],lw=1.5,c='k',transform=trans)
#     plt.savefig(plot_dir / 'enterobactin.pdf', dpi=300)

    
#     # addition of iron to BW and fepD
#     iron_meta = window_meta.query("drug_type=='feCl3' or drug_type=='fe2O12S3' or drug_type=='none'")
#     iron_meta['treatment'] = iron_meta[['food_type','drug_type','imaging_plate_drug_conc']
#                                         ].agg('-'.join, axis=1)
#     plot_df = iron_meta.join(features.reindex(iron_meta.index))
    
#     # boxplots comparing addition of 1mM and 4mM iron (feCl3 or fe2O12S3) to BW and fepD
#     plt.close('all')
#     fig, ax = plt.subplots(figsize=(12,8))
#     col_dict = dict(zip(iron_treatment_list, sns.color_palette('tab10', len(iron_treatment_list))))
#     sns.boxplot(x='treatment', y=feature, order=iron_treatment_list, ax=ax, data=plot_df,
#                 palette=col_dict, showfliers=False) 
#                 #hue='is_dead', hue_order=is_dead_list, dodge=True
#     sns.stripplot(x='treatment', y=feature, order=iron_treatment_list, ax=ax, data=plot_df,
#                   s=5, marker='D', color='k')
#     legend = ax.legend()
#     legend.remove()

#     ax.set_xlabel('')
#     ax.set_ylabel(feature.replace('_',' '), fontsize=12, labelpad=10)
#     # ax.set_ylim(0.28,1.1)
#     ax.set_title('Addition of feCl3 and fe2O12S3 to BW and fepD', pad=30, fontsize=18)
#     # annotate p-values - load t-test results for each treatment vs BW control
#     ttest_path = stats_dir / 'iron' / 'treatment_ttest_results.csv'
#     ttest_df = pd.read_csv(ttest_path, index_col=0)
#     for i, treatment in enumerate(iron_treatment_list):
#         assert ax.get_xticklabels()[i].get_text() == treatment
#         if treatment == 'BW-none-nan':
#             continue
#         p = ttest_df.loc[feature, 'pvals_' + treatment]
#         p_text = '***\nP < 0.001' if p < 0.001 else sig_asterix([p])[0] + '\nP = %.3f' % p
#         trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
#         ax.text(i, 1.01, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
#     ax.set_xticklabels([s.get_text().replace('-','\n') for s in ax.get_xticklabels()]) #rotation=45
#     plt.savefig(plot_dir / 'iron.pdf', dpi=300)
    
    
#     # addition of paraquat to BW and fepD
#     paraquat_meta = window_meta.query("drug_type=='paraquat' or drug_type=='none'")
#     paraquat_meta['treatment'] = paraquat_meta[['food_type','drug_type','imaging_plate_drug_conc']
#                                                ].agg('-'.join, axis=1)
#     plot_df = paraquat_meta.join(features.reindex(paraquat_meta.index))
    
#     # boxplots comparing addition of paraquat (0.5, 1, 2 and 4 mM) to BW and fepD
#     plt.close('all')
#     fig, ax = plt.subplots(figsize=(12,8))
#     col_dict = dict(zip(paraquat_treatment_list, sns.color_palette('tab10_r', 
#                                                                    len(paraquat_treatment_list))))
#     sns.boxplot(x='treatment', y=feature, order=paraquat_treatment_list, ax=ax, data=plot_df,
#                 palette=col_dict, showfliers=False) 
#     sns.stripplot(x='treatment', y=feature, order=paraquat_treatment_list, ax=ax, data=plot_df,
#                   s=5, marker='D', color='k')
#     legend = ax.legend()
#     legend.remove()
#     # annotate p-values - load t-test results for each treatment vs BW control
#     ttest_path = stats_dir / 'paraquat' / 'treatment_ttest_results.csv'
#     ttest_df = pd.read_csv(ttest_path, index_col=0)
#     for i, treatment in enumerate(paraquat_treatment_list):
#         assert ax.get_xticklabels()[i].get_text() == treatment
#         if treatment == 'BW-none-nan':
#             continue
#         p = ttest_df.loc[feature, 'pvals_' + treatment]
#         p_text = '***\nP < 0.001' if p < 0.001 else sig_asterix([p])[0] + '\nP = %.3f' % p
#         trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
#         ax.text(i, 1.01, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
#     ax.set_xlabel('')
#     ax.set_ylabel(feature.replace('_',' '), fontsize=12, labelpad=10)
#     # ax.set_ylim(0.28,1.1)
#     ax.set_title('Addition of paraquat to BW and fepD', pad=30, fontsize=18)
#     ax.set_xticklabels([s.get_text().replace('-','\n') for s in ax.get_xticklabels()])#rotation=45,ha='right'
#     plt.savefig(plot_dir / 'paraquat.pdf', dpi=300)
    
#     return

# def supplements_timeseries(metadata, project_dir=PROJECT_DIR, save_dir=SAVE_DIR, window=WINDOW_NUMBER):
#     """ Timeseries plots for addition of enterobactin, paraquat, and iron to BW and fepD """
    
#     # # Plot BW and fepD together for each treatment: none, enterobactin, paraquat, iron    
#     # for drug in tqdm(drug_type_list):
#     #     print("Plotting timeseries comparing BW to fepD when %s is added" % drug)
        
#     #     drug_meta = metadata.query("window==@window and  drug_type==@drug")
                
#     #     # boxplots for all treatments vs BW control
#     #     drug_meta['treatment'] = drug_meta[['food_type','drug_type','imaging_plate_drug_conc']
#     #                                        ].agg('-'.join, axis=1) 
                
#     #     treatment_order = sorted(drug_meta['treatment'].unique())
        
#     #     # get timeseries for control data
#     #     control_timeseries = get_strain_timeseries(metadata[metadata['treatment']==all_treatment_control], 
#     #                                                project_dir=project_dir, 
#     #                                                strain=all_treatment_control,
#     #                                                group_by='treatment',
#     #                                                only_wells=None,
#     #                                                save_dir=Path(save_dir) / 'Data' / all_treatment_control,
#     #                                                verbose=False)
        
#     #     for mode in motion_modes:
            
#     #         plt.close('all')
#     #         fig, ax = plt.subplots(figsize=(15,5), dpi=200)
#     #         BW_treatments = [t for t in treatment_order if 'BW' in t]
#     #         fepD_treatments = [t for t in treatment_order if 'fepD' in t]
#     #         col_dict = dict(zip(BW_treatments, sns.color_palette('Blues', len(BW_treatments))))
#     #         col_dict.update(dict(zip(fepD_treatments, sns.color_palette('Greens', len(fepD_treatments)))))
            
#     #         # col_dict = dict(zip(treatment_order, sns.color_palette("plasma", len(treatment_order))))
#     #         bluelight_frames = [(i*60*FPS, i*60*FPS+10*FPS) for i in BLUELIGHT_TIMEPOINTS_MINUTES]

#     #         for treatment in treatment_order:
                    
#     #             # get timeseries data for treatment data
#     #             treatment_metadata = drug_meta[drug_meta['treatment']==treatment]
#     #             treatment_timeseries = get_strain_timeseries(treatment_metadata, 
#     #                                                          project_dir=project_dir, 
#     #                                                          strain=treatment,
#     #                                                          group_by='treatment',
#     #                                                          only_wells=None,
#     #                                                          save_dir=Path(save_dir) / 'Data' / treatment,
#     #                                                          verbose=False)

#     #             print("Plotting timeseries '%s' fraction for %s" % (mode, treatment))
#     #             ax = plot_timeseries_motion_mode(df=treatment_timeseries,
#     #                                              window=SMOOTH_WINDOW_SECONDS*FPS,
#     #                                              error=True,
#     #                                              mode=mode,
#     #                                              max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
#     #                                              title=None,
#     #                                              saveAs=None,
#     #                                              ax=ax,
#     #                                              bluelight_frames=bluelight_frames,
#     #                                              colour=col_dict[treatment],
#     #                                              alpha=0.25)
            
#     #         xticks = np.linspace(0, VIDEO_LENGTH_SECONDS*FPS, int(VIDEO_LENGTH_SECONDS/60)+1)
#     #         ax.set_xticks(xticks)
#     #         ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
#     #         ax.set_xlabel('Time (minutes)', fontsize=12, labelpad=10)
#     #         ax.set_ylabel('Fraction {}'.format(mode), fontsize=12, labelpad=10)
#     #         ax.legend(treatment_order, fontsize=12, frameon=False, loc='best')
    
#     #         if BLUELIGHT_WINDOWS_ONLY_TS:
#     #             ts_plot_dir = Path(save_dir) / 'Plots' / 'timeseries_bluelight' / treatment
#     #             ax.set_xlim([min(BLUELIGHT_TIMEPOINTS_MINUTES)*60*FPS-60*FPS, 
#     #                          max(BLUELIGHT_TIMEPOINTS_MINUTES)*60*FPS+70*FPS])
#     #         else:
#     #             ts_plot_dir = Path(save_dir) / 'Plots' / 'timeseries' / treatment
    
#     #         plt.tight_layout()
#     #         ts_plot_dir.mkdir(exist_ok=True, parents=True)
#     #         plt.savefig(ts_plot_dir / '{0}_{1}.pdf'.format(treatment, mode))  

#     ### Each treatment vs control: BW vs BW + lysate // BW vs BW + supernatant
#     window_meta = metadata.query("window==@window")
#     window_meta['treatment'] = window_meta[['food_type','drug_type',
#                                             'imaging_plate_drug_conc']].agg('-'.join, axis=1)     
#     control_list = list(np.concatenate([np.array(['fepD-none-nan']), np.repeat('BW-none-nan', 19)]))
#     treatment_list = ['fepD-enterobactin-nan',
#                       'fepD-none-nan','BW-enterobactin-nan',
#                       'BW-fe2O12S3-1.0','BW-fe2O12S3-4.0',
#                       'BW-feCl3-1.0','BW-feCl3-4.0',
#                       'BW-paraquat-0.5','BW-paraquat-1.0',
#                       'BW-paraquat-2.0','BW-paraquat-4.0',
#                       'fepD-enterobactin-nan',
#                       'fepD-fe2O12S3-1.0','fepD-fe2O12S3-4.0',
#                       'fepD-feCl3-1.0','fepD-feCl3-4.0',
#                       'fepD-paraquat-0.5','fepD-paraquat-1.0',
#                       'fepD-paraquat-2.0','fepD-paraquat-4.0']
#     title_list = ['fepD vs fepD + enterobactin',
#                   'BW vs fepD','Addition of enterobactin (BW control)',
#                   'Addition of 1mM fe2O12S3 (BW control)','Addition of 4mM fe2O12S3 (BW control)',
#                   'Addition of 1mM feCl3 (BW control)','Addition of 4mM feCl3 (BW control)',
#                   'Addition of 0.5mM paraquat (BW control)','Addition of 1mM paraquat (BW control)',
#                   'Addition of 2mM paraquat (BW control)','Addition of 4mM paraquat (BW control)',
#                   'BW vs fepD + enterobactin',
#                   'BW vs fepD + 1mM fe2O12S3','BW vs fepD + 4mM fe2O12S3',
#                   'BW vs fepD + 1mM feCl3','BW vs fepD + 4mM feCl3',
#                   'BW vs fepD + 0.5mM paraquat','BW vs fepD + 1mM paraquat',
#                   'BW vs fepD + 2mM paraquat','BW vs fepD + 4mM paraquat']
#     labs = [('fepD', 'fepD + enterobactin'),
#             ('BW', 'fepD'),('BW', 'BW + enterobactin'),
#             ('BW', 'BW + 1mM fe2O12S3'),('BW', 'BW + 4mM fe2O12S3'),
#             ('BW', 'BW + 1mM feCl3'),('BW', 'BW + 4mM feCl3'),
#             ('BW', 'BW + 0.5mM paraquat'),('BW', 'BW + 1mM paraquat'),
#             ('BW', 'BW + 2mM paraquat'),('BW', 'BW + 4mM paraquat'),
#             ('BW', 'fepD + enterobactin'),
#             ('BW', 'fepD + 1mM fe2O12S3'),('BW', 'fepD + 4mM fe2O12S3'),
#             ('BW', 'fepD + 1mM feCl3'),('BW', 'fepD + 4mM feCl3'),
#             ('BW', 'fepD + 0.5mM paraquat'),('BW', 'fepD + 1mM paraquat'),
#             ('BW', 'fepD + 2mM paraquat'),('BW', 'fepD + 4mM paraquat')]


#     for control, treatment, title, lab in tqdm(zip(control_list, treatment_list, title_list, labs)):
        
#         #get timeseries for control data
#         control_ts = get_strain_timeseries(window_meta[window_meta['treatment']==control], 
#                                             project_dir=project_dir, 
#                                             strain=control,
#                                             group_by='treatment',
#                                             n_wells=6,
#                                             save_dir=Path(save_dir) / 'Data' / control,
#                                             verbose=False)
        
#         # get timeseries for treatment data
#         treatment_ts = get_strain_timeseries(window_meta[window_meta['treatment']==treatment], 
#                                               project_dir=project_dir, 
#                                               strain=treatment,
#                                               group_by='treatment',
#                                               n_wells=6,
#                                               save_dir=Path(save_dir) / 'Data' / treatment,
#                                               verbose=False)
 
#         colour_dict = dict(zip([control, treatment], sns.color_palette("pastel", 2)))
#         bluelight_frames = [(i*60*FPS, i*60*FPS+10*FPS) for i in BLUELIGHT_TIMEPOINTS_MINUTES]

#         for mode in motion_modes:
                    
#             print("Plotting timeseries '%s' fraction for '%s' vs '%s'..." %\
#                   (mode, treatment, control))
    
#             plt.close('all')
#             fig, ax = plt.subplots(figsize=(15,5), dpi=200)
    
#             ax = plot_timeseries_motion_mode(df=control_ts,
#                                               window=SMOOTH_WINDOW_SECONDS*FPS,
#                                               error=True,
#                                               mode=mode,
#                                               max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
#                                               title=None,
#                                               saveAs=None,
#                                               ax=ax,
#                                               bluelight_frames=bluelight_frames,
#                                               colour=colour_dict[control],
#                                               alpha=0.25)
            
#             ax = plot_timeseries_motion_mode(df=treatment_ts,
#                                               window=SMOOTH_WINDOW_SECONDS*FPS,
#                                               error=True,
#                                               mode=mode,
#                                               max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
#                                               title=None,
#                                               saveAs=None,
#                                               ax=ax,
#                                               bluelight_frames=bluelight_frames,
#                                               colour=colour_dict[treatment],
#                                               alpha=0.25)
        
#             xticks = np.linspace(0, VIDEO_LENGTH_SECONDS*FPS, int(VIDEO_LENGTH_SECONDS/60)+1)
#             ax.set_xticks(xticks)
#             ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
#             ax.set_xlabel('Time (minutes)', fontsize=12, labelpad=10)
#             ax.set_ylabel('Fraction {}'.format(mode), fontsize=12, labelpad=10)
#             ax.set_title(title, fontsize=12, pad=10)
#             ax.legend([lab[0], lab[1]], fontsize=12, frameon=False, loc='best')
    
#             if BLUELIGHT_WINDOWS_ONLY_TS:
#                 ts_plot_dir = Path(save_dir) / 'Plots' / 'timeseries_bluelight' / treatment
#                 ax.set_xlim([min(BLUELIGHT_TIMEPOINTS_MINUTES)*60*FPS-60*FPS, 
#                               max(BLUELIGHT_TIMEPOINTS_MINUTES)*60*FPS+70*FPS])
#             else:   
#                 ts_plot_dir = Path(save_dir) / 'Plots' / 'timeseries' / treatment
    
#             #plt.tight_layout()
#             ts_plot_dir.mkdir(exist_ok=True, parents=True)
#             save_path = ts_plot_dir / '{0}_{1}.pdf'.format(treatment, mode)
#             print("Saving to: %s" % save_path)
#             plt.savefig(save_path)  

#     return

#%% Main

if __name__ == '__main__':
    
    AUX_DIR = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    RES_DIR = Path(PROJECT_DIR) / 'Results'

    META_PATH = Path(SAVE_DIR) / 'metadata.csv'
    FEAT_PATH = Path(SAVE_DIR) / 'features.csv'
    
    if not META_PATH.exists() and not FEAT_PATH.exists():
        Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)

        # compile metadata
        metadata, metadata_path = compile_metadata(aux_dir=AUX_DIR, 
                                                   imaging_dates=None, 
                                                   n_wells=N_WELLS,
                                                   add_well_annotations=False,
                                                   from_source_plate=True)
        
        # compile window summaries
        features, metadata = process_feature_summaries(metadata_path=metadata_path, 
                                                       results_dir=RES_DIR, 
                                                       compile_day_summaries=True,
                                                       imaging_dates=None, 
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

    treatment_cols = ['food_type','drug_type','imaging_plate_drug_conc','solvent']
    metadata['treatment'] = metadata[treatment_cols].astype(str).agg('-'.join, axis=1) 
    order = ['BW-none-nan-nan','BW-none-nan-DMSO',
             'fepD-none-nan-nan','fepD-none-nan-DMSO',
             'BW-paraquat-0.5-H2O','BW-paraquat-1.0-H2O',
             'BW-paraquat-2.0-H2O','BW-paraquat-4.0-H2O',
             'fepD-paraquat-0.5-H2O','fepD-paraquat-1.0-H2O',
             'fepD-paraquat-2.0-H2O','fepD-paraquat-4.0-H2O',
             'BW-enterobactin-nan-DMSO','fepD-enterobactin-nan-DMSO',
             'BW-feCl3-1.0-NGM','BW-feCl3-4.0-NGM','BW-feCl3-4.0-H2O',
             'fepD-feCl3-1.0-NGM','fepD-feCl3-4.0-NGM','fepD-feCl3-4.0-H2O',
             'BW-fe2O12S3-1.0-NGM','BW-fe2O12S3-4.0-NGM','BW-fe2O12S3-4.0-H2O',
             'fepD-fe2O12S3-1.0-NGM','fepD-fe2O12S3-4.0-NGM','fepD-fe2O12S3-4.0-H2O']
    control = 'BW-none-nan-nan'

    # subset metadata results for bluelight videos only 
    bluelight_videos = [i for i in metadata['imgstore_name'] if 'bluelight' in i]
    metadata = metadata[metadata['imgstore_name'].isin(bluelight_videos)]

    metadata['window'] = metadata['window'].astype(int)
    window_list = list(metadata['window'].unique())

    for window in tqdm(window_list):
        meta_window = metadata[metadata['window']==window]
        feat_window = features.reindex(meta_window.index)
        
        stats_dir = Path(SAVE_DIR) / 'Stats' / WINDOW_NAME_DICT[window]
        plot_dir = Path(SAVE_DIR) / 'Plots' / WINDOW_NAME_DICT[window]

        # T-tests comparing supplementation of BW and fepD lawns with enterobactin, iron and paraquat
        # against BW and fepD withoout the drugs added. 
        # - BW vs BW + enterobactin
        # - BW vs BW + iron
        # - BW vs BW + paraquat
        # - fepD vs fepD + enterobactin
        # - fepD vs fepD + iron
        # - fepD vs fepD + paraquat        
        # paraquat: fepD vs BW
        
        meta_paraquat = meta_window[np.logical_or(meta_window['drug_type']=='paraquat',
                                                  np.logical_and(meta_window['drug_type']=='none',
                                                                 meta_window['solvent']!='DMSO'))]
        feat_paraquat = feat_window.reindex(meta_paraquat.index)
        supplements_stats(meta_paraquat,
                          feat_paraquat,
                          group_by='treatment',
                          control=control,
                          save_dir=stats_dir / 'paraquat',
                          feature_set=FEATURE_SET,
                          pvalue_threshold=0.05,
                          fdr_method='fdr_bh')
        colour_labels = sns.color_palette('tab10', 2)
        groups = meta_paraquat['treatment'].unique()
        colours = [colour_labels[0] if 'BW' in s else colour_labels[1] for s in groups]
        colour_dict = {key:col for (key,col) in zip(groups, colours)}
        all_in_one_boxplots(meta_paraquat,
                            feat_paraquat,
                            group_by='treatment',
                            control=control,
                            save_dir=plot_dir / 'all-in-one' / 'paraquat',
                            ttest_path=stats_dir / 'paraquat' / 't-test' / 't-test_results.csv',
                            feature_set=feature_list,
                            pvalue_threshold=0.05,
                            order=[s for s in order if s in groups],
                            colour_dict=colour_dict,
                            figsize=(30,10),
                            # ylim_minmax=(-20,130),
                            vline_boxpos=[1],
                            fontsize=20,
                            subplots_adjust={'bottom':0.4,'top':0.95,'left':0.05,'right':0.98})
        
        # enterobactin: fepD vs BW
        meta_ent = meta_window[np.logical_or(meta_window['drug_type']=='enterobactin',
                                             meta_window['drug_type']=='none')]
        feat_ent = feat_window.reindex(meta_ent.index)
        supplements_stats(meta_ent,
                          feat_ent,
                          group_by='treatment',
                          control=control,
                          save_dir=stats_dir / 'enterobactin',
                          feature_set=FEATURE_SET,
                          pvalue_threshold=0.05,
                          fdr_method='fdr_bh')
        colour_labels = sns.color_palette('tab10', 2)
        groups = meta_ent['treatment'].unique()
        colours = [colour_labels[0] if 'BW' in s else colour_labels[1] for s in groups]
        colour_dict = {key:col for (key,col) in zip(groups, colours)}
        all_in_one_boxplots(meta_ent,
                            feat_ent,
                            group_by='treatment',
                            control=control,
                            save_dir=plot_dir / 'all-in-one' / 'enterobactin',
                            ttest_path=stats_dir / 'enterobactin' / 't-test' / 't-test_results.csv',
                            feature_set=feature_list,
                            pvalue_threshold=0.05,
                            order=[s for s in order if s in groups],
                            colour_dict=colour_dict,
                            figsize=(30,10),
                            # ylim_minmax=(-20,130),
                            vline_boxpos=[3],
                            fontsize=20,
                            subplots_adjust={'bottom':0.4,'top':0.95,'left':0.05,'right':0.98})
        
        # iron: fepD vs BW (H2O solvent only)
        meta_iron = meta_window[np.logical_or(np.logical_or(meta_window['drug_type']=='feCl3',
                                                            meta_window['drug_type']=='fe2O12S3'),
                                              np.logical_and(meta_window['drug_type']=='none',
                                                             meta_window['solvent']!='DMSO'))]
        meta_iron = meta_iron[meta_iron['solvent']!='NGM']
        feat_iron = feat_window.reindex(meta_iron.index)
        supplements_stats(meta_iron,
                          feat_iron,
                          group_by='treatment',
                          control=control,
                          save_dir=stats_dir / 'iron',
                          feature_set=FEATURE_SET,
                          pvalue_threshold=0.05,
                          fdr_method='fdr_bh')
        colour_labels = sns.color_palette('tab10', 2)
        groups = meta_iron['treatment'].unique()
        colours = [colour_labels[0] if 'BW' in s else colour_labels[1] for s in groups]
        colour_dict = {key:col for (key,col) in zip(groups, colours)}
        all_in_one_boxplots(meta_iron,
                            feat_iron,
                            group_by='treatment',
                            control=control,
                            save_dir=plot_dir / 'all-in-one' / 'iron',
                            ttest_path=stats_dir / 'iron' / 't-test' / 't-test_results.csv',
                            feature_set=feature_list,
                            pvalue_threshold=0.05,
                            order=[s for s in order if s in groups],
                            colour_dict=colour_dict,
                            figsize=(20,10),
                            # ylim_minmax=(-20,130),
                            vline_boxpos=[1,3],
                            fontsize=20,
                            subplots_adjust={'bottom':0.4,'top':0.95,'left':0.05,'right':0.98})

    metadata = metadata[metadata['window']==0]
        
    # timeseries plots of speed for fepD vs BW control
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
                                   ylim_minmax=(-20,220),
                                   xlim_crop_around_bluelight_seconds=(120,300),
                                   video_length_seconds=VIDEO_LENGTH_SECONDS)

    # # Check length/area of tracked objects - prop bad skeletons
    # results_df = check_tracked_objects(metadata, 
    #                                    length_minmax=(200, 2000), 
    #                                    width_minmax=(20, 500),
    #                                    save_to=Path(SAVE_DIR) / 'tracking_checks.csv')

