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
from matplotlib import transforms
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from visualisation.plotting_helper import sig_asterix
from statistical_testing.stats_helper import do_stats
from time_series.time_series_helper import get_strain_timeseries
from time_series.plot_timeseries import plot_timeseries_motion_mode
from analysis.keio_screen.check_keio_screen_worm_trajectories import check_tracked_objects

#%% Globals

PROJECT_DIR = '/Volumes/hermes$/Keio_Supplements_6WP'
SAVE_DIR = '/Users/sm5911/Documents/Keio_Supplements'
IMAGING_DATES = ['20220414','20220418']
N_WELLS = 6

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
MIN_NSKEL_PER_VIDEO = None
MIN_NSKEL_SUM = 500

FEATURE = 'motion_mode_forward_fraction'
motion_modes = ['forwards']

WINDOW_DICT_SECONDS = {0:(1790,1800), 1:(1805,1815), 2:(1815,1825),
                       3:(1850,1860), 4:(1865,1875), 5:(1875,1885),
                       6:(1910,1920), 7:(1925,1935), 8:(1935,1945)}

WINDOW_DICT_STIM_TYPE = {0:'prestim\n(30min)',1:'bluelight\n(30min)',2:'poststim\n(30min)',
                         3:'prestim\n(31min)',4:'bluelight\n(31min)',5:'poststim\n(31min)',
                         6:'prestim\n(32min)',7:'bluelight\n(32min)',8:'poststim\n(32min)'}

WINDOW_NUMBER = 3

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
                      save_dir,
                      window=WINDOW_NUMBER,
                      feature=FEATURE,
                      pvalue_threshold=0.05,
                      fdr_method='fdr_by'):
    """ T-tests comparing supplementation of BW and fepD lawns with enterobactin, iron and paraquat
        against BW and fepD withoout the drugs added. 
        - BW vs BW + enterobactin
        - BW vs BW + iron
        - BW vs BW + paraquat
        - fepD vs fepD + enterobactin
        - fepD vs fepD + iron
        - fepD vs fepD + paraquat
    """
    
    # subset for window of interest
    window_meta = metadata.query("window==@window")
    save_dir = Path(save_dir) / 'window_{}'.format(window)

    
    # testing difference in motion mode forwards on fepD vs BW
    no_drug_meta = window_meta.query("drug_type=='none' and solvent!='DMSO'")
    do_stats(metadata=no_drug_meta,
             features=features.reindex(no_drug_meta.index),
             group_by='food_type',
             control='BW',
             save_dir=save_dir / 'fepD_vs_BW',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)
    
    # testing addition of enterobactin to BW and fepD lawns - all compared to BW control
    enterobactin_meta = window_meta.query("drug_type=='enterobactin' or drug_type=='none'")
    enterobactin_meta['treatment'] = enterobactin_meta[['food_type','drug_type']].agg('-'.join, axis=1)
    do_stats(metadata=enterobactin_meta, 
             features=features.reindex(enterobactin_meta.index), 
             group_by='treatment',
             control='BW-none',
             save_dir=save_dir / 'enterobactin',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)
    
    # testing addition of enterobactin to fepD lawns - compared to fepD
    fepD_meta = window_meta.query("food_type=='fepD'")
    fepD_enterobactin_meta = fepD_meta.query("drug_type=='enterobactin' or drug_type=='none'")
    fepD_enterobactin_meta['treatment'] = fepD_enterobactin_meta[['food_type','drug_type']
                                                                 ].agg('-'.join, axis=1)
    do_stats(metadata=fepD_enterobactin_meta, 
             features=features.reindex(fepD_enterobactin_meta.index), 
             group_by='treatment',
             control='fepD-none',
             save_dir=save_dir / 'enterobactin' / 'fepD',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)
    
    # testing addition of iron to BW and fepD lawns at different conc's
    iron_meta = window_meta.query("drug_type=='feCl3' or drug_type=='fe2O12S3' or drug_type=='none'")
    iron_meta['treatment'] = iron_meta[['food_type','drug_type','imaging_plate_drug_conc']
                                        ].agg('-'.join, axis=1)
    do_stats(metadata=iron_meta,
             features=features.reindex(iron_meta.index),
             group_by='treatment',
             control='BW-none-nan',
             save_dir=save_dir / 'iron',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)
    
    # testing addition of paraquat at different conc's
    paraquat_meta = window_meta.query("drug_type=='paraquat' or drug_type=='none'")
    paraquat_meta['treatment'] = paraquat_meta[['food_type','drug_type','imaging_plate_drug_conc']
                                               ].agg('-'.join, axis=1)
    do_stats(metadata=paraquat_meta,
             features=features.reindex(paraquat_meta.index),
             group_by='treatment',
             control='BW-none-nan',
             save_dir=save_dir / 'paraquat',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method,
             ttest_if_nonsig=True)
    
    return

def supplements_plots(metadata,
                      features,
                      plot_dir,
                      stats_dir,
                      window=WINDOW_NUMBER,
                      feature=FEATURE):
    """ Boxplots showing results of supplementation experiments using enterobactin, iron and 
        paraquat
    """
    
    assert metadata.shape[0] == features.shape[0]
        
    window_meta = metadata.query("window==@window")
    stats_dir = Path(stats_dir) / 'window_{}'.format(window)
    plot_dir = Path(plot_dir) / 'window_{}'.format(window)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # difference in motion mode forwards on fepD vs BW
    no_drug_meta = window_meta.query("drug_type=='none' and solvent!='DMSO'")
    plot_df = no_drug_meta.join(features.reindex(no_drug_meta.index))

    # boxplots for BW vs fepD
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12,8))
    sns.boxplot(x='food_type', y=feature, order=food_type_list, ax=ax, data=plot_df,
                palette='tab10', showfliers=False) 
                #hue='is_dead', hue_order=is_dead_list, dodge=True
    sns.stripplot(x='food_type', y=feature, order=food_type_list, ax=ax, data=plot_df,
                  s=5, marker='D', color='k')
    ax.set_xlabel('')
    ax.set_ylabel(feature.replace('_',' '), fontsize=12, labelpad=10)
    ax.set_title('fepD vs BW (supplements experiment control)', pad=30, fontsize=18)
    # annotate p-values - load t-test results for each treatment vs BW control
    ttest_path = stats_dir / 'fepD_vs_BW' / 'food_type_ttest_results.csv'
    ttest_df = pd.read_csv(ttest_path, index_col=0)
    p = ttest_df.loc[feature, 'pvals_fepD']
    assert ax.get_xticklabels()[1].get_text() == 'fepD'
    p_text = '***\nP < 0.001' if p < 0.001 else sig_asterix([p])[0] + '\nP = %.3f' % p
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(1, 1.01, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
    plt.savefig(plot_dir / 'fepD_vs_BW.pdf', dpi=300)


    # addition of enterobactin to BW and fepD
    enterobactin_meta = window_meta.query("drug_type=='enterobactin' or drug_type=='none'")
    enterobactin_meta['treatment'] = enterobactin_meta[['food_type','drug_type']].agg('-'.join, axis=1)
    plot_df = enterobactin_meta.join(features.reindex(enterobactin_meta.index))
    
    # boxplots comparing addition of enterobactin to BW and fepD
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12,8))
    col_dict = dict(zip(drug_type_list[:2], sns.color_palette('Paired', len(drug_type_list[:2]))))
    sns.boxplot(x='food_type', y=feature, order=food_type_list, ax=ax, data=plot_df,
                hue='drug_type', hue_order=drug_type_list[:2], dodge=True,
                palette=col_dict, showfliers=False) 
                #hue='is_dead', hue_order=is_dead_list, dodge=True
    sns.stripplot(x='food_type', y=feature, order=food_type_list, ax=ax, data=plot_df,
                  hue='drug_type', hue_order=drug_type_list[:2], dodge=True,
                  s=5, marker='D', color='k')
    handles = []
    for label in col_dict.keys():
        handles.append(mpatches.Patch(color=col_dict[label]))
    ax.legend(handles, col_dict.keys(), loc='best', frameon=False)

    ax.set_xlabel('')
    ax.set_ylabel(feature.replace('_',' '), fontsize=12, labelpad=10)
    # scale plot to exclude outliers > 2.5 * IQR
    grouped_strain = plot_df.groupby(['food_type','drug_type'])
    y_bar = grouped_strain[feature].median() # median is less skewed by outliers
    Q1, Q3 = grouped_strain[feature].quantile(0.25), grouped_strain[feature].quantile(0.75)
    IQR = Q3 - Q1
    plt.ylim(min(y_bar) - 2.5 * max(IQR), max(y_bar) + 2.5 * max(IQR))
    ax.set_title('Addition of enterobactin to BW and fepD', pad=30, fontsize=18)
    # annotate p-values - load t-test results for each treatment vs BW control
    ttest_path = stats_dir / 'enterobactin' / 'treatment_ttest_results.csv'
    ttest_df = pd.read_csv(ttest_path, index_col=0)
    for i, food in enumerate(food_type_list):
        assert ax.get_xticklabels()[i].get_text() == food
        for ii, drug in enumerate(drug_type_list[:2]):
            if food=='BW' and drug=='none':
                continue
            p = ttest_df.loc[feature, 'pvals_' + food + '-' + drug]
            p_text = '***\nP < 0.001' if p < 0.001 else sig_asterix([p])[0] + '\nP = %.3f' % p
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            offset = -.2 if ii==0 else .2
            ax.text(i + offset, 1.01, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
    ttest_path = stats_dir / 'enterobactin' / 'fepD' / 'treatment_ttest_results.csv'
    ttest_df = pd.read_csv(ttest_path, index_col=0)
    p = ttest_df.loc[feature, 'pvals_fepD-enterobactin']
    p_text = '***\nP < 0.001' if p < 0.001 else sig_asterix([p])[0] + '\nP = %.3f' % p
    ax.text(1, 0.94, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
    plt.plot([0.8,0.8,1.2,1.2],[0.92,0.93,0.93,0.92],lw=1.5,c='k',transform=trans)
    plt.savefig(plot_dir / 'enterobactin.pdf', dpi=300)

    
    # addition of iron to BW and fepD
    iron_meta = window_meta.query("drug_type=='feCl3' or drug_type=='fe2O12S3' or drug_type=='none'")
    iron_meta['treatment'] = iron_meta[['food_type','drug_type','imaging_plate_drug_conc']
                                        ].agg('-'.join, axis=1)
    plot_df = iron_meta.join(features.reindex(iron_meta.index))
    
    # boxplots comparing addition of 1mM and 4mM iron (feCl3 or fe2O12S3) to BW and fepD
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12,8))
    col_dict = dict(zip(iron_treatment_list, sns.color_palette('tab10', len(iron_treatment_list))))
    sns.boxplot(x='treatment', y=feature, order=iron_treatment_list, ax=ax, data=plot_df,
                palette=col_dict, showfliers=False) 
                #hue='is_dead', hue_order=is_dead_list, dodge=True
    sns.stripplot(x='treatment', y=feature, order=iron_treatment_list, ax=ax, data=plot_df,
                  s=5, marker='D', color='k')
    legend = ax.legend()
    legend.remove()

    ax.set_xlabel('')
    ax.set_ylabel(feature.replace('_',' '), fontsize=12, labelpad=10)
    # ax.set_ylim(0.28,1.1)
    ax.set_title('Addition of feCl3 and fe2O12S3 to BW and fepD', pad=30, fontsize=18)
    # annotate p-values - load t-test results for each treatment vs BW control
    ttest_path = stats_dir / 'iron' / 'treatment_ttest_results.csv'
    ttest_df = pd.read_csv(ttest_path, index_col=0)
    for i, treatment in enumerate(iron_treatment_list):
        assert ax.get_xticklabels()[i].get_text() == treatment
        if treatment == 'BW-none-nan':
            continue
        p = ttest_df.loc[feature, 'pvals_' + treatment]
        p_text = '***\nP < 0.001' if p < 0.001 else sig_asterix([p])[0] + '\nP = %.3f' % p
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(i, 1.01, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
    ax.set_xticklabels([s.get_text().replace('-','\n') for s in ax.get_xticklabels()]) #rotation=45
    plt.savefig(plot_dir / 'iron.pdf', dpi=300)
    
    
    # addition of paraquat to BW and fepD
    paraquat_meta = window_meta.query("drug_type=='paraquat' or drug_type=='none'")
    paraquat_meta['treatment'] = paraquat_meta[['food_type','drug_type','imaging_plate_drug_conc']
                                               ].agg('-'.join, axis=1)
    plot_df = paraquat_meta.join(features.reindex(paraquat_meta.index))
    
    # boxplots comparing addition of paraquat (0.5, 1, 2 and 4 mM) to BW and fepD
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12,8))
    col_dict = dict(zip(paraquat_treatment_list, sns.color_palette('tab10_r', 
                                                                   len(paraquat_treatment_list))))
    sns.boxplot(x='treatment', y=feature, order=paraquat_treatment_list, ax=ax, data=plot_df,
                palette=col_dict, showfliers=False) 
    sns.stripplot(x='treatment', y=feature, order=paraquat_treatment_list, ax=ax, data=plot_df,
                  s=5, marker='D', color='k')
    legend = ax.legend()
    legend.remove()
    # annotate p-values - load t-test results for each treatment vs BW control
    ttest_path = stats_dir / 'paraquat' / 'treatment_ttest_results.csv'
    ttest_df = pd.read_csv(ttest_path, index_col=0)
    for i, treatment in enumerate(paraquat_treatment_list):
        assert ax.get_xticklabels()[i].get_text() == treatment
        if treatment == 'BW-none-nan':
            continue
        p = ttest_df.loc[feature, 'pvals_' + treatment]
        p_text = '***\nP < 0.001' if p < 0.001 else sig_asterix([p])[0] + '\nP = %.3f' % p
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(i, 1.01, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
    ax.set_xlabel('')
    ax.set_ylabel(feature.replace('_',' '), fontsize=12, labelpad=10)
    # ax.set_ylim(0.28,1.1)
    ax.set_title('Addition of paraquat to BW and fepD', pad=30, fontsize=18)
    ax.set_xticklabels([s.get_text().replace('-','\n') for s in ax.get_xticklabels()])#rotation=45,ha='right'
    plt.savefig(plot_dir / 'paraquat.pdf', dpi=300)
    
    return

def supplements_timeseries(metadata, project_dir=PROJECT_DIR, save_dir=SAVE_DIR, window=WINDOW_NUMBER):
    """ Timeseries plots for addition of enterobactin, paraquat, and iron to BW and fepD """
    
    # # Plot BW and fepD together for each treatment: none, enterobactin, paraquat, iron    
    # for drug in tqdm(drug_type_list):
    #     print("Plotting timeseries comparing BW to fepD when %s is added" % drug)
        
    #     drug_meta = metadata.query("window==@window and  drug_type==@drug")
                
    #     # boxplots for all treatments vs BW control
    #     drug_meta['treatment'] = drug_meta[['food_type','drug_type','imaging_plate_drug_conc']
    #                                        ].agg('-'.join, axis=1) 
                
    #     treatment_order = sorted(drug_meta['treatment'].unique())
        
    #     # get timeseries for control data
    #     control_timeseries = get_strain_timeseries(metadata[metadata['treatment']==all_treatment_control], 
    #                                                project_dir=project_dir, 
    #                                                strain=all_treatment_control,
    #                                                group_by='treatment',
    #                                                only_wells=None,
    #                                                save_dir=Path(save_dir) / 'Data' / all_treatment_control,
    #                                                verbose=False)
        
    #     for mode in motion_modes:
            
    #         plt.close('all')
    #         fig, ax = plt.subplots(figsize=(15,5), dpi=200)
    #         BW_treatments = [t for t in treatment_order if 'BW' in t]
    #         fepD_treatments = [t for t in treatment_order if 'fepD' in t]
    #         col_dict = dict(zip(BW_treatments, sns.color_palette('Blues', len(BW_treatments))))
    #         col_dict.update(dict(zip(fepD_treatments, sns.color_palette('Greens', len(fepD_treatments)))))
            
    #         # col_dict = dict(zip(treatment_order, sns.color_palette("plasma", len(treatment_order))))
    #         bluelight_frames = [(i*60*FPS, i*60*FPS+10*FPS) for i in BLUELIGHT_TIMEPOINTS_MINUTES]

    #         for treatment in treatment_order:
                    
    #             # get timeseries data for treatment data
    #             treatment_metadata = drug_meta[drug_meta['treatment']==treatment]
    #             treatment_timeseries = get_strain_timeseries(treatment_metadata, 
    #                                                          project_dir=project_dir, 
    #                                                          strain=treatment,
    #                                                          group_by='treatment',
    #                                                          only_wells=None,
    #                                                          save_dir=Path(save_dir) / 'Data' / treatment,
    #                                                          verbose=False)

    #             print("Plotting timeseries '%s' fraction for %s" % (mode, treatment))
    #             ax = plot_timeseries_motion_mode(df=treatment_timeseries,
    #                                              window=SMOOTH_WINDOW_SECONDS*FPS,
    #                                              error=True,
    #                                              mode=mode,
    #                                              max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
    #                                              title=None,
    #                                              saveAs=None,
    #                                              ax=ax,
    #                                              bluelight_frames=bluelight_frames,
    #                                              colour=col_dict[treatment],
    #                                              alpha=0.25)
            
    #         xticks = np.linspace(0, VIDEO_LENGTH_SECONDS*FPS, int(VIDEO_LENGTH_SECONDS/60)+1)
    #         ax.set_xticks(xticks)
    #         ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
    #         ax.set_xlabel('Time (minutes)', fontsize=12, labelpad=10)
    #         ax.set_ylabel('Fraction {}'.format(mode), fontsize=12, labelpad=10)
    #         ax.legend(treatment_order, fontsize=12, frameon=False, loc='best')
    
    #         if BLUELIGHT_WINDOWS_ONLY_TS:
    #             ts_plot_dir = Path(save_dir) / 'Plots' / 'timeseries_bluelight' / treatment
    #             ax.set_xlim([min(BLUELIGHT_TIMEPOINTS_MINUTES)*60*FPS-60*FPS, 
    #                          max(BLUELIGHT_TIMEPOINTS_MINUTES)*60*FPS+70*FPS])
    #         else:
    #             ts_plot_dir = Path(save_dir) / 'Plots' / 'timeseries' / treatment
    
    #         plt.tight_layout()
    #         ts_plot_dir.mkdir(exist_ok=True, parents=True)
    #         plt.savefig(ts_plot_dir / '{0}_{1}.pdf'.format(treatment, mode))  

    ### Each treatment vs control: BW vs BW + lysate // BW vs BW + supernatant
    window_meta = metadata.query("window==@window")
    window_meta['treatment'] = window_meta[['food_type','drug_type',
                                            'imaging_plate_drug_conc']].agg('-'.join, axis=1)     
    control_list = list(np.concatenate([np.array(['fepD-none-nan']), np.repeat('BW-none-nan', 19)]))
    treatment_list = ['fepD-enterobactin-nan',
                      'fepD-none-nan','BW-enterobactin-nan',
                      'BW-fe2O12S3-1.0','BW-fe2O12S3-4.0',
                      'BW-feCl3-1.0','BW-feCl3-4.0',
                      'BW-paraquat-0.5','BW-paraquat-1.0',
                      'BW-paraquat-2.0','BW-paraquat-4.0',
                      'fepD-enterobactin-nan',
                      'fepD-fe2O12S3-1.0','fepD-fe2O12S3-4.0',
                      'fepD-feCl3-1.0','fepD-feCl3-4.0',
                      'fepD-paraquat-0.5','fepD-paraquat-1.0',
                      'fepD-paraquat-2.0','fepD-paraquat-4.0']
    title_list = ['fepD vs fepD + enterobactin',
                  'BW vs fepD','Addition of enterobactin (BW control)',
                  'Addition of 1mM fe2O12S3 (BW control)','Addition of 4mM fe2O12S3 (BW control)',
                  'Addition of 1mM feCl3 (BW control)','Addition of 4mM feCl3 (BW control)',
                  'Addition of 0.5mM paraquat (BW control)','Addition of 1mM paraquat (BW control)',
                  'Addition of 2mM paraquat (BW control)','Addition of 4mM paraquat (BW control)',
                  'BW vs fepD + enterobactin',
                  'BW vs fepD + 1mM fe2O12S3','BW vs fepD + 4mM fe2O12S3',
                  'BW vs fepD + 1mM feCl3','BW vs fepD + 4mM feCl3',
                  'BW vs fepD + 0.5mM paraquat','BW vs fepD + 1mM paraquat',
                  'BW vs fepD + 2mM paraquat','BW vs fepD + 4mM paraquat']
    labs = [('fepD', 'fepD + enterobactin'),
            ('BW', 'fepD'),('BW', 'BW + enterobactin'),
            ('BW', 'BW + 1mM fe2O12S3'),('BW', 'BW + 4mM fe2O12S3'),
            ('BW', 'BW + 1mM feCl3'),('BW', 'BW + 4mM feCl3'),
            ('BW', 'BW + 0.5mM paraquat'),('BW', 'BW + 1mM paraquat'),
            ('BW', 'BW + 2mM paraquat'),('BW', 'BW + 4mM paraquat'),
            ('BW', 'fepD + enterobactin'),
            ('BW', 'fepD + 1mM fe2O12S3'),('BW', 'fepD + 4mM fe2O12S3'),
            ('BW', 'fepD + 1mM feCl3'),('BW', 'fepD + 4mM feCl3'),
            ('BW', 'fepD + 0.5mM paraquat'),('BW', 'fepD + 1mM paraquat'),
            ('BW', 'fepD + 2mM paraquat'),('BW', 'fepD + 4mM paraquat')]
    
    for control, treatment, title, lab in tqdm(zip(control_list, treatment_list, title_list, labs)):
        
        # get timeseries for control data
        control_ts = get_strain_timeseries(window_meta[window_meta['treatment']==control], 
                                           project_dir=project_dir, 
                                           strain=control,
                                           group_by='treatment',
                                           n_wells=6,
                                           save_dir=Path(save_dir) / 'Data' / control,
                                           verbose=False)
        
        # get timeseries for treatment data
        treatment_ts = get_strain_timeseries(window_meta[window_meta['treatment']==treatment], 
                                             project_dir=project_dir, 
                                             strain=treatment,
                                             group_by='treatment',
                                             n_wells=6,
                                             save_dir=Path(save_dir) / 'Data' / treatment,
                                             verbose=False)
 
        col_dict = dict(zip([control, treatment], sns.color_palette("pastel", 2)))
        bluelight_frames = [(i*60*FPS, i*60*FPS+10*FPS) for i in BLUELIGHT_TIMEPOINTS_MINUTES]

        for mode in motion_modes:
                    
            print("Plotting timeseries '%s' fraction for '%s' vs '%s'..." %\
                  (mode, treatment, control))
    
            plt.close('all')
            fig, ax = plt.subplots(figsize=(15,5), dpi=200)
    
            ax = plot_timeseries_motion_mode(df=control_ts,
                                             window=SMOOTH_WINDOW_SECONDS*FPS,
                                             error=True,
                                             mode=mode,
                                             max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
                                             title=None,
                                             saveAs=None,
                                             ax=ax,
                                             bluelight_frames=bluelight_frames,
                                             colour=col_dict[control],
                                             alpha=0.25)
            
            ax = plot_timeseries_motion_mode(df=treatment_ts,
                                             window=SMOOTH_WINDOW_SECONDS*FPS,
                                             error=True,
                                             mode=mode,
                                             max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
                                             title=None,
                                             saveAs=None,
                                             ax=ax,
                                             bluelight_frames=bluelight_frames,
                                             colour=col_dict[treatment],
                                             alpha=0.25)
        
            xticks = np.linspace(0, VIDEO_LENGTH_SECONDS*FPS, int(VIDEO_LENGTH_SECONDS/60)+1)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
            ax.set_xlabel('Time (minutes)', fontsize=12, labelpad=10)
            ax.set_ylabel('Fraction {}'.format(mode), fontsize=12, labelpad=10)
            ax.set_title(title, fontsize=12, pad=10)
            ax.legend([lab[0], lab[1]], fontsize=12, frameon=False, loc='best')
    
            if BLUELIGHT_WINDOWS_ONLY_TS:
                ts_plot_dir = Path(save_dir) / 'Plots' / 'timeseries_bluelight' / treatment
                ax.set_xlim([min(BLUELIGHT_TIMEPOINTS_MINUTES)*60*FPS-60*FPS, 
                             max(BLUELIGHT_TIMEPOINTS_MINUTES)*60*FPS+70*FPS])
            else:   
                ts_plot_dir = Path(save_dir) / 'Plots' / 'timeseries' / treatment
    
            #plt.tight_layout()
            ts_plot_dir.mkdir(exist_ok=True, parents=True)
            save_path = ts_plot_dir / '{0}_{1}.pdf'.format(treatment, mode)
            print("Saving to: %s" % save_path)
            plt.savefig(save_path)  

    return

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
                                                   imaging_dates=IMAGING_DATES, 
                                                   n_wells=N_WELLS,
                                                   add_well_annotations=N_WELLS==96,
                                                   from_source_plate=True)
        
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
                                                   max_value_cap=1e15, # TODO: check if needed, delete if not
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

    metadata['imaging_plate_drug_conc'] = metadata['imaging_plate_drug_conc'].astype(str)

    supplements_stats(metadata, 
                      features, 
                      save_dir=Path(SAVE_DIR) / "Stats",
                      window=WINDOW_NUMBER,
                      feature=FEATURE)

    # TODO: Test effect of solvents! DMSO vs PBS
    
    supplements_plots(metadata,
                      features,
                      plot_dir=Path(SAVE_DIR) / "Plots",
                      stats_dir=Path(SAVE_DIR) / "Stats",
                      window=WINDOW_NUMBER,
                      feature=FEATURE)

    supplements_timeseries(metadata,
                           project_dir=Path(PROJECT_DIR),
                           save_dir=Path(SAVE_DIR),
                           window=WINDOW_NUMBER)

    # # Check length/area of tracked objects - prop bad skeletons
    # results_df = check_tracked_objects(metadata, 
    #                                    length_minmax=(200, 2000), 
    #                                    width_minmax=(20, 500),
    #                                    save_to=Path(SAVE_DIR) / 'tracking_checks.csv')

