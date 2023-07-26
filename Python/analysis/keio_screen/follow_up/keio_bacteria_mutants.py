#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse bacterial mutants that were created based on results of fepD proteomics analysis in order
to identify the pathway(s) responsible for arousal on fepD

@author: sm5911
@date: 11/07/2023

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
from visualisation.plotting_helper import sig_asterix
from time_series.plot_timeseries import plot_timeseries, get_strain_timeseries
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
# from preprocessing.find_raw_videos import find_raw_videos_from_metadata, copy_raw_videos_to_folder
# from tierpsytools.plot.plot_plate_from_raw_video import plot_plates_from_metadata

#%% Globals 

PROJECT_DIR = "/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Bacteria_Mutants"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Bacteria_Mutants"

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
MIN_NSKEL_PER_VIDEO = None
MIN_NSKEL_SUM = 500
P_VALUE_THRESHOLD = 0.05
FDR_METHOD = 'fdr_bh'

FEATURE = 'speed_50th'
N_WELLS = 6
DPI = 600
FPS = 25

BLUELIGHT_TIMEPOINTS_SECONDS = [(60, 70),(160, 170),(260, 270)]
WINDOW_DICT = {0:(290,300)}
WINDOW_NAME_DICT = {0:"20-30 seconds after blue light 3"}

#%% Functions 

def stats(metadata,
          features,
          group_by='treatment',
          control='BW',
          feat='speed_50th',
          pvalue_threshold=0.05,
          fdr_method='fdr_bh'):
    
    """ Perform ANOVA and t-tests to compare worm speed on each treatment vs control """
        
    assert all(metadata.index == features.index)
    features = features[[feat]]

    # print mean sample size
    sample_size = metadata.groupby(group_by).count()
    print("Mean sample size of %s/window: %d" % (group_by, 
                                                 int(sample_size[sample_size.columns[-1]].mean())))

    n = len(metadata[group_by].unique())
    if n > 2:
   
        # Perform ANOVA - is there variation among strains?
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

        # compile results
        anova_results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
        anova_results.columns = ['stats','effect_size','pvals','reject']     
        anova_results['significance'] = sig_asterix(anova_results['pvals'])
        anova_results = anova_results.sort_values(by=['pvals'], ascending=True) # rank by p-value
             
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
        
    nsig = sum(reject_t.sum(axis=1) > 0)
    print("%d significant features between any %s vs %s (t-test, P<%.2f, %s)" %\
          (nsig, group_by, control, pvalue_threshold, fdr_method))

    return anova_results, ttest_results


def main():
    
    metadata_path_local = Path(SAVE_DIR) / 'metadata.csv'
    features_path_local = Path(SAVE_DIR) / 'features.csv'
    
    aux_dir = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    res_dir = Path(PROJECT_DIR) / 'Results'
    
    if not metadata_path_local.exists() or not features_path_local.exists():

        # compile metadata and feature summaries
        metadata, metadata_path = compile_metadata(aux_dir, 
                                                   n_wells=N_WELLS, 
                                                   add_well_annotations=False,
                                                   from_source_plate=True)
                
        features, metadata = process_feature_summaries(metadata_path, 
                                                       results_dir=res_dir, 
                                                       compile_day_summaries=True, 
                                                       imaging_dates=None, 
                                                       align_bluelight=False, 
                                                       window_summaries=True,
                                                       n_wells=N_WELLS)

        # Clean results - Remove bad well data + features with too many NaNs/zero std + impute NaNs
        features, metadata = clean_summary_results(features, 
                                                   metadata,
                                                   feature_columns=None,
                                                   nan_threshold_row=NAN_THRESHOLD_ROW,
                                                   nan_threshold_col=NAN_THRESHOLD_COL,
                                                   max_value_cap=1e15,
                                                   imputeNaN=True,
                                                   min_nskel_per_video=None,
                                                   min_nskel_sum=None,
                                                   drop_size_related_feats=False,
                                                   norm_feats_only=False)
        
        # save clean metadata and features
        metadata.to_csv(metadata_path_local, index=False)
        features.to_csv(features_path_local, index=False)
        
    else:
        metadata = pd.read_csv(metadata_path_local, header=0, index_col=None, dtype={'comments':str})
        features = pd.read_csv(features_path_local, header=0, index_col=None)

    assert not features.isna().sum(axis=1).any()
    assert not (features.std(axis=1) == 0).any()    
    
    # delete empty well data from plate 35 (no bacteria strain info)
    metadata = metadata[~metadata['bacteria_strain'].isna()]

    # subset metadata results for bluelight videos only 
    bluelight_videos = [i for i in metadata['imgstore_name'] if 'bluelight' in i]
    metadata = metadata[metadata['imgstore_name'].isin(bluelight_videos)]
    
    # create treatment column
    metadata['treatment'] = metadata[['bacteria_strain','drug_type']
                                     ].astype(str).agg('-'.join, axis=1)
    metadata['treatment'] = [i.replace('-nan','') for i in metadata['treatment']]

    # for Cassandra:
    # selected_treatments = ['BW','fepD','fepD; entA','fepD; entF','fepD; entS']
    # video_dict = find_raw_videos_from_metadata(metadata, 
    #                                            treatment_col='treatment', 
    #                                            treatment_list=selected_treatments,
    #                                            n_videos=5,
    #                                            rand_seed=101)
    # dst = "/Users/sm5911/OneDrive - Imperial College London/Collaborations/2023_keio_bacteria_mutants"
    # copy_raw_videos_to_folder(video_dict, dst)
    
    # plot_plates_from_metadata(metadata, save_dir=Path(SAVE_DIR), dpi=600)
    
    KO_mutants = [i for i in metadata['treatment'].unique() if not 'IPTG' in i]
    OE_mutants = [i for i in metadata['treatment'].unique() if 'IPTG' in i]
    
    # Analyse KO mutants
    meta_KO = metadata[metadata['treatment'].isin(KO_mutants)]
    feat_KO = features.reindex(meta_KO.index)

    strain_KO_list = ['BW','fepD'] + [i for i in sorted(KO_mutants) if i not in ['BW','fepD']]

    # boxplot (KO strains)
    plot_df = meta_KO.join(feat_KO)
    colour_dict_KO = dict(zip(strain_KO_list, 
                              sns.color_palette(palette='tab10', n_colors=len(strain_KO_list))))

    plt.close('all')
    sns.set_style('ticks')
    fig = plt.figure(figsize=[12,18])
    ax = fig.add_subplot(1,1,1)
    sns.boxplot(x='speed_50th',
                y='treatment',
                data=plot_df, 
                order=strain_KO_list,
                palette=colour_dict_KO,
                showfliers=False, 
                showmeans=False,
                meanprops={"marker":"x", 
                           "markersize":5,
                           "markeredgecolor":"k"},
                flierprops={"marker":"x", 
                            "markersize":15, 
                            "markeredgecolor":"r"})
    sns.stripplot(x='speed_50th',
                  y='treatment',
                  data=plot_df,
                  s=12,
                  order=strain_KO_list,
                  hue=None,
                  palette=None,
                  color='dimgray',
                  marker=".",
                  edgecolor='k',
                  linewidth=0.3) #facecolors="none"
    
    ax.axes.get_yaxis().get_label().set_visible(False) # remove y axis label
    ax.axes.set_yticklabels([l.get_text() for l in ax.axes.get_yticklabels()], fontsize=25)
    ax.tick_params(axis='y', which='major', pad=15)
    ax.axes.set_xlabel('Speed (µm s$^{-1}$)', fontsize=30, labelpad=25)                             
    plt.xticks(fontsize=20)
    plt.xlim(-20, 250)
        
    # do stats
    control = 'BW'
    anova_results, ttest_results = stats(meta_KO,
                                         feat_KO,
                                         group_by='treatment',
                                         control=control,
                                         feat='speed_50th',
                                         pvalue_threshold=P_VALUE_THRESHOLD,
                                         fdr_method=FDR_METHOD)

    anova_path = Path(SAVE_DIR) / 'Stats_KO' / 'ANOVA' / 'anova_results_vs_{}.csv'.format(control)
    anova_path.parent.mkdir(exist_ok=True, parents=True)
    anova_results.to_csv(anova_path, header=True, index=True)
    
    ttest_path = Path(SAVE_DIR) / 'Stats_KO' / 't-test' / 't-test_results_vs_{}.csv'.format(control)
    ttest_path.parent.mkdir(exist_ok=True, parents=True)
    ttest_results.to_csv(ttest_path, header=True, index=True)
    
    # t-test pvals
    pvals = ttest_results[[c for c in ttest_results.columns if 'pval' in c]]
    pvals.columns = [c.replace('pvals_','') for c in pvals.columns]
    
    # scale x axis for annotations    
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData) #x=scaled
    
    # add pvalues to plot
    for i, strain in enumerate(strain_KO_list, start=0):
        if strain == control:
            continue
        else:
            p = pvals.loc['speed_50th', strain]
            text = ax.get_yticklabels()[i]
            assert text.get_text() == strain
            p_text = sig_asterix([p])[0]
            ax.text(1.03, i, p_text, fontsize=35, ha='left', va='center', transform=trans)
            
    plt.subplots_adjust(left=0.3, right=0.9)
    boxplot_path = Path(SAVE_DIR) / 'Plots_KO' / 'speed_50th_vs_{}.png'.format(control)
    boxplot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(boxplot_path, dpi=DPI)  
    

    # Analyse OE mutants
    meta_OE = metadata[metadata['treatment'].isin(OE_mutants)]
    feat_OE = features.reindex(meta_OE.index)

    strain_OE_list = ['BW-IPTG','fepD-IPTG'] + [i for i in sorted(OE_mutants) if i not in 
                                                ['BW-IPTG','fepD-IPTG']]

    # boxplot (OE strains)
    plot_df = meta_OE.join(feat_OE)
    colour_dict_OE = dict(zip(strain_OE_list, 
                              sns.color_palette(palette='tab10', n_colors=len(strain_OE_list))))

    plt.close('all')
    sns.set_style('ticks')
    fig = plt.figure(figsize=[18,12])
    ax = fig.add_subplot(1,1,1)
    sns.boxplot(x='speed_50th',
                y='treatment',
                data=plot_df, 
                order=strain_OE_list,
                palette=colour_dict_OE,
                showfliers=False, 
                showmeans=False,
                meanprops={"marker":"x", 
                           "markersize":5,
                           "markeredgecolor":"k"},
                flierprops={"marker":"x", 
                            "markersize":15, 
                            "markeredgecolor":"r"})
    sns.stripplot(x='speed_50th',
                  y='treatment',
                  data=plot_df,
                  s=12,
                  order=strain_OE_list,
                  hue=None,
                  palette=None,
                  color='dimgray',
                  marker=".",
                  edgecolor='k',
                  linewidth=0.3) #facecolors="none"
    
    ax.axes.get_yaxis().get_label().set_visible(False) # remove y axis label
    ax.axes.set_yticklabels([l.get_text() for l in ax.axes.get_yticklabels()], fontsize=25)
    ax.tick_params(axis='y', which='major', pad=15)
    ax.axes.set_xlabel('Speed (µm s$^{-1}$)', fontsize=30, labelpad=25)                             
    plt.xticks(fontsize=20)
    plt.xlim(-20, 250)
        
    # do stats
    control = 'BW-IPTG'
    anova_results, ttest_results = stats(meta_OE,
                                         feat_OE,
                                         group_by='treatment',
                                         control=control,
                                         feat='speed_50th',
                                         pvalue_threshold=P_VALUE_THRESHOLD,
                                         fdr_method=FDR_METHOD)

    anova_path = Path(SAVE_DIR) / 'Stats_OE' / 'ANOVA' / 'anova_results_vs_{}.csv'.format(control)
    anova_path.parent.mkdir(exist_ok=True, parents=True)
    anova_results.to_csv(anova_path, header=True, index=True)
    
    ttest_path = Path(SAVE_DIR) / 'Stats_OE' / 't-test' / 't-test_results_vs_{}.csv'.format(control)
    ttest_path.parent.mkdir(exist_ok=True, parents=True)
    ttest_results.to_csv(ttest_path, header=True, index=True)
    
    # t-test pvals
    pvals = ttest_results[[c for c in ttest_results.columns if 'pval' in c]]
    pvals.columns = [c.replace('pvals_','') for c in pvals.columns]
    
    # scale x axis for annotations    
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData) #x=scaled
    
    # add pvalues to plot
    for i, strain in enumerate(strain_OE_list, start=0):
        if strain == control:
            continue
        else:
            p = pvals.loc['speed_50th', strain]
            text = ax.get_yticklabels()[i]
            assert text.get_text() == strain
            p_text = sig_asterix([p])[0]
            ax.text(1.03, i, p_text, fontsize=35, ha='left', va='center', transform=trans)
            
    plt.subplots_adjust(left=0.3, right=0.9)
    boxplot_path = Path(SAVE_DIR) / 'Plots_OE' / 'speed_50th_vs_{}.png'.format(control)
    boxplot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(boxplot_path, dpi=DPI)  
    

    # timeseries (KO strains)
    
    for bacteria in tqdm(strain_KO_list[2:]):
        
        groups = ['BW','fepD', bacteria]
        bluelight_frames = [(i*FPS, j*FPS) for (i, j) in BLUELIGHT_TIMEPOINTS_SECONDS]
        feature = 'speed'
    
        save_dir = Path(SAVE_DIR) / 'timeseries-speed_KO'
        save_dir.mkdir(exist_ok=True, parents=True)
        save_path = save_dir / '{}_speed_bluelight.pdf'.format(bacteria)
        
        if not save_path.exists():
        
            print("Plotting timeseries speed for %s" % bacteria)
        
            plt.close('all')
            fig, ax = plt.subplots(figsize=(15,6), dpi=300)
        
            for group in groups:
                
                # get control timeseries
                group_ts = get_strain_timeseries(meta_KO,
                                                 project_dir=Path(PROJECT_DIR),
                                                 strain=group,
                                                 group_by='treatment',
                                                 feature_list=[feature],
                                                 save_dir=save_dir,
                                                 n_wells=N_WELLS,
                                                 verbose=True)
                
                ax = plot_timeseries(df=group_ts,
                                     feature=feature,
                                     error=True,
                                     max_n_frames=360*FPS, # 6-minute bluelight videos
                                     smoothing=10*FPS, 
                                     ax=ax,
                                     bluelight_frames=bluelight_frames,
                                     colour=colour_dict_KO[group])
        
            plt.ylim(-10, 300)
            xticks = np.linspace(0, 360*FPS, int(360/60)+1)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
            ax.set_xlabel('Time (minutes)', fontsize=20, labelpad=10)
            ylab = feature.replace('_50th'," (µm s$^{-1}$)")
            ax.set_ylabel(ylab, fontsize=20, labelpad=10)
            ax.legend(groups, fontsize=12, frameon=False, loc='best', handletextpad=1)
            plt.subplots_adjust(left=0.1, top=0.98, bottom=0.15, right=0.98)
        
            # save plot
            print("Saving to: %s" % save_path)
            plt.savefig(save_path)

    
    # timeseries (OE strains)

    for bacteria in tqdm(strain_OE_list[2:]):
        
        groups = ['BW-IPTG','fepD-IPTG', bacteria]
        bluelight_frames = [(i*FPS, j*FPS) for (i, j) in BLUELIGHT_TIMEPOINTS_SECONDS]
        feature = 'speed'
    
        save_dir = Path(SAVE_DIR) / 'timeseries-speed_KO'
        save_dir.mkdir(exist_ok=True, parents=True)
        save_path = save_dir / '{}_speed_bluelight.pdf'.format(bacteria)
        
        if not save_path.exists():
        
            print("Plotting timeseries speed for %s" % bacteria)
        
            plt.close('all')
            fig, ax = plt.subplots(figsize=(15,6), dpi=300)
        
            for group in groups:
                
                # get control timeseries
                group_ts = get_strain_timeseries(meta_OE,
                                                 project_dir=Path(PROJECT_DIR),
                                                 strain=group,
                                                 group_by='treatment',
                                                 feature_list=[feature],
                                                 save_dir=save_dir,
                                                 n_wells=N_WELLS,
                                                 verbose=True)
                
                ax = plot_timeseries(df=group_ts,
                                     feature=feature,
                                     error=True,
                                     max_n_frames=360*FPS, # 6-minute bluelight videos
                                     smoothing=10*FPS, 
                                     ax=ax,
                                     bluelight_frames=bluelight_frames,
                                     colour=colour_dict_OE[group])
        
            plt.ylim(-10, 300)
            xticks = np.linspace(0, 360*FPS, int(360/60)+1)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
            ax.set_xlabel('Time (minutes)', fontsize=20, labelpad=10)
            ylab = feature.replace('_50th'," (µm s$^{-1}$)")
            ax.set_ylabel(ylab, fontsize=20, labelpad=10)
            ax.legend(groups, fontsize=12, frameon=False, loc='best', handletextpad=1)
            plt.subplots_adjust(left=0.1, top=0.98, bottom=0.15, right=0.98)
        
            # save plot
            print("Saving to: %s" % save_path)
            plt.savefig(save_path)    
    
    
    return

#%% Main

if __name__ == '__main__':
    main()
