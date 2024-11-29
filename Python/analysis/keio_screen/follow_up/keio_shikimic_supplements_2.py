#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse results of FepD_supplementation data (collected by Alan and Riju) in which Shikimic acid,
Gentisic acid and 2,3-dihydroxybenzoic acid (or none - control) was added to the following bacteria:
    - BW, fepD, entA, entE, fepD_entA, and fepD_entE

This script:
    - compiles project metadata and feature summaries
    - cleans the summary results
    - calculates statistics for speed_50th across treatment groups (t-tests and ANOVA)
    - plots box plots of speed_50th (all treatments together)
    - plots time-series of speed throughout bluelight video (each treatment overlayed with BW and fepD)

@author: Saul Moore (sm5911)
@date: 03/07/2024 (updated: 21/11/2024)
    
"""

#%% Imports 

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from matplotlib import transforms
from matplotlib import pyplot as plt
# from matplotlib import patches
from scipy.stats import zscore

from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from visualisation.plotting_helper import sig_asterix
from clustering.hierarchical_clustering import plot_clustermap
from time_series.plot_timeseries import plot_timeseries, get_strain_timeseries
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
# from tierpsytools.plot.plot_plate_from_raw_video import plot_plates_from_metadata
from tierpsytools.preprocessing.filter_data import select_feat_set

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Shikimic_Supplements_2"
SAVE_DIR = "/Users/sm5911/Documents/PhD_DLBG/39_Keio_Shikimic_Supplements_2"

NAN_THRESHOLD_ROW = 0.8  # Drop samples with too many NaN/Inf values across features
NAN_THRESHOLD_COL = 0.05 # Drop features with too many NaN/Inf values across samples
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
          save_dir=None,
          feature_list=['speed_50th'],
          p_value_threshold=0.05,
          fdr_method='fdr_bh'):
        
    assert all(metadata.index == features.index)
    features = features[feature_list]

    # print mean sample size
    sample_size = metadata.groupby(group_by).count()
    print("Mean sample size of treatment group: %d" % int(sample_size[sample_size.columns[-1]].mean()))
    n = len(metadata[group_by].unique())
    
    if n > 2:
        
        # Perform ANOVA - is there variation among strains?
        stats, pvals, reject = univariate_tests(X=features, 
                                                y=metadata[group_by], 
                                                control=control, 
                                                test='ANOVA',
                                                comparison_type='multiclass',
                                                multitest_correction=fdr_method,
                                                alpha=p_value_threshold,
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

        if save_dir is not None:
            anova_path = Path(save_dir) / 'ANOVA_results.csv'
            anova_path.parent.mkdir(parents=True, exist_ok=True)
            anova_results.to_csv(anova_path, header=True, index=True)
             
    # Perform t-tests
    stats_t, pvals_t, reject_t = univariate_tests(X=features,
                                                  y=metadata[group_by],
                                                  control=control,
                                                  test='t-test',
                                                  comparison_type='binary_each_group',
                                                  multitest_correction=fdr_method,
                                                  alpha=p_value_threshold)
    
    effect_sizes_t = get_effect_sizes(X=features,
                                      y=metadata[group_by],
                                      control=control,
                                      linked_test='t-test')
    
    stats_t.columns = ['stats_' + str(c) for c in stats_t.columns]
    pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
    reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
    effect_sizes_t.columns = ['effect_size_' + str(c) for c in effect_sizes_t.columns]
    ttest_results = pd.concat([stats_t, pvals_t, reject_t, effect_sizes_t], axis=1)
       
    if save_dir is not None:
        ttest_path = Path(save_dir) / 't-test_results.csv'
        ttest_path.parent.mkdir(exist_ok=True, parents=True)
        ttest_results.to_csv(ttest_path, header=True, index=True)
    
    nsig = sum(reject_t.sum(axis=1) > 0)
    print("%d significant features between any %s vs %s (t-test, P<%.2f, %s)" %\
          (nsig, group_by, control, p_value_threshold, fdr_method))

    if n > 2:
        return anova_results, ttest_results
    else:
        return ttest_results  


def main():
    
    aux_dir = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    res_dir = Path(PROJECT_DIR) / 'Results'
    
    metadata_path_local = Path(SAVE_DIR) / 'metadata.csv'
    features_path_local = Path(SAVE_DIR) / 'features.csv'
    
    if not metadata_path_local.exists() or not features_path_local.exists():
        # compile metadata
        metadata, metadata_path = compile_metadata(aux_dir, 
                                                   n_wells=N_WELLS, 
                                                   add_well_annotations=False,
                                                   from_source_plate=True)
        
        # compile feature summaries
        features, metadata = process_feature_summaries(metadata_path, 
                                                       results_dir=res_dir, 
                                                       compile_day_summaries=True, 
                                                       imaging_dates=None, 
                                                       align_bluelight=False, 
                                                       window_summaries=True,
                                                       n_wells=N_WELLS)

        # clean results - remove bad well data + features with too many NaNs/zero std + impute NaNs
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
    
    assert not metadata['bacteria_strain'].isna().any()

    # plot full plate view by tiling first raw video frame from each camera
    #plot_plates_from_metadata(metadata, save_dir=Path(SAVE_DIR), dpi=600)

    # subset metadata results for bluelight videos only 
    bluelight_videos = [i for i in metadata['imgstore_name'] if 'bluelight' in i]
    metadata = metadata[metadata['imgstore_name'].isin(bluelight_videos)]
        
    # subset features for metadata subset
    features = features.reindex(metadata.index)
            
    treatment_cols = ['bacteria_strain','supplement']
    metadata['treatment'] = metadata[treatment_cols].astype(str).agg('-'.join, axis=1)
    metadata['treatment'] = [i.replace('-none','') for i in metadata['treatment']]

    treatment_list = ['BW','fepD'] + [i for i in sorted(metadata['treatment'].unique()) if i not in 
                                      ['BW','fepD']]
    
    stats_dir = Path(SAVE_DIR) / 'Stats'
    plots_dir = Path(SAVE_DIR) / 'Plots'

    
    # HCA heatmaps for Tierpsy Top256 features
    
    # load Tierpsy Top256 feature set + subset for Top256 features only
    features256 = select_feat_set(features, 
                                  tierpsy_set_name='tierpsy_256', 
                                  append_bluelight=False) # NB: results for only 251/256 features
    
    # z-normalise
    featZ = features256.apply(zscore, axis=0)
    
    heatmap_path = plots_dir / 'heatmap_256.pdf'
    heatmap_path.parent.mkdir(exist_ok=True, parents=True)
    
    fig = plot_clustermap(featZ, metadata, 
                          group_by='treatment',
                          row_colours=None,
                          method='complete', 
                          metric='euclidean',
                          figsize=[15,20],
                          sub_adj={'bottom':0.1,'left':0,'top':1,'right':0.83},
                          saveto=heatmap_path,
                          label_size=(3,10),
                          show_xlabels=True,
                          bluelight_col_colours=False)

    # stats vs BW
    _, ttest_results = stats(metadata,
                             features,
                             group_by='treatment',
                             control='BW',
                             save_dir=stats_dir / 'supplements_vs_BW',
                             feature_list=[FEATURE],
                             p_value_threshold=P_VALUE_THRESHOLD,
                             fdr_method=FDR_METHOD)

    # t-test pvals
    pvals = ttest_results[[c for c in ttest_results.columns if 'pval' in c]]
    pvals.columns = [c.replace('pvals_','') for c in pvals.columns]

    # boxplot - supplement treatments vs BW
    plot_df = metadata.join(features)
    lut = dict(zip(['BW','fepD'], sns.color_palette(palette='tab10', n_colors=2)))
    colour_dict = {i:(lut['fepD'] if 'fepD' in i else lut['BW']) for i in treatment_list}

    plt.close('all')
    sns.set_style('ticks')
    fig = plt.figure(figsize=[15,18])
    ax = fig.add_subplot(1,1,1)
    sns.boxplot(x='speed_50th',
                y='treatment',
                data=plot_df, 
                order=treatment_list,
                palette=colour_dict,
                showfliers=False, 
                showmeans=False,
                meanprops={"marker":"x", 
                           "markersize":5,
                           "markeredgecolor":"k"},
                flierprops={"marker":"x", 
                            "markersize":15, 
                            "markeredgecolor":"r"})
    dates = sorted(plot_df['date_yyyymmdd'].unique())
    date_lut = dict(zip(dates, sns.color_palette(palette="Greys", n_colors=len(dates))))
    for date in dates:
        date_df = plot_df[plot_df['date_yyyymmdd']==date]
        sns.stripplot(x='speed_50th',
                      y='treatment',
                      data=date_df,
                      order=treatment_list,
                      palette=[date_lut[date]] * len(treatment_list),
                      color='dimgray',
                      marker=".",
                      edgecolor='k',
                      linewidth=0.3,
                      s=12) #facecolors="none"
# =============================================================================
#         
#     patch_list = []
#     for day in list(plot_df['date_yyyymmdd'].unique()):
#         day_df = plot_df[plot_df['date_yyyymmdd']==day]
#         sns.stripplot(x='speed_50th',
#                       y='treatment',
#                       data=day_df,
#                       s=12,
#                       order=treatment_list,
#                       hue=None,
#                       palette=None,
#                       color=day_colour_dict[day],
#                       marker=".",
#                       edgecolor='k',
#                       linewidth=0.3) #facecolors="none"
#         patch_list.append(patches.Patch(color=day_colour_dict[day], label=str(day)))
# =============================================================================
    ax.axes.get_yaxis().get_label().set_visible(False) # remove y axis label
    ax.axes.set_yticklabels([l.get_text() for l in ax.axes.get_yticklabels()], fontsize=25)
    ax.tick_params(axis='y', which='major', pad=15)
    ax.axes.set_xlabel('Speed (µm s$^{-1}$)', fontsize=30, labelpad=25)                             
    plt.xticks(fontsize=20)
    plt.xlim(-20, 320)
    
    # scale x axis for annotations    
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData) #x=scaled
    
    # add pvalues to plot
    for i, strain in enumerate(treatment_list, start=0):
        if strain == 'BW':
            continue
        else:
            p = pvals.loc['speed_50th', strain]
            text = ax.get_yticklabels()[i]
            assert text.get_text() == strain
            p_text = sig_asterix([p])[0]
            ax.text(1.03, i, p_text, fontsize=20, ha='left', va='center', transform=trans)

    # add day legend key to plot
    # plt.legend(title="Date (YYYYMMDD)", handles=patch_list)

    # save plot            
    plt.subplots_adjust(left=0.5, right=0.9)
    boxplot_path = plots_dir / 'speed_50th_Shikimic_supplements_vs_BW.svg'
    boxplot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(boxplot_path, bbox_inches='tight', transparent=True)


    # time-series plots
    
    for treatment in tqdm(treatment_list[2:]):
        
        groups = ['BW','fepD', treatment]
        colour_dict_ts = dict(zip(groups, 
                                  sns.color_palette(palette='tab10', n_colors=len(groups))))
        bluelight_frames = [(i*FPS, j*FPS) for (i, j) in BLUELIGHT_TIMEPOINTS_SECONDS]
        feature = 'speed'
    
        save_dir = Path(SAVE_DIR) / 'timeseries-speed'
        save_dir.mkdir(exist_ok=True, parents=True)
        save_path = save_dir / '{}_speed_bluelight.pdf'.format(treatment)
        
        if not save_path.exists():
        
            print("Plotting timeseries speed for %s" % treatment)
        
            plt.close('all')
            fig, ax = plt.subplots(figsize=(15,6), dpi=300)
        
            for group in groups:
                
                # get control timeseries
                group_ts = get_strain_timeseries(metadata,
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
                                     colour=colour_dict_ts[group])
        
            plt.ylim(-10, 350)
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