#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse results of data collected by John for nhr worm mutants on BW and fepD bacteria.
The following worm mutants were tested for arousal on fepD relative to BW control:

This script:
    - compiles project metadata and window eature summaries
    - cleans the summary results
    - calculates statistics for speed_50th across strains 
      (t-tests comparing fepD vs BW for each worm strain)
    - plots box plots of speed_50th (all strains together)
    - plots time-series of speed throughout bluelight video

@author: Saul Moore (sm5911)
@date: 04/10/2024
    
"""

#%% Imports 

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from matplotlib import transforms
from matplotlib import pyplot as plt
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

PROJECT_DIR = "/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Worm_NHR_Mutants"
SAVE_DIR = "/Users/sm5911/Documents/PhD_DLBG/40_Keio_Worm_NHR_Mutants"

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
# WINDOW_DICT = {0:(290,300)}
# WINDOW_NAME_DICT = {0:"20-30 seconds after blue light 3"}

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
    
    assert not metadata['bacteria_strain'].isna().any()
    assert not metadata['worm_gene'].isna().any()    

# =============================================================================
#     # plot full plate view by tiling first raw video frame (prestim) from each camera
#     plot_plates_from_metadata(metadata, save_dir=Path(SAVE_DIR), dpi=600)
# =============================================================================

    # subset metadata results for bluelight videos only 
    bluelight_videos = [i for i in metadata['imgstore_name'] if 'bluelight' in i]
    metadata = metadata[metadata['imgstore_name'].isin(bluelight_videos)]
    features = features.reindex(metadata.index) # reindex features for new metadata subset
    
    # standardise gene names in metadata
    metadata['bacteria_strain'] = ['fepD' if i=='FepD' else i for i in 
                                   metadata['bacteria_strain'].copy()]
    metadata['bacteria_strain'] = ['BW' if i.upper().startswith('BW') else i for i in
                                   metadata['bacteria_strain'].copy()]
        
    # combine into single list of treatment combinations
    treatment_cols = ['worm_gene','bacteria_strain']
    metadata['treatment'] = metadata[treatment_cols].astype(str).agg('_'.join, axis=1)
    metadata['treatment'] = [i.replace('-none','') for i in metadata['treatment']]
    treatment_list = ['N2_BW','N2_fepD'] + [i for i in sorted(metadata['treatment'].unique()) 
                                            if i not in ['N2_BW','N2_fepD']]
    
    # HCA heatmaps for Tierpsy Top256 features
    
    # load Tierpsy Top256 feature set + subset for Top256 features only
    features256 = select_feat_set(features, 
                                  tierpsy_set_name='tierpsy_256', 
                                  append_bluelight=False) # NB: results for only 251/256 features
    
    # z-normalise
    featZ = features256.apply(zscore, axis=0)
    
    heatmap_path = Path(SAVE_DIR) / 'Plots' / 'heatmap_256.pdf'
    heatmap_path.parent.mkdir(exist_ok=True, parents=True)
    
    fig = plot_clustermap(featZ, metadata, 
                          group_by='treatment',
                          row_colours=None,
                          method='complete', 
                          metric='euclidean',
                          figsize=[15,20],
                          sub_adj={'bottom':0.1,'left':0,'top':1,'right':0.9},
                          saveto=heatmap_path,
                          label_size=(3,10),
                          show_xlabels=True,
                          bluelight_col_colours=False)

    # boxplot
    plot_df = metadata.join(features)
    colour_dict = dict(zip(treatment_list, 
                           sns.color_palette(palette='tab10', n_colors=len(treatment_list))))

    plt.close('all')
    sns.set_style('ticks')
    fig = plt.figure(figsize=[12,18])
    ax = fig.add_subplot(1,1,1)
    sns.boxplot(x=FEATURE,
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
    date_lut = dict(zip(dates, sns.color_palette(palette="Set1", n_colors=len(dates))))
    for date in dates:
        date_df = plot_df[plot_df['date_yyyymmdd']==date]
        sns.stripplot(x=FEATURE,
                      y='treatment',
                      data=date_df,
                      s=12,
                      order=treatment_list,
                      hue=None,
                      palette=None,
                      color=date_lut[date],
                      marker=".",
                      edgecolor='k',
                      linewidth=0.3) #facecolors="none"
    ax.axes.get_yaxis().get_label().set_visible(False) # remove y axis label
    ax.axes.set_yticklabels([l.get_text() for l in ax.axes.get_yticklabels()], fontsize=25)
    ax.tick_params(axis='y', which='major', pad=15)
    if FEATURE == 'speed_50th':
        ax.axes.set_xlabel('Speed (µm s$^{-1}$)', fontsize=30, labelpad=25)                             
    plt.xticks(fontsize=20)
    plt.xlim(-20, 300)
        
    # do stats
    control = 'N2_BW'
    anova_results, ttest_results = stats(metadata,
                                         features,
                                         group_by='treatment',
                                         control=control,
                                         feat=FEATURE,
                                         pvalue_threshold=P_VALUE_THRESHOLD,
                                         fdr_method=FDR_METHOD)

    anova_path = Path(SAVE_DIR) / 'Stats' / 'ANOVA' / 'anova_results_vs_{}.csv'.format(control)
    anova_path.parent.mkdir(exist_ok=True, parents=True)
    anova_results.to_csv(anova_path, header=True, index=True)
    
    ttest_path = Path(SAVE_DIR) / 'Stats' / 't-test' / 't-test_results_vs_{}.csv'.format(control)
    ttest_path.parent.mkdir(exist_ok=True, parents=True)
    ttest_results.to_csv(ttest_path, header=True, index=True)
    
    # t-test pvals
    pvals = ttest_results[[c for c in ttest_results.columns if 'pval' in c]]
    pvals.columns = [c.replace('pvals_','') for c in pvals.columns]
    
    # scale x axis for annotations    
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData) #x=scaled
    
    # add pvalues to plot
    for i, treatment in enumerate(treatment_list, start=0):
        if treatment == control:
            continue
        else:
            p = pvals.loc[FEATURE, treatment]
            text = ax.get_yticklabels()[i]
            assert text.get_text() == treatment
            p_text = sig_asterix([p])[0]
            ax.text(1.03, i, p_text, fontsize=35, ha='left', va='center', transform=trans)
            
    plt.subplots_adjust(left=0.3, right=0.9)
    boxplot_path = Path(SAVE_DIR) / 'Plots' / '{0}_vs_{1}.png'.format(FEATURE, control)
    boxplot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(boxplot_path, dpi=DPI)      


    # time-series speed plots
    feature = 'speed'
    for worm in tqdm(sorted(metadata['worm_gene'].unique())[1:]):
        
        groups = ['N2_BW','N2_fepD',worm+'_BW',worm+'_fepD']
        colour_dict_ts = dict(zip(groups, 
                                  sns.color_palette(palette='tab10', n_colors=len(groups))))
        bluelight_frames = [(i*FPS, j*FPS) for (i, j) in BLUELIGHT_TIMEPOINTS_SECONDS]
    
        save_dir = Path(SAVE_DIR) / 'timeseries-speed'
        save_dir.mkdir(exist_ok=True, parents=True)
        save_path = save_dir / '{0}_{1}_bluelight.pdf'.format(worm, feature)
        
        if not save_path.exists():
        
            print("Plotting timeseries %s for %s" % (feature, worm))
        
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
        
            plt.ylim(-10, 300)
            xticks = np.linspace(0, 360*FPS, int(360/60)+1)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
            ax.set_xlabel('Time (minutes)', fontsize=20, labelpad=10)
            ax.set_ylabel(feature + " (µm s$^{-1}$)", fontsize=20, labelpad=10)
            ax.legend(groups, fontsize=12, frameon=False, loc='best', handletextpad=1)
            plt.subplots_adjust(left=0.1, top=0.98, bottom=0.15, right=0.98)
        
            # save plot
            print("Saving to: %s" % save_path)
            plt.savefig(save_path)    
    
    return

#%% Main

if __name__ == '__main__':
    main()