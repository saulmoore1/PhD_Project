#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASH lite-1 intesity tests - to find a suitable blue light intensity that does not result in a 
saturated response on BW and fepD with ChR2-expressing ASH lite-1 worms (+ retinal)

Hydra rig blue light intensity settings: 1,2,3,5,8

@author: sm5911
@date: 09/01/2023 (updated: 24/11/2024)

"""

#%% Imports 

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
#from matplotlib import transforms
from matplotlib import pyplot as plt

from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from visualisation.plotting_helper import sig_asterix#, boxplots_sigfeats
#from write_data.write import write_list_to_file
from time_series.plot_timeseries import plot_timeseries, get_strain_timeseries
# from time_series.plot_timeseries import plot_timeseries_feature

from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from tierpsytools.analysis.statistical_tests import _multitest_correct
from tierpsytools.preprocessing.filter_data import select_feat_set

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_ASH_Intensity_Tests"
SAVE_DIR = "/Users/sm5911/Documents/PhD_DLBG/26_Keio_ASH_Intensity_Tests"

N_WELLS = 6
FPS = 25

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
P_VALUE_THRESHOLD = 0.05
FDR_METHOD = 'fdr_bh'

FEATURE_SET = ['speed_50th']

BLUELIGHT_TIMEPOINTS_SECONDS = [(300, 310),(360, 370),(420, 430)]
VIDEO_LENGTH_SECONDS = 780

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


# =============================================================================
# def boxplots(metadata,
#              features,
#              group_by='treatment',
#              control='BW',
#              save_dir=None,
#              stats_dir=None,
#              feature_set=None,
#              pvalue_threshold=0.05):
#         
#     feature_set = features.columns.tolist() if feature_set is None else feature_set
#     assert isinstance(feature_set, list) and all(f in features.columns for f in feature_set)
#                     
#     # load t-test results for window
#     if stats_dir is not None:
#         ttest_path = Path(stats_dir) / 't-test' / 't-test_results.csv'
#         ttest_df = pd.read_csv(ttest_path, header=0, index_col=0)
#         pvals = ttest_df[[c for c in ttest_df.columns if 'pval' in c]]
#         pvals.columns = [c.replace('pvals_','') for c in pvals.columns]
#     
#     boxplots_sigfeats(features,
#                       y_class=metadata[group_by],
#                       control=control,
#                       pvals=pvals if stats_dir is not None else None,
#                       z_class=None,
#                       feature_set=feature_set,
#                       saveDir=Path(save_dir),
#                       drop_insignificant=True if feature_set is None else False,
#                       p_value_threshold=pvalue_threshold,
#                       scale_outliers=True,
#                       append_ranking_fname=False)
# 
#     return
# =============================================================================

def main():
    
    aux_dir = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    res_dir = Path(PROJECT_DIR) / 'Results'
    
    metadata_path_local = Path(SAVE_DIR) / 'metadata.csv'
    features_path_local = Path(SAVE_DIR) / 'features.csv'
    
    if not metadata_path_local.exists() or not features_path_local.exists():
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

    treatment_cols = ['bacteria_strain','rig_intensity']
    metadata['treatment'] = metadata.loc[:,treatment_cols].astype(str).agg('-'.join, axis=1)
    intensity_list = sorted(metadata['rig_intensity'].unique())
    strain_list = sorted(metadata['bacteria_strain'].unique())
    treatment_list = sorted(metadata['treatment'].unique())
    
    stats_dir = Path(SAVE_DIR) / 'Stats'
    plots_dir = Path(SAVE_DIR) / 'Plots'
    
    # stats - pairwise t-tests comparing speed on fepD vs BW during arousal window at each light 
    # intensity (p-values corrected for multiple testing afterwards)
    pvalues_dict = {}
    for light_intensity in intensity_list:
        meta = metadata[metadata['rig_intensity']==light_intensity]
        feat = features.reindex(meta.index)
        
        ttest_results = stats(meta, 
                              feat,
                              group_by='treatment',
                              control='BW-'+str(light_intensity),
                              save_dir=stats_dir / 'intensity_{}'.format(light_intensity),
                              feature_list=feature_list,
                              p_value_threshold=P_VALUE_THRESHOLD,
                              fdr_method=None)
        pvalues_dict[light_intensity] = ttest_results.loc['speed_50th',
                                                          'pvals_fepD-' + str(light_intensity)]
    # correct p-values for multiple comparisons - pairwise BW vs fepD across 5 light intensities
    pvals = pd.DataFrame.from_dict(pvalues_dict, orient='index', columns=['pvals'])
    reject, corrected_pvals = _multitest_correct(pvals['pvals'], multitest_method='fdr_bh', fdr=0.05)
    ttest_results_corrected = pd.concat([corrected_pvals, reject], axis=1
                                        ).rename(columns={0:'fepD',1:'reject'})
    save_path = stats_dir / 't-test_results_corrected.csv'
    save_path.parent.mkdir(exist_ok=True, parents=True)
    ttest_results_corrected.to_csv(save_path)
    
    # boxplot
    plot_df = metadata.join(features)
    lut = dict(zip(strain_list, sns.color_palette('tab10',2)))

    plt.close('all')
    sns.set_style('ticks')
    fig = plt.figure(figsize=[15,8])
    ax = fig.add_subplot(1,1,1)
    sns.boxplot(x='rig_intensity',
                y='speed_50th',
                data=plot_df, 
                order=intensity_list,
                hue='bacteria_strain',
                hue_order=strain_list,
                palette=lut,
                dodge=True,
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
        sns.stripplot(x='rig_intensity',
                      y='speed_50th',
                      data=date_df,
                      s=8,
                      order=intensity_list,
                      hue='bacteria_strain',
                      hue_order=strain_list,
                      dodge=True,
                      palette=[date_lut[date]] * len(intensity_list),
                      color=None,
                      marker=".",
                      edgecolor='k',
                      linewidth=0.3) #facecolors="none"

    # # scale y axis for annotations    
    # trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled
    ymax = plot_df.groupby('rig_intensity')['speed_50th'].max()
    
    # add pvalues to plot - fepD vs BW for each rig intensity
    for i, text in enumerate(ax.axes.get_xticklabels()):
        intensity = int(text.get_text())
        p = ttest_results_corrected.loc[intensity, 'fepD']
        p_text = sig_asterix([p])[0]
        y = ymax[intensity]
        ax.text(i, y+20, p_text, fontsize=20, ha='center', va='center')#, transform=trans)
    
        # Plot the bar: [x1,x1,x2,x2],[bar_tips,bar_height,bar_height,bar_tips]
        plt.plot([i-0.2, i-0.2, i+0.2, i+0.2],[y+10, y+15, y+15, y+10], lw=1, c='k')#, transform=trans)

    ax.axes.set_xticklabels(intensity_list, fontsize=20)
    ax.axes.set_xlabel('Rig Intensity (1-Low, 10-High)', fontsize=25, labelpad=20)   
    ax.tick_params(axis='x', which='major', pad=10)     
    ax.axes.set_ylabel('Speed (µm s$^{-1}$)', fontsize=25, labelpad=20)                     
    ax.tick_params(axis='y', which='major', pad=10)
    plt.yticks(fontsize=20)
    plt.ylim(0, 230)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc='lower right', frameon=False, fontsize=20)

    boxplot_path = plots_dir / 'rig_intensities_fepD_vs_BW.svg'
    boxplot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(boxplot_path, bbox_inches='tight', transparent=True)
    
    # timeseries - speed
    
    for intensity in tqdm(intensity_list):
        groups = ['BW-' + str(intensity), 'fepD-' + str(intensity)]
        print("Plotting timeseries speed for rig intensity: %s" % intensity)
        
        bluelight_frames = [(i*FPS, j*FPS) for (i, j) in BLUELIGHT_TIMEPOINTS_SECONDS]
        feature = 'speed'
    
        save_dir = Path(SAVE_DIR) / 'timeseries-speed'
        ts_plot_dir = save_dir / 'Plots'
        ts_plot_dir.mkdir(exist_ok=True, parents=True)
        save_path = ts_plot_dir / 'speed_bluelight_intensity_{}.pdf'.format(intensity)
        
        plt.close('all')
        fig, ax = plt.subplots(figsize=(15,6), dpi=300)
        col_dict = dict(zip(groups, sns.color_palette('tab10', len(groups))))
    
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
                                 max_n_frames=VIDEO_LENGTH_SECONDS*FPS, 
                                 smoothing=10*FPS, 
                                 ax=ax,
                                 bluelight_frames=bluelight_frames,
                                 colour=col_dict[group])
    
        plt.ylim(-200, 350)
        xticks = np.linspace(0, VIDEO_LENGTH_SECONDS*FPS, int(VIDEO_LENGTH_SECONDS/60)+1)
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


    # timeseries - absolute speed
    
    for intensity in tqdm(intensity_list):
        groups = ['BW-' + str(intensity), 'fepD-' + str(intensity)]
        print("Plotting timeseries absolute speed for rig intensity: %s" % intensity)
        
        bluelight_frames = [(i*FPS, j*FPS) for (i, j) in BLUELIGHT_TIMEPOINTS_SECONDS]
        feature = 'speed'
    
        save_dir = Path(SAVE_DIR) / 'timeseries-speed'
        ts_plot_dir = save_dir / 'Plots'
        ts_plot_dir.mkdir(exist_ok=True, parents=True)
        save_path = ts_plot_dir / 'absolute_speed_bluelight_intensity_{}.pdf'.format(intensity)
        
        plt.close('all')
        fig, ax = plt.subplots(figsize=(15,6), dpi=300)
        col_dict = dict(zip(groups, sns.color_palette('tab10', len(groups))))
    
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
            
            # convert to absolute speed
            group_ts['speed'] = np.abs(group_ts['speed'])

            
            ax = plot_timeseries(df=group_ts,
                                 feature=feature,
                                 error=True,
                                 max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
                                 smoothing=10*FPS, 
                                 ax=ax,
                                 bluelight_frames=bluelight_frames,
                                 colour=col_dict[group])
    
        plt.ylim(-20, 350)
        xticks = np.linspace(0, VIDEO_LENGTH_SECONDS*FPS, int(VIDEO_LENGTH_SECONDS/60)+1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
        ax.set_xlabel('Time (minutes)', fontsize=20, labelpad=10)
        ax.set_ylabel("Absolute Speed (µm s$^{-1}$)", fontsize=20, labelpad=10)
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