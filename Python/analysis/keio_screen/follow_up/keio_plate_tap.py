#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C. elegans plate tap experiments (+ paraquat dichloride)

@author: sm5911
@date: 08/08/2022 (updated: 11/11/2024)

"""

#%% Imports

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import transforms
from matplotlib import pyplot as plt

from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
#from write_data.write import write_list_to_file
from visualisation.plotting_helper import sig_asterix #, boxplots_sigfeats
from time_series.plot_timeseries import plot_timeseries_feature, plot_timeseries
from time_series.time_series_helper import get_strain_timeseries
#from analysis.keio_screen.follow_up.lawn_leaving_rate import fraction_on_food, timeseries_on_food

from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
#from tierpsytools.preprocessing.filter_data import select_feat_set

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Plate_Tap"
SAVE_DIR = "/Users/sm5911/Documents/PhD_DLBG/19_Keio_Plate_Tap"

N_WELLS = 6
FPS = 25

nan_threshold_row = 0.8
nan_threshold_col = 0.05

THRESHOLD_FILTER_DURATION = 25 # threshold trajectory length (n frames) / 25 fps => 1 second
THRESHOLD_FILTER_MOVEMENT = 10 # threshold movement (n pixels) * 12.4 microns per pixel => 124 microns
THRESHOLD_LEAVING_DURATION = 50 # n frames a worm has to leave food for => a true leaving event

#%% Functions

def stats(metadata,
          features,
          group_by='treatment',
          control='BW',
          feat='speed_50th',
          save_dir=None,
          pvalue_threshold=0.05,
          fdr_method='fdr_bh'):
        
    assert all(metadata.index == features.index)
    features = features[[feat]]
    
    # check case-sensitivity
    assert len(metadata[group_by].unique()) == len(metadata[group_by].str.upper().unique())

    # print mean sample size
    sample_size = metadata.groupby(group_by).count()
    print("Mean sample size per %s: %d" % (group_by, int(sample_size.max(axis=1).mean())))

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

        # compile ANOVA results
        anova_results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
        anova_results.columns = ['stats','effect_size','pvals','reject']     
        anova_results['significance'] = sig_asterix(anova_results['pvals'])
        anova_results = anova_results.sort_values(by=['pvals'], ascending=True) # rank by p-value

        if save_dir is not None:
            anova_path = Path(save_dir) / 'ANOVA_results.csv'
            anova_path.parent.mkdir(parents=True, exist_ok=True)
            anova_results.to_csv(anova_path, header=True, index=True)

        # # use reject mask to find significant feature set
        # fset = pvals.loc[reject['ANOVA']].sort_values(by='ANOVA', ascending=True).index.to_list()            
        # if len(fset) > 0:
        #     print("%d significant features found by ANOVA by '%s' (P<%.2f, %s)" %\
        #           (len(fset), group_by, pvalue_threshold, fdr_method))
        #     if save_dir is not None:
        #         anova_sigfeats_path = anova_path.parent / 'ANOVA_sigfeats.txt'
        #         write_list_to_file(fset, anova_sigfeats_path)
             
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
    if save_dir is not None:
        ttest_path = Path(save_dir) / 't-test' / 't-test_results.csv'
        ttest_path.parent.mkdir(exist_ok=True, parents=True)
        ttest_results.to_csv(ttest_path, header=True, index=True)
    
    # nsig = sum(reject_t.sum(axis=1) > 0)
    # print("%d significant features between any %s vs %s (t-test, P<%.2f, %s)" %\
    #       (nsig, group_by, control, pvalue_threshold, fdr_method))

    return anova_results, ttest_results

# =============================================================================
# def tap_boxplots(metadata,
#                   features,
#                   group_by='treatment',
#                   control='BW',
#                   save_dir=None,
#                   stats_dir=None,
#                   feature_set=None,
#                   pvalue_threshold=0.05,
#                   drop_insignificant=False,
#                   scale_outliers=False,
#                   ylim_minmax=None):
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
#                       pvals=pvals,
#                       z_class=None,
#                       feature_set=feature_set,
#                       saveDir=Path(save_dir),
#                       drop_insignificant=drop_insignificant,
#                       p_value_threshold=pvalue_threshold,
#                       scale_outliers=scale_outliers,
#                       ylim_minmax=ylim_minmax)
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
                                                   nan_threshold_row=nan_threshold_row,
                                                   nan_threshold_col=nan_threshold_col,
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
    
    # # load feature set
    # if FEATURE_SET is not None:
    #     # subset for selected feature set (and remove path curvature features)
    #     if isinstance(FEATURE_SET, int) and FEATURE_SET in [8,16,256]:
    #         features = select_feat_set(features, 'tierpsy_{}'.format(FEATURE_SET), append_bluelight=True)
    #         features = features[[f for f in features.columns if 'path_curvature' not in f]]
    #     elif isinstance(FEATURE_SET, list) or isinstance(FEATURE_SET, set):
    #         assert all(f in features.columns for f in FEATURE_SET)
    #         features = features[FEATURE_SET].copy()
    # feature_list = features.columns.tolist()
    
    # TODO: For no tap control use 20220803 mutant worm experiments (no bluelight, use prestim)
    
    treatment_cols = ['food_type','drug_type'] # 'tap_stimulus'
    metadata['treatment'] = metadata[treatment_cols].astype(str).agg('-'.join, axis=1)
    metadata['treatment'] = [i.replace('-nan','') for i in metadata['treatment']]
    treatment_list = sorted(metadata['treatment'].unique())
        
    stats_dir = Path(SAVE_DIR) / 'Stats'
    plots_dir = Path(SAVE_DIR) / 'Plots'
    
    # perform anova and t-tests comparing each treatment to BW control
    _, ttest_results = stats(metadata,
                             features,
                             group_by='treatment',
                             control='BW',
                             save_dir=stats_dir,
                             feat='speed_50th',
                             pvalue_threshold=0.05,
                             fdr_method='fdr_by')

    # extract t-test pvals
    pvals = ttest_results[[c for c in ttest_results.columns if 'pval' in c]]
    pvals.columns = [c.replace('pvals_','') for c in pvals.columns]
    
    # boxplot
    plot_df = metadata.join(features[['speed_50th']])
    colour_dict = dict(zip(treatment_list, 
                           np.repeat(sns.color_palette(palette='tab10', n_colors=2), 2, axis=0)))

    plt.close('all')
    sns.set_style('ticks')
    fig = plt.figure(figsize=[12,8])
    ax = fig.add_subplot(1,1,1)
    sns.boxplot(x='treatment',
                y='speed_50th',
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
    sns.stripplot(x='treatment',
                  y='speed_50th',
                  data=plot_df,
                  order=treatment_list,
                  color='dimgray',
                  marker=".",
                  edgecolor='k',
                  linewidth=0.3,
                  s=12) #facecolors="none"
        
    ax.axes.get_xaxis().get_label().set_visible(False) # remove y axis label
    ax.axes.set_xticklabels([l.get_text() for l in ax.axes.get_xticklabels()], fontsize=20)
    ax.tick_params(axis='x', which='major', pad=15)
    ax.axes.set_ylabel('Speed (µm s$^{-1}$)', fontsize=25, labelpad=25)                             
    plt.yticks(fontsize=20)
    plt.ylim(0, 230)
                
    # scale y axis for annotations    
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled
        
    # add pvalues to plot
    for i, text in enumerate(ax.axes.get_xticklabels()):
        treatment = text.get_text()
        if treatment == 'BW':
            continue
        else:
            p = pvals.loc['speed_50th', treatment]
            p_text = sig_asterix([p])[0]
            ax.text(i, 1.03, p_text, fontsize=20, ha='center', va='center', transform=trans)

    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[:2], labels[:2], loc='best', frameon=False, fontsize=15)
    boxplot_path = plots_dir / 'plate_tap_response.svg'
    boxplot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(boxplot_path, bbox_inches='tight', transparent=True)
    
    # timeseries plots of speed for each treatment vs control
    plot_timeseries_feature(metadata,
                            project_dir=Path(PROJECT_DIR),
                            save_dir=Path(SAVE_DIR) / 'timeseries-speed',
                            group_by='treatment',
                            control='BW',
                            groups_list=treatment_list,
                            feature='speed',
                            n_wells=6,
                            bluelight_stim_type=None,
                            video_length_seconds=360,
                            bluelight_timepoints_seconds=None,
                            smoothing=10,
                            fps=FPS,
                            ylim_minmax=(0,200))
        
    # bespoke timeseries    
    groups = ['BW-nan','fepD-nan','BW-Paraquat']        
    feature = 'speed'
    save_dir = Path(SAVE_DIR) / 'timeseries-speed' / 'rescues'
    ts_plot_dir = save_dir / 'Plots'
    ts_plot_dir.mkdir(exist_ok=True, parents=True)
    save_path = ts_plot_dir / 'speed_bluelight.pdf'
    
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
                             max_n_frames=300*FPS, 
                             smoothing=10*FPS, 
                             ax=ax,
                             bluelight_frames=None,
                             colour=col_dict[group])

    plt.ylim(0, 200)
    xticks = np.linspace(0, 300*FPS, int(300/60)+1)
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
