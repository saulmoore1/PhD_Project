#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keio Tap Response Tests

@author: sm5911
@date: 05/07/2022

"""

#%% Imports

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt

from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from write_data.write import write_list_to_file
from visualisation.plotting_helper import sig_asterix, boxplots_sigfeats
from time_series.time_series_helper import get_strain_timeseries
from time_series.plot_timeseries import plot_timeseries_motion_mode


from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from tierpsytools.preprocessing.filter_data import select_feat_set

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Keio_Tap_Tests_6WP"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Tap_Tests"

N_WELLS = 6

nan_threshold_row = 0.8
nan_threshold_col = 0.05

FEATURE_SET = ['motion_mode_forward_fraction']

#%% Functions

def tap_response_stats(metadata,
                       features,
                       group_by='treatment',
                       control='BW-none',
                       save_dir=None,
                       feature_set=None,
                       pvalue_threshold=0.05,
                       fdr_method='fdr_by'):
    
    # check case-sensitivity
    assert len(metadata[group_by].unique()) == len(metadata[group_by].str.upper().unique())
    
    if feature_set is not None:
        feature_set = [feature_set] if isinstance(feature_set, str) else feature_set
        assert isinstance(feature_set, list)
        assert(all(f in features.columns for f in feature_set))
    else:
        feature_set = features.columns.tolist()
    features = features[feature_set]

    # print mean sample size
    sample_size = metadata.groupby(group_by).count()
    print("Mean sample size of %s/window: %d" % (group_by, 
                                                 int(sample_size[sample_size.columns[-1]].mean())))

    fset = []
    n = len(metadata[group_by].unique())
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

    return #anova_results, ttest_results

def tap_response_boxplots(metadata,
                          features,
                          group_by='treatment',
                          control='BW-none',
                          save_dir=None,
                          stats_dir=None,
                          feature_set=None,
                          pvalue_threshold=0.05):
    
    
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

def tap_response_timeseries(metadata, 
                            project_dir, 
                            save_dir, 
                            group_by='tap_stimulus',
                            control='none',
                            strain_list=None,
                            n_wells=6,
                            video_length_seconds=60,
                            motion_modes=['forwards','stationary','backwards'],
                            smoothing=10,
                            fps=25):
    """ Timeseries plots for standard imaging and bluelight delivery protocol for the initial and 
        confirmation screening of Keio Collection. Bluelight stimulation is delivered after 5 mins
        pre-stimulus, 10 secs stimulus every 60 secs, repeated 3 times (6 mins total), 
        followed by 5 mins post-stimulus (16 minutes total)
    """
            
    if strain_list is None:
        strain_list = list(metadata[group_by].unique())
    else:
        assert isinstance(strain_list, list)
        assert all(s in metadata[group_by].unique() for s in strain_list)
    strain_list = [s for s in strain_list if s != control]
        
    # remove entries with missing video filename info
    n_nan = len([s for s in metadata['imgstore_name'].unique() if not isinstance(s, str)])
    if n_nan > 1:
        print("WARNING: Ignoring {} entries with missing imgstore_name info".format(n_nan))
        metadata = metadata[~metadata['imgstore_name'].isna()]
        
    # get timeseries for BW
    control_ts = get_strain_timeseries(metadata[metadata[group_by]==control], 
                                  project_dir=project_dir, 
                                  strain=control,
                                  group_by=group_by,
                                  n_wells=n_wells,
                                  save_dir=Path(save_dir) / 'Data' / control)
    
    for strain in tqdm(strain_list):
        col_dict = dict(zip([control, strain], sns.color_palette("pastel", 2)))

        # get timeseries for strain
        strain_ts = get_strain_timeseries(metadata[metadata[group_by]==strain], 
                                          project_dir=project_dir, 
                                          strain=strain,
                                          group_by=group_by,
                                          n_wells=n_wells,
                                          save_dir=Path(save_dir) / 'Data' / strain)
    
        for mode in motion_modes:
            print("Plotting timeseries for motion mode %s fraction for %s vs %s.." % \
                  (mode, strain, control))

            plt.close('all')
            fig, ax = plt.subplots(figsize=(12,5), dpi=200)
    
            ax = plot_timeseries_motion_mode(df=control_ts,
                                             window=smoothing*fps,
                                             error=True,
                                             mode=mode,
                                             max_n_frames=video_length_seconds*fps,
                                             title=None,
                                             saveAs=None,
                                             ax=ax,
                                             bluelight_frames=None,
                                             colour=col_dict[control],
                                             alpha=0.25)
            
            ax = plot_timeseries_motion_mode(df=strain_ts,
                                             window=smoothing*fps,
                                             error=True,
                                             mode=mode,
                                             max_n_frames=video_length_seconds*fps,
                                             title=None,
                                             saveAs=None,
                                             ax=ax,
                                             bluelight_frames=None,
                                             colour=col_dict[strain],
                                             alpha=0.25)
        
            xticks = np.linspace(0, video_length_seconds*fps, int(video_length_seconds/60)+1)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(x/fps/60)) for x in xticks])   
            ax.set_xlabel('Time (minutes)', fontsize=12, labelpad=10)
            ax.set_ylabel('Fraction {}'.format(mode), fontsize=12, labelpad=10)
            ax.set_title('{0} vs {1}'.format(control, strain), fontsize=12, pad=10)
            ax.legend([control, strain], fontsize=12, frameon=False, loc='best')
            #TODO: plt.subplots_adjust(left=0.01,top=0.9,bottom=0.1,left=0.2)
    
            # save plot
            ts_plot_dir = save_dir / 'Plots' / '{0}'.format(strain)
            ts_plot_dir.mkdir(exist_ok=True, parents=True)
            save_path = ts_plot_dir / 'motion_mode_{0}.pdf'.format(mode)
            print("Saving to: %s" % save_path)
            plt.savefig(save_path)

    return

#%% Main

if __name__ == '__main__':
    
    aux_dir = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    res_dir = Path(PROJECT_DIR) / 'Results'
    
    metadata_path_local = Path(SAVE_DIR) / 'metadata.csv'
    features_path_local = Path(SAVE_DIR) / 'features.csv'
    
    if not metadata_path_local.exists() and not features_path_local.exists():
        metadata, metadata_path = compile_metadata(aux_dir, n_wells=N_WELLS, from_source_plate=True)
        
        features, metadata = process_feature_summaries(metadata_path, 
                                                       results_dir=res_dir, 
                                                       compile_day_summaries=True, 
                                                       imaging_dates=None, 
                                                       align_bluelight=False, 
                                                       window_summaries=False,
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

    # perform anova and t-tests comparing each treatment to BW control
    metadata['treatment'] = metadata[['food_type','tap_stimulus']].astype(str).agg('-'.join, axis=1)
    control = 'BW-none'
    
    # tap_response_stats(metadata,
    #                    features,
    #                    group_by='treatment',
    #                    control=control,
    #                    save_dir=Path(SAVE_DIR) / 'Stats',
    #                    feature_set=feature_list,
    #                    pvalue_threshold=0.05,
    #                    fdr_method='fdr_by')
    
    #XXX not enough data for individual treatment-wise comparison, so instead lump data together for 
    # each tap stimulus type to get an idea whether the tap stimuli are causing arousal
    tap_response_stats(metadata,
                       features,
                       group_by='tap_stimulus',
                       control='none',
                       save_dir=Path(SAVE_DIR) / 'Stats',
                       feature_set=feature_list,
                       pvalue_threshold=0.05,
                       fdr_method='fdr_by')
    tap_response_stats(metadata,
                       features,
                       group_by='food_type',
                       control='BW',
                       save_dir=Path(SAVE_DIR) / 'Stats',
                       feature_set=feature_list,
                       pvalue_threshold=0.05,
                       fdr_method='fdr_by')
    
    # boxplots comparing each treatment to BW control for each feature
    tap_response_boxplots(metadata,
                          features,
                          group_by='tap_stimulus',
                          control='none',
                          save_dir=Path(SAVE_DIR) / 'Plots',
                          stats_dir=Path(SAVE_DIR) / 'Stats',
                          feature_set=feature_list,
                          pvalue_threshold=0.05)
    
    # timeseries motion mode fraction for each treatment vs BW control
    # strain_list = list(metadata['treatment'].unique())
    tap_response_timeseries(metadata, 
                            project_dir=Path(PROJECT_DIR), 
                            save_dir=Path(SAVE_DIR) / 'timeseries', 
                            group_by='tap_stimulus',
                            control='none',
                            strain_list=None,
                            n_wells=6,
                            video_length_seconds=60,
                            motion_modes=['forwards','stationary','backwards'],
                            smoothing=4,
                            fps=25)
    
