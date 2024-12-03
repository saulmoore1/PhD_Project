#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Confirmation screen speed window summaries - boxplots, timeseries, and cluster analysis of hit 
strains for each feature

Performs a cluster analysis to identify the strains with the biggest differences in terms 
of each feature
Andre: It might make more sense to just do this for a reduced set of features 
(e.g. one feature from each of the obvious clusters in the hits heatmap)


@author: sm5911
@date: 19/10/2022 (updated: 21/10/2024)

"""

#%% Imports

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from write_data.write import write_list_to_file
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from visualisation.plotting_helper import sig_asterix, boxplots_sigfeats
from time_series.plot_timeseries import plot_timeseries_feature

from tierpsytools.preprocessing.filter_data import select_feat_set

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Screen_Confirmation"
SAVE_DIR = "/Users/sm5911/Documents/PhD_DLBG/4_Keio_Screen_Confirmation/feature_hits_for_Filipe"

# PROJECT_DIR = "/Volumes/hermes$/KeioScreen2_96WP"
# SAVE_DIR = "/Users/sm5911/Documents/Keio_Conf_Screen"

N_WELLS = 96
MAX_VALUE_CAP = 1e15
nan_threshold_row = 0.8
nan_threshold_col = 0.05
min_nskel_sum = 6000
add_well_annotations = True
pval_threshold = 0.05
fdr_method = 'fdr_by'
FPS = 25
RENAME_DICT = {"FECE" : "fecE",
               "BW" : "wild_type"}

feature_list = ['speed_50th']

BLUELIGHT_TIMEPOINTS_SECONDS = [(60, 70),(160, 170),(260, 270)]
WINDOW_DICT = {0:(290,300)}
WINDOW_NAME_DICT = {0:"20-30 seconds after blue light 3"}


#%% Functions

def confirmation_window_stats(metadata,
                              features,
                              group_by='gene_name',
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
        
    fset = []
   
    # Perform ANOVA - is there variation among strains at each window?
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
    anova_test_results = test_results.sort_values(by=['pvals'], ascending=True) # rank by p-value
    
    if save_dir is not None:
        anova_path = Path(save_dir) / 'ANOVA' / 'ANOVA_results.csv'
        anova_path.parent.mkdir(parents=True, exist_ok=True)
        anova_test_results.to_csv(anova_path, header=True, index=True)

    # use reject mask to find significant feature set
    fset = pvals.loc[reject['ANOVA']].sort_values(by='ANOVA', ascending=True).index.to_list()

    if len(fset) > 0:
        print("%d significant features found by ANOVA by '%s' (P<%.2f, %s)" %\
              (len(fset), group_by, pvalue_threshold, fdr_method))
        if save_dir is not None:
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
    if save_dir is not None:
        ttest_path = Path(save_dir) / 't-test' / 't-test_results.csv'
        ttest_path.parent.mkdir(exist_ok=True, parents=True)
        ttest_results.to_csv(ttest_path, header=True, index=True)
    
    nsig = sum(reject_t.sum(axis=1) > 0)
    print("%d significant features between any %s vs %s (t-test, P<%.2f, %s)" %\
          (nsig, group_by, control, pvalue_threshold, fdr_method))

    return anova_test_results, ttest_results

def confirmation_window_boxplots(metadata,
                                 features,
                                 group_by='gene_name',
                                 control='BW',
                                 save_dir=None,
                                 stats_dir=None,
                                 feature_set=None,
                                 pvalue_threshold=0.05,
                                 drop_insignificant=False,
                                 scale_outliers=False,
                                 ylim_minmax=None):
    
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
                      pvals=pvals,
                      z_class=None,
                      feature_set=feature_set,
                      saveDir=Path(save_dir),
                      drop_insignificant=drop_insignificant,
                      p_value_threshold=pvalue_threshold,
                      scale_outliers=scale_outliers,
                      ylim_minmax=ylim_minmax,
                      append_ranking_fname=False)
    
    return

def hit_strains_by_feature(metadata, features, feature_list=None):
    """ 
    Performs a cluster analysis to identify the strains with the biggest differences in terms 
    of each feature
    
    Andre: It might make more sense to just do this for a reduced set of features 
    (e.g. one feature from each of the obvious clusters in the hits heatmap)
    
    Inputs
    -------------------    
    metadata, features - type: pd.DataFrame - The number of features in the features dataframe
    provided will be used for the cluster analysis
    
    feature_list - type: list - All features in the features dataframe will be used for the cluster 
    analysis. However, a subset list of features may be provided to return hit strains for those 
    features only
    -------------------
    
    Outputs
    -------------------
    hits_by_feature_dict - type: dict - A dictionary of hit strains (values) for each feature (keys)
    -------------------   
    """
    
    assert np.array_equal(metadata.index, features.index)
    
    # perform cluster analysis
    
    # average data for each strain
    from clustering.nearest_neighbours import average_strain_data, dropNaN
    feat, meta = average_strain_data(features, metadata, groups_column='gene_name')
    
    # z-normalise features
    
    
    
    
    
    return #hits_by_feature_dict

#%% Main

if __name__ == '__main__':
    
    aux_dir = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    res_dir = Path(PROJECT_DIR) / 'Results'
    
    metadata_path_local = Path(SAVE_DIR) / 'metadata_clean.csv' #'window_metadata.csv'
    features_path_local = Path(SAVE_DIR) / 'features_clean.csv' #'window_features.csv'
    
    if not metadata_path_local.exists() and not features_path_local.exists():
        metadata, metadata_path = compile_metadata(aux_dir, 
                                                   n_wells=N_WELLS, 
                                                   add_well_annotations=True,
                                                   from_source_plate=False)
        
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
                                                   norm_feats_only=False,
                                                   no_nan_cols=['gene_name'])
        
        # save clean metadata and features
        metadata.to_csv(metadata_path_local, index=False)
        features.to_csv(features_path_local, index=False)
        
    else:
        metadata = pd.read_csv(metadata_path_local, header=0, index_col=None, dtype={'comments':str})
        features = pd.read_csv(features_path_local, header=0, index_col=None)

    assert not features.isna().sum(axis=1).any()
    assert not (features.std(axis=1) == 0).any()
    
    # remove entries for results with missing gene name metadata
    n_rows = metadata.shape[0]
    metadata = metadata[~metadata['gene_name'].isna()]
    if metadata.shape[0] < n_rows:
        print("Removed %d row entries with missing gene name (or empty wells)" % (
            n_rows - metadata.shape[0]))
    
    # subset metadata results for bluelight videos only 
    if not 'bluelight' in metadata.columns:
        metadata['bluelight'] = [i.split('_run')[-1].split('_')[1] for i in metadata['imgstore_name']]
    metadata = metadata[metadata['bluelight']=='bluelight']
    
    features = features.reindex(metadata.index)

    metadata['window'] = metadata['window'].astype(int)    
    meta_window = metadata[metadata['window']==0]
    feat_window = features.reindex(meta_window.index)
 
    strain_list = sorted(list(metadata['gene_name'].unique()))

    stats_dir = Path(SAVE_DIR) / 'Stats'
    plot_dir = Path(SAVE_DIR) / 'Plots'

    # all gene_names
    confirmation_window_stats(meta_window,
                              feat_window,
                              group_by='gene_name',
                              control='BW',
                              save_dir=stats_dir,
                              feature_set=feature_list,
                              pvalue_threshold=pval_threshold,
                              fdr_method=fdr_method)
    
    confirmation_window_boxplots(meta_window,
                                 feat_window,
                                 group_by='gene_name',
                                 control='BW',
                                 save_dir=plot_dir,
                                 stats_dir=stats_dir,
                                 feature_set=feature_list,
                                 pvalue_threshold=pval_threshold,
                                 scale_outliers=False,
                                 ylim_minmax=None) # ylim_minmax for speed feature only 
    
    # perform cluster analysis to find hit strains for each feature
    top16 = select_feat_set(features, tierpsy_set_name='tierpsy_16')
    hit_strains_by_feature(metadata, top16)
    
    strain_list = ['fepD']
    for strain in tqdm(strain_list):
        plot_timeseries_feature(metadata,
                                project_dir=Path(PROJECT_DIR),
                                save_dir=Path(SAVE_DIR) / 'timeseries_speed',
                                group_by='gene_name',
                                control='BW',
                                groups_list=['BW', strain],
                                feature='speed',
                                n_wells=6,
                                bluelight_stim_type='bluelight',
                                video_length_seconds=360,
                                bluelight_timepoints_seconds=BLUELIGHT_TIMEPOINTS_SECONDS,
                                smoothing=10,
                                fps=FPS,
                                ylim_minmax=(-20,330)) # ylim_minmax for speed feature only
 
        