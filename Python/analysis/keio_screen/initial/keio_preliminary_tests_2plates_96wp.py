#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of Preliminary Keio tests with just Plate 13A and Plate 13B of the Keio library

Aims: To decide on protocol for:
    1. dispensing worms/worm life stage:  L4 in 10ul  vs  D1 with COPAS

@author: sm5911
@date: 20200303

"""

#%% Imports

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import zscore

from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from write_data.write import write_list_to_file
from visualisation.plotting_helper import sig_asterix, boxplots_sigfeats
from analysis.keio_screen.follow_up.uv_paraquat_antioxidant import masked_video_list_from_metadata
from clustering.hierarchical_clustering import plot_clustermap, plot_barcode_heatmap
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from tierpsytools.preprocessing.filter_data import select_feat_set

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Preliminary_Tests_2plates_96WP"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Preliminary_Tests_2plates_96WP"

IMAGING_DATES = ['20200303']
N_WELLS = 96
FPS = 25
FEATURE_SET = 256

nan_threshold_row = 0.8
nan_threshold_col = 0.2

THRESHOLD_FILTER_DURATION = 25 # threshold trajectory length (n frames) / 25 fps => 1 second
THRESHOLD_FILTER_MOVEMENT = 10 # threshold movement (n pixels) * 12.4 microns per pixel => 124 microns
THRESHOLD_LEAVING_DURATION = 50 # n frames a worm has to leave food for => a true leaving event

METHOD = 'complete'
METRIC = 'euclidean'

#%% Functions

def preliminary_stats(metadata,
                      features,
                      group_by='food_type',
                      control='WT',
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

def preliminary_boxplots(metadata,
                         features,
                         group_by='food_type',
                         control='WT',
                         save_dir=None,
                         stats_dir=None,
                         feature_set=None,
                         pvalue_threshold=0.05,
                         drop_insignificant=False,
                         max_sig_feats=10,
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
                      max_sig_feats=max_sig_feats,
                      drop_insignificant=drop_insignificant,
                      p_value_threshold=pvalue_threshold,
                      scale_outliers=scale_outliers,
                      ylim_minmax=ylim_minmax)
    
    return


def main():
    
    aux_dir = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    res_dir = Path(PROJECT_DIR) / 'Results'
    
    metadata_path_local = Path(SAVE_DIR) / 'metadata.csv'
    features_path_local = Path(SAVE_DIR) / 'features.csv'

    if not metadata_path_local.exists() and not features_path_local.exists():
        metadata, metadata_path = compile_metadata(aux_dir, 
                                                   n_wells=N_WELLS, 
                                                   imaging_dates=IMAGING_DATES,
                                                   add_well_annotations=False,
                                                   from_source_plate=False)
        
        features, metadata = process_feature_summaries(metadata_path, 
                                                       results_dir=res_dir, 
                                                       compile_day_summaries=True, 
                                                       imaging_dates=IMAGING_DATES, 
                                                       align_bluelight=True, 
                                                       window_summaries=False,
                                                       n_wells=N_WELLS)
    
        # drop instances where food type == '0'
        metadata = metadata[metadata['food_type'] != '0']

        # drop instances where any imgstore_name is missing
        imgstore_cols = ['imgstore_name_prestim','imgstore_name_bluelight','imgstore_name_poststim']
        for i in imgstore_cols:
            metadata = metadata[~metadata[i].isna()]
            
        features = features.reindex(metadata.index)
        
        # Clean results - remove bad well data + features with too many NaNs/zero std + impute NaNs
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
    
        assert not metadata['food_type'].isna().any()    
        assert not metadata['worm_strain'].isna().any()
        assert not features.isna().sum(axis=1).any()
        assert not (features.std(axis=1) == 0).any()
            
        # save clean metadata and features
        metadata.to_csv(metadata_path_local, index=False)
        features.to_csv(features_path_local, index=False)
        
    else:
        metadata = pd.read_csv(metadata_path_local, header=0, index_col=None, dtype={'comments':str})
        features = pd.read_csv(features_path_local, header=0, index_col=None)
        
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
    
    treatment_cols = ['food_type','worm_stage_when_dispensed']
    metadata['treatment'] = metadata[treatment_cols].astype(str).agg('-'.join, axis=1)
    control = 'WT-Day1'
    
    # save video file list for treatments (for manual inspection)
    _ = masked_video_list_from_metadata(metadata, 
                                        group_by='treatment', 
                                        groups_list=None,
                                        imgstore_col='imgstore_name_bluelight',
                                        project_dir=Path(PROJECT_DIR),
                                        save_dir=Path(SAVE_DIR) / 'video_filenames')
    
    plot_dir = Path(SAVE_DIR) / 'Plots'
    stats_dir = Path(SAVE_DIR) / 'Stats'
    
    ### Compare L4 vs Day1 ###
    
    WT_meta = metadata.query("food_type=='WT'")
    WT_feat = features.reindex(WT_meta.index)
    
    preliminary_stats(WT_meta,
                      WT_feat,
                      group_by='treatment',
                      control=control,
                      save_dir=stats_dir / 'WT_L4_vs_D1',
                      feature_set=feature_list,
                      pvalue_threshold=0.05,
                      fdr_method='fdr_by')
    
    # boxplots comparing each treatment to control for each feature
    preliminary_boxplots(WT_meta,
                         WT_feat,
                         group_by='treatment',
                         control=control,
                         save_dir=plot_dir / 'WT_L4_vs_D1',
                         stats_dir=stats_dir / 'WT_L4_vs_D1',
                         feature_set=feature_list,
                         pvalue_threshold=0.05,
                         scale_outliers=False,
                         drop_insignificant=True,
                         max_sig_feats=None)
    
    ### Compare all strains ###
        
    # perform anova and t-tests comparing each treatment to control
    preliminary_stats(metadata,
                      features,
                      group_by='treatment',
                      control=control,
                      save_dir=stats_dir / 'all_treatment',
                      feature_set=feature_list,
                      pvalue_threshold=0.05,
                      fdr_method='fdr_by')
    
    # boxplots comparing each treatment to control for each feature
    preliminary_boxplots(metadata,
                         features,
                         group_by='treatment',
                         control=control,
                         save_dir=plot_dir / 'all_treatment',
                         stats_dir=stats_dir / 'all_treatment',
                         feature_set=feature_list,
                         pvalue_threshold=0.05,
                         scale_outliers=False,
                         drop_insignificant=True,
                         max_sig_feats=None)

    # subset for Day 1 worms only (+4 hours on food)

    D1_meta = metadata.query("worm_stage_when_dispensed=='Day1' and imaging_run_number==3")
    D1_feat = features.reindex(D1_meta.index)
    
    # perform anova and t-tests comparing each treatment to control
    preliminary_stats(D1_meta,
                      D1_feat,
                      group_by='food_type',
                      control='WT',
                      save_dir=stats_dir / 'all_strains_Day1',
                      feature_set=feature_list,
                      pvalue_threshold=0.05,
                      fdr_method='fdr_by')
    
    # boxplots comparing each treatment to control for each feature
    preliminary_boxplots(D1_meta,
                         D1_feat,
                         group_by='food_type',
                         control='WT',
                         save_dir=plot_dir / 'all_strains_Day1',
                         stats_dir=stats_dir / 'all_strains_Day1',
                         feature_set=feature_list,
                         pvalue_threshold=0.05,
                         scale_outliers=False,
                         drop_insignificant=True,
                         max_sig_feats=None) 
    
    # subset for L4 worms only 
    
    L4_meta = metadata.query("worm_stage_when_dispensed=='L4'")
    L4_feat = features.reindex(L4_meta.index)
    
    # perform anova and t-tests comparing each treatment to control
    preliminary_stats(L4_meta,
                      L4_feat,
                      group_by='food_type',
                      control='WT',
                      save_dir=stats_dir / 'all_strains_L4',
                      feature_set=feature_list,
                      pvalue_threshold=0.05,
                      fdr_method='fdr_by')
    
    # boxplots comparing each treatment to control for each feature
    preliminary_boxplots(L4_meta,
                         L4_feat,
                         group_by='food_type',
                         control='WT',
                         save_dir=plot_dir / 'all_strains_L4',
                         stats_dir=stats_dir / 'all_strains_L4',
                         feature_set=feature_list,
                         pvalue_threshold=0.05,
                         scale_outliers=False,
                         drop_insignificant=True,
                         max_sig_feats=None)     

    ##### Hierarchical Clustering Analysis #####
            
    # Z-normalise data
    featZ = D1_feat.apply(zscore, axis=0)

    # plot clustermap Day 1
    print("\nPlotting clustermap")
    
    clustermap_path = plot_dir / 'heatmaps' / 'treatment_clustermap.pdf'
    cg = plot_clustermap(featZ, D1_meta,
                         group_by='food_type',
                         method=METHOD, 
                         metric=METRIC,
                         figsize=[30,12],
                         sub_adj={'bottom':0.05,'left':0,'top':1,'right':0.85},
                         saveto=clustermap_path,
                         label_size=12,
                         show_xlabels=False)
 
    clustermap_path = plot_dir / 'heatmaps' / 'treatment_clustermap_labels.pdf'
    cg = plot_clustermap(featZ, D1_meta,
                         group_by='food_type',
                         method=METHOD, 
                         metric=METRIC,
                         figsize=[30,12],
                         sub_adj={'bottom':0.15,'left':0,'top':1,'right':0.85},
                         saveto=clustermap_path,
                         label_size=(2,12),
                         show_xlabels=True)
    
    # save feature order to file
    feature_order = np.array(featZ.columns)[cg.dendrogram_col.reordered_ind]
    feature_order_df = pd.DataFrame(columns=['feature_name'], 
                                    index=range(1, len(feature_order) + 1),
                                    data=feature_order)
    feature_order_path = clustermap_path.parent / (clustermap_path.stem + '_feature_order.csv')
    feature_order_df.to_csv(feature_order_path, header=True, index=True)

    # plot heatmap - features (x-axis columns)
    
    print("\nLoading statistics results")
    anova_path = stats_dir / 'all_strains_Day1' / 'ANOVA' / 'ANOVA_results.csv'
    anova_table = pd.read_csv(anova_path, index_col=0)            
    pvals = anova_table.sort_values(by='pvals', ascending=True)['pvals'] # rank features by p-value
    fset = pvals[pvals < 0.05].index.to_list()
    print("\n%d significant features found by ANOVA (P<0.05, fdr_by)" % len(fset))
    pvals_heatmap = anova_table.loc[feature_order, 'pvals']
    pvals_heatmap.name = 'P < 0.05'
    assert all(f in featZ.columns for f in pvals_heatmap.index)
            
    # Plot heatmap (averaged for each sample)
    print("\nPlotting barcode heatmap")
    heatmap_path = plot_dir / 'heatmaps' / 'treatment_heatmap.pdf'
    plot_barcode_heatmap(featZ=featZ[feature_order], 
                         meta=D1_meta, 
                         group_by='food_type', 
                         pvalues_series=pvals_heatmap,
                         p_value_threshold=0.05,
                         selected_feats=None, # fset if len(fset) > 0 else None
                         saveto=heatmap_path,
                         figsize=[30,12],
                         sns_colour_palette="Pastel1",
                         label_size=20,
                         sub_adj={'top': 0.95, 'bottom': 0.05, 'left': 0.2, 'right': 0.9})        
    
    return

#%% Main

if __name__ == '__main__':
    main()
    
    
    