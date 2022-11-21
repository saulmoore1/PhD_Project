#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of Preliminary Keio tests with 4 strains

Aims: To decide on optimal conditions for imaging:
    1. Refrigerated vs freshly seeded lawns: old vs new
    2. Lawn growth time: 8 vs 24 hours
    3. Worm life stage: L4 vs D1

# Experiment 1
# imaging_dates: ['20201020','20201021']
# lawn_storage_type: old vs new 
# lawn_growth_time: 8hrs vs 24hrs
# worm_life_stage: L4 vs D1

# Experiment 2 - 8hrs lawn growth time chosen
# imaging_dates: ['20201208','20201209']
# lawn_storage_type: old vs new
# worm_life_stage: L4 vs D1

@author: sm5911
@date: 20201208

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
# from feature_extraction.decomposition.pca import plot_pca, remove_outliers_pca
# from feature_extraction.decomposition.tsne import plot_tSNE
# from feature_extraction.decomposition.umap import plot_umap
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from tierpsytools.preprocessing.filter_data import select_feat_set

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Keio_Tests_4strains_96WP"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Tests_4_strains"

IMAGING_DATES = ['20201020','20201021','20201208','20201209']
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

# REMOVE_OUTLIERS_PCA = False
# BLUELIGHT_TIMEPOINTS_SECONDS = [(60, 70),(160, 170),(260, 270)]

#%% Functions

def preliminary_stats(metadata,
                      features,
                      group_by='food_type',
                      control='BW25113',
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
                         control='BW25113',
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
                      drop_insignificant=drop_insignificant,
                      max_sig_feats=max_sig_feats,
                      p_value_threshold=pvalue_threshold,
                      scale_outliers=scale_outliers,
                      ylim_minmax=ylim_minmax)
    
    return

#%% Main

if __name__ == '__main__':
    
    aux_dir = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    res_dir = Path(PROJECT_DIR) / 'Results'
    
    metadata_path_local = Path(SAVE_DIR) / 'metadata.csv'
    features_path_local = Path(SAVE_DIR) / 'features.csv'

    if not metadata_path_local.exists() and not features_path_local.exists():
        metadata, metadata_path = compile_metadata(aux_dir, 
                                                   n_wells=N_WELLS, 
                                                   imaging_dates=IMAGING_DATES,
                                                   add_well_annotations=True,
                                                   from_source_plate=False)
        
        features, metadata = process_feature_summaries(metadata_path, 
                                                       results_dir=res_dir, 
                                                       compile_day_summaries=True, 
                                                       imaging_dates=IMAGING_DATES, 
                                                       align_bluelight=True, 
                                                       window_summaries=False,
                                                       n_wells=N_WELLS)
    
        # drop instances where any imgstore_name is missing
        imgstore_cols = ['imgstore_name_prestim','imgstore_name_bluelight','imgstore_name_poststim']
        for i in imgstore_cols:
            metadata = metadata[~metadata[i].isna()]
        features = features.reindex(metadata.index)
   
        ## replace _ with ∆
        # metadata['food_type'] = [f.replace('_','\u0394') for f in metadata['food_type']]


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
        
        assert not metadata['worm_strain'].isna().any()
        assert not metadata['food_type'].isna().any()
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
    
    treatment_cols = ['food_type','worm_life_stage','lawn_storage_type']
    metadata['treatment'] = metadata[treatment_cols].astype(str).agg('-'.join, axis=1)  
    control = 'BW25113-D1-old'
        
    # save video file list for treatments (for manual inspection)
    video_dict = masked_video_list_from_metadata(metadata, 
                                                 group_by='treatment', 
                                                 groups_list=None,
                                                 imgstore_col='imgstore_name_bluelight',
                                                 project_dir=Path(PROJECT_DIR),
                                                 save_dir=Path(SAVE_DIR) / 'video_filenames')
    
    plot_dir = Path(SAVE_DIR) / 'Plots'
    stats_dir = Path(SAVE_DIR) / 'Stats'
    
    # perform anova and t-tests comparing each treatment to control
    # preliminary_stats(metadata,
    #                   features,
    #                   group_by='treatment',
    #                   control=control,
    #                   save_dir=Path(SAVE_DIR) / 'Stats',
    #                   feature_set=feature_list,
    #                   pvalue_threshold=0.05,
    #                   fdr_method='fdr_by')
    
    # feature_list = ['motion_mode_paused_fraction_bluelight',
    #                 'motion_mode_forward_fraction_bluelight',
    #                 'motion_mode_backward_fraction_bluelight',
    #                 'width_midbody_50th_bluelight',
    #                 'minor_axis_w_forward_50th_bluelight',
    #                 'relative_to_head_base_angular_velocity_head_tip_abs_50th_bluelight',
    #                 'curvature_head_abs_50th_bluelight']
    
    # # boxplots comparing each treatment to control for each feature
    # preliminary_boxplots(metadata,
    #                      features,
    #                      group_by='treatment',
    #                      control=control,
    #                      save_dir=Path(SAVE_DIR) / 'Plots',
    #                      stats_dir=Path(SAVE_DIR) / 'Stats',
    #                      feature_set=feature_list,
    #                      pvalue_threshold=0.05,
    #                      scale_outliers=False,
    #                      drop_insignificant=True)

    # # extract Day 1 worms and compare across strains
    # D1_meta = metadata.query("worm_life_stage=='D1'")
    # D1_feat = features.reindex(D1_meta.index)
    # D1_meta['treatment'] = D1_meta[['food_type','lawn_storage_type']].astype(str).agg('-'.join, axis=1)  
    # control = 'BW25113-new'
    # preliminary_stats(D1_meta,
    #                   D1_feat,
    #                   group_by='treatment',
    #                   control='BW25113-new',
    #                   save_dir=stats_dir / 'lawn_storage_type_D1',
    #                   feature_set=feature_list,
    #                   pvalue_threshold=0.05,
    #                   fdr_method='fdr_by')
    # preliminary_boxplots(D1_meta,
    #                      D1_feat,
    #                      group_by='treatment',
    #                      control='BW25113-new',
    #                      save_dir=plot_dir / 'lawn_storage_type_D1',
    #                      stats_dir=stats_dir / 'lawn_storage_type_D1',
    #                      feature_set=feature_list,
    #                      pvalue_threshold=0.05,
    #                      scale_outliers=False,
    #                      drop_insignificant=True,
    #                      max_sig_feats=None)
    
    # extract just BW control data and compare 
    # old vs new lawns for BW D1 after 5 hours
    BW_D1_meta = metadata.query("food_type=='BW25113' and " + 
                                "worm_life_stage=='D1' and " +
                                "time_recording=='11:10:00'")
    BW_D1_feat = features.reindex(BW_D1_meta.index)
    
    preliminary_stats(BW_D1_meta,
                      BW_D1_feat,
                      group_by='lawn_storage_type',
                      control='new',
                      save_dir=stats_dir / 'lawn_storage_type_BW_D1',
                      feature_set=feature_list,
                      pvalue_threshold=0.05,
                      fdr_method='fdr_by')
    
    preliminary_boxplots(BW_D1_meta,
                         BW_D1_feat,
                         group_by='lawn_storage_type',
                         control='new',
                         save_dir=plot_dir / 'lawn_storage_type_BW_D1',
                         stats_dir=stats_dir / 'lawn_storage_type_BW_D1',
                         feature_set=feature_list,
                         pvalue_threshold=0.05,
                         scale_outliers=False,
                         drop_insignificant=True,
                         max_sig_feats=None)
    
    # old vs new lawns for BW L4
    BW_L4_meta = metadata.query("food_type=='BW25113' and " + 
                                "worm_life_stage=='L4'")
    BW_L4_feat = features.reindex(BW_L4_meta.index)
    
    preliminary_stats(BW_L4_meta,
                      BW_L4_feat,
                      group_by='lawn_storage_type',
                      control='new',
                      save_dir=stats_dir / 'lawn_storage_type_BW_L4',
                      feature_set=feature_list,
                      pvalue_threshold=0.05,
                      fdr_method='fdr_by')
    
    preliminary_boxplots(BW_L4_meta,
                         BW_L4_feat,
                         group_by='lawn_storage_type',
                         control='new',
                         save_dir=plot_dir / 'lawn_storage_type_BW_L4',
                         stats_dir=stats_dir / 'lawn_storage_type_BW_L4',
                         feature_set=feature_list,
                         pvalue_threshold=0.05,
                         scale_outliers=False,
                         drop_insignificant=True,
                         max_sig_feats=None)
    
    # L4 vs D1 worms for BW old
    BW_old_meta = metadata.query("food_type=='BW25113' and " + 
                                 "lawn_storage_type=='old' and " +
                                 "time_recording=='11:10:00' or time_recording=='11:50:00'")
    BW_old_feat = features.reindex(BW_old_meta.index)
    
    preliminary_stats(BW_old_meta,
                      BW_old_feat,
                      group_by='worm_life_stage',
                      control='D1',
                      save_dir=stats_dir / 'worm_life_stage_BW_old',
                      feature_set=feature_list,
                      pvalue_threshold=0.05,
                      fdr_method='fdr_by')
    
    preliminary_boxplots(BW_old_meta,
                         BW_old_feat,
                         group_by='worm_life_stage',
                         control='D1',
                         save_dir=plot_dir / 'worm_life_stage_BW_old',
                         stats_dir=stats_dir / 'worm_life_stage_BW_old',
                         feature_set=feature_list,
                         pvalue_threshold=0.05,
                         scale_outliers=False,
                         drop_insignificant=True,
                         max_sig_feats=None)
    
    # L4 vs D1 worms for BW new
    BW_new_meta = metadata.query("food_type=='BW25113' and " + 
                                 "lawn_storage_type=='new'")
    BW_new_feat = features.reindex(BW_new_meta.index)
    
    preliminary_stats(BW_new_meta,
                      BW_new_feat,
                      group_by='worm_life_stage',
                      control='D1',
                      save_dir=stats_dir / 'worm_life_stage_BW_new',
                      feature_set=feature_list,
                      pvalue_threshold=0.05,
                      fdr_method='fdr_by')
    
    preliminary_boxplots(BW_new_meta,
                         BW_new_feat,
                         group_by='worm_life_stage',
                         control='D1',
                         save_dir=plot_dir / 'worm_life_stage_BW_new',
                         stats_dir=stats_dir / 'worm_life_stage_BW_new',
                         feature_set=feature_list,
                         pvalue_threshold=0.05,
                         scale_outliers=False,
                         drop_insignificant=True,
                         max_sig_feats=None)
    
    # Compare food type on old lawns for D1 worms
    D1_old_meta = metadata.query("worm_life_stage=='D1' and " + 
                                 "lawn_storage_type=='old' and " + 
                                 "time_recording=='16:10:00'")
    D1_old_feat = features.reindex(D1_old_meta.index)
    
    preliminary_stats(D1_old_meta,
                      D1_old_feat,
                      group_by='food_type',
                      control='BW25113',
                      save_dir=stats_dir / 'food_type_D1_old',
                      feature_set=feature_list,
                      pvalue_threshold=0.05,
                      fdr_method='fdr_by')
    
    preliminary_boxplots(D1_old_meta,
                         D1_old_feat,
                         group_by='food_type',
                         control='BW25113_icd',
                         save_dir=plot_dir / 'food_type_D1_old',
                         stats_dir=stats_dir / 'food_type_D1_old',
                         feature_set=feature_list,
                         pvalue_threshold=0.05,
                         scale_outliers=False,
                         drop_insignificant=True,
                         max_sig_feats=None)
        
    ##### Hierarchical Clustering Analysis #####
            
    # Z-normalise data
    featZ = features.apply(zscore, axis=0)

    # plot clustermap
    print("\nPlotting clustermap")
    
    clustermap_path = plot_dir / 'heatmaps' / 'treatment_clustermap.pdf'
    cg = plot_clustermap(featZ, metadata,
                         group_by='treatment',
                         method=METHOD, 
                         metric=METRIC,
                         figsize=[20,6],
                         sub_adj={'bottom':0.05,'left':0,'top':1,'right':0.85},
                         saveto=clustermap_path,
                         label_size=15,
                         show_xlabels=False)
 
    clustermap_path = plot_dir / 'heatmaps' / 'treatment_clustermap_labels.pdf'
    cg = plot_clustermap(featZ, metadata,
                         group_by='treatment',
                         method=METHOD, 
                         metric=METRIC,
                         figsize=[20,6],
                         sub_adj={'bottom':0.2,'left':0,'top':1,'right':0.85},
                         saveto=clustermap_path,
                         label_size=(1.5,15),
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
    anova_path = stats_dir / 'ANOVA' / 'ANOVA_results.csv'
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
                         meta=metadata, 
                         group_by='treatment', 
                         pvalues_series=pvals_heatmap,
                         p_value_threshold=0.05,
                         selected_feats=None, # fset if len(fset) > 0 else None
                         saveto=heatmap_path,
                         figsize=[20,30],
                         sns_colour_palette="Pastel1",
                         label_size=25,
                         sub_adj={'top': 0.95, 'bottom': 0.05, 'left': 0.2, 'right': 0.9})        


    ### D1 worms new vs old ###
    
    # Z-normalise data
    
    featZ = BW_D1_feat.apply(zscore, axis=0)

    # plot clustermap
    print("\nPlotting clustermap")
    
    clustermap_path = plot_dir / 'heatmaps' / 'BW_D1_new_vs_old_clustermap.pdf'
    cg = plot_clustermap(featZ, BW_D1_meta,
                         group_by='lawn_storage_type',
                         method=METHOD, 
                         metric=METRIC,
                         figsize=[30,6],
                         sub_adj={'bottom':0.05,'left':0,'top':1,'right':0.85},
                         saveto=clustermap_path,
                         label_size=30,
                         show_xlabels=False)
 
    clustermap_path = plot_dir / 'heatmaps' / 'BW_D1_new_vs_old_clustermap_labels.pdf'
    cg = plot_clustermap(featZ, BW_D1_meta,
                         group_by='lawn_storage_type',
                         method=METHOD, 
                         metric=METRIC,
                         figsize=[30,6],
                         sub_adj={'bottom':0.2,'left':0,'top':1,'right':0.85},
                         saveto=clustermap_path,
                         label_size=(1.5,30),
                         show_xlabels=True)


    ### L4 worms new vs old ###
    
    # Z-normalise data
    featZ = BW_L4_feat.apply(zscore, axis=0)

    # plot clustermap
    print("\nPlotting clustermap")
    
    clustermap_path = plot_dir / 'heatmaps' / 'BW_L4_new_vs_old_clustermap.pdf'
    cg = plot_clustermap(featZ, BW_L4_meta,
                         group_by='lawn_storage_type',
                         method=METHOD, 
                         metric=METRIC,
                         figsize=[30,6],
                         sub_adj={'bottom':0.05,'left':0,'top':1,'right':0.85},
                         saveto=clustermap_path,
                         label_size=15,
                         show_xlabels=False)
 
    clustermap_path = plot_dir / 'heatmaps' / 'BW_L4_new_vs_old_clustermap_labels.pdf'
    cg = plot_clustermap(featZ, BW_L4_meta,
                         group_by='lawn_storage_type',
                         method=METHOD, 
                         metric=METRIC,
                         figsize=[30,6],
                         sub_adj={'bottom':0.2,'left':0,'top':1,'right':0.85},
                         saveto=clustermap_path,
                         label_size=(1.5,15),
                         show_xlabels=True)
    
 
    ### BW old L4 vs D1 worms ###
    
    # Z-normalise data
    featZ = BW_old_feat.apply(zscore, axis=0)

    # plot clustermap
    print("\nPlotting clustermap")
    
    clustermap_path = plot_dir / 'heatmaps' / 'BW_old_D1_vs_L4_clustermap.pdf'
    cg = plot_clustermap(featZ, BW_old_meta,
                         group_by='worm_life_stage',
                         method=METHOD, 
                         metric=METRIC,
                         figsize=[30,6],
                         sub_adj={'bottom':0.05,'left':0,'top':1,'right':0.85},
                         saveto=clustermap_path,
                         label_size=15,
                         show_xlabels=False)
 
    clustermap_path = plot_dir / 'heatmaps' / 'BW_old_D1_vs_L4_clustermap_labels.pdf'
    cg = plot_clustermap(featZ, BW_old_meta,
                         group_by='worm_life_stage',
                         method=METHOD, 
                         metric=METRIC,
                         figsize=[30,6],
                         sub_adj={'bottom':0.2,'left':0,'top':1,'right':0.85},
                         saveto=clustermap_path,
                         label_size=(1.5,15),
                         show_xlabels=True)
    

    ### BW new L4 vs D1 worms ###
    
    # Z-normalise data
    featZ = BW_new_feat.apply(zscore, axis=0)

    # plot clustermap
    print("\nPlotting clustermap")
    
    clustermap_path = plot_dir / 'heatmaps' / 'BW_new_D1_vs_L4_clustermap.pdf'
    cg = plot_clustermap(featZ, BW_new_meta,
                         group_by='worm_life_stage',
                         method=METHOD, 
                         metric=METRIC,
                         figsize=[30,6],
                         sub_adj={'bottom':0.05,'left':0,'top':1,'right':0.85},
                         saveto=clustermap_path,
                         label_size=15,
                         show_xlabels=False)
 
    clustermap_path = plot_dir / 'heatmaps' / 'BW_new_D1_vs_L4_clustermap_labels.pdf'
    cg = plot_clustermap(featZ, BW_new_meta,
                         group_by='worm_life_stage',
                         method=METHOD, 
                         metric=METRIC,
                         figsize=[30,6],
                         sub_adj={'bottom':0.2,'left':0,'top':1,'right':0.85},
                         saveto=clustermap_path,
                         label_size=(1.5,15),
                         show_xlabels=True)  


    ### D1 old food_type ###
    
    # Z-normalise data
    featZ = D1_old_feat.apply(zscore, axis=0)

    # plot clustermap
    print("\nPlotting clustermap")
    
    clustermap_path = plot_dir / 'heatmaps' / 'D1_old_food_type_clustermap.pdf'
    cg = plot_clustermap(featZ, D1_old_meta,
                         group_by='food_type',
                         method=METHOD, 
                         metric=METRIC,
                         figsize=[30,6],
                         sub_adj={'bottom':0.05,'left':0,'top':1,'right':0.85},
                         saveto=clustermap_path,
                         label_size=15,
                         show_xlabels=False)
 
    clustermap_path = plot_dir / 'heatmaps' / 'D1_old_food_type_clustermap_labels.pdf'
    cg = plot_clustermap(featZ, D1_old_meta,
                         group_by='food_type',
                         method=METHOD, 
                         metric=METRIC,
                         figsize=[30,6],
                         sub_adj={'bottom':0.2,'left':0,'top':1,'right':0.85},
                         saveto=clustermap_path,
                         label_size=(1.5,15),
                         show_xlabels=True)                  

     
    ##### Principal Components Analysis #####

    # pca_dir = plot_dir / 'PCA'
    
    # # remove outlier samples from PCA
    # if REMOVE_OUTLIERS_PCA:
    #     outlier_path = pca_dir / 'mahalanobis_outliers.pdf'
    #     features, inds = remove_outliers_pca(df=features, saveto=outlier_path)
    #     metadata = metadata.reindex(features.index) # reindex metadata
    #     featZ = features.apply(zscore, axis=0) # re-normalise data

    #     # Drop features with NaN values after normalising
    #     n_cols = len(featZ.columns)
    #     featZ.dropna(axis=1, inplace=True)
    #     n_dropped = n_cols - len(featZ.columns)
    #     if n_dropped > 0:
    #         print("Dropped %d features after normalisation (NaN)" % n_dropped)

    # coloured_strains_pca = list(metadata['treatment'].unique())

    # _ = plot_pca(featZ, metadata, 
    #              group_by='treatment', 
    #              control=control,
    #              var_subset=coloured_strains_pca, 
    #              saveDir=pca_dir,
    #              PCs_to_keep=10,
    #              n_feats2print=10,
    #              kde=False,
    #              sns_colour_palette="plasma",
    #              n_dims=2,
    #              label_size=8,
    #              sub_adj={'bottom':0.13,'left':0.13,'top':0.95,'right':0.88},
    #              legend_loc=[1.02,0.6],
    #              hypercolor=False)

    # ##### t-distributed Stochastic Neighbour Embedding #####   
    
    # print("\nPerforming tSNE")
    # tsne_dir = plot_dir / 'tSNE'
    # perplexities = [150] # NB: should be roughly equal to group size    
    # _ = plot_tSNE(featZ, metadata,
    #               group_by='treatment',
    #               var_subset=coloured_strains_pca,
    #               saveDir=tsne_dir,
    #               perplexities=perplexities,
    #               figsize=[8,8],
    #               label_size=8,
    #               # marker_size=20,
    #               sns_colour_palette="plasma")
    
    # ##### Uniform Manifold Projection #####  
    
    # print("\nPerforming UMAP")
    # umap_dir = plot_dir / 'UMAP'
    # n_neighbours = [150] # NB: should be roughly equal to group size
    # min_dist = 0.1 # Minimum distance parameter    
    # _ = plot_umap(featZ, metadata,
    #               group_by='treatment',
    #               var_subset=coloured_strains_pca,
    #               saveDir=umap_dir,
    #               n_neighbours=n_neighbours,
    #               min_dist=min_dist,
    #               figsize=[8,8],
    #               label_size=8,
    #               # marker_size=20,
    #               sns_colour_palette="plasma")   


    # TODO: Test optimal imaging timepoint - time on food 
    # 1. L4 at 25hrs vs D1 at 1 hour
    # 2. D1 sat 1, 3, 5, and 24 hrs
    
    # subset for imaging dates of experiment
    
    imaging_dates = ['20201208','20201209']
    metadata2 = metadata[metadata['date_yyyymmdd'].astype(str).isin(imaging_dates)]
    
    # extract time on food from filename
    metadata2['time_on_food'] = [Path(f).parent.name.split('hr_')[0].split('_')[-1] for f in 
                                 metadata2.featuresN_filename]
    
    # 1. 
    # compare between foods separately for L4 at 24 hours and Day1 at 24 hours
    old_lawn_meta2 = metadata2.query("lawn_storage_type=='old'")
    D1_old_24hr_meta2 = old_lawn_meta2.query("worm_life_stage=='D1' and time_on_food=='24'")
    D1_old_24hr_feat2 = features.reindex(D1_old_24hr_meta2.index)
    
    preliminary_stats(D1_old_24hr_meta2,
                      D1_old_24hr_feat2,
                      group_by='food_type',
                      control='BW25113',
                      save_dir=stats_dir / 'food_type_D1_24hr_old',
                      feature_set=feature_list,
                      pvalue_threshold=0.05,
                      fdr_method='fdr_by')
    
    preliminary_boxplots(D1_old_24hr_meta2,
                         D1_old_24hr_feat2,
                         group_by='food_type',
                             control='BW25113',
                         save_dir=plot_dir / 'food_type_D1_24hr_old',
                         stats_dir=stats_dir / 'food_type_D1_24hr_old',
                         feature_set=feature_list,
                         pvalue_threshold=0.05,
                         scale_outliers=False,
                         drop_insignificant=True,
                         max_sig_feats=None)
    
    # L4 at 24 hours
    L4_old_24hr_meta2 = old_lawn_meta2.query("worm_life_stage=='L4' and time_on_food=='24'")
    L4_old_24hr_feat2 = features.reindex(L4_old_24hr_meta2.index)
    
    preliminary_stats(L4_old_24hr_meta2,
                      L4_old_24hr_feat2,
                      group_by='food_type',
                      control='BW25113',
                      save_dir=stats_dir / 'food_type_L4_24hr_old',
                      feature_set=feature_list,
                      pvalue_threshold=0.05,
                      fdr_method='fdr_by')
    
    preliminary_boxplots(L4_old_24hr_meta2,
                         L4_old_24hr_feat2,
                         group_by='food_type',
                         control='BW25113',
                         save_dir=plot_dir / 'food_type_L4_24hr_old',
                         stats_dir=stats_dir / 'food_type_L4_24hr_old',
                         feature_set=feature_list,
                         pvalue_threshold=0.05,
                         scale_outliers=False,
                         drop_insignificant=True,
                         max_sig_feats=None)
    
    
    # 2. 
    # compare between food type at 1, 3, 5 and 24 hours on food for D1 worms on old lawns
    D1_old_meta2 = metadata2.query("worm_life_stage=='D1' and lawn_storage_type=='old'")
    D1_old_feat2 = features.reindex(D1_old_meta2.index)
    
    
    # 1 hour on food
    D1_old_1hr_meta2 = D1_old_meta2.query("time_on_food=='1'")
    D1_old_1hr_feat2 = features.reindex(D1_old_1hr_meta2.index)
    
    preliminary_stats(D1_old_1hr_meta2,
                      D1_old_1hr_feat2,
                      group_by='food_type',
                      control='BW25113',
                      save_dir=stats_dir / '1hr_on_food_D1_old',
                      feature_set=feature_list,
                      pvalue_threshold=0.05,
                      fdr_method='fdr_by')
    
    preliminary_boxplots(D1_old_1hr_meta2,
                         D1_old_1hr_feat2,
                         group_by='food_type',
                         control='BW25113',
                         save_dir=plot_dir / '1hr_on_food_D1_old',
                         stats_dir=stats_dir / '1hr_on_food_D1_old',
                         feature_set=feature_list,
                         pvalue_threshold=0.05,
                         scale_outliers=False,
                         drop_insignificant=True,
                         max_sig_feats=None) 
    
    # 3 hours on food
    D1_old_3hr_meta2 = D1_old_meta2.query("time_on_food=='3'")
    D1_old_3hr_feat2 = features.reindex(D1_old_3hr_meta2.index)
    
    preliminary_stats(D1_old_3hr_meta2,
                      D1_old_3hr_feat2,
                      group_by='food_type',
                      control='BW25113',
                      save_dir=stats_dir / '3hr_on_food_D1_old',
                      feature_set=feature_list,
                      pvalue_threshold=0.05,
                      fdr_method='fdr_by')
    
    preliminary_boxplots(D1_old_3hr_meta2,
                         D1_old_3hr_feat2,
                         group_by='food_type',
                         control='BW25113',
                         save_dir=plot_dir / '3hr_on_food_D1_old',
                         stats_dir=stats_dir / '3hr_on_food_D1_old',
                         feature_set=feature_list,
                         pvalue_threshold=0.05,
                         scale_outliers=False,
                         drop_insignificant=True,
                         max_sig_feats=None) 
    
    # 5 hours on food
    D1_old_5hr_meta2 = D1_old_meta2.query("time_on_food=='5'")
    D1_old_5hr_feat2 = features.reindex(D1_old_5hr_meta2.index)
    
    preliminary_stats(D1_old_5hr_meta2,
                      D1_old_5hr_feat2,
                      group_by='food_type',
                      control='BW25113',
                      save_dir=stats_dir / '5hr_on_food_D1_old',
                      feature_set=feature_list,
                      pvalue_threshold=0.05,
                      fdr_method='fdr_by')
    
    preliminary_boxplots(D1_old_5hr_meta2,
                         D1_old_5hr_feat2,
                         group_by='food_type',
                         control='BW25113',
                         save_dir=plot_dir / '5hr_on_food_D1_old',
                         stats_dir=stats_dir / '5hr_on_food_D1_old',
                         feature_set=feature_list,
                         pvalue_threshold=0.05,
                         scale_outliers=False,
                         drop_insignificant=True,
                         max_sig_feats=None)     
    
    