#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse between strain variation

@author: saul.moore11@lms.mrc.ac.uk
@date: 13/01/2021
"""

#%% Imports

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import (ttest_ind,
                         f_oneway,
                         kruskal,
                         zscore)

# Custom functions
from helper import (process_metadata, 
                    process_feature_summaries,
                    clean_features_summaries,
                    load_top256,
                    shapiro_normality_test,
                    ranksumtest,
                    ttest_by_feature,
                    anova_by_feature,
                    linear_mixed_model,
                    swarmplot_random_effect_variation,
                    barplot_sigfeats,
                    boxplots_sigfeats,
                    boxplots_by_strain,
                    plot_clustermap,
                    plot_barcode_clustermap,
                    plot_pca,
                    remove_outliers_pca,
                    plot_tSNE,
                    plot_umap)
   
# Path to Github helper functions (USER-DEFINED path to local copy of Github repo)
PATH_LIST = ['/Users/sm5911/Tierpsy_Versions/tierpsy-tools-python/']
for sysPath in PATH_LIST:
    if sysPath not in sys.path:
        sys.path.insert(0, sysPath)

from tierpsytools.analysis.significant_features import k_significant_feat

#%% Globals

control_dict = {'worm_strain': 'N2',
                'drug_type': 'DMSO',
                'food_type': 'BW25113', #'OP50',
                'instrument_name': 'Hydra01',
                'worm_life_stage': 'D1',
                'lawn_growth_duration_hours': '8',
                'lawn_storage_type': 'old'}
    
#%% Main

if __name__ == "__main__":
    
    # Accept command-line inputs
    parser = argparse.ArgumentParser(description='Analyse Tierpsy results (96-well)')
    parser.add_argument('--project_dir', help="Project root directory,\
                        containing 'AuxiliaryFiles', 'RawVideos',\
                        'MaskedVideos' and 'Results' folders",
                        default='/Volumes/hermes$/Filipe_Tests_96WP', type=str) # '/Volumes/hermes$/KeioScreen_96WP'
    parser.add_argument('--analyse_variables', help="List of categorical \
                        variables that you wish to investigate", nargs='+',
                        default=['worm_strain','instrument_name']) #['food_type','instrument_name']
    parser.add_argument('--compile_day_summaries', help="Compile full feature summaries from \
                        day feature summary results", default=False, action='store_true')
    # Keio = ['food_type','instrument_name','lawn_growth_duration_hours','lawn_storage_type']
    parser.add_argument('--add_well_annotations', help="Add 'is_bad_well' labels \
                        from WellAnnotator GUI", default=True,
                        action='store_false')
    parser.add_argument('--omit_strains', help="List of strains in 'analyse_variables' \
                        to omit from the analysis", nargs='+', default=None)
    parser.add_argument('--dates', help="List of imaging dates to use for analysis.\
                        If None, all imaging dates will be investigated", nargs='+',
                        default=None) # ['20201208', '20201209']
    parser.add_argument('--runs', help="List of imaging run numbers to use for \
                        analysis. If None, all imaging runs will be investigated",
                        nargs='+', default=None)
    parser.add_argument('--use_top256', help="Use Tierpsy Top256 features only",
                        default=False, action='store_true')
    parser.add_argument('--drop_size_features', help="Remove size-related Tierpsy \
                        features from analysis", default=False, 
                        action='store_true')
    parser.add_argument('--norm_features_only', help="Use only normalised \
                        size-invariant features ('_norm') for analysis",
                        default=False, action='store_true')
    parser.add_argument('--check_normal', help="Perform Shapiro-Wilks test for \
                        normality to decide between parametric/non-parametric \
                        statistics", default=True, action='store_false')
    parser.add_argument('--feature_means_only', help="Use feature summaries for distribution means \
                        only", default=False, action='store_true')
    parser.add_argument('--remove_outliers', help="Use Mahalanobis distance to \
                        exclude outliers from analysis", default=False, 
                        action='store_true')  
    parser.add_argument('--nan_threshold', help="Threshold proportion of NaN values \
                        to drop feature from analysis", default=0.2, type=float)
    parser.add_argument('--pval_threshold', help="Threshold p-value for statistical \
                        significance", default=0.05, type=float)
    parser.add_argument('--k_sig_features', help="Number of most significantly \
                        different features to plot", default=100, type=int)  
    args = parser.parse_args()
    # FIXME: Add optional args for feature means only from feeature summaries -- see discussion about percentile testing
    
    print("\nInputs:\n")
    for arg in list(args._get_kwargs()):
        print('%s: %s' % (arg[0].upper(), str(arg[1])))
    print("\n")
        
    PROJECT_DIR = Path(args.project_dir)                        # str
    IMAGING_DATES = args.dates                                  # list
    COMPILE_DAY_SUMMARIES = args.compile_day_summaries          # bool
    IMAGING_RUNS = args.runs                                    # list
    VARIABLES_TO_INVESTIGATE = args.analyse_variables           # list
    OMIT_STRAIN_LIST = args.omit_strains                        # list
    USE_TOP256 = args.use_top256                                # bool
    FILTER_SIZE_FEATS = args.drop_size_features                 # bool
    NORM_FEATS_ONLY = args.norm_features_only                   # bool
    FEAT_MEANS_ONLY = args.feature_means_only                   # bool
    ADD_WELL_ANNOTATIONS = args.add_well_annotations            # bool
    CHECK_NORMAL = args.check_normal                            # bool
    REMOVE_OUTLIERS = args.remove_outliers                      # bool
    NAN_THRESHOLD = args.nan_threshold                          # float
    P_VALUE_THRESHOLD = args.pval_threshold                     # float
    K_SIG_FEATS = args.k_sig_features                           # int
    
    FDR_METHOD = 'fdr_by' # Benjamini-Yekutieli correction for multiple testing
    
    # IO paths
    aux_dir = PROJECT_DIR / "AuxiliaryFiles"
    results_dir = PROJECT_DIR / "Results"
    save_dir = Path('/Users/sm5911/Documents/tmp_analysis/Filipe') #  = results_dir

    #%% Compile and clean results
    
    # Process metadata    
    metadata = process_metadata(aux_dir=aux_dir, 
                                imaging_dates=IMAGING_DATES, 
                                add_well_annotations=ADD_WELL_ANNOTATIONS)
       
    # Process feature summary results
    features, metadata = process_feature_summaries(metadata, 
                                                   results_dir,
                                                   compile_day_summaries=COMPILE_DAY_SUMMARIES,
                                                   imaging_dates=IMAGING_DATES,
                                                   add_bluelight=True)
    
    # Clean: remove data with too many NaNs/zero std and impute remaining NaNs
    features, metadata = clean_features_summaries(features, 
                                                  metadata,
                                                  feature_columns=None,
                                                  imputeNaN=True,
                                                  nan_threshold=NAN_THRESHOLD,
                                                  max_value_cap=1e15,
                                                  drop_size_related_feats=FILTER_SIZE_FEATS,
                                                  norm_feats_only=NORM_FEATS_ONLY)
    # Save full results to file
    full_results_path = save_dir / 'full_results.csv' 
    if not full_results_path.exists():
        fullresults = metadata.join(features) # join metadata + results
        print("Saving full results (features/metadata) to:\n '%s'" % full_results_path)     
        fullresults.to_csv(full_results_path, index=False)
    
    # Load Tierpsy Top256 feature set
    if USE_TOP256:     
        top256_path = aux_dir / 'top256_tierpsy_no_blob_no_eigen_only_abs_no_norm.csv'
        top256 = load_top256(top256_path, remove_path_curvature=True, add_bluelight=True)
        
        # Ensure results exist for features in featurelist
        top256_feat_list = [feat for feat in top256 if feat in features.columns]
        print("Dropped %d features in Top256 that are missing from results" %\
              (len(top256)-len(top256_feat_list)))
        
        # Select features for analysis
        features = features[top256_feat_list]
    
    # Subset results (rows) for imaging dates of interest
    if IMAGING_DATES:
        metadata = metadata[metadata['date_yyyymmdd'].isin(IMAGING_DATES)]
        features = features.reindex(metadata.index)
    
    # Subset results (rows) for imaging run of interest
    if IMAGING_RUNS:
        imaging_run_list = IMAGING_RUNS if type(IMAGING_RUNS) == list else [IMAGING_RUNS]
    else:
        imaging_run_list = list(metadata['imaging_run_number'].unique().astype(int))
        print("Found %d imaging runs to analyse: %s" % (len(imaging_run_list), imaging_run_list))
        
    # Subset results (rows) for strains of interest (check case-sensitive)
    assert len(metadata['worm_strain'].unique()) == len(metadata['worm_strain'].str.upper().unique())
    if OMIT_STRAIN_LIST:
        OMIT_STRAIN_LIST = [o.upper() for o in OMIT_STRAIN_LIST]
        strain_list = [s for s in metadata['worm_strain'].unique() if s.upper() not in OMIT_STRAIN_LIST]
    else:
        strain_list = list(metadata['worm_strain'].unique())
        
    metadata = metadata[metadata['worm_strain'].isin(strain_list)]
    features = features.reindex(metadata.index)
    
    # Subset results (cols) for '_norm' features only
    if NORM_FEATS_ONLY:
        norm_feat_list = [f for f in features.columns if '_norm' in f]
        features = features[norm_feat_list]
    
    # feature sets for each stimulus type
    featsets = {}
    for stim in ['_prestim','_bluelight','_poststim']:
        featsets[stim] = [f for f in features.columns if stim in f]
        
    # Subset results (cols) for mean feature summaries only
    # FIXME
    # if FEAT_MEANS_ONLY:
    #     mean_feats_list = []
    #     for stim, fset in featsets.items():
    #         omit_parts = ['10th','']
            
    #         [f for f in fset if]
    #         split_stems = [f.split('_').split('_bluelight').split('_prestim') for f in fset]
    #         fstem_list = ['_'.join(fstem) for fstem in ]
    #         fstem_list = list(np.unique(fstem_list))
    #         for fstem in fstem_list:
    #             _fset = [f for f in fset if f.startswith(fstem)]
    #             print(_fset)
    #             features[(fstem + stim)] = features[_fset].mean(axis=1)
    #             mean_feats_list.append((fstem + stim))
    #     features = features[list(np.unique(mean_feats_list))]

    #%% Analyse variables

    for GROUPING_VAR in VARIABLES_TO_INVESTIGATE:
        print("\nInvestigating '%s' variation" % GROUPING_VAR)
        
        control = control_dict[GROUPING_VAR]
        
        for run in imaging_run_list:
            print("\nAnalysing imaging run %d" % run)
            
            # Subset results to investigate single imaging run
            meta_df = metadata[metadata['imaging_run_number']==run]
            feat_df = features.reindex(meta_df.index)
            
            # Clean subsetted data: drop NaNs, zero std, etc
            feat_df, meta_df = clean_features_summaries(feat_df, 
                                                        meta_df, 
                                                        max_value_cap=False,
                                                        imputeNaN=False)
            # Save paths
            ftname = 'Top256' if USE_TOP256 else 'All_features'
            ftname = ftname + '_noSize' if FILTER_SIZE_FEATS else ftname
            ftname = ftname + '_norm' if NORM_FEATS_ONLY else ftname
            ftname = ftname + '_noOutliers' if REMOVE_OUTLIERS else ftname
            stats_dir = save_dir / "Stats" / "Run_{}".format(run) /\
                        (GROUPING_VAR + '_variation') / ftname
            plot_dir = save_dir / "Plots" / "Run_{}".format(run) /\
                       (GROUPING_VAR + '_variation') / ftname

            #%% Check normality: Look to see if response data are 
            #   homoscedastic / normally distributed
            if CHECK_NORMAL:
                normtest_savepath = stats_dir / "shapiro_normality_test_results.csv"
                normtest_savepath.parent.mkdir(exist_ok=True, parents=True) # make folder if it does not exist
                (prop_features_normal, 
                 is_normal) = shapiro_normality_test(features_df=feat_df,
                                                     metadata_df=meta_df,
                                                     group_by=GROUPING_VAR,
                                                     p_value_threshold=P_VALUE_THRESHOLD,
                                                     verbose=True)             
                # Save normailty test results to file
                prop_features_normal.to_csv(normtest_savepath, 
                                            index=True, 
                                            index_label=GROUPING_VAR, 
                                            header='prop_normal')
            else:
                is_normal = False # Default non-parametric
            
            #%%     One-way ANOVA/Kruskal-Wallis tests for significantly different 
            #       features across groups (e.g. strains)
        
            # When comparing more than 2 groups, perform ANOVA and proceed only to 
            # pairwise two-sample t-tests if there is significant variability among 
            # all groups for any feature
            var_list = list(meta_df[GROUPING_VAR].unique())
            
            if len(var_list) > 2:
                pvalues_anova, anova_sigfeats_list = anova_by_feature(feat_df=feat_df, 
                                                                      meta_df=meta_df, 
                                                                      group_by=GROUPING_VAR, 
                                                                      strain_list=var_list, 
                                                                      p_value_threshold=P_VALUE_THRESHOLD, 
                                                                      is_normal=is_normal, 
                                                                      fdr_method=FDR_METHOD)
                            
                # Record name of statistical test used (kruskal/f_oneway)
                TEST = f_oneway if is_normal else kruskal
                test_name = str(TEST).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]
                    
                # Save test statistics to file
                stats_outpath = stats_dir / '{}_results.csv'.format(test_name)
                sigfeats_outpath = Path(str(stats_outpath).replace('_results.csv',
                                                                   '_significant_features.csv'))
                pvalues_anova.to_csv(stats_outpath) # Save test results as CSV
                anova_sigfeats_list.to_csv(sigfeats_outpath, index=False) # Save feature list as text file
                
                # Record number of signficant features by ANOVA
                n_sigfeats_anova = len(anova_sigfeats_list)    
            else:
                n_sigfeats_anova = 0
                    
        
            #%%     t-tests/rank-sum tests for significantly different features between 
            #       each strains vs control
         
            if n_sigfeats_anova > 0 or len(var_list) == 2:
         
                # Record name of statistical test
                TEST = ttest_ind if is_normal else ranksumtest
                test_name = str(TEST).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]
                
                (pvalues_ttest, 
                 sigfeats_table, 
                 sigfeats_list) = ttest_by_feature(feat_df, 
                                                   meta_df, 
                                                   group_by=GROUPING_VAR, 
                                                   control_strain=control, 
                                                   is_normal=is_normal, 
                                                   p_value_threshold=P_VALUE_THRESHOLD,
                                                   fdr_method=FDR_METHOD)
                
                # Print number of significant features
                print("%d significant features for (run %d, %s, %s, p<%.2f)"\
                      % (len(sigfeats_list), run, GROUPING_VAR, 
                         test_name, P_VALUE_THRESHOLD))
                # Save test statistics to file
                stats_outpath = stats_dir / '{}_results.csv'.format(test_name)
                stats_outpath.parent.mkdir(exist_ok=True, parents=True)
                sigfeats_outpath = Path(str(stats_outpath).replace('_results.csv',
                                                                   '_significant_features.csv'))
                sigfeats_list.to_csv(sigfeats_outpath, index=False) # Save feature list to file
                pvalues_ttest.to_csv(stats_outpath) # Save test results to CSV
                                
                # Barplot of number of significantly different features for each strain   
                prop_sigfeats = barplot_sigfeats(test_pvalues_df=pvalues_ttest, 
                                                 saveDir=plot_dir,
                                                 p_value_threshold=P_VALUE_THRESHOLD)

                #%% Boxplots of most significantly different features for each strain vs control
                #   features ranked by t-test pvalue significance (lowest first)
                    
                # Load test results (pvalues) for plotting
                # NB: Non-parametric ranksum test preferred over t-test as many features may not be normally distributed
                test_name = 'ttest_ind' if is_normal else 'ranksumtest'    
                stats_inpath = stats_dir / '{}_results.csv'.format(test_name)
                pvalues_ttest = pd.read_csv(stats_inpath, index_col=0)
                print("Loaded '%s' results." % test_name)
                
                boxplots_sigfeats(feat_meta_df=meta_df.join(feat_df), 
                                  test_pvalues_df=pvalues_ttest, 
                                  group_by=GROUPING_VAR, 
                                  control_strain=control, 
                                  saveDir=plot_dir,
                                  n_sig_feats_to_plot=K_SIG_FEATS,
                                  p_value_threshold=P_VALUE_THRESHOLD)
                        
                # from tierpsytools.analysis.significant_features import plot_feature_boxplots
                # plot_feature_boxplots(feat_to_plot=fset,
                #                       y_class=GROUPING_VAR,
                #                       scores=pvalues_ttest.rank(axis=1),
                #                       feat_df=feat_df,
                #                       pvalues=np.asarray(pvalues_ttest).flatten(),
                #                       saveto=None,
                #                       close_after_plotting=False)

            #%% Linear Mixed Models (LMMs) to account for day-to-day variation when comparing 
            #   between groups
            RANDOM_EFFECT = 'date_yyyymmdd'
            pvalues_lmm = linear_mixed_model(feat_df=feat_df,
                                             meta_df=meta_df,
                                             group_by=GROUPING_VAR,
                                             control=control,
                                             random_effect=RANDOM_EFFECT,
                                             fdr=P_VALUE_THRESHOLD,
                                             fdr_method=FDR_METHOD,
                                             comparison_type='infer',
                                             n_jobs=-1)
            
            # Save LMM significant features
            lmm_path = stats_dir / 'LMM_{}.csv'.format(RANDOM_EFFECT)
            lmm_path.parent.mkdir(exist_ok=True, parents=True)
            pvalues_lmm.to_csv(lmm_path, header=True, index=True)
            
            # FIXME: Determine correct way of selecting reduced set of significant features
            fset = pvalues_lmm[pvalues_lmm.columns[np.where(pvalues_lmm < P_VALUE_THRESHOLD)[1]]]
            fset = fset.iloc[0].sort_values(ascending=True).index.to_list()
            
            swarmDir = plot_dir / 'swarmplots_by_{}'.format(RANDOM_EFFECT)
            swarmplot_random_effect_variation(feat_df=feat_df,
                                              meta_df=meta_df,
                                              group_by=GROUPING_VAR,
                                              test_pvalues_df=pvalues_lmm,
                                              control=control,
                                              features2plot=fset[:K_SIG_FEATS],
                                              p_value_threshold=0.05,
                                              saveDir=swarmDir,
                                              sns_colour_palette="tab10",
                                              dodge=None)
                    
            #%% K significant features
    
            k_sigfeat_dir = plot_dir / 'k_sig_feats'
            k_sigfeat_dir.mkdir(exist_ok=True, parents=True)
                
            # Infer feature set
            fset, (scores, pvalues), support = k_significant_feat(feat=feat_df, 
                                                                  y_class=meta_df[GROUPING_VAR], 
                                                                  k=K_SIG_FEATS, 
                                                                  score_func='f_classif', 
                                                                  scale=None, 
                                                                  feat_names=None, 
                                                                  plot=False, 
                                                                  k_to_plot=None, 
                                                                  close_after_plotting=True,
                                                                  saveto=None, 
                                                                  figsize=None, 
                                                                  title=None, 
                                                                  xlabel=None)
            pvalues_ksig = pd.Series(data=pvalues, index=fset, name='k_significant_features')
 
            # Save k most significant features
            fset_out = pd.DataFrame(pvalues_ksig).reset_index(drop=False)
            fset_out.columns = ['feature','p_value']
            fset_out.to_csv(stats_dir / 'k_significant_features.csv', header=True, index=False)   
   
            # OPTIONAL: Plot cherry-picked features
            #fset = ['speed_50th','curvature_neck_abs_50th','major_axis_50th','angular_velocity_neck_abs_50th']
            #fset = pvalues_ttest.columns[np.where((pvalues_ttest < P_VALUE_THRESHOLD).any(axis=0))]            
            
            boxplots_by_strain(df=meta_df.join(feat_df), 
                               group_by=GROUPING_VAR,
                               test_pvalues_df=pvalues_ttest,
                               control_group=control,
                               features2plot=fset,
                               saveDir=k_sigfeat_dir,
                               max_features_plot_cap=K_SIG_FEATS, 
                               p_value_threshold=0.05,
                               figsize=[8,12])    
            
            #%% Hierarchical Clustering Analysis
            #   - Clustermap of features by strain, to see if data cluster into groups
            #   - Control data is clustered first, feature order is stored and ordering applied to 
            #     full data for comparison
            
            # Extract data for control
            control_feat_df = feat_df[meta_df[GROUPING_VAR]==control]
            control_meta_df = meta_df.reindex(control_feat_df.index)
            
            control_feat_df, control_meta_df = clean_features_summaries(features=control_feat_df,
                                                                        metadata=control_meta_df)
            
            # Ensure no NaNs or features with zero standard deviation before normalisation
            assert not control_feat_df.isna().sum(axis=0).any()
            assert not (control_feat_df.std(axis=0) == 0).any()
            
            #zscores = (df-df.mean())/df.std() # minus mean, divide by std
            controlZ_feat_df = control_feat_df.apply(zscore, axis=0)
            
            # Drop features with NaN values after normalising
            n_cols = len(controlZ_feat_df.columns)
            controlZ_feat_df.dropna(axis=1, inplace=True)
            n_dropped = n_cols - len(controlZ_feat_df.columns)
            if n_dropped > 0:
                print("Dropped %d features after normalisation (NaN)" % n_dropped)

            # plot clustermap for control
            # NB: methods=[linkage, complete, average, weighted, centroid]
            control_heatmap_path = plot_dir / 'HCA' / '{}_cluster_heatmap.eps'.format(control)               
            cg = plot_clustermap(featZ=controlZ_feat_df, 
                                 meta=control_meta_df, 
                                 group_by=[GROUPING_VAR,'date_yyyymmdd'],
                                 figsize=[18,6],
                                 saveto=control_heatmap_path)

            # Extract linkage + clustered features
            col_linkage = cg.dendrogram_col.calculated_linkage
            clustered_features = np.array(controlZ_feat_df.columns)[cg.dendrogram_col.reordered_ind]
            
            # Clustermap of full data using control clustered features order
            assert not feat_df.isna().sum(axis=0).any()
            assert not (feat_df.std(axis=0) == 0).any()
            
            featZ_df = feat_df.apply(zscore, axis=0)
            
            # Drop features with NaN values after normalising
            n_cols = len(featZ_df.columns)
            featZ_df.dropna(axis=1, inplace=True)
            n_dropped = n_cols - len(featZ_df.columns)
            if n_dropped > 0:
                print("Dropped %d features after normalisation (NaN)" % n_dropped)
                
            full_heatmap_path = plot_dir / 'HCA' / '{}_cluster_heatmap.eps'.format(GROUPING_VAR)                             
            clustered_features = plot_clustermap(featZ=featZ_df, 
                                                 meta=meta_df, 
                                                 group_by=GROUPING_VAR,
                                                 col_linkage=col_linkage,
                                                 figsize=[20,5],
                                                 saveto=full_heatmap_path)

            # Use lmm/anova pvalues for heatmap comparison if comparing 2/>2 groups, respectively
            if len(var_list) > 2:
                pvalues_heatmap = pvalues_anova.loc['pval',:]
            elif len(var_list) == 2:
                pvalues_heatmap = pvalues_lmm.iloc[0]
            pvalues_heatmap.name = 'P < {}'.format(P_VALUE_THRESHOLD)
            
            assert set(pvalues_heatmap.index) == set(featZ_df.columns)

            ranked_pvals = pvalues_heatmap.sort_values(ascending=True)
            selected_features = ranked_pvals[ranked_pvals < P_VALUE_THRESHOLD].index
            
            # Plot barcode clustermap
            barcode_heatmap_path = Path(str(full_heatmap_path).replace('.eps', '_barcode.eps'))
            plot_barcode_clustermap(featZ=featZ_df, 
                                    meta=meta_df, 
                                    group_by=[GROUPING_VAR,'date_yyyymmdd'], 
                                    pvalues_series=pvalues_heatmap,
                                    p_value_threshold=P_VALUE_THRESHOLD,
                                    selected_feats=selected_features,
                                    saveto=barcode_heatmap_path,
                                    figsize=[18,6],
                                    sns_colour_palette="tab10")
            plt.show()

            #%% Principal Components Analysis (PCA)
    
            if REMOVE_OUTLIERS:
                outlier_path = plot_dir / 'mahalanobis_outliers.eps'
                feat_df, inds = remove_outliers_pca(df=feat_df, 
                                                    features_to_analyse=None, 
                                                    saveto=outlier_path)
                meta_df = meta_df.reindex(feat_df.index)
                featZ_df = feat_df.apply(zscore, axis=0)

    
            # plot PCA
            #from tierpsytools.analysis.decomposition import plot_pca
            pca_dir = plot_dir / 'PCA'
            projected_df = plot_pca(featZ=featZ_df, 
                                    meta=meta_df, 
                                    group_by=GROUPING_VAR, 
                                    n_dims=2,
                                    var_subset=None, 
                                    saveDir=pca_dir,
                                    PCs_to_keep=10,
                                    n_feats2print=10)            
             
            #%%     t-distributed Stochastic Neighbour Embedding (tSNE)
    
            perplexities = [5,10,20,50] # tSNE: Perplexity parameter for tSNE mapping
    
            tsne_dir = plot_dir / 'tSNE'
            tSNE_df = plot_tSNE(featZ=featZ_df,
                                meta=meta_df,
                                group_by=GROUPING_VAR,
                                var_subset=None,
                                saveDir=tsne_dir,
                                perplexities=perplexities)
            
            #%%     Uniform Manifold Projection (UMAP)
    
            n_neighbours = [5,10,20,50] # UMAP: N-neighbours parameter for UMAP projections                                            
            min_dist = 0.3 # Minimum distance parameter for UMAP projections    
            
            umap_dir = plot_dir / 'UMAP'
            umap_df = plot_umap(featZ=featZ_df,
                                meta=meta_df,
                                group_by=GROUPING_VAR,
                                var_subset=None,
                                saveDir=umap_dir,
                                n_neighbours=n_neighbours,
                                min_dist=min_dist)
   
    #%% Investigate selected features
    #   - Read in selected features list
    
    # TODO: Timeseries analysis of feature across timepoints/stimulus windows/on-off food/etc
    
    # TODO: sns.relplot and sns.jointplot and sns.lineplot for visualising covariance/corrrelation between selected features


