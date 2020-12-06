#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse Filipe's Tests - N2 vs R60K (ribosomal mutant) behaviour on E. coli OP50

@author: sm5911
@date: 20/11/2020
"""

#%% Imports

import sys
import argparse
#import numpy as np
import pandas as pd
#from tqdm import tqdm
from pathlib import Path
from scipy.stats import (kruskal,
                         ttest_ind, 
                         f_oneway, 
                         zscore)
#from matplotlib import pyplot as plt

# Import custom helper functions
from helper import (process_metadata, 
                    process_feature_summaries,
                    clean_features_summaries,
                    load_top256,
                    shapiro_normality_test,
                    ranksumtest,
                    ttest_by_feature,
                    anova_by_feature,
                    barplot_sigfeats_ttest,
                    boxplots_top_feats,
                    boxplots_by_strain,
                    plot_clustermap,
                    plot_pca,
                    remove_outliers_mahalanobis,
                    plot_tSNE,
                    plot_umap)
   
# Path to Github helper functions (USER-DEFINED path to local copy of Github repo)
PATH_LIST = ['/Users/sm5911/Tierpsy_Versions/tierpsy-tools-python/']
for sysPath in PATH_LIST:
    if sysPath not in sys.path:
        sys.path.insert(0, sysPath)

from tierpsytools.analysis.significant_features import k_significant_feat
from tierpsytools.feature_processing.filter_features import feat_filter_std
        
#%% Globals

SHOW_PLOTS = False

#%% Functions

            
#%% Main

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Analyse Tierpsy results (96-well)')
    parser.add_argument('--project_dir', help="Project root directory,\
                        containing 'AuxiliaryFiles', 'RawVideos',\
                        'MaskedVideos' and 'Results' folders,\
                        eg. /Volumes/hermes$/KeioScreen_96WP",
                        default="/Volumes/hermes$/KeioScreen_96WP", type=str)
    parser.add_argument('--group_by', help="Treatment variable by which you \
                        want to group results, eg. 'worm_strain' or 'food_type'",
                        default='food_type', type=str)
    parser.add_argument('--omit_strains', help="List of strains in 'group_by' \
                        to ignore in the analysis", default=None)
    parser.add_argument('--dates', help="List of imaging dates to use for analysis.\
                        If None, all imaging dates will be investigated",
                        default=None)
    parser.add_argument('--timepoints', help="List of timepoints to subset \
                        results for analysis. If None, all timepoints will be \
                        investigated", default=None) # TODO: Change to imaging run number
    parser.add_argument('--top256', help="Use Tierpsy Top256 features only",
                        default=True, type=bool)
    parser.add_argument('--drop_size_related', help="Remove size-related Tierpsy \
                        features from analysis", default=False, type=bool)
    parser.add_argument('--well_annotations', help="Add 'is_bad_well' labels \
                        from WellAnnotator GUI", default=True, type=bool)
    parser.add_argument('--check_normal', help="Perform Shapiro-Wilks test for \
                        normality to decide between parametric/non-parametric \
                        statistics", default=True, type=bool)
    parser.add_argument('--nan_thresh', help="Threshold proportion of NaN values \
                        to drop feature from analysis", default=0.2, type=float)
    parser.add_argument('--pval_thresh', help="Threshold p-value for statistical \
                        significance", default=0.05, type=float)
    parser.add_argument('--k_sig_feats', help="Number of most significantly \
                        different features to plot", default=10, type=int)
    args = parser.parse_args()
    
    for arg in list(args._get_kwargs()):
        print('%s: %s' % (arg[0], str(arg[1])))
        
    PROJECT_DIR = Path(args.project_dir)                # str
    GROUPING_VAR = args.group_by                        # str
    OMIT_STRAIN_LIST = args.omit_strains                # list
    IMAGING_DATES = args.dates                          # list
    TIMEPOINT = args.timepoints                         # list
    USE_TOP256 = args.top256                            # bool
    FILTER_SIZE_RELATED_FEATS = args.drop_size_related  # bool
    ADD_WELL_ANNOTATIONS = args.well_annotations        # bool
    CHECK_NORMAL = args.check_normal                    # bool
    NAN_THRESHOLD = args.nan_thresh                     # float
    P_VALUE_THRESHOLD = args.pval_thresh                # float
    K_SIG_FEATS = args.k_sig_feats                      # int
    
    # PROJECT_DIR = Path("/Volumes/hermes$/Filipe_Tests_96WP") # Filipe
    # GROUPING_VAR = 'worm_strain' # Filipe
    IMAGING_DATES = ['20201020','20201021'] # Keio
     

    aux_dir = PROJECT_DIR / "AuxiliaryFiles"
    results_dir = PROJECT_DIR / "Results"
    
    # Process metadata    
    metadata = process_metadata(aux_dir=aux_dir, 
                                imaging_dates=IMAGING_DATES, 
                                add_well_annotations=ADD_WELL_ANNOTATIONS)
    
    # Analysis is case-sensitive - ensure that there is no confusion in strain names
    assert len(metadata[GROUPING_VAR].unique()) == len(metadata[GROUPING_VAR].str.upper().unique())
    
    # Process feature summary results
    features, metadata = process_feature_summaries(metadata, 
                                                   results_dir,
                                                   imaging_dates=IMAGING_DATES,
                                                   add_bluelight=True)
    
    # Clean: remove data with too many NaNs/all zeros + impute remaining NaNs
    features, metadata = clean_features_summaries(features, 
                                                  metadata,
                                                  feature_columns=None,
                                                  imputeNaN=True,
                                                  nan_threshold=NAN_THRESHOLD,
                                                  drop_size_related_feats=FILTER_SIZE_RELATED_FEATS)
    
    # Record strain names
    strain_list = [strain for strain in list(metadata[GROUPING_VAR].unique())]
    if OMIT_STRAIN_LIST:
        strain_list = [strain for strain in strain_list if strain not in OMIT_STRAIN_LIST]
    
    # Save full results to file
    full_results_path = results_dir / 'full_results.csv' 
    if not full_results_path.exists():
        fullresults = metadata.join(features) # join metadata + results
        
        print("Saving full results (features/metadata) to:\n '%s'" % full_results_path)     
        fullresults.to_csv(full_results_path, index=False)

#%% Subset
    
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
    
    # Subset results for strains of interest
    metadata = metadata[metadata[GROUPING_VAR].isin(strain_list)]
    features = features.loc[metadata.index]
    
    # Subset results for imaging dates of interest
    if IMAGING_DATES:
        metadata = metadata[metadata['date_yyyymmdd'].isin(IMAGING_DATES)]
        features = features.loc[metadata.index]
    
    if TIMEPOINT:
        timepoints_list = TIMEPOINT if type(TIMEPOINT) == list else [TIMEPOINT]
    else:
        print("No timepoint provided, analysing each timepoint separately")
        timepoints_list = list(metadata['imaging_run_number'].unique().astype(int))

#%%
    for timepoint in timepoints_list:
        print("\nAnalysing timepoint = %d" % timepoint)
        
        # Subset results to investigate single timepoint
        meta_df = metadata[metadata['imaging_run_number']==timepoint]
        feat_df = features.loc[meta_df.index]
        
        ftname = 'Top256' if USE_TOP256 else 'All_features'
        ftname = ftname + '_noSize' if FILTER_SIZE_RELATED_FEATS else ftname
        stats_dir = results_dir / "Stats" / "Timepoint_{}".format(timepoint) / ftname
        plot_dir = results_dir / "Plots" / "Timepoint_{}".format(timepoint) / ftname

#%%     Check normality  
        
        # Look to see if response data are homoscedastic / normally distributed
        if CHECK_NORMAL:
            normtest_savepath = stats_dir / "shapiro_normality_test_results.csv"
            normtest_savepath.parent.mkdir(exist_ok=True, parents=True) # make folder if it does not exist
            (prop_features_normal, 
             is_normal) = shapiro_normality_test(features_df=feat_df,
                                                 metadata_df=meta_df,
                                                 group_by=GROUPING_VAR,
                                                 p_value_threshold=P_VALUE_THRESHOLD)                
            # Save normailty test results to file
            prop_features_normal.to_csv(normtest_savepath, 
                                        index=True, 
                                        index_label='food_type', 
                                        header='prop_normal')
        else:
            is_normal = False # Default non-parametric
        
        # Identify control and test strains for statistics
        control_strain_list = ['OP50','N2','DMSO','BW25113'] # analysis control strains, if worm/bacteria/drug
        test_strains = [strain for strain in strain_list if strain not in control_strain_list]
        control_strain = [strain for strain in strain_list if strain in control_strain_list]
        assert len(control_strain) == 1
        control_strain = control_strain[0]

#%%     t-tests/rank-sum tests for significantly different features between 
#       each strains vs control
 
        
        # Record name of statistical test
        TEST = ttest_ind if is_normal else ranksumtest
        test_name = str(TEST).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]
        
        (pvalues_ttest, 
         sigfeats_table, 
         sigfeats_list) = ttest_by_feature(feat_df, 
                                           meta_df, 
                                           group_by=GROUPING_VAR, 
                                           control_strain=control_strain, 
                                           is_normal=is_normal, 
                                           p_value_threshold=P_VALUE_THRESHOLD,
                                           fdr_method='fdr_by')
        # Save test statistics to file
        stats_outpath = stats_dir / '{}_results.csv'.format(test_name)
        sigfeats_outpath = Path(str(stats_outpath).replace('_results.csv',
                                                           '_significant_features.csv'))
        stats_outpath.parent.mkdir(exist_ok=True, parents=True) # Create save directory if it does not exist
        pvalues_ttest.to_csv(stats_outpath) # Save test results to CSV
        sigfeats_list.to_csv(sigfeats_outpath, index=False) # Save feature list to text file
        
        # Barplot of number of significantly different features for each strain   
        prop_sigfeats = barplot_sigfeats_ttest(test_pvalues_df=pvalues_ttest, 
                                               saveDir=plot_dir,
                                               p_value_threshold=P_VALUE_THRESHOLD)

#%%     One-way ANOVA/Kruskal-Wallis tests for significantly different 
#       features across strains
        
        pvalues_anova, sigfeats_list = anova_by_feature(feat_df, 
                                                        meta_df, 
                                                        group_by=GROUPING_VAR, 
                                                        strain_list=strain_list, 
                                                        p_value_threshold=P_VALUE_THRESHOLD, 
                                                        is_normal=is_normal, 
                                                        fdr_method='fdr_by')
        
        # Record name of statistical test used (kruskal/f_oneway)
        TEST = f_oneway if is_normal else kruskal
        test_name = str(TEST).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]
        
        # Save test statistics to file
        stats_outpath = stats_dir / '{}_results.csv'.format(test_name)
        sigfeats_outpath = Path(str(stats_outpath).replace('_results.csv',
                                                           '_significant_features.csv'))
        pvalues_anova.to_csv(stats_outpath) # Save test results as CSV
        sigfeats_list.to_csv(sigfeats_outpath, index=False) # Save feature list as text file
        
        # TODO: Perform post-hoc analyses (eg.Tukey HSD) for pairwise comparisons 
        # between strains for each feature?

#%%     Boxplots of most significantly different features for each strain vs control
#       features ranked by t-test pvalue significance (lowest first)
        
        # Load test results (pvalues) for plotting
        # NB: Non-parametric ranksum test preferred over t-test as many features may not be normally distributed
        test_name = 'ttest_ind' if is_normal else 'ranksumtest'    
        stats_inpath = stats_dir / '{}_results.csv'.format(test_name)
        pvalues_ttest = pd.read_csv(stats_inpath, index_col=0)
        print("Loaded '%s' results." % test_name)
                
        # from tierpsytools.analysis.significant_features import plot_feature_boxplots
        # plot_feature_boxplots(feat_to_plot=fset,
        #                       y_class=GROUPING_VAR,
        #                       scores=pvalues_ttest.rank(axis=1),
        #                       feat_df=feat_df,
        #                       pvalues=np.asarray(pvalues_ttest).flatten(),
        #                       saveto=None,
        #                       close_after_plotting=False)
        
        boxplots_top_feats(feat_meta_df=meta_df.join(feat_df), 
                           test_pvalues_df=pvalues_ttest, 
                           group_by=GROUPING_VAR, 
                           control_strain=control_strain, 
                           saveDir=plot_dir, 
                           p_value_threshold=0.05, 
                           n_top_features=50)
        
#%%     K significant features

        k_sigfeat_dir = plot_dir / 'k_sig_feats'
        k_sigfeat_dir.mkdir(exist_ok=True, parents=True)

        # Infer feature set
        fset, (scores, pvalues), support = k_significant_feat(feat=feat_df, 
                                                              y_class=meta_df[GROUPING_VAR], 
                                                              k=K_SIG_FEATS, 
                                                              score_func='f_classif', 
                                                              scale=None, 
                                                              feat_names=None, 
                                                              plot=True, 
                                                              k_to_plot=None, 
                                                              close_after_plotting=True,
                                                              saveto=None, 
                                                              figsize=None, 
                                                              title=None, 
                                                              xlabel=None)        
        # OPTIONAL: Plot cherry-picked features
        #fset = ['speed_50th','curvature_neck_abs_50th','major_axis_50th','angular_velocity_neck_abs_50th']
        #fset = pvalues_ttest.columns[np.where((pvalues_ttest < P_VALUE_THRESHOLD).any(axis=0))]
        boxplots_by_strain(df=meta_df.join(feat_df), 
                           group_by=GROUPING_VAR,
                           test_pvalues_df=pvalues_ttest,
                           control_strain=control_strain,
                           features2plot=fset,
                           saveDir=k_sigfeat_dir,
                           max_features_plot_cap=K_SIG_FEATS, 
                           max_groups_plot_cap=48,
                           p_value_threshold=0.05)   
        
        # Save k most significant features
        fset_out = pd.Series(fset)
        fset_out.name = 'k_significant_features'
        fset_out = pd.DataFrame(fset_out)
        fset_out.to_csv(stats_dir / 'k_significant_features.csv', header=0, 
                        index=None)   
        
#%%     Hierarchical Clustering (Heatmap)
#       - clustermap of features by strain, to see if data cluster into groups
        
        # Ensure no NaN values in features
        assert not feat_df.isna().sum(axis=0).any()
        
        # Drop features with zero standard deviation before normalisation
        feat_df = feat_filter_std(feat_df, threshold=0.0) 
        
        # Normalise features results 
        #zscores = (feat_df-feat_df.mean())/feat_df.std() # minus mean, divide by std
        featZ_df = feat_df.apply(zscore, axis=0)
        
        # Drop features with NaN values after normalising
        n_cols = len(featZ_df.columns)
        featZ_df.dropna(axis=1, inplace=True)
        print("Dropped %d features after normalisation (NaN)" %\
              (n_cols-len(featZ_df.columns)))
        
        # TODO: import fastcluster
        # eg. linkage, complete, average, weighted, centroid
        # Plot clustermap
        heatmap_path = plot_dir / 'hierarchical_clustermap.eps'
        clustered_features = plot_clustermap(featZ=featZ_df, 
                                             meta=meta_df, 
                                             group_by=GROUPING_VAR, 
                                             saveto=heatmap_path)

#%%     Principal Components Analysis (PCA)

        pca_dir = plot_dir / 'PCA'
        projected_df = plot_pca(featZ=featZ_df, 
                                meta=meta_df, 
                                group_by=GROUPING_VAR, 
                                n_dims=2,
                                var_subset=strain_list, 
                                saveDir=pca_dir,
                                PCs_to_keep=10,
                                n_feats2print=10)
        
        # TODO: Remove outliers from PCA
        #remove_outliers_pca(projected_df, feat_df)
 
# #%%     t-distributed Stochastic Neighbour Embedding (tSNE)

#         perplexities = [5,10,20,30] # tSNE: Perplexity parameter for tSNE mapping

#         tsne_dir = plot_dir / 'tSNE'
#         plot_tSNE(featZ=featZ_df,
#                   meta=meta_df,
#                   group_by=GROUPING_VAR,
#                   var_subset=strain_list,
#                   saveDir=tsne_dir,
#                   perplexities=perplexities)
        
# #%%     Uniform Manifold Projection (UMAP)

#         n_neighbours = [5,10,20,30] # UMAP: N-neighbours parameter for UMAP projections                                            
#         min_dist = 0.3 # Minimum distance parameter for UMAP projections    
        
#         umap_dir = plot_dir / 'UMAP'
#         plot_umap(featZ=featZ_df,
#                   meta=meta_df,
#                   group_by=GROUPING_VAR,
#                   var_subset=strain_list,
#                   saveDir=tsne_dir,
#                   n_neighbours=n_neighbours,
#                   min_dist=min_dist)
     
