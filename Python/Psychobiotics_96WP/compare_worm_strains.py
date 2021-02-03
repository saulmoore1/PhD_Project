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
import warnings
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
                    plot_day_variation,
                    barplot_sigfeats,
                    boxplots_sigfeats,
                    boxplots_grouped,
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
from tierpsytools.drug_screenings.filter_compounds import compounds_with_low_effect_univariate #lmm

#%% Globals

CONTROL_DICT = {'worm_strain': 'N2',
                'drug_type': 'DMSO',
                'food_type': 'BW25113', #'OP50',
                'instrument_name': 'Hydra01',
                'worm_life_stage': 'D1',
                'lawn_growth_duration_hours': '8',
                'lawn_storage_type': 'old'}

FDR_METHOD = 'fdr_by' # Benjamini-Yekutieli correction for multiple testing
RANDOM_EFFECT = 'date_yyyymmdd'

#%% Main

if __name__ == "__main__":
    
    # Accept command-line inputs
    parser = argparse.ArgumentParser(description='Analyse Tierpsy results (96-well)')
    parser.add_argument('--project_dir', help="Project root directory,\
                        containing 'AuxiliaryFiles', 'RawVideos',\
                        'MaskedVideos' and 'Results' folders",
                        default='/Volumes/hermes$/Filipe_Tests_96WP', type=str)
    parser.add_argument('--analyse_variables', help="List of categorical \
                        variables that you wish to investigate", nargs='+',
                        default=['worm_strain','instrument_name']) #'food_type'
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
    parser.add_argument('--feature_means_only', help="Use only 50th percentile feature summaries \
                        for each feature", default=True, action='store_false')
    parser.add_argument('--drop_size_features', help="Remove size-related Tierpsy \
                        features from analysis", default=True, action='store_false')
    parser.add_argument('--norm_features_only', help="Use only normalised \
                        size-invariant features ('_norm') for analysis",
                        default=True, action='store_false')
    parser.add_argument('--check_normal', help="Perform Shapiro-Wilks test for \
                        normality to decide between parametric/non-parametric \
                        statistics", default=True, action='store_false')
    parser.add_argument('--remove_outliers', help="Use Mahalanobis distance to \
                        exclude outliers from analysis", default=False, 
                        action='store_true')  
    parser.add_argument('--nan_threshold', help="Threshold proportion of NaN values \
                        to drop feature from analysis", default=0.2, type=float)
    parser.add_argument('--pval_threshold', help="Threshold p-value for statistical \
                        significance", default=0.05, type=float)
    parser.add_argument('--k_sig_features', help="Number of most significantly \
                        different features to plot", default=100, type=int)  
    parser.add_argument('--selected_features_path', help="Path to manually selected intuitive \
                        features for publication", default='/Users/sm5911/Documents/tmp_analysis/Filipe/manually_selected_features.csv', type=str)
    args = parser.parse_args()
    
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
        
    # IO paths
    aux_dir = PROJECT_DIR / "AuxiliaryFiles"
    results_dir = PROJECT_DIR / "Results"
    save_dir = Path('/Users/sm5911/Documents/tmp_analysis/Filipe') # PROJECT_DIR / 'Analysis'

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
                                                  norm_feats_only=NORM_FEATS_ONLY,
                                                  mean_feats_only=FEAT_MEANS_ONLY)
    # Save full results to file
    full_results_path = save_dir / 'full_results.csv' 
    if not full_results_path.exists():
        fullresults = metadata.join(features) # join metadata + results
        print("Saving full results (features/metadata) to:\n '%s'" % full_results_path)     
        fullresults.to_csv(full_results_path, index=False)

    #%% Subset results
    
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
    assert len(metadata['worm_strain'].unique())==len(metadata['worm_strain'].str.upper().unique())
    if OMIT_STRAIN_LIST:
        OMIT_STRAIN_LIST = [o.upper() for o in OMIT_STRAIN_LIST]
        strain_list = [s for s in metadata['worm_strain'].unique()
                       if s.upper() not in OMIT_STRAIN_LIST]
    else:
        strain_list = list(metadata['worm_strain'].unique())
        
    metadata = metadata[metadata['worm_strain'].isin(strain_list)]
    features = features.reindex(metadata.index)

    #%% Analyse variables

    for GROUPING_VAR in VARIABLES_TO_INVESTIGATE:
        print("\nInvestigating '%s' variation" % GROUPING_VAR)
        
        control = CONTROL_DICT[GROUPING_VAR]
        
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
            fn = 'Top256' if USE_TOP256 else 'All_features'
            fn = fn + '_noSize' if FILTER_SIZE_FEATS else fn
            fn = fn + '_norm' if NORM_FEATS_ONLY else fn
            fn = fn + '_50th' if FEAT_MEANS_ONLY else fn
            fn = fn + '_noOutliers' if REMOVE_OUTLIERS else fn
            stats_dir = save_dir / "Stats" / "Run_{}".format(run) /\
                        (GROUPING_VAR + '_variation') / fn
            plot_dir = save_dir / "Plots" / "Run_{}".format(run) /\
                       (GROUPING_VAR + '_variation') / fn

            #%% Check normality: Look to see if response data are 
            #   homoscedastic / normally distributed
            #   NB: Non-parametric ranksum test preferred over t-test if many features are not
            #       normally distributed -- see percentile stats discussion
            if CHECK_NORMAL:
                normtest_savepath = stats_dir / "shapiro_normality_test_results.csv"
                normtest_savepath.parent.mkdir(exist_ok=True, parents=True)
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
            
            #%% STATISTICS
            #       One-way ANOVA/Kruskal-Wallis tests for significantly different 
            #       features across groups (e.g. strains)
        
            # When comparing more than 2 groups, perform ANOVA and proceed only to 
            # pairwise two-sample t-tests if there is significant variability among 
            # all groups for any feature
            var_list = list(meta_df[GROUPING_VAR].unique())
            
            if len(var_list) > 2:
                (pvalues_anova, 
                anova_sigfeats_list) = anova_by_feature(feat_df=feat_df, 
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
                pvalues_anova.to_csv(stats_outpath)
                anova_sigfeats_list.to_csv(sigfeats_outpath, index=False)
                
                # Record number of signficant features by ANOVA
                fset_anova = list(pvalues_anova.columns[pvalues_anova.loc['pval'] < 
                                                        P_VALUE_THRESHOLD])
            else:
                fset_anova = []     
        
            #   t-tests/rank-sum tests for significantly different features between 
            #   each group vs control
         
            if len(fset_anova) > 0 or len(var_list) == 2:
         
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
                print("%d significant features found for %s (run %d, %s, P<%.2f, %s)"\
                      % (len(sigfeats_list), GROUPING_VAR.replace('_',' '), run, 
                         test_name, P_VALUE_THRESHOLD, FDR_METHOD))
                # Save test statistics to file
                stats_outpath = stats_dir / '{}_results.csv'.format(test_name)
                stats_outpath.parent.mkdir(exist_ok=True, parents=True)
                sigfeats_outpath = Path(str(stats_outpath).replace('_results.csv',
                                                                   '_significant_features.csv'))
                sigfeats_list.to_csv(sigfeats_outpath, index=False) # Save feature list to file
                pvalues_ttest.to_csv(stats_outpath) # Save test results to CSV
                
                fset_ttest = list(pvalues_ttest.columns[(pvalues_ttest < 
                                                         P_VALUE_THRESHOLD).sum(axis=0) > 0])
            else:
                fset_ttest = []
                
            #   Linear Mixed Models (LMMs) to account for day-to-day variation when comparing 
            #   between worm/food/drug type
            
            if GROUPING_VAR in ['worm_strain','food_type','drug_type']:
                test_name = 'LMM'
                with warnings.catch_warnings():
                    # Filter warnings as parameter is often on the boundary
                    warnings.filterwarnings("ignore")
                    #warnings.simplefilter("ignore", ConvergenceWarning)
                    (signif_effect, 
                     low_effect, 
                     error, 
                     mask, 
                     pvalues_lmm)=compounds_with_low_effect_univariate(feat=feat_df, 
                                                              drug_name=meta_df[GROUPING_VAR], 
                                                              drug_dose=None, 
                                                              random_effect=meta_df[RANDOM_EFFECT], 
                                                              control=control, 
                                                              test=test_name, 
                                                              comparison_type='multiclass',
                                                              multitest_method=FDR_METHOD,
                                                              ignore_names=None, 
                                                              return_pvals=True)
                assert len(error) == 0
                
                # Save LMM significant features
                lmm_path = stats_dir / '{}_results.csv'.format(test_name)
                lmm_path.parent.mkdir(exist_ok=True, parents=True)
                pvalues_lmm.to_csv(lmm_path, header=True, index=True)
                
                # Ideally report as: parameter | beta | lower-95 | upper-95 | random effect (SD)
                fset_lmm = list(pvalues_lmm.columns[(pvalues_lmm < P_VALUE_THRESHOLD).any() > 0])
                
                if len(signif_effect) > 0:
                    print(("%d significant features found (%d significant %ss vs %s, "\
                          % (len(fset_lmm), len(signif_effect), GROUPING_VAR.replace('_',' '), 
                             control) if len(signif_effect) > 0 else\
                          "No significant differences found between %s "\
                          % GROUPING_VAR.replace('_',' '))
                          + "after accounting for %s variation (run %d, %s, P<%.2f, %s)"\
                          % (RANDOM_EFFECT.split('_yyyymmdd')[0], run, test_name, P_VALUE_THRESHOLD,
                             FDR_METHOD))
            else:
                fset_lmm = []

            # Prefer LMM over ttest if applicable, and compare with k sig feats
            # If there is little overlap, investigate why...
            if ((len(var_list) > 2 and len(fset_anova) == 0) or len(fset_ttest)==0):
                print("NO SIGNIFICANT FEATURES (%s, run %d)" % (GROUPING_VAR.replace('_',' '), run))
                fset = []
            elif GROUPING_VAR in ['worm_strain','food_type','drug_type']:
                if len(fset_lmm) == 0:
                    # No significant features after accounting for day variation (LMM)
                    print("NO SIGNIFICANT FEATURES (%s, run %d)" % (GROUPING_VAR.replace('_',
                                                                                         ' '), run))
                    fset = []
                else:
                    fset = fset_lmm
            elif len(fset_anova) > 0 or len(var_list) == 2:
                fset = fset_ttest
            else:
                assert len(fset_anova) > 0
                fset = fset_anova
                
            #%% K significant features
            #   Compare feature set overlap with k significant features
    
            k_sigfeat_dir = plot_dir / 'k_sig_feats'
            k_sigfeat_dir.mkdir(exist_ok=True, parents=True)
                
            # Infer feature set
            K_SIG_FEATS = len(fset) if (fset is not None and len(fset)>K_SIG_FEATS) else K_SIG_FEATS
            fset_ksig, (scores, pvalues_ksig), support = k_significant_feat(feat=feat_df, 
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
            pvalues_ksig = pd.DataFrame(pd.Series(data=pvalues_ksig, 
                                                  index=fset_ksig, 
                                                  name='k_significant_features')).T
            # Save k most significant features
            pvalues_ksig.to_csv(stats_dir / 'k_significant_features.csv', header=True, index=False)   

            if len(fset) > 0:
                fset_overlap = set(fset).intersection(set(fset_ksig))
                prop_overlap = len(fset_overlap) / len(fset)
                if prop_overlap < 0.5 and len(fset) > 100:
                    raise Warning("Insufficient consistency in statistics for feature set agreement!") 
                else:
                    fset = pvalues_ksig.loc['k_significant_features', 
                                            fset_overlap].sort_values(axis=0, ascending=True).index
            else:
                print("NO SIGNIFICANT FEATURES FOUND! Using 'k_significant_feat' features instead")
                fset = fset_ksig
                
            #%% Plot day variation
                    
            print("Loading %s model p-values for plotting" % test_name)
            pvalues = pd.read_csv((stats_dir / '{}_results.csv'.format(test_name)), index_col=0)
            assert all(f in pvalues.columns for f in feat_df.columns)             
            
            swarmDir = plot_dir / '{}_variation'.format(RANDOM_EFFECT.split('_yyyymmdd')[0])
            plot_day_variation(feat_df=feat_df,
                               meta_df=meta_df,
                               group_by=GROUPING_VAR,
                               test_pvalues_df=pvalues,
                               control=control,
                               day_var='date_yyyymmdd',
                               fset=fset,
                               p_value_threshold=P_VALUE_THRESHOLD,
                               saveDir=swarmDir,
                               sns_colour_palette="tab10",
                               dodge=True, 
                               ranked=True)
                                                                   
            #%% Boxplots of most significantly different features for each strain vs control
            #   features ranked by test pvalue significance (lowest first)

            # Barplot of number of significantly different features for each strain   
            prop_sigfeats = barplot_sigfeats(test_pvalues_df=pvalues, 
                                             saveDir=plot_dir,
                                             p_value_threshold=P_VALUE_THRESHOLD)
            
            # Boxplots of significant features (for each group vs control)
            boxplots_sigfeats(feat_meta_df=meta_df.join(feat_df), 
                              test_pvalues_df=pvalues, 
                              group_by=GROUPING_VAR, 
                              control_strain=control, 
                              saveDir=plot_dir / 'paired_boxplots',
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
            
            boxplots_grouped(feat_meta_df=meta_df.join(feat_df), 
                             group_by=GROUPING_VAR,
                             test_pvalues_df=pvalues,
                             control_group=control,
                             fset=fset,
                             saveDir= (plot_dir / 'grouped_boxplots'),
                             max_features_plot_cap=K_SIG_FEATS, 
                             p_value_threshold=0.05,
                             figsize=[8,12],
                             saveFormat='png')    
            
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
            control_heatmap_path = plot_dir / 'HCA' / '{}_cluster_heatmap.eps'.format(control)
            cg = plot_clustermap(featZ=controlZ_feat_df,
                                 meta=control_meta_df,
                                 group_by=[GROUPING_VAR,'date_yyyymmdd'],
                                 col_linkage=None,
                                 order=None,
                                 method='complete',#[linkage, complete, average, weighted, centroid]
                                 figsize=[18,6],
                                 saveto=control_heatmap_path)

            # Extract linkage + clustered features
            col_linkage = cg.dendrogram_col.calculated_linkage
            clustered_features = np.array(controlZ_feat_df.columns)[cg.dendrogram_col.reordered_ind]
                        
            # Clustermap of full data using control clustered features order
            assert not feat_df.isna().sum(axis=0).any()
            assert not (feat_df.std(axis=0) == 0).any()
            
            featZ_df = feat_df.apply(zscore, axis=0)

            featZ_df = featZ_df[clustered_features]

            # Drop features with NaN values after normalising
            n_cols = len(featZ_df.columns)
            featZ_df.dropna(axis=1, inplace=True)
            n_dropped = n_cols - len(featZ_df.columns)
            if n_dropped > 0:
                print("Dropped %d features after normalisation (NaN)" % n_dropped)
            
            full_heatmap_path = plot_dir / 'HCA' / '{}_cluster_heatmap.eps'.format(GROUPING_VAR)
            cg = plot_clustermap(featZ=featZ_df, 
                                 meta=meta_df, 
                                 group_by=GROUPING_VAR,
                                 col_linkage=col_linkage,
                                 order=pd.Series(clustered_features),
                                 method='complete',
                                 figsize=[20,5],
                                 saveto=full_heatmap_path)
            
            if len(var_list) > 2:
                pvalues_heatmap = pvalues_anova.loc['pval', clustered_features]
            elif len(var_list) == 2:
                pvalues_heatmap = pvalues.loc[pvalues.index[0], clustered_features]
            pvalues_heatmap.name = 'P < {}'.format(P_VALUE_THRESHOLD)
            
            assert set(pvalues_heatmap.index) == set(featZ_df.columns)
            
            # Plot barcode clustermap
            barcode_heatmap_path = Path(str(full_heatmap_path).replace('.eps', '_barcode.eps'))
            plot_barcode_clustermap(featZ=featZ_df, 
                                    meta=meta_df, 
                                    group_by=[GROUPING_VAR,'date_yyyymmdd'], 
                                    pvalues_series=pvalues_heatmap,
                                    p_value_threshold=P_VALUE_THRESHOLD,
                                    selected_feats=fset,
                                    saveto=barcode_heatmap_path,
                                    figsize=[18,6],
                                    sns_colour_palette="Pastel1")

            #%% Heatmap barcode with selected features
            #   - Read in selected features list
            
            if args.selected_features_path is not None and run==3 and GROUPING_VAR=='worm_strain':
                selected_features = pd.read_csv(Path(args.selected_features_path), index_col=None)
                selected_features = [s for s in selected_features['feature'] 
                                     if s in featZ_df.columns]              
                selected_barcode_heatmap_path = plot_dir / 'HCA' /\
                    '{}_cluster_heatmap_barcode_selected_features.eps'.format(GROUPING_VAR)
                plot_barcode_clustermap(featZ=featZ_df, 
                                        meta=meta_df, 
                                        group_by=[GROUPING_VAR], 
                                        pvalues_series=pvalues_heatmap,
                                        p_value_threshold=P_VALUE_THRESHOLD,
                                        selected_feats=selected_features,
                                        saveto=selected_barcode_heatmap_path,
                                        figsize=[18,6],
                                        sns_colour_palette="Pastel1")                
                # TODO: Timeseries analysis of feature across timepoints/stimulus windows/on-off food/etc
            
            # feature sets for each stimulus type
            featsets = {}
            for stim in ['_prestim','_bluelight','_poststim']:
                featsets[stim] = [f for f in features.columns if stim in f]
            
        # TODO: sns.relplot and sns.jointplot and sns.lineplot for visualising covariance/corrrelation
        # between selected features
        
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
    
            perplexities = [5,15,30,50] # tSNE: Perplexity parameter for tSNE mapping
    
            tsne_dir = plot_dir / 'tSNE'
            tSNE_df = plot_tSNE(featZ=featZ_df,
                                meta=meta_df,
                                group_by=GROUPING_VAR,
                                var_subset=None,
                                saveDir=tsne_dir,
                                perplexities=perplexities)
            
            #%%     Uniform Manifold Projection (UMAP)
    
            n_neighbours = [5,15,30,50] # N-neighbours parameter
            min_dist = 0.1 # Minimum distance parameter
            
            umap_dir = plot_dir / 'UMAP'
            umap_df = plot_umap(featZ=featZ_df,
                                meta=meta_df,
                                group_by=GROUPING_VAR,
                                var_subset=None,
                                saveDir=umap_dir,
                                n_neighbours=n_neighbours,
                                min_dist=min_dist)

