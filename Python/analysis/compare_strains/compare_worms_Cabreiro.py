#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare worm or bacterial strains to control (behavioural phenotype analysis)

Stats:
    ANOVA/Kruskal - for significant features among all strains
    t-test/ranksum - for significant features between  each strain and the control
    Linear mixed models - for significant features among all strains, accounting for day variation
    
Plots:
    Boxplots of all strains for significant features by ANONVA/Kruskal/LMM
    Boxplots of each strain vs control for significant features by t-test/ranksum
    Heatmaps of all strains
    PCA/tSNE/UMAP of all strains

@author: saul.moore11@lms.mrc.ac.uk
@date: 13/01/2021
"""

#%% Imports

import argparse
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import zscore # ttest_ind, f_oneway, kruskal

# Custom imports
from read_data.read import load_json, load_top256
from write_data.write import write_list_to_file
from preprocessing.compile_hydra_data import process_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results, subset_results
from statistical_testing.stats_helper import shapiro_normality_test
from feature_extraction.decomposition.pca import plot_pca, remove_outliers_pca
from feature_extraction.decomposition.tsne import plot_tSNE
from feature_extraction.decomposition.umap import plot_umap
from feature_extraction.decomposition.hierarchical_clustering import (plot_clustermap, 
                                                                      plot_barcode_heatmap)
from visualisation.super_plots import superplot
from visualisation.plotting_helper import (sig_asterix, 
                                           plot_day_variation, 
                                           barplot_sigfeats, 
                                           boxplots_sigfeats,
                                           boxplots_grouped)

from tierpsytools.analysis.significant_features import k_significant_feat
from tierpsytools.analysis.statistical_tests import univariate_tests
from tierpsytools.drug_screenings.filter_compounds import compounds_with_low_effect_univariate

#%% Globals

JSON_PARAMETERS_PATH = "analysis/20201028_parameters_filipe_tests.json"
            
#%% Main

# TODO: Make so that you can provide a manual set of selected features for fset and it will plot for those features only

if __name__ == "__main__":
    # Accept command-line inputs
    parser = argparse.ArgumentParser(description='Analyse Tierpsy results (96-well)')
    parser.add_argument('-j', '--json', help="Path to JSON parameters file for analysis",
                        default=JSON_PARAMETERS_PATH, type=str)
    args = parser.parse_args()  
    
    # Load params from JSON file + convert to python object
    args = load_json(args.json)

    assert args.project_dir is not None
    AUX_DIR = Path(args.project_dir) / "AuxiliaryFiles"
    RESULTS_DIR = Path(args.project_dir) / "Results"
    SAVE_DIR = (Path(args.save_dir) if args.save_dir is not None else 
                Path(args.project_dir) / "Analysis")

    # Update save path according to JSON parameters for features to use
    fn = 'Top256' if args.use_top256 else 'All_features'
    fn = fn + '_noSize' if args.drop_size_features else fn
    fn = fn + '_norm' if args.norm_features_only else fn
    fn = fn + '_' + args.percentile_to_use if args.percentile_to_use is not None else fn
    fn = fn + '_noOutliers' if args.remove_outliers else fn
    
    GROUPING_VAR = args.grouping_variable # categorical variable to investigate, eg.'worm_strain'
    
    CONTROL = args.control_dict[GROUPING_VAR] # control strain to use

    IS_NORMAL = args.is_normal 
    # Perform t-tests/ANOVA if True, ranksum/kruskal if False. If None, Shapiro-Wilks tests for 
    # normality are perofrmed to decide between parametric/non-parametric tests

    TEST_NAME = args.test # str, Choose between 'LMM' (if >1 day replicate), 'ANOVA' or 'Kruskal' 
    # Kruskal tests are performed instead of ANOVA if check_normal and data is not normally distributed) 
    # If significant features are found, pairwise t-tests are performed

    #%% Compile and clean results
        
    # Process metadata    
    metadata, metadata_path = process_metadata(aux_dir=AUX_DIR,
                                               imaging_dates=args.dates, 
                                               add_well_annotations=args.add_well_annotations)
    
    # Process feature summary results
    features, metadata = process_feature_summaries(metadata_path, 
                                                   RESULTS_DIR,
                                                   compile_day_summaries=args.compile_day_summaries,
                                                   imaging_dates=args.dates,
                                                   align_bluelight=args.align_bluelight)
    
    # Clean: remove data with too many NaNs/zero std and impute remaining NaNs
    features, metadata = clean_summary_results(features, 
                                               metadata,
                                               feature_columns=None,
                                               imputeNaN=args.impute_nans,
                                               nan_threshold=args.nan_threshold,
                                               max_value_cap=args.max_value_cap,
                                               drop_size_related_feats=args.drop_size_features,
                                               norm_feats_only=args.norm_features_only,
                                               percentile_to_use=args.percentile_to_use)
          
    # # Calculate duration on food + duration in L1 diapause
    # metadata = duration_on_food(metadata) 
    # metadata = duration_L1_diapause(metadata)

    #%% Subset results
    
    # Subset results (rows) to remove NA groups
    metadata = metadata[~metadata[GROUPING_VAR].isna()]
    features = features.reindex(metadata.index)
    
    # Check case-sensitivity
    assert len(metadata[GROUPING_VAR].unique())==len(metadata[GROUPING_VAR].str.upper().unique())
    
    # Subset results (rows) to omit selected strains
    if args.omit_strains is not None:
        features, metadata = subset_results(features, metadata, GROUPING_VAR, args.omit_strains)

    # Subset results (rows) for imaging dates of interest    
    if args.dates is not None:
        features, metadata = subset_results(features, metadata, 'date_yyyymmdd', args.dates)
    
    # Load Tierpsy Top256 feature set + subset (columns) for Top256 features only
    if args.use_top256:
        top256_path = AUX_DIR / 'top256_tierpsy_no_blob_no_eigen_only_abs_no_norm.csv'
        top256 = load_top256(top256_path, add_bluelight=args.align_bluelight)
        
        # Ensure results exist for features in featurelist
        top256_feat_list = [feat for feat in top256 if feat in features.columns]
        print("Dropped %d features in Top256 that are missing from results" %\
              (len(top256)-len(top256_feat_list)))
        
        # Select features for analysis
        features = features[top256_feat_list]

    # Record imaging runs to analyse
    run_list = list(metadata['imaging_run_number'].unique())
    if args.runs is not None:
        assert isinstance(args.runs, list)
        assert all(run in run_list for run in args.runs)
    else:
        print("Found %d imaging runs to analyse: %s" % (len(run_list), run_list))
        args.runs = run_list
    
    # Stats test to use
    assert TEST_NAME in ['ANOVA','Kruskal','LMM']
    # If 'LMM' is chosen, ensure that there are multiple day replicates to compare at each timepoint
    if TEST_NAME == 'LMM':
        assert GROUPING_VAR in ['worm_strain','food_type','drug_type']
        assert all(len(metadata.loc[metadata['imaging_run_number']==timepoint, 
                   args.lmm_random_effect].unique()) > 1 for timepoint in args.runs)

    #%% Check for normality + update decision of which test to use (parametric/non-parametric)
    
    CHECK_NORMAL = True if IS_NORMAL is None else False # Check normality if None

    if CHECK_NORMAL:        
        # Sample data from a random run to see if normal. If not, use Kruskal-Wallis test instead.
        _r = np.random.choice(args.runs, size=1)[0]
        _rMeta = metadata[metadata['imaging_run_number']==_r]
        _rFeat = features.reindex(_rMeta.index)
        
        (prop_features_normal, IS_NORMAL) = shapiro_normality_test(features_df=_rFeat,
                                                            metadata_df=_rMeta,
                                                            group_by=GROUPING_VAR,
                                                            p_value_threshold=args.pval_threshold,
                                                            verbose=True)  

        normtest_savepath = SAVE_DIR / (fn + "_run{}_shapiro_results.csv".format(_r))
        normtest_savepath.parent.mkdir(exist_ok=True, parents=True)

        # Save normailty test results to file
        prop_features_normal.to_csv(normtest_savepath, 
                                    index=True, 
                                    index_label=GROUPING_VAR, 
                                    header='prop_normal')
        
    if IS_NORMAL:
        TEST_NAME = 'ANOVA' if TEST_NAME == 'Kruskal' else TEST_NAME
        T_TEST_NAME = 't-test'
    else:
        TEST_NAME = 'Kruskal' if TEST_NAME == 'ANOVA' else TEST_NAME
        T_TEST_NAME = 'Mann-Whitney' # aka. Wilcoxon rank-sums
        print("WARNING: Data is not normal! Kruskal-Wallis tests will be used instead of ANOVA")

    #%% Analyse variables

    print("\nInvestigating '%s' variation" % GROUPING_VAR)    
    for run in args.runs:
        print("\nAnalysing imaging run %d" % run)
        
        # Subset results to investigate single imaging run
        meta_df = metadata[metadata['imaging_run_number']==run]
        feat_df = features.reindex(meta_df.index)
        
        # Record n wells for imaging run
        print("n = %d wells for run %d" % (meta_df.shape[0], run))
        
        # Record mean sample size per group
        mean_sample_size = int(np.round(meta_df.join(feat_df).groupby([GROUPING_VAR, 
                               'date_yyyymmdd'], as_index=False).size().mean()))
        print("Mean sample size: %d" % mean_sample_size)
        
        # Clean data after subset - zero std, etc
        feat_df, meta_df = clean_summary_results(feat_df, 
                                                 meta_df, 
                                                 max_value_cap=False,
                                                 imputeNaN=False)
        # Record strain list
        run_strain_list = list(meta_df[GROUPING_VAR].unique())
        
        # Save paths
        run_save_dir = SAVE_DIR / fn / "Run_{}".format(run) / (GROUPING_VAR + '_variation')
        stats_dir =  run_save_dir / "Stats"
        plot_dir = run_save_dir / "Plots"

        #%% STATISTICS
        #   One-way ANOVA / Kruskal-Wallis tests for significantly different features across groups
                    
        stats_path = stats_dir / '{}_results.csv'.format(TEST_NAME) # LMM/ANOVA/Kruskal  
        ttest_path = stats_dir / '{}_results.csv'.format(T_TEST_NAME)
        
        if not np.logical_and(stats_path.exists(), ttest_path.exists()):
            stats_path.parent.mkdir(exist_ok=True, parents=True)

            # Create table to store statistics results
            grouped = feat_df.join(meta_df[GROUPING_VAR]).groupby(by=GROUPING_VAR)
            stats_table = grouped.mean().T
            mean_cols = ['mean ' + v for v in stats_table.columns.to_list()]
            stats_table.columns = mean_cols
            for group in grouped.size().index: # store sample size
                stats_table['sample size {}'.format(group)] = grouped.size().loc[group]
            
            # ANOVA / Kruskal-Wallis tests
            if (TEST_NAME == "ANOVA" or TEST_NAME == "Kruskal"):
                if len(run_strain_list) > 2:                    
                    stats, pvals, reject = univariate_tests(X=feat_df, 
                                                            y=meta_df[GROUPING_VAR], 
                                                            control=CONTROL, 
                                                            test=TEST_NAME,
                                                            comparison_type='multiclass',
                                                            multitest_correction=args.fdr_method, 
                                                            fdr=0.05,
                                                            n_jobs=-1)
                                                    
                    # Record name of statistical test used (kruskal/f_oneway)
                    col = '{} p-value'.format(TEST_NAME)
                    stats_table[col] = pvals.loc[stats_table.index, TEST_NAME]
        
                    # Sort pvals + record significant features
                    pvals = pvals.sort_values(by=[TEST_NAME], ascending=True)
                    fset = list(pvals.index[np.where(pvals < args.pval_threshold)[0]])
                    if len(fset) > 0:
                        print("\n%d significant features found by %s for %s (run %d, P<%.2f, %s)" %\
                              (len(fset), TEST_NAME, GROUPING_VAR, run, args.pval_threshold, 
                               args.fdr_method))
                else:
                    fset = []
                    print("\nWARNING: Not enough groups for ANOVA (n=%d groups, run %d)" %\
                          (len(run_strain_list), run))
        
            # Linear Mixed Models (LMMs), accounting for day-to-day variation
            # NB: Ideally report:  parameter | beta | lower-95 | upper-95 | random effect (SD)
            elif TEST_NAME == 'LMM':
                with warnings.catch_warnings():
                    # Filter warnings as parameter is often on the boundary
                    warnings.filterwarnings("ignore")
                    #warnings.simplefilter("ignore", ConvergenceWarning)
                    (signif_effect, low_effect, 
                     error, mask, pvals)=compounds_with_low_effect_univariate(feat=feat_df, 
                                                    drug_name=meta_df[GROUPING_VAR], 
                                                    drug_dose=None, 
                                                    random_effect=meta_df[args.lmm_random_effect], 
                                                    control=CONTROL, 
                                                    test=TEST_NAME, 
                                                    comparison_type='multiclass',
                                                    multitest_method=args.fdr_method,
                                                    ignore_names=None, 
                                                    return_pvals=True)
                assert len(error) == 0
                if len(run_strain_list) == 2:
                    col = '{} p-value'.format(TEST_NAME)
                    stats_table[col] = pvals.loc[pvals.index[0], stats_table.index]  

                # Significant features = if significant for ANY strain vs control
                fset = list(pvals.columns[(pvals < args.pval_threshold).any()])
                
                if len(signif_effect) > 0:
                    print(("%d significant features found (%d significant %ss vs %s control, "\
                          % (len(fset), len(signif_effect), GROUPING_VAR.replace('_',' '), 
                              CONTROL) if len(signif_effect) > 0 else\
                          "No significant differences found between %s "\
                          % GROUPING_VAR.replace('_',' '))
                          + "after accounting for %s variation, run %d, %s, P<%.2f, %s)"\
                          % (args.lmm_random_effect.split('_yyyymmdd')[0], run, TEST_NAME, 
                             args.pval_threshold, args.fdr_method))

            # TODO: Use get_effect_sizes from tierpsytools

            # Save statistics results + significant feature set to file
            pvals.to_csv(stats_path, header=True, index=True)
            
            sigfeats_path = Path(str(stats_path).replace('_results.csv', 
                                                         '_significant_features.txt'))
            if len(fset) > 0:
                write_list_to_file(fset, sigfeats_path)
    
            # T-TESTS: If significance is found by ANOVA/LMM, or only 2 groups, perform t-tests or 
            # rank-sum tests for significant features between each group vs control
            # When comparing >2 groups, perform ANOVA and proceed only to pairwise 2-sample t-tests 
            # if there is significant variability among all groups for any feature
            if len(fset) > 0 or len(run_strain_list) == 2:
                if not ttest_path.exists():
                    ttest_path.parent.mkdir(exist_ok=True, parents=True)
                    ttest_sigfeats_outpath = Path(str(ttest_path).replace('_results.csv',
                                                                          '_significant_features.csv'))
                    # t-tests: each strain vs control
                    stats_t, pvals_t, reject_t = univariate_tests(X=feat_df, 
                                                                  y=meta_df[GROUPING_VAR], 
                                                                  control=CONTROL, 
                                                                  test=T_TEST_NAME,
                                                                  comparison_type='binary_each_group',
                                                                  multitest_correction=args.fdr_method, 
                                                                  fdr=0.05,
                                                                  n_jobs=-1)
    
                    # Record significant feature set
                    fset_ttest = list(pvals_t.columns[(pvals_t < args.pval_threshold).sum(axis=0) > 0])
                    if len(fset_ttest) > 0:
                        print("%d signficant features found for any %s vs %s (run %d, %s, P<%.2f)" %\
                              (len(fset_ttest), GROUPING_VAR, CONTROL, run, T_TEST_NAME, 
                               args.pval_threshold))
                    elif len(fset_ttest) == 0:
                        print("No significant features found for any %s vs %s (run %d, %s, P<%.2f)" %\
                              (GROUPING_VAR, CONTROL, run, T_TEST_NAME, args.pval_threshold))
                                         
                    # Save t-test results to file
                    pvals_t.T.to_csv(ttest_path) # Save test results to CSV
                    if len(fset_ttest) > 0:
                        write_list_to_file(fset_ttest, ttest_sigfeats_outpath)
                    
                    # Barplot of number of significantly different features for each strain   
                    prop_sigfeats = barplot_sigfeats(test_pvalues_df=pvals_t, 
                                                     saveDir=plot_dir,
                                                     p_value_threshold=args.pval_threshold,
                                                     test_name=T_TEST_NAME)
     
                # Add pvalues to stats table
                if len(run_strain_list) == 2:
                    col = '{} p-value'.format(T_TEST_NAME)
                    stats_table[col] = pvals.loc[pvals.index[0], stats_table.index]
    
            # Add stats results to stats table
            if len(run_strain_list) == 2:
                stats_table['significance'] = sig_asterix(pvals.values)
            else:
                stats_table['significance'] = sig_asterix(pvals.loc[stats_table.index, 
                                                                    TEST_NAME].values)
                
            #%% K significant features
            
            # k_sigfeat_dir = plot_dir / 'k_sig_feats'
            # k_sigfeat_dir.mkdir(exist_ok=True, parents=True)      
            fset_ksig, (scores, pvalues_ksig), support = k_significant_feat(feat=feat_df, 
                                                            y_class=meta_df[GROUPING_VAR], 
                                                            k=(len(fset) if len(fset) > 
                                                               args.k_sig_features else 
                                                               args.k_sig_features), 
                                                            score_func='f_classif', 
                                                            scale=None, 
                                                            feat_names=None, 
                                                            plot=False, 
                                                            k_to_plot=None, 
                                                            close_after_plotting=True,
                                                            saveto=None, #k_sigfeat_dir
                                                            figsize=None, 
                                                            title=None, 
                                                            xlabel=None)
            
            pvalues_ksig = pd.DataFrame(pd.Series(data=pvalues_ksig, 
                                                  index=fset_ksig, 
                                                  name='k_significant_features')).T
            # Save k most significant features
            pvalues_ksig.to_csv(stats_dir / 'k_significant_features.csv', header=True, index=False)   
            
            # col = 'Top{} k-significant p-value'.format(args.k_sig_features)
            # stats_table[col] = np.nan
            # stats_table.loc[fset_ksig,col] = pvalues_ksig.loc['k_significant_features',fset_ksig]
            
            if len(fset) > 0:
                fset_overlap = set(fset).intersection(set(fset_ksig))
                prop_overlap = len(fset_overlap) / len(fset)
                if prop_overlap < 0.5 and len(fset) > 100:
                    raise Warning("Inconsistency in statistics for feature set agreement between "
                                  + "%s and k significant features!" % (T_TEST_NAME if 
                                  len(run_strain_list) == 2 else TEST_NAME)) 
                elif args.use_k_sig_feats_overlap:
                    fset = pvalues_ksig.loc['k_significant_features', 
                                            fset_overlap].sort_values(axis=0, ascending=True).index
            else:
                print("NO SIGNIFICANT FEATURES FOUND! "
                      + "Falling back on 'k_significant_feat' feature set for plotting.")
                fset = fset_ksig
            
        #%% Load statistics results
        
        # # Read feature set from file
        # fset_in = run_save_dir / '{}_significant_feature_set.csv'.format(load_test_name)
        # fset = pd.read_csv()
        
        # Read ANOVA results and record significant features
        print("Loading statistics results")
        if stats_path.exists() and len(run_strain_list) > 2:
            pval_stats = pd.read_csv(stats_path, index_col=0)
            
            try: # tierpsytools version
                pvals = pval_stats.sort_values(by=TEST_NAME, ascending=True)
            except: # old version
                pvals = pval_stats.loc['pval']
                pvals = pvals.sort_values(ascending=True) # sort p-values
            
            # Record significant features by ANOVA
            fset = pvals.index[pvals[TEST_NAME].values < args.pval_threshold].to_list()
            print("%d significant features found by %s (run %d, P<%.2f)" %\
                  (len(fset), TEST_NAME, run, args.pval_threshold))
        
        # Read t-test results and record significant features
        if ttest_path.exists():
            pvals_t = pd.read_csv(ttest_path, index_col=0)
            assert all(f in pvals_t.columns for f in feat_df.columns)
                
            # Record significant features by t-test
            fset_ttest = list(pvals_t.columns[(pvals_t < args.pval_threshold).sum(axis=0)>0])
            
            # Use t-test significant feature set if comparing just 2 strains
            if len(run_strain_list) == 2:
                pvals = pvals_t.T
                fset = fset_ttest
                                      
        #%% Plot day variation - visualisation with super-plots!
     
        superplot_dir = plot_dir / 'superplots'    
        for feat in fset[:args.k_sig_features]:

            # TODO: Add t-test/LMM pvalues to superplots!
            
            # strain vs date yyyymmdd
            superplot(features, metadata, feat, 
                      x1=GROUPING_VAR, x2='date_yyyymmdd',
                      plot_type='box', #show_points=True, sns_colour_palettes=["plasma","viridis"]
                      dodge=True, saveDir=superplot_dir)

            # plate ID vs run number
            superplot(features, metadata, feat, 
                      x1=GROUPING_VAR, x2='imaging_run_number',
                      dodge=True, saveDir=superplot_dir)
        
        # # TODO: Look into why these plots take so long?!
        # swarmDir = plot_dir / '{}_variation'.format(args.lmm_random_effect.split('_yyyymmdd')[0])
        # plot_day_variation(feat_df=feat_df,
        #                    meta_df=meta_df,
        #                    group_by=GROUPING_VAR,
        #                    test_pvalues_df=pvalues_ttest,
        #                    control=CONTROL,
        #                    day_var='date_yyyymmdd',
        #                    feature_set=fset,
        #                    max_features_plot_cap=args.max_features_plot_cap,
        #                    p_value_threshold=args.pval_threshold,
        #                    saveDir=swarmDir,
        #                    figsize=[(len(run_strain_list)/3 if len(run_strain_list)>10 else 6), 6],
        #                    sns_colour_palette="tab10",
        #                    dodge=False, 
        #                    ranked=True,
        #                    drop_insignificant=False)
                                                               
        #%% Boxplots of most significantly different features for each strain vs control
        # features ranked by test pvalue significance (lowest first)
        
        # Boxplots of significant features by ANOVA/LMM (across all groups)
        boxplots_grouped(feat_meta_df=meta_df.join(feat_df), 
                         group_by=GROUPING_VAR,
                         control_group=CONTROL,
                         test_pvalues_df=pvals_t,
                         feature_set=fset,
                         saveDir=(plot_dir / 'grouped_boxplots'),
                         max_features_plot_cap=None, 
                         max_groups_plot_cap=None,
                         p_value_threshold=args.pval_threshold,
                         drop_insignificant=False,
                         sns_colour_palette="tab10",
                         figsize=[6, (len(run_strain_list)/3 if len(run_strain_list)>10 else 12)],
                         saveFormat='png')
                
        # Boxplots of significant features by pairwise t-test (for each group vs control)
        boxplots_sigfeats(feat_meta_df=meta_df.join(feat_df), 
                          test_pvalues_df=pvals_t, 
                          group_by=GROUPING_VAR, 
                          control_strain=CONTROL, 
                          feature_set=fset, #['speed_norm_50th_bluelight'],
                          saveDir=plot_dir / 'paired_boxplots',
                          max_features_plot_cap=args.k_sig_features,
                          p_value_threshold=args.pval_threshold,
                          drop_insignificant=True,
                          verbose=False)
            
        # from tierpsytools.analysis.significant_features import plot_feature_boxplots
        # plot_feature_boxplots(feat_to_plot=fset,
        #                       y_class=GROUPING_VAR,
        #                       scores=pvalues.rank(axis=1),
        #                       feat_df=feat_df,
        #                       pvalues=np.asarray(pvalues).flatten(),
        #                       saveto=None,
        #                       close_after_plotting=False)
        
        #%% Hierarchical Clustering Analysis
        #   - Clustermap of features by strain, to see if data cluster into groups
        #   - Control data is clustered first, feature order is stored and ordering applied to 
        #     full data for comparison
        
        heatmap_saveFormat = 'pdf'
        
        # Extract data for control
        control_feat_df = feat_df[meta_df[GROUPING_VAR]==CONTROL]
        control_meta_df = meta_df.reindex(control_feat_df.index)
        
        control_feat_df, control_meta_df = clean_summary_results(features=control_feat_df,
                                                                 metadata=control_meta_df,
                                                                 imputeNaN=False)
        
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
        if len(control_meta_df[args.lmm_random_effect].unique()) > 1:
            control_clustermap_path = plot_dir / 'HCA' / ('{}_clustermap'.format(CONTROL) + 
                                                          '.{}'.format(heatmap_saveFormat))
            cg = plot_clustermap(featZ=controlZ_feat_df,
                                 meta=control_meta_df,
                                 group_by=[GROUPING_VAR,'date_yyyymmdd'],
                                 col_linkage=None,
                                 method='complete',#[linkage, complete, average, weighted, centroid]
                                 figsize=[18,6],
                                 saveto=control_clustermap_path)
    
            # Extract linkage + clustered features
            col_linkage = cg.dendrogram_col.calculated_linkage
            clustered_features = np.array(controlZ_feat_df.columns)[cg.dendrogram_col.reordered_ind]
        else:
            clustered_features = None
        
        assert not feat_df.isna().sum(axis=0).any()
        assert not (feat_df.std(axis=0) == 0).any()
        
        featZ_df = feat_df.apply(zscore, axis=0)
        
        # Drop features with NaN values after normalising
        # TODO: Do we need these checks?
        #assert not any(featZ_df.isna(axis=1))
        n_cols = len(featZ_df.columns)
        featZ_df.dropna(axis=1, inplace=True)
        n_dropped = n_cols - len(featZ_df.columns)
        if n_dropped > 0:
            print("Dropped %d features after normalisation (NaN)" % n_dropped)

        # Save stats table to CSV
        stats_table_path = stats_dir / 'stats_summary_table.csv'
        if not stats_path.exists():
            # Add z-normalised values
            z_stats = featZ_df.join(meta_df[GROUPING_VAR]).groupby(by=GROUPING_VAR).mean().T
            z_mean_cols = ['z-mean ' + v for v in z_stats.columns.to_list()]
            z_stats.columns = z_mean_cols
            stats_table = stats_table.join(z_stats)
            first_cols = [m for m in stats_table.columns if 'mean' in m]
            last_cols = [c for c in stats_table.columns if c not in first_cols]
            first_cols.extend(last_cols)
            stats_table = stats_table[first_cols].reset_index()
            first_cols.insert(0, 'feature')
            stats_table.columns = first_cols
            stats_table['feature'] = [' '.join(f.split('_')) for f in stats_table['feature']]
            stats_table = stats_table.sort_values(by='{} p-value'.format((T_TEST_NAME if 
                                         len(run_strain_list) == 2 else TEST_NAME)), ascending=True)
            stats_table.to_csv(stats_table_path, header=True, index=None)
        
        # Clustermap of full data       
        full_clustermap_path = plot_dir / 'HCA' / ('{}_full_clustermap'.format(GROUPING_VAR) + 
                                                   '.{}'.format(heatmap_saveFormat))
        fg = plot_clustermap(featZ=featZ_df, 
                             meta=meta_df, 
                             group_by=GROUPING_VAR,
                             col_linkage=None,
                             method='complete',
                             figsize=[20, (len(run_strain_list) / 4 if 
                                           len(run_strain_list) > 10 else 6)],
                             saveto=full_clustermap_path)
        if not clustered_features:
            # If no control clustering (due to no day variation) then use clustered features for 
            # all strains to order barcode heatmaps
            clustered_features = np.array(featZ_df.columns)[fg.dendrogram_col.reordered_ind]
        
        if len(run_strain_list) > 2:
            pvalues_heatmap = pvals.loc[clustered_features, TEST_NAME]
        elif len(run_strain_list) == 2:
            pvalues_heatmap = pvals_t.loc[pvals_t.index[0], clustered_features]
        pvalues_heatmap.name = 'P < {}'.format(args.pval_threshold)

        assert all(f in featZ_df.columns for f in pvalues_heatmap.index)

        # Heatmap barcode with selected features, ordered by control clustered feature order
        #   - Read in selected features list  
        if args.selected_features_path is not None and run == 3 and GROUPING_VAR == 'worm_strain':
            fset = pd.read_csv(Path(args.selected_features_path), index_col=None)
            fset = [s for s in fset['feature'] if s in featZ_df.columns] 
            # TODO: assert all(s in featZ_df.columns for s in fset['feature'])
            
        # Plot barcode heatmap (grouping by date)
        if len(control_meta_df[args.lmm_random_effect].unique()) > 1:
            heatmap_date_path = plot_dir / 'HCA' / ('{}_date_heatmap'.format(GROUPING_VAR) + 
                                                    '.{}'.format(heatmap_saveFormat))
            plot_barcode_heatmap(featZ=featZ_df[clustered_features], 
                                 meta=meta_df, 
                                 group_by=['date_yyyymmdd',GROUPING_VAR], 
                                 pvalues_series=pvalues_heatmap,
                                 p_value_threshold=args.pval_threshold,
                                 selected_feats=fset if len(fset) > 0 else None,
                                 saveto=heatmap_date_path,
                                 figsize=[20, (len(run_strain_list) / 4 if 
                                               len(run_strain_list) > 10 else 6)],
                                 sns_colour_palette="Pastel1")
        
        # Plot group-mean heatmap (averaged across days)
        heatmap_path = plot_dir / 'HCA' / ('{}_heatmap'.format(GROUPING_VAR) + 
                                           '.{}'.format(heatmap_saveFormat))
        plot_barcode_heatmap(featZ=featZ_df[clustered_features], 
                             meta=meta_df, 
                             group_by=[GROUPING_VAR], 
                             pvalues_series=pvalues_heatmap,
                             p_value_threshold=args.pval_threshold,
                             selected_feats=fset if len(fset) > 0 else None,
                             saveto=heatmap_path,
                             figsize=[20, (len(run_strain_list) / 4 if 
                                           len(run_strain_list) > 10 else 6)],
                             sns_colour_palette="Pastel1")        
                        
        #%% Principal Components Analysis (PCA)

        if args.remove_outliers:
            outlier_path = plot_dir / 'mahalanobis_outliers.pdf'
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
                                control=CONTROL,
                                var_subset=None, 
                                saveDir=pca_dir,
                                PCs_to_keep=10,
                                n_feats2print=10,
                                sns_colour_palette="tab10",
                                hypercolor=False) 
        # TODO: Ensure sns colour palette doees not plot white points
         
        #%%     t-distributed Stochastic Neighbour Embedding (tSNE)

        tsne_dir = plot_dir / 'tSNE'
        perplexities = [5,15,30]
        
        tSNE_df = plot_tSNE(featZ=featZ_df,
                            meta=meta_df,
                            group_by=GROUPING_VAR,
                            var_subset=None,
                            saveDir=tsne_dir,
                            perplexities=perplexities,
                             # NB: perplexity parameter should be roughly equal to group size
                            sns_colour_palette="plasma")
   
        #%%     Uniform Manifold Projection (UMAP)

        umap_dir = plot_dir / 'UMAP'
        n_neighbours = [5,15,30]
        min_dist = 0.1 # Minimum distance parameter
        
        umap_df = plot_umap(featZ=featZ_df,
                            meta=meta_df,
                            group_by=GROUPING_VAR,
                            var_subset=None,
                            saveDir=umap_dir,
                            n_neighbours=n_neighbours,
                            # NB: n_neighbours parameter should be roughly equal to group size
                            min_dist=min_dist,
                            sns_colour_palette="tab10")
            