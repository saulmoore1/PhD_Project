#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perform Keio screen statistics
- Significant features across all groups by ANOVA/Kruskal
- Significant features for each strain vs control using t-tests/Mann-Whitney

@author: sm5911
@date: 21/05/2021
"""

#%% IMPORTS

import argparse
import warnings
import numpy as np
import pandas as pd
from time import time
from pathlib import Path
from read_data.read import load_json, load_top256
from write_data.write import write_list_to_file
from filter_data.clean_feature_summaries import subset_results
from visualisation.plotting_helper import sig_asterix

from tierpsytools.analysis.significant_features import k_significant_feat
from tierpsytools.analysis.statistical_tests import univariate_tests
from tierpsytools.drug_screenings.filter_compounds import compounds_with_low_effect_univariate

#%% GLOBALS

JSON_PARAMETERS_PATH = "analysis/20210406_parameters_keio_screen.json"
FEATURES_PATH = "/Users/sm5911/Documents/Keio_Screen/features.csv"
METADATA_PATH = "/Users/sm5911/Documents/Keio_Screen/metadata.csv"

#%% MAIN

def do_stats(features, metadata, args):

    AUX_DIR = Path(args.project_dir) / "AuxiliaryFiles"
    
    GROUPING_VAR = args.grouping_variable # categorical variable to investigate, eg.'gene_name' / 'worm_strain'
    assert len(metadata[GROUPING_VAR].unique()) == len(metadata[GROUPING_VAR].str.upper().unique())
    print("\nInvestigating '%s' variation" % GROUPING_VAR)    
    
    # Subset results (rows) to omit selected strains
    if args.omit_strains is not None:
        features, metadata = subset_results(features, metadata, GROUPING_VAR, args.omit_strains)

    # Load Tierpsy Top256 feature set + subset (columns) for Top256
    if args.use_top256:
        top256_path = AUX_DIR / 'top256_tierpsy_no_blob_no_eigen_only_abs_no_norm.csv'
        top256 = load_top256(top256_path, add_bluelight=args.align_bluelight)
        
        # Ensure results exist for features in featurelist
        top256_feat_list = [feat for feat in top256 if feat in features.columns]
        print("Dropped %d features in Top256 that are missing from results" %\
              (len(top256)-len(top256_feat_list)))
        
        # Use Top256 features for analysis
        features = features[top256_feat_list]
          
    # Update save path according to JSON parameters for features to use
    fn = 'Top256' if args.use_top256 else 'All_features'
    fn = fn + '_noSize' if args.drop_size_features else fn
    fn = fn + '_norm' if args.norm_features_only else fn
    fn = fn + '_' + args.percentile_to_use if args.percentile_to_use is not None else fn
    fn = fn + '_noOutliers' if args.remove_outliers else fn

    SAVE_DIR = (Path(args.save_dir) if args.save_dir is not None else Path(args.project_dir) / 
                "Analysis") / fn

    stats_dir =  SAVE_DIR / (args.grouping_variable) / "Stats"
    
    # Stats test to use
    assert args.test in ['ANOVA','Kruskal','LMM']
    if args.test == 'LMM':
        # If 'LMM' is chosen, ensure there are multiple day replicates to compare at each timepoint
        assert all(len(metadata.loc[metadata['imaging_run_number']==timepoint, 
                   args.lmm_random_effect].unique()) > 1 for timepoint in args.runs)
        
    T_TEST_NAME = 't-test' if args.test == 'ANOVA' else 'Mann-Whitney' # aka. Wilcoxon rank-sum

    CONTROL = args.control_dict[GROUPING_VAR] # control strain to use
                            
    # Record mean sample size per group
    mean_sample_size = int(np.round(metadata.join(features).groupby([GROUPING_VAR], as_index=False).size().mean()))
    print("Mean sample size: %d" % mean_sample_size)

    ##### STATISTICS #####
    #   One-way ANOVA / Kruskal-Wallis tests for significantly different features across groups
                
    stats_path = stats_dir / '{}_results.csv'.format(args.test) # LMM/ANOVA/Kruskal  
    ttest_path = stats_dir / '{}_results.csv'.format(T_TEST_NAME) #t-test/Mann-Whitney
    
    if not np.logical_and(stats_path.exists(), ttest_path.exists()):
        stats_path.parent.mkdir(exist_ok=True, parents=True)
        print("Saving stats results to: %s" % stats_dir)

        # Create table to store statistics results
        grouped = features.join(metadata[GROUPING_VAR]).groupby(by=GROUPING_VAR)
        stats_table = grouped.mean().T
        mean_cols = ['mean ' + v for v in stats_table.columns.to_list()]
        stats_table.columns = mean_cols
        for group in grouped.size().index: # store sample size
            stats_table['sample size {}'.format(group)] = grouped.size().loc[group]
        
        # ANOVA / Kruskal-Wallis tests
        if (args.test == "ANOVA" or args.test == "Kruskal"):
            stats, pvals, reject = univariate_tests(X=features, 
                                                    y=metadata[GROUPING_VAR], 
                                                    control=CONTROL, 
                                                    test=args.test,
                                                    comparison_type='multiclass',
                                                    multitest_correction=args.fdr_method, 
                                                    alpha=0.05)
                                            
            # Record name of statistical test used (kruskal/f_oneway)
            col = '{} p-value'.format(args.test)
            stats_table[col] = pvals.loc[stats_table.index, args.test]

            # Sort pvals + record significant features
            pvals = pvals.sort_values(by=[args.test], ascending=True)
            fset = list(pvals.index[np.where(pvals < args.pval_threshold)[0]])
            if len(fset) > 0:
                print("\n%d significant features found by %s for %s (P<%.2f, %s)" %\
                      (len(fset), args.test, GROUPING_VAR, args.pval_threshold, args.fdr_method))
    
        # Linear Mixed Models (LMMs), accounting for day-to-day variation
        # NB: Ideally report:  parameter | beta | lower-95 | upper-95 | random effect (SD)
        elif args.test == 'LMM':
            with warnings.catch_warnings():
                # Filter warnings as parameter is often on the boundary
                warnings.filterwarnings("ignore")
                #warnings.simplefilter("ignore", ConvergenceWarning)
                (signif_effect, low_effect, 
                 error, mask, pvals)=compounds_with_low_effect_univariate(feat=features, 
                                                drug_name=metadata[GROUPING_VAR], 
                                                drug_dose=None, 
                                                random_effect=metadata[args.lmm_random_effect], 
                                                control=CONTROL, 
                                                test=args.test, 
                                                comparison_type='multiclass',
                                                multitest_method=args.fdr_method,
                                                ignore_names=None, 
                                                return_pvals=True)
            assert len(error) == 0

            # Significant features = if significant for ANY strain vs control
            fset = list(pvals.columns[(pvals < args.pval_threshold).any()])
            
            if len(signif_effect) > 0:
                print(("%d significant features found (%d significant %ss vs %s control, "\
                      % (len(fset), len(signif_effect), GROUPING_VAR.replace('_',' '), 
                          CONTROL) if len(signif_effect) > 0 else\
                      "No significant differences found between %s "\
                      % GROUPING_VAR.replace('_',' '))
                      + "after accounting for %s variation, %s, P<%.2f, %s)"\
                      % (args.lmm_random_effect.split('_yyyymmdd')[0], args.test, 
                         args.pval_threshold, args.fdr_method))

        # TODO: Use get_effect_sizes from tierpsytools
        
        # Add significance results to stats table
        stats_table['significance'] = sig_asterix(pvals.loc[stats_table.index, args.test].values)

        # Save statistics results + significant feature set to file
        pvals.to_csv(stats_path, header=True, index=True)
        
        sigfeats_path = Path(str(stats_path).replace('_results.csv', '_significant_features.txt'))
        if len(fset) > 0:
            write_list_to_file(fset, sigfeats_path)

        # T-TESTS: If significance is found by ANOVA/LMM, perform t-tests or 
        # rank-sum tests for significant features between each group vs control
        # Perform ANOVA and proceed only to pairwise 2-sample t-tests 
        # if there is significant variability among all groups for any feature
        if len(fset) > 0 and not ttest_path.exists():
            ttest_path.parent.mkdir(exist_ok=True, parents=True)
            ttest_sigfeats_outpath = Path(str(ttest_path).replace('_results.csv',
                                                                  '_significant_features.csv'))
            # t-tests: each strain vs control
            stats_t, pvals_t, reject_t = univariate_tests(X=features, 
                                                          y=metadata[GROUPING_VAR], 
                                                          control=CONTROL, 
                                                          test=T_TEST_NAME,
                                                          comparison_type='binary_each_group',
                                                          multitest_correction=args.fdr_method, 
                                                          alpha=0.05)

            # Record significant feature set
            fset_ttest = list(pvals_t.index[(pvals_t < args.pval_threshold).sum(axis=1) > 0])
            if len(fset_ttest) > 0:
                print("%d signficant features found for any %s vs %s (%s, P<%.2f)" %\
                      (len(fset_ttest), GROUPING_VAR, CONTROL, T_TEST_NAME, args.pval_threshold))
            elif len(fset_ttest) == 0:
                print("No significant features found for any %s vs %s (%s, P<%.2f)" %\
                      (GROUPING_VAR, CONTROL, T_TEST_NAME, args.pval_threshold))
                                 
            # Save t-test results to file
            pvals_t.T.to_csv(ttest_path) # Save test results to CSV
            if len(fset_ttest) > 0:
                write_list_to_file(fset_ttest, ttest_sigfeats_outpath)
            
            # # Barplot of number of significantly different features for each strain   
            # _ = barplot_sigfeats(test_pvalues_df=pvals_t, 
            #                      saveDir=plot_dir,
            #                      p_value_threshold=args.pval_threshold,
            #                      test_name=T_TEST_NAME)
             
        ##### K-significant features #####
        
        # k_sigfeat_dir = plot_dir / 'k_sig_feats'
        # k_sigfeat_dir.mkdir(exist_ok=True, parents=True)      
        fset_ksig, (scores, pvalues_ksig), support = k_significant_feat(feat=features, 
                                                        y_class=metadata[GROUPING_VAR], 
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
        
        # Save k most significant features
        pvalues_ksig = pd.DataFrame(pd.Series(data=pvalues_ksig, index=fset_ksig, 
                                              name='k_significant_features')).T
        pvalues_ksig.to_csv(stats_dir / 'k_significant_features.csv', header=True, index=False)   
        
        if len(fset) > 0:
            fset_overlap = set(fset).intersection(set(fset_ksig))
            prop_overlap = len(fset_overlap) / len(fset)
            if prop_overlap < 0.5 and len(fset) > 100:
                raise Warning("Inconsistency in statistics for feature set agreement between "
                              + "%s and k significant features!" % args.test) 
            if args.use_k_sig_feats_overlap:
                fset = pvalues_ksig.loc['k_significant_features', fset_overlap].sort_values(
                       axis=0, ascending=True).index
        else:
            print("NO SIGNIFICANT FEATURES FOUND! "
                  + "Falling back on 'k_significant_feat' feature set for plotting.")
            fset = fset_ksig
            
#%% MAIN

if __name__ == "__main__":
    tic = time()
    parser = argparse.ArgumentParser(description="Find 'hit' Keio knockout strains that alter worm behaviour")
    parser.add_argument('-j', '--json', help="Path to JSON parameters file", 
                        default=JSON_PARAMETERS_PATH, type=str)
    parser.add_argument('--features_path', help="Path to feature summaries file", 
                        default=FEATURES_PATH, type=str)
    parser.add_argument('--metadata_path', help="Path to metadata file", 
                        default=METADATA_PATH, type=str)
    args = parser.parse_args()
    
    args = load_json(args.json)
    
    # Read feature summaries + metadata
    features = pd.read_csv(FEATURES_PATH)
    metadata = pd.read_csv(METADATA_PATH, dtype={'comments':str, 'source_plate_id':str})
    
    do_stats(features, metadata, args)
    
    toc = time()
    print("\nDone in %.1f seconds (%.1f minutes)" % (toc-tic, (toc-tic)/60))
