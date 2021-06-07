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
#import warnings
import numpy as np
import pandas as pd
from time import time
from pathlib import Path

from read_data.paths import get_save_dir
from read_data.read import load_json, load_topfeats
from write_data.write import write_list_to_file
from filter_data.clean_feature_summaries import subset_results
from visualisation.plotting_helper import sig_asterix
from statistical_testing.stats_helper import levene_f_test
from tierpsytools.analysis.significant_features import k_significant_feat
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes, _multitest_correct
#from tierpsytools.drug_screenings.filter_compounds import compounds_with_low_effect_univariate

#%% GLOBALS

JSON_PARAMETERS_PATH = "analysis/20210406_parameters_keio_screen.json"
FEATURES_PATH = "/Users/sm5911/Documents/Keio_Screen/features.csv"
METADATA_PATH = "/Users/sm5911/Documents/Keio_Screen/metadata.csv"

#%% MAIN

def average_control_keio(features, metadata):
    """ Average control data for each experiment day to yield a single mean datapoint for the 
        control plate. This reduces the control sample size to equal the test strain sample size, 
        for t-test comparison
    """
    
    # Subset results for control data
    control_metadata = metadata[metadata['source_plate_id']=='BW']
    control_features = features.reindex(control_metadata.index)

    # Take mean of control for each day = collapse to single datapoint for strain comparison
    mean_control = control_metadata[['gene_name', 'date_yyyymmdd']].join(features).groupby(
                                    by=['gene_name', 'date_yyyymmdd']).mean().reset_index()
    
    # Append remaining control metadata column info (with first well data for each date)
    remaining_cols = [c for c in control_metadata.columns.to_list() if c not in ['gene_name', 
                                                                                 'date_yyyymmdd']]
    mean_control_row_data = []
    for i in mean_control.index:
        date = mean_control.loc[i, 'date_yyyymmdd']
        control_date_meta = control_metadata.loc[control_metadata['date_yyyymmdd'] == date]
        first_well = control_date_meta.loc[control_date_meta.index[0], remaining_cols]
        first_well_mean = first_well.append(mean_control.loc[mean_control['date_yyyymmdd'] ==\
                                                             date].squeeze(axis=0))
        mean_control_row_data.append(first_well_mean)
    
    control_mean = pd.DataFrame.from_records(mean_control_row_data)
    control_metadata = control_mean[control_metadata.columns.to_list()]
    control_features = control_mean[control_features.columns.to_list()]

    features = pd.concat([features.loc[metadata['source_plate_id'] != 'BW', :], 
                          control_features], axis=0).reset_index(drop=True)        
    metadata = pd.concat([metadata.loc[metadata['source_plate_id'] != 'BW', :], 
                          control_metadata.loc[:, metadata.columns.to_list()]], 
                          axis=0).reset_index(drop=True)
    
    return features, metadata

def keio_stats(features, metadata, args):

    SAVE_DIR = get_save_dir(args)

    GROUPING_VAR = args.grouping_variable # categorical variable to investigate, eg.'gene_name'
    assert len(metadata[GROUPING_VAR].unique()) == len(metadata[GROUPING_VAR].str.upper().unique())
    print("\nInvestigating '%s' variation" % GROUPING_VAR)    
    
    # Subset results (rows) to omit selected strains
    if args.omit_strains is not None:
        features, metadata = subset_results(features, metadata, GROUPING_VAR, args.omit_strains)

    # Load Tierpsy Top feature set + subset (columns) for top feats only
    if args.n_top_feats is not None:
        top_feats_path = Path(args.tierpsy_top_feats_dir) / "tierpsy_{}.csv".format(str(args.n_top_feats))
        topfeats = load_topfeats(top_feats_path, add_bluelight=True, 
                                 remove_path_curvature=True, header=None)
        
        # Drop features that are not in results
        top_feats_list = [feat for feat in list(topfeats) if feat in features.columns]
        features = features[top_feats_list]

    STRAIN_LIST = list(metadata[GROUPING_VAR].unique())
    CONTROL = args.control_dict[GROUPING_VAR] # control strain to use
    assert CONTROL in STRAIN_LIST

    ##### STATISTICS #####

    stats_dir =  SAVE_DIR / GROUPING_VAR / "Stats"                  

    # F-test for equal variances
    levene_stats_path = stats_dir / 'levene_results.csv'
    levene_stats = levene_f_test(features, metadata, GROUPING_VAR, 
                                 p_value_threshold=args.pval_threshold, 
                                 multitest_method=args.fdr_method,
                                 saveto=levene_stats_path,
                                 del_if_exists=False)

    # if p < 0.05 then variances are not equal, and sample size matters
    prop_eqvar = (levene_stats['pval'] > args.pval_threshold).sum() / len(levene_stats['pval'])
    print("Percentage equal variance %.1f%%" % (prop_eqvar * 100))

    if args.collapse_control:
        print("Collapsing control data (mean of each day)")
        features, metadata = average_control_keio(features, metadata)

    # Record mean sample size per group
    mean_sample_size = int(np.round(metadata.join(features).groupby([GROUPING_VAR], as_index=False).size().mean()))
    print("Mean sample size: %d" % mean_sample_size)
          
    ### ANOVA / Kruskal-Wallis tests for significantly different features across groups
    stats_path = stats_dir / '{}_results.csv'.format(args.test) # LMM/ANOVA/Kruskal  
    if not stats_path.exists():
        stats_path.parent.mkdir(exist_ok=True, parents=True)
    
        if (args.test == "ANOVA" or args.test == "Kruskal"):
            if len(STRAIN_LIST) > 2:                    
                stats, pvals, reject = univariate_tests(X=features, 
                                                        y=metadata[GROUPING_VAR], 
                                                        control=CONTROL, 
                                                        test=args.test,
                                                        comparison_type='multiclass',
                                                        multitest_correction=args.fdr_method, 
                                                        alpha=0.05)
                # get effect sizes
                effect_sizes = get_effect_sizes(X=features, 
                                                y=metadata[GROUPING_VAR],
                                                control=CONTROL,
                                                effect_type=None,
                                                linked_test=args.test)
                                            
                anova_table = pd.concat([stats, pvals, effect_sizes, reject], axis=1)
                anova_table.columns = ['stats','pvals','effect_size','reject']     
                anova_table['significance'] = sig_asterix(anova_table['pvals'])
    
                # Sort pvals + record significant features
                anova_table = anova_table.sort_values(by=['pvals'], ascending=True)
                fset = list(anova_table['pvals'].index[np.where(anova_table['pvals'] < 
                                                                args.pval_threshold)[0]])
                
                # Save statistics results + significant feature set to file
                anova_table.to_csv(stats_path, header=True, index=True)
        
                if len(fset) > 0:
                    anova_sigfeats_path = Path(str(stats_path).replace('_results.csv', '_sigfeats.txt'))
                    write_list_to_file(fset, anova_sigfeats_path)
                    print("\n%d significant features found by %s for '%s' (P<%.2f, %s)" %\
                          (len(fset), args.test, GROUPING_VAR, args.pval_threshold, args.fdr_method))
            else:
                fset = []
                print("\nWARNING: Not enough groups for %s for '%s' (n=%d groups)" %\
                      (args.test, GROUPING_VAR, len(STRAIN_LIST)))
        else:
            raise IOError("Test not recognised")
    else:
        # Load ANOVA results
        anova_table = pd.read_csv(stats_path, index_col=0)
        pvals = anova_table.sort_values(by='pvals', ascending=True)['pvals']
        fset = pvals[pvals < args.pval_threshold].index.to_list()

    # TODO: LMMs using compounds_with_low_effect_univariate
# =============================================================================
#         # Linear Mixed Models (LMMs), accounting for day-to-day variation
#         # NB: Ideally report:  parameter | beta | lower-95 | upper-95 | random effect (SD)
#         elif args.test == 'LMM':
#             with warnings.catch_warnings():
#                 # Filter warnings as parameter is often on the boundary
#                 warnings.filterwarnings("ignore")
#                 #warnings.simplefilter("ignore", ConvergenceWarning)
#                 (signif_effect, low_effect, 
#                  error, mask, pvals)=compounds_with_low_effect_univariate(feat=features, 
#                                                 drug_name=metadata[GROUPING_VAR], 
#                                                 drug_dose=None, 
#                                                 random_effect=metadata[args.lmm_random_effect], 
#                                                 control=CONTROL, 
#                                                 test=args.test, 
#                                                 comparison_type='multiclass',
#                                                 multitest_method=args.fdr_method,
#                                                 ignore_names=None, 
#                                                 return_pvals=True)
#             assert len(error) == 0
#         
#             # Significant features = if significant for ANY strain vs control
#             fset = list(pvals.columns[(pvals < args.pval_threshold).any()])
#             
#             if len(signif_effect) > 0:
#                 print(("%d significant features found (%d significant %ss vs %s control, "\
#                       % (len(fset), len(signif_effect), GROUPING_VAR.replace('_',' '), 
#                           CONTROL) if len(signif_effect) > 0 else\
#                       "No significant differences found between %s "\
#                       % GROUPING_VAR.replace('_',' '))
#                       + "after accounting for %s variation, %s, P<%.2f, %s)"\
#                       % (args.lmm_random_effect.split('_yyyymmdd')[0], args.test, 
#                          args.pval_threshold, args.fdr_method))
# =============================================================================
    
    ### t-tests / Mann-Whitney tests
    
    # t-test to use        
    t_test = 't-test' if args.test == 'ANOVA' else 'Mann-Whitney' # aka. Wilcoxon rank-sum                            

    ttest_path = stats_dir / '{}_results.csv'.format(t_test)
    if not ttest_path.exists():    
        ttest_path.parent.mkdir(exist_ok=True, parents=True)

        if len(fset) > 0 or len(STRAIN_LIST) == 2:
            stats_t, pvals_t, reject_t = univariate_tests(X=features, 
                                                          y=metadata[GROUPING_VAR], 
                                                          control=CONTROL, 
                                                          test=t_test,
                                                          comparison_type='binary_each_group',
                                                          multitest_correction=args.fdr_method, 
                                                          alpha=0.05)
            effect_sizes_t =  get_effect_sizes(X=features, y=metadata[GROUPING_VAR], 
                                               control=CONTROL,
                                               effect_type=None,
                                               linked_test=t_test)
            
            stats_t.columns = ['stats_' + str(c) for c in stats_t.columns]
            pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
            reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
            effect_sizes_t.columns = ['effect_size_' + str(c) for c in effect_sizes_t.columns]
            
            ttest_table = pd.concat([stats_t, pvals_t, effect_sizes_t, reject_t], axis=1)

            # Record t-test significant feature set (NOT ORDERED)
            fset_ttest = list(pvals_t.index[(pvals_t < args.pval_threshold).sum(axis=1) > 0])
            print("%d signficant features found for any %s vs %s (%s, P<%.2f)" %\
                  (len(fset_ttest), GROUPING_VAR, CONTROL, t_test, args.pval_threshold))
            
            # Save t-test results to file
            ttest_table.to_csv(ttest_path, header=True, index=True) # Save test results to CSV

            if len(fset_ttest) > 0:
                ttest_sigfeats_path = Path(str(ttest_path).replace('_results.csv', '_sigfeats.txt'))
                write_list_to_file(fset_ttest, ttest_sigfeats_path)
                                 
    ### K significant features
    
    k_sigfeats_path = stats_dir / 'k_significant_features.csv'
    if not k_sigfeats_path.exists():
        fset_ksig, (scores, pvalues_ksig), support = k_significant_feat(feat=features, 
                                                                        y_class=metadata[GROUPING_VAR], 
                                                                        k=(len(fset) if len(fset) > 
                                                                           args.n_sig_features else 
                                                                           args.n_sig_features), 
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
        
        ksig_table = pd.concat([pd.Series(scores), pd.Series(pvalues_ksig)], axis=1)
        ksig_table.columns = ['scores','pvals']
        ksig_table.index = fset_ksig
        
        # Correct for multiple comparisons
        _, ksig_table['pvals'] = _multitest_correct(ksig_table['pvals'], 
                                                    multitest_method=args.fdr_method,
                                                    fdr=args.pval_threshold)
        
        # Save k most significant features
        k_sigfeats_path.parent.mkdir(exist_ok=True, parents=True)      
        ksig_table.to_csv(k_sigfeats_path, header=True, index=True)   

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
    
    keio_stats(features, metadata, args)
    
    toc = time()
    print("\nDone in %.1f seconds (%.1f minutes)" % (toc - tic, (toc - tic) / 60))
