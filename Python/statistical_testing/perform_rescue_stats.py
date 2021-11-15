#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perform Keio Rescue statistics - Analyse variation in worm behaviour on strains treated with 
different antioxidants

For each strain/treatment combination, compare each to BW control with no antioxidant (wild-type/None)
and also each strain with/without antioxidant treatment

- Significant features across samples/treatments by ANOVA/Kruskal
- Significant features for each strain/treatment vs control using t-tests/Mann-Whitney

@author: sm5911
@date: 12/11/2021
"""

#%% IMPORTS

import argparse
import numpy as np
import pandas as pd
from time import time
from pathlib import Path

from read_data.paths import get_save_dir
from read_data.read import load_json, load_topfeats
from write_data.write import write_list_to_file
from visualisation.plotting_helper import sig_asterix
from statistical_testing.perform_keio_stats import df_summary_stats

from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes, _multitest_correct

#%% GLOBALS

JSON_PARAMETERS_PATH = "analysis/20211102_parameters_keio_rescue.json"

CONTROL_STRAIN = 'wild_type'
STRAIN_COLNAME = 'gene_name'
CONTROL_TREATMENT = 'None'
TREATMENT_COLNAME = 'antioxidant'
    
#%% FUNCTIONS

def pairwise_ttest(control_df, strain_df, feature_list, group_by='antioxidant', 
                   fdr_method='fdr_by', fdr=0.05):
    """ """
    from scipy.stats import ttest_ind

    groups = control_df[group_by].unique()

    strain_pvals_list = []
    strain_stats_list = []
    for group in groups:
        test_control = control_df[control_df[group_by]==group]
        test_strain = strain_df[strain_df[group_by]==group]
        
        pvals = []
        stats = []
        for feature in feature_list:
            _stat, _pval = ttest_ind(test_control[feature], test_strain[feature], axis=0)
            pvals.append(_pval)
            stats.append(_stat)
        
        pvals = pd.DataFrame(np.array(pvals).T, index=feature_list, columns=[group])
        stats = pd.DataFrame(np.array(stats).T, index=feature_list, columns=[group])
            
        strain_pvals_list.append(pvals)
        strain_stats_list.append(stats)
        
    strain_pvals = pd.concat(strain_pvals_list, axis=1)
    strain_stats = pd.concat(strain_stats_list, axis=1)

    # correct for multiple feature/antioxidant comparisons
    if fdr_method is not None:
        strain_reject, strain_pvals = _multitest_correct(strain_pvals, fdr_method, fdr)

    return strain_stats, strain_pvals, strain_reject

#%% 
def antioxidant_stats(features, metadata, args):
    """ Perform statistical analyses on Keio antioxidant rescue experiment results:
        - ANOVA tests for significant feature variation between strains (for each antioxidant treatment in turn)
        - ANOVA tests for significant feature variation in antioxidant treatment (for each strain in turn)
        - t-tests for each feature comparing each strain vs control for paired antioxidant treatment conditions
        - t-tests for each feature comparing each strain antioxidant treatment to negative control (no antioxidant)
        
        Inputs
        ------
        features, metadata : pd.DataFrame
            Clean feature summaries and accompanying metadata
        
        args : Object
            Python object with the following attributes:
            - drop_size_features : bool
            - norm_features_only : bool
            - percentile_to_use : str
            - remove_outliers : bool
            - control_dict : dict
            - n_top_feats : int
            - tierpsy_top_feats_dir (if n_top_feats) : str
            - test : str
            - f_test : bool
            - pval_threshold : float
            - fdr_method : str
            - n_sig_features : int           
    """

    # categorical variables to investigate: 'gene_name' and 'antioxidant'
    print("\nInvestigating variation in worm behaviour on hit strains treated with different antioxidants")    

    # assert there will be no errors due to case-sensitivity
    assert len(metadata[STRAIN_COLNAME].unique()) == len(metadata[STRAIN_COLNAME].str.upper().unique())
    assert len(metadata[TREATMENT_COLNAME].unique()) == len(metadata[TREATMENT_COLNAME].str.upper().unique())
    
    # load Tierpsy top feature set + subset (columns) for top feats only
    if args.n_top_feats is not None:
        top_feats_path = Path(args.tierpsy_top_feats_dir) / "tierpsy_{}.csv".format(str(args.n_top_feats))
        topfeats = load_topfeats(top_feats_path, add_bluelight=args.align_bluelight, 
                                 remove_path_curvature=True, header=None)
        
        # drop features that are not in results
        top_feats_list = [feat for feat in list(topfeats) if feat in features.columns]
        features = features[top_feats_list]
    
    assert not features.isna().any().any()
    n_feats = features.shape[1]
    
    strain_list = list(metadata[STRAIN_COLNAME].unique())
    antioxidant_list = list(metadata[TREATMENT_COLNAME].unique())
    assert CONTROL_STRAIN in strain_list and CONTROL_TREATMENT in antioxidant_list

    # print mean sample size
    sample_size = df_summary_stats(metadata, columns=[STRAIN_COLNAME, TREATMENT_COLNAME])
    print("Mean sample size of %s: %d" % (STRAIN_COLNAME, int(sample_size['n_samples'].mean())))
    
    # construct save paths (args.save_dir / topfeats? etc)
    save_dir = get_save_dir(args)
    stats_dir =  save_dir / "Stats" / args.fdr_method

    ### For each antioxidant treatment in turn...
    
    for antiox in antioxidant_list:
        meta_antiox = metadata[metadata[TREATMENT_COLNAME]==antiox]
        feat_antiox = features.reindex(meta_antiox.index)

        ### ANOVA tests for significant feature variation between strains
        
        # make path to save ANOVA results
        test_path_unncorrected = stats_dir / '{}_uncorrected.csv'.format((args.test + '_' + antiox))
        test_path = stats_dir / '{}_results.csv'.format((args.test + '_' + antiox))   
        test_path.parent.mkdir(exist_ok=True, parents=True)    
 
        if len(meta_antiox[STRAIN_COLNAME].unique()) > 2:   
            # perform ANOVA + record results before & after correcting for multiple comparisons               
            stats, pvals, reject = univariate_tests(X=feat_antiox, 
                                                    y=meta_antiox[STRAIN_COLNAME], 
                                                    test=args.test,
                                                    control=CONTROL_STRAIN,
                                                    comparison_type='multiclass',
                                                    multitest_correction=None, # uncorrected
                                                    alpha=args.pval_threshold,
                                                    n_permutation_test=None) # 'all'
        
            # get effect sizes
            effect_sizes = get_effect_sizes(X=feat_antiox, 
                                            y=meta_antiox[STRAIN_COLNAME],
                                            control=CONTROL_STRAIN,
                                            effect_type=None,
                                            linked_test=args.test)
        
            # compile + save results (uncorrected)
            test_results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
            test_results.columns = ['stats','effect_size','pvals','reject']     
            test_results['significance'] = sig_asterix(test_results['pvals'])
            test_results = test_results.sort_values(by=['pvals'], ascending=True) # rank pvals
            test_results.to_csv(test_path_unncorrected, header=True, index=True)
        
            # correct for multiple comparisons
            reject_corrected, pvals_corrected = _multitest_correct(pvals, 
                                                                   multitest_method=args.fdr_method,
                                                                   fdr=args.pval_threshold)
                                        
            # compile + save results (corrected)
            test_results = pd.concat([stats, effect_sizes, pvals_corrected, reject_corrected], axis=1)
            test_results.columns = ['stats','effect_size','pvals','reject']     
            test_results['significance'] = sig_asterix(test_results['pvals'])
            test_results = test_results.sort_values(by=['pvals'], ascending=True) # rank pvals
            test_results.to_csv(test_path, header=True, index=True)
        
            # use reject mask to find significant feature set
            fset = pvals.loc[reject[args.test]].sort_values(by=args.test, ascending=True).index.to_list()
            print("%d (%.1f%%) significant features found by %s for %s across '%s' (P<%.2f, %s)" %\
                  (len(fset), (len(fset)/n_feats*100), args.test, antiox, STRAIN_COLNAME, 
                  args.pval_threshold, args.fdr_method))

            if len(fset) > 0:
                anova_sigfeats_path = stats_dir / '{}_sigfeats.txt'.format((args.test + '_' + antiox))
                write_list_to_file(fset, anova_sigfeats_path)
        else:
            fset = []
            print("\nWARNING: Not enough groups for %s for '%s' (n=%d groups)" %\
                  (args.test, STRAIN_COLNAME, len(strain_list)))                      

        ### t-tests for each feature comparing each strain vs control for paired antioxidant treatment conditions
                
        # t-test to use        
        t_test = 't-test' if args.test == 'ANOVA' else 'Mann-Whitney' # aka. Wilcoxon rank-sum      
        ttest_path_uncorrected = stats_dir / '{}_uncorrected.csv'.format((t_test + '_' + antiox))
        ttest_path = stats_dir / '{}_results.csv'.format((t_test + '_' + antiox))  
        ttest_path.parent.mkdir(exist_ok=True, parents=True)

        # perform t-tests (without correction for multiple testing)
        stats_t, pvals_t, reject_t = univariate_tests(X=feat_antiox, 
                                                      y=meta_antiox[STRAIN_COLNAME], 
                                                      control=CONTROL_STRAIN, 
                                                      test=t_test,
                                                      comparison_type='binary_each_group',
                                                      multitest_correction=None, 
                                                      alpha=0.05)
        # get effect sizes for comparisons
        effect_sizes_t =  get_effect_sizes(X=feat_antiox, 
                                           y=meta_antiox[STRAIN_COLNAME], 
                                           control=CONTROL_STRAIN,
                                           effect_type=None,
                                           linked_test=t_test)
        
        # compile + save t-test results (uncorrected)
        stats_t.columns = ['stats_' + str(c) for c in stats_t.columns]
        pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
        reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
        effect_sizes_t.columns = ['effect_size_' + str(c) for c in effect_sizes_t.columns]
        ttest_uncorrected = pd.concat([stats_t, effect_sizes_t, pvals_t, reject_t], axis=1)
        ttest_uncorrected.to_csv(ttest_path_uncorrected, header=True, index=True)
        
        # correct for multiple comparisons
        pvals_t.columns = [c.split("_")[-1] for c in pvals_t.columns]
        reject_t, pvals_t = _multitest_correct(pvals_t, 
                                               multitest_method=args.fdr_method,
                                               fdr=args.pval_threshold)

        # compile + save t-test results (corrected)
        pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
        reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
        ttest_corrected = pd.concat([stats_t, effect_sizes_t, pvals_t, reject_t], axis=1)
        ttest_corrected.to_csv(ttest_path, header=True, index=True)

        # record t-test significant features (not ordered)
        fset_ttest = pvals_t[np.asmatrix(reject_t)].index.unique().to_list()
        #assert set(fset_ttest) == set(pvals_t.index[(pvals_t < args.pval_threshold).sum(axis=1) > 0])
        print("%d significant features on %s for any %s vs %s (%s, P<%.2f)" % (len(fset_ttest),
              antiox, STRAIN_COLNAME, CONTROL_STRAIN, t_test, args.pval_threshold))

        if len(fset_ttest) > 0:
            ttest_sigfeats_path = stats_dir / '{}_sigfeats.txt'.format((t_test + '_' + antiox))
            write_list_to_file(fset_ttest, ttest_sigfeats_path)

    
    ### For each strain in turn...
    
    for strain in strain_list:
        meta_strain = metadata[metadata[STRAIN_COLNAME]==strain]
        feat_strain = features.reindex(meta_strain.index)
        
        ### ANOVA tests for significant feature variation in antioxidant treatment
    
        # make path to save ANOVA results
        test_path_unncorrected = stats_dir / '{}_uncorrected.csv'.format((args.test + '_' + strain))
        test_path = stats_dir / '{}_results.csv'.format((args.test + '_' + strain))
        test_path.parent.mkdir(exist_ok=True, parents=True)
    
        if len(meta_strain[TREATMENT_COLNAME].unique()) > 2:   
            # perform ANOVA + record results before & after correcting for multiple comparisons               
            stats, pvals, reject = univariate_tests(X=feat_strain, 
                                                    y=meta_strain[TREATMENT_COLNAME], 
                                                    test=args.test,
                                                    control=CONTROL_TREATMENT,
                                                    comparison_type='multiclass',
                                                    multitest_correction=None, # uncorrected
                                                    alpha=args.pval_threshold,
                                                    n_permutation_test=None) # 'all'

            # get effect sizes
            effect_sizes = get_effect_sizes(X=feat_strain, 
                                            y=meta_strain[TREATMENT_COLNAME],
                                            control=CONTROL_TREATMENT,
                                            effect_type=None,
                                            linked_test=args.test)

            # compile + save results (uncorrected)
            test_results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
            test_results.columns = ['stats','effect_size','pvals','reject']     
            test_results['significance'] = sig_asterix(test_results['pvals'])
            test_results = test_results.sort_values(by=['pvals'], ascending=True) # rank pvals
            test_results.to_csv(test_path_unncorrected, header=True, index=True)

            # correct for multiple comparisons
            reject_corrected, pvals_corrected = _multitest_correct(pvals, 
                                                                   multitest_method=args.fdr_method,
                                                                   fdr=args.pval_threshold)
                                        
            # compile + save results (corrected)
            test_results = pd.concat([stats, effect_sizes, pvals_corrected, reject_corrected], axis=1)
            test_results.columns = ['stats','effect_size','pvals','reject']     
            test_results['significance'] = sig_asterix(test_results['pvals'])
            test_results = test_results.sort_values(by=['pvals'], ascending=True) # rank pvals
            test_results.to_csv(test_path, header=True, index=True)
    
            # use reject mask to find significant feature set
            fset = pvals.loc[reject[args.test]].sort_values(by=args.test, ascending=True).index.to_list()
            #assert set(fset) == set(anova_corrected['pvals'].index[np.where(anova_corrected['pvals'] < 
            #args.pval_threshold)[0]])

            print("%d (%.1f%%) significant features found by %s for %s across '%s' (P<%.2f, %s)" %\
                  (len(fset), (len(fset)/n_feats*100), args.test, strain, TREATMENT_COLNAME, 
                   args.pval_threshold, args.fdr_method))
                
            if len(fset) > 0:
                anova_sigfeats_path = stats_dir / '{}_sigfeats.txt'.format((args.test + '_' + strain))
                write_list_to_file(fset, anova_sigfeats_path)
        else:
            fset = []
            print("\nWARNING: Not enough groups for %s for '%s' (n=%d groups)" %\
                  (args.test, TREATMENT_COLNAME, len(antioxidant_list)))                      
                                                        
        ### t-tests for each feature comparing each antioxidant treatment to no antioxidant for each strain 
 
        # t-test to use        
        t_test = 't-test' if args.test == 'ANOVA' else 'Mann-Whitney' # aka. Wilcoxon rank-sum      
        ttest_path_uncorrected = stats_dir / '{}_uncorrected.csv'.format((t_test + '_' + strain))
        ttest_path = stats_dir / '{}_results.csv'.format((t_test + '_' + strain))  
        ttest_path.parent.mkdir(exist_ok=True, parents=True)

        # perform t-tests (without correction for multiple testing)
        stats_t, pvals_t, reject_t = univariate_tests(X=feat_strain, 
                                                      y=meta_strain[TREATMENT_COLNAME], 
                                                      control=CONTROL_TREATMENT, 
                                                      test=t_test,
                                                      comparison_type='binary_each_group',
                                                      multitest_correction=None, 
                                                      alpha=0.05)
        # get effect sizes for comparisons
        effect_sizes_t =  get_effect_sizes(X=feat_strain, 
                                           y=meta_strain[TREATMENT_COLNAME], 
                                           control=CONTROL_TREATMENT,
                                           effect_type=None,
                                           linked_test=t_test)
        
        # compile + save t-test results (uncorrected)
        stats_t.columns = ['stats_' + str(c) for c in stats_t.columns]
        pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
        reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
        effect_sizes_t.columns = ['effect_size_' + str(c) for c in effect_sizes_t.columns]
        ttest_uncorrected = pd.concat([stats_t, effect_sizes_t, pvals_t, reject_t], axis=1)
        ttest_uncorrected.to_csv(ttest_path_uncorrected, header=True, index=True)
        
        # correct for multiple comparisons
        pvals_t.columns = [c.split("_")[-1] for c in pvals_t.columns]
        reject_t, pvals_t = _multitest_correct(pvals_t, 
                                               multitest_method=args.fdr_method,
                                               fdr=args.pval_threshold)

        # compile + save t-test results (corrected)
        pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
        reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
        ttest_corrected = pd.concat([stats_t, effect_sizes_t, pvals_t, reject_t], axis=1)
        ttest_corrected.to_csv(ttest_path, header=True, index=True)

        # record t-test significant features (not ordered)
        fset_ttest = pvals_t[np.asmatrix(reject_t)].index.unique().to_list()
        #assert set(fset_ttest) == set(pvals_t.index[(pvals_t < args.pval_threshold).sum(axis=1) > 0])
        print("%d significant features for %s on any %s vs %s (%s, %s, P<%.2f)" % (len(fset_ttest),
              strain, TREATMENT_COLNAME, CONTROL_TREATMENT, t_test, args.fdr_method, args.pval_threshold))

        if len(fset_ttest) > 0:
            ttest_sigfeats_path = stats_dir / '{}_sigfeats.txt'.format((t_test + '_' + strain))
            write_list_to_file(fset_ttest, ttest_sigfeats_path)
  
                    
    ### Pairwise t-tests comparing strain vs control behaviour on each antioxidant
    
    # subset for control data
    control_strain_meta = metadata[metadata[STRAIN_COLNAME] == CONTROL_STRAIN]
    control_strain_feat = features.reindex(control_strain_meta.index)
    control_df = control_strain_meta.join(control_strain_feat)

    for strain in strain_list:
        if strain == CONTROL_STRAIN:
            continue
                
        # subset for strain data
        strain_meta = metadata[metadata[STRAIN_COLNAME]==strain]
        strain_feat = features.reindex(strain_meta.index)
        strain_df = strain_meta.join(strain_feat)
        
        # perform pairwise t-tests comparing strain with control for each antioxidant treatment
        stats, pvals, reject = pairwise_ttest(control_df, strain_df, feature_list=features.columns, 
                                              group_by=TREATMENT_COLNAME, fdr_method=args.fdr_method, 
                                              fdr=args.pval_threshold)
        
        # compile table of results
        stats.columns = ['stats_' + str(c) for c in stats.columns]
        pvals.columns = ['pvals_' + str(c) for c in pvals.columns]
        reject.columns = ['reject_' + str(c) for c in reject.columns]
        test_results = pd.concat([stats, pvals, reject], axis=1)
        
        # save results
        ttest_strain_path = stats_dir / 'pairwise_ttests' / '{}_results.csv'.format(strain + 
                            "_vs_" + CONTROL_STRAIN)
        ttest_strain_path.parent.mkdir(parents=True, exist_ok=True)
        test_results.to_csv(ttest_strain_path, header=True, index=True)
        
        # record t-test significant features (not ordered)
        fset_ttest = pvals[np.asmatrix(reject)].index.unique().to_list()
        print("%d significant features between %s vs %s on any %s (%s, %s, P<%.2f)" % (len(fset_ttest),
              strain, CONTROL_STRAIN, TREATMENT_COLNAME, t_test, args.fdr_method, args.pval_threshold))
    
        if len(fset_ttest) > 0:
            ttest_sigfeats_path = stats_dir / 'pairwise_ttests' / '{}_sigfeats.txt'.format(strain + 
                                  "_vs_" + CONTROL_STRAIN)
            write_list_to_file(fset_ttest, ttest_sigfeats_path)


#%% MAIN

if __name__ == "__main__":
    tic = time()
    parser = argparse.ArgumentParser(description="Find 'hit' Keio knockout strains that alter worm behaviour")
    parser.add_argument('-j', '--json', help="Path to JSON parameters file", 
                        default=JSON_PARAMETERS_PATH, type=str)
    parser.add_argument('--features_path', help="Path to feature summaries file", 
                        default=None, type=str)
    parser.add_argument('--metadata_path', help="Path to metadata file", 
                        default=None, type=str)
    args = parser.parse_args()
    
    FEATURES_PATH = args.features_path
    METADATA_PATH = args.metadata_path
    
    args = load_json(args.json)

    if FEATURES_PATH is None:
        FEATURES_PATH = Path(args.save_dir) / 'features.csv'
    if METADATA_PATH is None:
        METADATA_PATH = Path(args.save_dir) / 'metadata.csv'
        
    # load feature summaries and metadata
    features = pd.read_csv(FEATURES_PATH)
    metadata = pd.read_csv(METADATA_PATH, dtype={'comments':str, 'source_plate_id':str})
    
    # subset for desired imaging dates
    if args.dates is not None:
        assert type(args.dates) == list
        metadata = metadata.loc[metadata['date_yyyymmdd'].astype(str).isin(args.dates)]
        features = features.reindex(metadata.index)

    antioxidant_stats(features, metadata, args)
    
    toc = time()
    print("\nDone in %.1f seconds (%.1f minutes)" % (toc - tic, (toc - tic) / 60))
