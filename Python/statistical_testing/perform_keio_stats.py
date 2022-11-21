#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perform Keio screen statistics
- Significant features across all groups by ANOVA/Kruskal
- Significant features for each strain vs control using t-tests/Mann-Whitney
- k-significant feature selection (for agreement with ANOVA significant feature set)

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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from read_data.paths import get_save_dir
from read_data.read import load_json
from write_data.write import write_list_to_file
from filter_data.clean_feature_summaries import subset_results
from visualisation.plotting_helper import sig_asterix
from statistical_testing.stats_helper import levene_f_test

from tierpsytools.analysis.significant_features import k_significant_feat, mRMR_feature_selection
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes, _multitest_correct
from tierpsytools.drug_screenings.filter_compounds import compounds_with_low_effect_univariate
from tierpsytools.preprocessing.filter_data import select_feat_set

#%% GLOBALS

# JSON_PARAMETERS_PATH = "analysis/20210406_parameters_keio_initial_screen.json"
JSON_PARAMETERS_PATH = "analysis/20210914_parameters_keio_confirmation_screen.json"

#%% FUNCTIONS

def average_plate_control_data(features, metadata, control='wild_type', grouping_var='gene_name', 
                               plate_var='imaging_plate_id'):
    """ Average data for control plate on each experiment day to yield a single mean datapoint for 
        the control. This reduces the control sample size to equal the test strain sample size, for 
        t-test comparison. Information for the first well in the control sample on each day is used 
        as the accompanying metadata for mean feature results. 
        
        Input
        -----
        features, metadata : pd.DataFrame
            Feature summary results and metadata dataframe with multiple entries per day
            
        Returns
        -------
        features, metadata : pd.DataFrame
            Feature summary results and metadata with control data averaged (single sample per day)
    """
        
    # Subset results for control data
    control_metadata = metadata[metadata[grouping_var]==control]
    control_features = features.reindex(control_metadata.index)

    # Take mean of control for each plate = collapse to single datapoint for strain comparison
    mean_control = control_metadata[[grouping_var, plate_var]].join(control_features).groupby(
                                    by=[grouping_var, plate_var]).mean().reset_index()
    
    # Append remaining control metadata column info (with first well data for each date)
    remaining_cols = [c for c in control_metadata.columns.to_list() if c not in [grouping_var, 
                                                                                 plate_var]]
    mean_control_row_data = []
    for i in mean_control.index:
        # TODO: use agg here?
        plate = mean_control.loc[i, plate_var]
        control_plate_meta = control_metadata.loc[control_metadata[plate_var] == plate]
        first_well = control_plate_meta.loc[control_plate_meta.index[0], remaining_cols]
        first_well_mean = first_well.append(mean_control.loc[mean_control[plate_var] == plate
                                                             ].squeeze(axis=0))
        mean_control_row_data.append(first_well_mean)
    
    control_mean = pd.DataFrame.from_records(mean_control_row_data)
    control_metadata = control_mean[control_metadata.columns.to_list()]
    control_features = control_mean[control_features.columns.to_list()]

    features = pd.concat([features.loc[metadata[grouping_var] != control, :], 
                          control_features], axis=0).reset_index(drop=True)        
    metadata = pd.concat([metadata.loc[metadata[grouping_var] != control, :], 
                          control_metadata.loc[:, metadata.columns.to_list()]], 
                          axis=0).reset_index(drop=True)
    
    return features, metadata

def keio_stats(features, metadata, args):
    """ Perform statistical analyses on Keio screen results:
        - ANOVA tests for significant between strain variation among all strains for each feature
        - t-tests for significant differences between each strain and control for each feature
        - k-significant feature selection for agreement with ANOVA significant feature set
        
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
            - omit_strains : list
            - grouping_variable : str
            - control_dict : dict
            - collapse_control : bool
            - n_top_feats : int
            - tierpsy_top_feats_dir (if n_top_feats) : str
            - test : str
            - f_test : bool
            - pval_threshold : float
            - fdr_method : str
            - n_sig_features : int           
    """

    # categorical variable to investigate, eg.'gene_name'
    grouping_var = args.grouping_variable
    print("\nInvestigating '%s' variation" % grouping_var)    

    # assert there will be no errors duee to case-sensitivity
    assert len(metadata[grouping_var].unique()) == len(metadata[grouping_var].str.upper().unique())
    
    # Subset results (rows) to omit selected strains
    if args.omit_strains is not None:
        features, metadata = subset_results(features, metadata, grouping_var, args.omit_strains)

    # Load Tierpsy Top feature set + subset (columns) for top feats only
    if args.n_top_feats is not None:
        assert args.n_top_feats in [8,16,256,'2k']
        features = select_feat_set(features, 
                                   tierpsy_set_name='tierpsy_{}'.format(args.n_top_feats), 
                                   append_bluelight=args.align_bluelight)
            
    assert not features.isna().any().any()
    
    strain_list = list(metadata[grouping_var].unique())
    control = args.control_dict[grouping_var] # control strain to use
    assert control in strain_list
    
    if args.collapse_control:
        print("Collapsing control data (mean of each day)")
        features, metadata = average_plate_control_data(features, 
                                                        metadata, 
                                                        control=control, 
                                                        grouping_var=grouping_var, 
                                                        plate_var='imaging_plate_id')

    # Record mean sample size per group
    mean_sample_size = int(np.round(metadata.join(features).groupby([grouping_var], 
                                                                    as_index=False).size().mean()))
    print("Mean sample size: %d" % mean_sample_size)

    # construct save paths (args.save_dir / topfeats? etc)
    save_dir = get_save_dir(args)
    stats_dir =  save_dir / grouping_var / "Stats" / args.fdr_method
    plot_dir = save_dir / grouping_var / "Plots" / args.fdr_method              

    # F-test for equal variances

    # Compare variance in samples with control (and correct for multiple comparisons)
    # Sample size matters in that unequal variances don't pose a problem for a t-test with 
    # equal sample sizes. So as long as your sample sizes are equal, you don't have to worry about 
    # homogeneity of variances. If they are not equal, perform F-tests first to see if variance is 
    # equal before doing a t-test
    if args.f_test:
        levene_stats_path = stats_dir / 'levene_results.csv'
        levene_stats = levene_f_test(features, metadata, grouping_var, 
                                      p_value_threshold=args.pval_threshold, 
                                      multitest_method=args.fdr_method,
                                      saveto=levene_stats_path,
                                      del_if_exists=False)
        # if p < 0.05 then variances are not equal, and sample size matters
        prop_eqvar = (levene_stats['pval'] > args.pval_threshold).sum() / len(levene_stats['pval'])
        print("Percentage equal variance %.1f%%" % (prop_eqvar * 100))
          
    # ANOVA / Kruskal-Wallis tests for significantly different features across groups

    test_path_unncorrected = stats_dir / '{}_results_uncorrected.csv'.format(args.test)
    test_path = stats_dir / '{}_results.csv'.format(args.test)
    
    if not (test_path.exists() and test_path_unncorrected.exists()):
        test_path.parent.mkdir(exist_ok=True, parents=True)
    
        if (args.test == "ANOVA" or args.test == "Kruskal"):
            if len(strain_list) > 2:   
                # perform ANOVA + record results before & after correcting for multiple comparisons               
                stats, pvals, reject = univariate_tests(X=features, 
                                                        y=metadata[grouping_var], 
                                                        control=control, 
                                                        test=args.test,
                                                        comparison_type='multiclass',
                                                        multitest_correction=None, # uncorrected
                                                        alpha=args.pval_threshold,
                                                        n_permutation_test=None)

                # get effect sizes
                effect_sizes = get_effect_sizes(X=features, 
                                                y=metadata[grouping_var],
                                                control=control,
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

                if len(fset) > 0:
                    print("%d significant features found by %s for '%s' (P<%.2f, %s)" % (len(fset), 
                          args.test, grouping_var, args.pval_threshold, args.fdr_method))
                    anova_sigfeats_path = stats_dir / '{}_sigfeats.txt'.format(args.test)
                    write_list_to_file(fset, anova_sigfeats_path)
            else:
                fset = []
                print("\nWARNING: Not enough groups for %s for '%s' (n=%d groups)" %\
                      (args.test, grouping_var, len(strain_list)))
                    
        # Linear Mixed Models (LMMs), accounting for day-to-day variation
        
        # NB: Ideally report:  parameter | beta | lower-95 | upper-95 | random effect (SD)
        elif args.test == 'LMM':
            with warnings.catch_warnings():
                # Filter warnings as parameter is often on the boundary
                warnings.filterwarnings("ignore")
                #warnings.simplefilter("ignore", ConvergenceWarning)
                (signif_effect, low_effect,  error, mask, pvals
                 ) = compounds_with_low_effect_univariate(feat=features, 
                                                          drug_name=metadata[grouping_var], 
                                                          drug_dose=None, 
                                                          random_effect=metadata[args.lmm_random_effect], 
                                                          control=control, 
                                                          test=args.test, 
                                                          comparison_type='multiclass',
                                                          multitest_method=args.fdr_method)
            assert len(error) == 0

            # save pvals
            pvals.to_csv(test_path_unncorrected, header=True, index=True)

            # save significant features -- if any strain significant for any feature
            fset = pvals.columns[(pvals < args.pval_threshold).any()].to_list()
            if len(fset) > 0:
                lmm_sigfeats_path = stats_dir / '{}_sigfeats.txt'.format(args.test)
                write_list_to_file(fset, lmm_sigfeats_path)

            # save significant effect strains
            if len(signif_effect) > 0:
                print(("%d significant features found (%d significant %ss vs %s control, "\
                      % (len(fset), len(signif_effect), grouping_var.replace('_',' '), 
                          control) if len(signif_effect) > 0 else\
                      "No significant differences found between %s "\
                      % grouping_var.replace('_',' '))
                      + "after accounting for %s variation, %s, P<%.2f, %s)"\
                      % (args.lmm_random_effect.split('_yyyymmdd')[0], args.test, 
                          args.pval_threshold, args.fdr_method))
                signif_effect_path = stats_dir / '{}_signif_effect_strains.txt'.format(args.test)
                write_list_to_file(signif_effect, signif_effect_path)
        
        else:
            raise IOError("Test '{}' not recognised".format(args.test))
    
    # t-tests / Mann-Whitney tests
    
    # t-test to use        
    t_test = 't-test' if args.test == 'ANOVA' else 'Mann-Whitney' # aka. Wilcoxon rank-sum      
    ttest_path_uncorrected = stats_dir / '{}_results_uncorrected.csv'.format(t_test)
    ttest_path = stats_dir / '{}_results.csv'.format(t_test)               

    if not (ttest_path_uncorrected.exists() and ttest_path.exists()):    
        ttest_path.parent.mkdir(exist_ok=True, parents=True)

        if len(fset) > 0 or len(strain_list) == 2:
            # perform t-tests (without correction for multiple testing)
            stats_t, pvals_t, reject_t = univariate_tests(X=features, 
                                                          y=metadata[grouping_var], 
                                                          control=control, 
                                                          test=t_test,
                                                          comparison_type='binary_each_group',
                                                          multitest_correction=None, 
                                                          alpha=0.05)
            # get effect sizes for comparisons
            effect_sizes_t =  get_effect_sizes(X=features, 
                                               y=metadata[grouping_var], 
                                               control=control,
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
            print("%d significant features found for any %s vs %s (%s, P<%.2f)" %\
                  (len(fset_ttest), grouping_var, control, t_test, args.pval_threshold))

            if len(fset_ttest) > 0:
                ttest_sigfeats_path = stats_dir / '{}_sigfeats.txt'.format(t_test)
                write_list_to_file(fset_ttest, ttest_sigfeats_path)
                                 
    # K significant features
    
    ksig_uncorrected_path = stats_dir / 'k_significant_features_uncorrected.csv'
    ksig_corrected_path = stats_dir / 'k_significant_features.csv'
    if not (ksig_uncorrected_path.exists() and ksig_corrected_path.exists()):
        ksig_corrected_path.parent.mkdir(exist_ok=True, parents=True)      
        fset_ksig, (scores, pvalues_ksig), support = k_significant_feat(feat=features, 
                                                                        y_class=metadata[grouping_var], 
                                                                        k=len(fset),
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
        # compile + save k-significant features (uncorrected) 
        ksig_table = pd.concat([pd.Series(scores), pd.Series(pvalues_ksig)], axis=1)
        ksig_table.columns = ['scores','pvals']
        ksig_table.index = fset_ksig
        ksig_table.to_csv(ksig_uncorrected_path, header=True, index=True)   
        
        # Correct for multiple comparisons
        _, ksig_table['pvals'] = _multitest_correct(ksig_table['pvals'], 
                                                    multitest_method=args.fdr_method,
                                                    fdr=args.pval_threshold)
        
        # save k-significant features (corrected)
        ksig_table.to_csv(ksig_corrected_path, header=True, index=True)   

    # mRMR feature selection: minimum Redunduncy, Maximum Relevance #####
    
    mrmr_dir = plot_dir / 'mrmr'
    mrmr_dir.mkdir(exist_ok=True, parents=True)
    mrmr_results_path = mrmr_dir / "mrmr_results.csv"

    if not mrmr_results_path.exists():
        estimator = Pipeline([('scaler', StandardScaler()), ('estimator', LogisticRegression())])
        y = metadata[grouping_var].values
        (mrmr_feat_set, 
          mrmr_scores, 
          mrmr_support) = mRMR_feature_selection(features, y_class=y, k=10,
                                                redundancy_func='pearson_corr',
                                                relevance_func='kruskal',
                                                n_bins=10, mrmr_criterion='MID',
                                                plot=True, k_to_plot=5, 
                                                close_after_plotting=True,
                                                saveto=mrmr_dir, figsize=None)
        # save results                                        
        mrmr_table = pd.concat([pd.Series(mrmr_feat_set), pd.Series(mrmr_scores)], axis=1)
        mrmr_table.columns = ['feature','score']
        mrmr_table.to_csv(mrmr_results_path, header=True, index=False)
        
        n_cv = 5
        cv_scores_mrmr = cross_val_score(estimator, features[mrmr_feat_set], y, cv=n_cv)
        cv_scores_mrmr = pd.DataFrame(cv_scores_mrmr, columns=['cv_score'])
        cv_scores_mrmr.to_csv(mrmr_dir / "cv_scores.csv", header=True, index=False)        
        print('MRMR CV Score: %f (n=%d)' % (np.mean(cv_scores_mrmr), n_cv))        
    else:
        # load mrmr results
        mrmr_table = pd.read_csv(mrmr_results_path)
        
    mrmr_feat_set = mrmr_table['feature'].to_list()
    print("\nTop %d features found by MRMR:" % len(mrmr_feat_set))
    for feat in mrmr_feat_set:
        print(feat)
        
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
        
    # Read feature summaries + metadata
    features = pd.read_csv(FEATURES_PATH)
    metadata = pd.read_csv(METADATA_PATH, dtype={'comments':str, 'source_plate_id':str})
    
    # Subset for desired imaging dates
    if args.dates is not None:
        assert type(args.dates) == list
        metadata = metadata.loc[metadata['date_yyyymmdd'].astype(str).isin(args.dates)]
        features = features.reindex(metadata.index)

    keio_stats(features, metadata, args)
    
    toc = time()
    print("\nDone in %.1f seconds (%.1f minutes)" % (toc - tic, (toc - tic) / 60))
    
