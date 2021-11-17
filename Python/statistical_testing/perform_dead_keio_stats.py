#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perform statistics for Dead Keio strains experiment - Analyse effects of UV-killed vs live hit Keio
strains on C. elegans behaviour

For each strain, compare live vs dead for differential effects on behaviour
Compare each strain to control for significant differences, both alive and dead

- Significant features differing between dead/live bacteria for each strain by t-test

@author: sm5911
@date: 15/11/2021
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

JSON_PARAMETERS_PATH = "analysis/20211109_parameters_keio_dead.json"

CONTROL_STRAIN = 'wild_type'
STRAIN_COLNAME = 'gene_name'
CONTROL_TREATMENT = False
TREATMENT_COLNAME = 'dead'
    
#%% FUNCTIONS

def dead_keio_stats(features, metadata, args):
    """ Perform statistical analyses on dead Keio experiment results:
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

    print("\nInvestigating variation in worm behaviour on dead vs alive hit Keio strains")  

    # assert there will be no errors due to case-sensitivity
    assert len(metadata[STRAIN_COLNAME].unique()) == len(metadata[STRAIN_COLNAME].str.upper().unique())
    assert all(type(b) == np.bool_ for b in metadata[TREATMENT_COLNAME].unique())
    
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
    assert CONTROL_STRAIN in strain_list

    # print mean sample size
    sample_size = df_summary_stats(metadata, columns=[STRAIN_COLNAME, TREATMENT_COLNAME])
    print("Mean sample size of %s: %d" % (STRAIN_COLNAME, int(sample_size['n_samples'].mean())))
    
    # construct save paths (args.save_dir / topfeats? etc)
    save_dir = get_save_dir(args)
    stats_dir =  save_dir / "Stats" / args.fdr_method

    for strain in strain_list:                                   
        strain_meta = metadata[metadata[STRAIN_COLNAME]==strain]
        strain_feat = features.reindex(strain_meta.index)
                     
        ### t-tests for each feature comparing live vs dead behaviour
 
        # t-test to use        
        t_test = 't-test'     
        ttest_path_uncorrected = stats_dir / '{}_uncorrected.csv'.format((t_test + '_' + strain))
        ttest_path = stats_dir / '{}_results.csv'.format((t_test + '_' + strain))  
        ttest_path.parent.mkdir(exist_ok=True, parents=True)

        # perform t-tests (without correction for multiple testing)
        stats_t, pvals_t, reject_t = univariate_tests(X=strain_feat, 
                                                      y=strain_meta[TREATMENT_COLNAME], 
                                                      control=CONTROL_TREATMENT, 
                                                      test=t_test,
                                                      comparison_type='binary_each_group',
                                                      multitest_correction=None, 
                                                      alpha=0.05)
        # get effect sizes for comparisons
        effect_sizes_t =  get_effect_sizes(X=strain_feat, 
                                           y=strain_meta[TREATMENT_COLNAME], 
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

    dead_keio_stats(features, metadata, args)
    
    toc = time()
    print("\nDone in %.1f seconds (%.1f minutes)" % (toc - tic, (toc - tic) / 60))
