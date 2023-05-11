#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse Keio antioxidant rescue experiment results

Please run the following scripts beforehand:
1. preprocessing/compile_keio_results.py
2. statistical_testing/perform_antioxidant_rescue_stats.py


THESE EXPERIMENTS FAILED: WILL REDO THEM WITH A LOWER CONCENTRATION OF ANTIOXIDANT

@author: sm5911
@date: 13/11/2021
"""

#%% IMPORTS

import argparse
import pandas as pd
from time import time
from pathlib import Path
# import numpy as np
# import seaborn as sns
# from tqdm import tqdm
# from matplotlib import pyplot as plt
# from matplotlib import transforms
# from scipy.stats import zscore

from read_data.paths import get_save_dir
from read_data.read import load_json
from visualisation.plotting_helper import boxplots_sigfeats
from write_data.write import write_list_to_file
from visualisation.plotting_helper import sig_asterix
from time_series.plot_timeseries import selected_strains_timeseries
# from clustering.hierarchical_clustering import plot_clustermap, plot_barcode_heatmap
# from feature_extraction.decomposition.pca import plot_pca, remove_outliers_pca
# from feature_extraction.decomposition.tsne import plot_tSNE
# from feature_extraction.decomposition.umap import plot_umap
# from analysis.keio_screen.initial.run_keio_analysis import COG_category_dict

from tierpsytools.preprocessing.filter_data import select_feat_set
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes


#%% GLOBALS

JSON_PARAMETERS_PATH = "analysis/20211102_parameters_keio_antioxidant_rescue.json"

STRAIN_COLNAME = 'gene_name'
TREATMENT_COLNAME = 'antioxidant'

feature_set = ['speed_50th_bluelight'] # motion_mode_forward_fraction_bluelight

METHOD = 'complete' # 'complete','linkage','average','weighted','centroid'
METRIC = 'euclidean' # 'euclidean','cosine','correlation'

BLUELIGHT_TIMEPOINTS_SECONDS = [(60, 70),(160, 170),(260, 270)]

#%% FUNCTIONS

def antioxidant_boxplots(metadata,
                         features,
                         control,
                         group_by='treatment',
                         stats_dir=None,
                         save_dir=None,
                         feature_set=None,
                         pvalue_threshold=0.05,
                         fdr_method='fdr_by'):
    
    feature_set = features.columns.tolist() if feature_set is None else feature_set
    assert isinstance(feature_set, list) and all(f in features.columns for f in feature_set)
                    
    # load t-test results for window
    if stats_dir is not None:
        ttest_path = Path(stats_dir) / 't-tests' / 't-test_results.csv'
        ttest_df = pd.read_csv(ttest_path, header=0, index_col=0)
        pvals = ttest_df[[c for c in ttest_df.columns if 'pval' in c]]
        pvals.columns = [c.replace('pvals_','') for c in pvals.columns]
    
    boxplots_sigfeats(features,
                      y_class=metadata[group_by],
                      control=control,
                      pvals=pvals if stats_dir is not None else None,
                      z_class=None,
                      feature_set=feature_set,
                      saveDir=Path(save_dir),
                      drop_insignificant=True if feature_set is None else False,
                      p_value_threshold=pvalue_threshold,
                      scale_outliers=True)
    
    return


def antioxidant_rescue_stats(metadata,
                             features,
                             group_by='treatment',
                             control='wild_type-None',
                             save_dir=None,
                             pvalue_threshold=0.05,
                             fdr_method='fdr_by'):
        
    fset = []
    n = len(metadata[group_by].unique())
    
    # Perform ANOVA - is there any variation among treatment groups
    if n > 2:
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
            print("%d significant features found by ANOVA for '%s' (P<%.2f, %s)" %\
                  (len(fset), group_by, pvalue_threshold, fdr_method))
            anova_sigfeats_path = anova_path.parent / 'ANOVA_sigfeats.txt'
            write_list_to_file(fset, anova_sigfeats_path)

    # perform t-tests comparing each strain to control
    if n == 2 or len(fset) > 0:
        ttest_path = Path(save_dir) / 't-tests' / 't-test_results.csv'
        ttest_path.parent.mkdir(parents=True, exist_ok=True)
        
        stats_t, pvals_t, reject_t = univariate_tests(X=features,
                                                      y=metadata[group_by],
                                                      control=control,
                                                      test='t-test',
                                                      comparison_type='binary_each_group',
                                                      multitest_correction=fdr_method,
                                                      alpha=pvalue_threshold,
                                                      n_permutation_test=None)
        
        effect_sizes_t = get_effect_sizes(X=features, 
                                          y=metadata[group_by],
                                          control=control,
                                          effect_type=None,
                                          linked_test='t-test')
        
        # compile results
        stats_t.columns = ['stats_' + str(c) for c in stats_t.columns]
        pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
        reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
        effect_sizes_t.columns = ['effect_size_' + str(c) for c in effect_sizes_t.columns]
        test_results = pd.concat([stats_t, pvals_t, reject_t, effect_sizes_t], axis=1)            
                        
        # save results
        test_results.to_csv(ttest_path, header=True, index=True)
            
        n_sig = sum(reject_t.sum(axis=1) > 0)
        print("%d significant features between any %s vs %s (t-test, P=%.2f, %s)" %\
              (n_sig, group_by, control, pvalue_threshold, fdr_method))

    return


def main():
    
    # Read clean feature summaries + metadata
    print("Loading metadata and feature summary results...")
    features = pd.read_csv(FEATURES_PATH)
    metadata = pd.read_csv(METADATA_PATH, dtype={'comments':str, 'source_plate_id':str})

    # Subset for desired imaging dates
    if args.dates is not None:
        assert type(args.dates) == list
        metadata = metadata.loc[metadata['date_yyyymmdd'].astype(str).isin(args.dates)]
        features = features.reindex(metadata.index)

    # Single feature only, or tierpsy feature set?
    if feature_set is not None:
        assert isinstance(feature_set, list) and all(f in features.columns for f in feature_set)
        features = features[feature_set]
    else:
        # Load Tierpsy feature set + subset (columns) for selected features only
        features = select_feat_set(features, 'tierpsy_{}'.format(args.n_top_feats), append_bluelight=True)
        features = features[[f for f in features.columns if 'path_curvature' not in f]]

    #compare_keio_rescue(features, metadata, args)
    metadata['treatment'] = metadata[['gene_name','antioxidant']].agg('-'.join, axis=1)
    control = args.control_dict['gene_name'] + '-' + args.control_dict['antioxidant']
            
    antioxidant_rescue_stats(metadata,
                             features,
                             group_by='treatment',
                             control=control,
                             save_dir=save_dir / 'Stats',
                             pvalue_threshold=args.pval_threshold,
                             fdr_method=args.fdr_method)
    
    antioxidant_boxplots(metadata,
                         features,
                         group_by='treatment',
                         control=control,
                         stats_dir=save_dir / 'Stats',
                         save_dir=save_dir / 'Plots',
                         feature_set=feature_set,
                         pvalue_threshold=args.pval_threshold,
                         fdr_method=args.fdr_method)

    selected_strains_timeseries(metadata, 
                                project_dir=Path(args.project_dir), 
                                save_dir=Path(args.save_dir) / 'timeseries', 
                                group_by='treatment',
                                control='wild_type-None',
                                strain_list=None,
                                n_wells=96,
                                bluelight_stim_type='bluelight',
                                video_length_seconds=6*60,
                                bluelight_timepoints_seconds=[(60,70),(160,170),(260,270)],
                                motion_modes=['forwards','stationary','backwards'],
                                smoothing=10)

    toc = time()
    print("\nDone in %.1f seconds (%.1f minutes)" % (toc - tic, (toc - tic) / 60))  
    
    return

    
#%% MAIN
if __name__ == "__main__":
    tic = time()
    parser = argparse.ArgumentParser(description="Read clean features and etadata and find 'hit' \
                                                  Keio knockout strains that alter worm behaviour")
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
    save_dir = get_save_dir(args)
    
    if FEATURES_PATH is None:
        FEATURES_PATH = Path(args.save_dir) / 'features.csv'
    if METADATA_PATH is None:
        METADATA_PATH = Path(args.save_dir) / 'metadata.csv'
     
    main(args)
    