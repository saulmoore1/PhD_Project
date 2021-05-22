#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse Keio Screen results

@author: sm5911
@date: 19/04/2021
"""

#%% IMPORTS

import argparse
import numpy as np
import pandas as pd
from time import time
from pathlib import Path
# from scipy.stats import zscore # ttest_ind, f_oneway, kruskal

from read_data.read import load_json, load_top256
# from write_data.write import write_list_to_file
from filter_data.clean_feature_summaries import subset_results
# from statistical_testing.stats_helper import shapiro_normality_test
# from feature_extraction.decomposition.pca import plot_pca, remove_outliers_pca
# from feature_extraction.decomposition.tsne import plot_tSNE
# from feature_extraction.decomposition.umap import plot_umap
# from feature_extraction.decomposition.hierarchical_clustering import (plot_clustermap, 
#                                                                       plot_barcode_heatmap)
from visualisation.super_plots import superplot
# from visualisation.plotting_helper import (sig_asterix, 
#                                            plot_day_variation, 
#                                            barplot_sigfeats, 
#                                            boxplots_sigfeats,
#                                            boxplots_grouped)

# from tierpsytools.analysis.significant_features import k_significant_feat
# from tierpsytools.analysis.statistical_tests import univariate_tests
# from tierpsytools.drug_screenings.filter_compounds import compounds_with_low_effect_univariate

#%% GLOBALS

JSON_PARAMETERS_PATH = "analysis/20210406_parameters_keio_screen.json"
FEATURES_PATH = "/Users/sm5911/Documents/Keio_Screen/features.csv"
METADATA_PATH = "/Users/sm5911/Documents/Keio_Screen/metadata.csv"

#%% MAIN

def main(features, metadata, args):

    AUX_DIR = Path(args.project_dir) / "AuxiliaryFiles"
    
    GROUPING_VAR = args.grouping_variable # categorical variable to investigate, eg.'gene_name' / 'worm_strain'
    assert len(metadata[GROUPING_VAR].unique()) == len(metadata[GROUPING_VAR].str.upper().unique())
    print("\nInvestigating '%s' variation" % GROUPING_VAR)    
    
    # Subset results (rows) to omit selected strains
    if args.omit_strains is not None:
        features, metadata = subset_results(features, metadata, GROUPING_VAR, args.omit_strains)

    STRAIN_LIST = list(metadata[GROUPING_VAR].unique())
    
    CONTROL = args.control_dict[GROUPING_VAR] # control strain to use

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

    #%% Control variation
    
    control_dir = SAVE_DIR / "Control"
            
    # Subset results for control data
    control_metadata = metadata[metadata['source_plate_id']=='BW']
    control_features = features.reindex(control_metadata.index)
    
    # from analysis.control_variation.control_variation import control_variation
    # control_variation(control_features, 
    #                   control_metadata, 
    #                   variables=['date_yyyymmdd','instrument_name'], 
    #                   saveDir=control_dir)
                    
    #%% Load statistics results

    stats_dir =  SAVE_DIR / (args.grouping_variable) / "Stats"
    
    # Stats test to use
    assert args.test in ['ANOVA','Kruskal','LMM']
    if args.test == 'LMM':
        # If 'LMM' is chosen, ensure there are multiple day replicates to compare at each timepoint
        assert all(len(metadata.loc[metadata['imaging_run_number']==timepoint, 
                   args.lmm_random_effect].unique()) > 1 for timepoint in args.runs)
        
    T_TEST_NAME = 't-test' if args.test == 'ANOVA' else 'Mann-Whitney' # aka. Wilcoxon rank-sum

    stats_path = stats_dir / '{}_results.csv'.format(args.test) # LMM/ANOVA/Kruskal  
    ttest_path = stats_dir / '{}_results.csv'.format(T_TEST_NAME) #t-test/Mann-Whitney
                
    # Record mean sample size per group
    mean_sample_size = int(np.round(metadata.join(features).groupby([GROUPING_VAR], as_index=False).size().mean()))
    print("Mean sample size: %d" % mean_sample_size)
    
    # Read ANOVA results and record significant features
    print("Loading statistics results")
    if stats_path.exists():
        
        pval_stats = pd.read_csv(stats_path, index_col=0)
        pvals = pval_stats.sort_values(by=args.test, ascending=True)
        
        # Record significant features by ANOVA
        fset = pvals.index[pvals[args.test].values < args.pval_threshold].to_list()
        print("%d significant features found by %s (P<%.2f)" % (len(fset), args.test,
                                                                args.pval_threshold))
    
    # Read t-test results and record significant features
    if ttest_path.exists():
        pvals_t = pd.read_csv(ttest_path, index_col=0)
        assert all(f in pvals_t.columns for f in features.columns)
            
        # Record significant features by t-test
        fset_ttest = list(pvals_t.columns[(pvals_t < args.pval_threshold).sum(axis=0) > 0])
        
    #%% Plotting

    plot_dir = SAVE_DIR / (args.grouping_variable) / "Plots"
    superplot_dir = plot_dir / 'superplots'    
    
    for feat in fset[:5]:
        superplot(features, metadata, feat, x1='source_plate_id', x2='date_yyyymmdd', 
                  saveDir=None, show_points=False, plot_means=False)
    
        superplot(features, metadata, feat, x1='source_plate_id', x2='imaging_run_number', 
                  saveDir=None, show_points=False, plot_means=False)
        
        superplot(features, metadata, feat, x1='instrument_name', x2='imaging_run_number', 
                  saveDir=None, show_points=False, plot_means=False)

        # TODO: Add t-test/LMM pvalues to superplots!
            
        # # strain vs date yyyymmdd
        # superplot(features, metadata, feat, 
        #           x1=GROUPING_VAR, x2='date_yyyymmdd',
        #           plot_type='box', #show_points=True, sns_colour_palettes=["plasma","viridis"]
        #           dodge=True, saveDir=superplot_dir)

        # # plate ID vs run number
        # superplot(features, metadata, feat, 
        #           x1=GROUPING_VAR, x2='imaging_run_number',
        #           dodge=True, saveDir=superplot_dir)

#%%
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
    
    main(features, metadata, args)
    
    toc = time()
    print("\nDone in %.1f seconds (%.1f minutes)" % (toc-tic, (toc-tic)/60))
