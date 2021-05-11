#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse Keio Screen results

@author: sm5911
@date: 19/04/2021

"""

#%% IMPORTS
import argparse
import pandas as pd
from time import time
from pathlib import Path
# from scipy.stats import zscore # ttest_ind, f_oneway, kruskal

from read_data.read import load_json, load_top256
# from write_data.write import write_list_to_file
from preprocessing.compile_hydra_data import process_metadata, process_feature_summaries
# from filter_data.clean_feature_summaries import clean_summary_results, subset_results
# from statistical_testing.stats_helper import shapiro_normality_test
# from feature_extraction.decomposition.pca import plot_pca, remove_outliers_pca
# from feature_extraction.decomposition.tsne import plot_tSNE
# from feature_extraction.decomposition.umap import plot_umap
# from feature_extraction.decomposition.hierarchical_clustering import (plot_clustermap, 
#                                                                       plot_barcode_heatmap)
# from visualisation.super_plots import superplot
# from visualisation.plotting_helper import (sig_asterix, 
#                                            #plot_day_variation, 
#                                            barplot_sigfeats, 
#                                            boxplots_sigfeats,
#                                            boxplots_grouped)

from tierpsytools.hydra.platechecker import fix_dtypes
# from tierpsytools.analysis.significant_features import k_significant_feat
# from tierpsytools.analysis.statistical_tests import univariate_tests
# from tierpsytools.drug_screenings.filter_compounds import compounds_with_low_effect_univariate

#%% GLOBALS
JSON_PARAMETERS_PATH = "analysis/20210406_parameters_keio_screen.json"

#%% MAIN
def main(args):
    
    assert args.project_dir is not None
    AUX_DIR = Path(args.project_dir) / "AuxiliaryFiles"
    RESULTS_DIR = Path(args.project_dir) / "Results"
    
    # Update save path according to JSON parameters for features to use
    fn = 'Top256' if args.use_top256 else 'All_features'
    fn = fn + '_noSize' if args.drop_size_features else fn
    fn = fn + '_norm' if args.norm_features_only else fn
    fn = fn + '_' + args.percentile_to_use if args.percentile_to_use is not None else fn
    fn = fn + '_noOutliers' if args.remove_outliers else fn

    SAVE_DIR = (Path(args.save_dir) if args.save_dir is not None else Path(args.project_dir) / 
                "Analysis") / fn / (args.grouping_variable + '_variation')
    
    GROUPING_VAR = args.grouping_variable # categorical variable to investigate, eg.'worm_strain'
    
    CONTROL = args.control_dict[GROUPING_VAR] # control strain to use

    IS_NORMAL = args.is_normal 
    # Perform t-tests/ANOVA if True, ranksum/kruskal if False. If None, Shapiro-Wilks tests for 
    # normality are perofrmed to decide between parametric/non-parametric tests

    TEST_NAME = args.test # str, Choose between 'LMM' (if >1 day replicate), 'ANOVA' or 'Kruskal' 
    # Kruskal tests are performed instead of ANOVA if check_normal and data is not normally distributed) 
    # If significant features are found, pairwise t-tests are performed

    #%% Compile results
        
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
    
    # fix data types
    metadata = fix_dtypes(metadata) 

    
    stats_dir =  SAVE_DIR / "Stats"
    plot_dir = SAVE_DIR / "Plots"
    
#%%
if __name__ == "__main__":
    tic = time()
    parser = argparse.ArgumentParser(description="Find 'hit' Keio knockout strains that alter worm \
                                     behaviour")
    parser.add_argument('-j', '--json', help="Path to JSON parameters file for analysis",
                        default=JSON_PARAMETERS_PATH, type=str)
    args = parser.parse_args()
    
    args = load_json(args.json)
    
    main(args)
    
    toc = time()
    print("\nDone in %.1f seconds (%.1f minutes)" % (toc-tic, (toc-tic)/60))
