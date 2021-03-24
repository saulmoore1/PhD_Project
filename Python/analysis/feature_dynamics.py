#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Selected feature dynamics

@author: sm5911
@date: 23/03/2021

"""

# TODO: Timeseries analysis of feature across timepoints/stimulus windows/on-off food/etc
# TODO: sns.relplot / sns.jointplot / sns.lineplot for visualising covariance/correlation
# between selected features
        
#%% Imports
import argparse

from read_data.read import read_list_from_file

#%% Globals    

EXAMPLE_PATH_FEATURE_LIST = ("/Users/sm5911/Documents/tmp_analysis/Keio/All_features_noSize_norm/"+
                             "Run_3/food_type_variation/Stats/Kruskal_significant_features.txt")

# Mapping stimulus order for plotting
STIMULUS_DICT = {'prestim' : 0, 
                 'bluelight' : 1, 
                 'poststim' : 2}

#%% Functions


#%% Main
if __name__ == "__main__":
    # Accept command-line inputs
    parser = argparse.ArgumentParser(description='Analyse selected feature dynamics')
    parser.add_argument('-p', '--path_feature_list', help=("Path to text file containing list of" + 
                        " selected features to analyse"),default=EXAMPLE_PATH_FEATURE_LIST,type=str)
    parser.add_argument('-f', '--feature_list', help=("List of features to analyse"), nargs="+", 
                        default=None, type=list)
    args = parser.parse_args()

    # Load selected feature set    
    if args.feature_list is not None:
        fset = list(args.feature_list)
    else:
        fset = read_list_from_file(args.path_feature_list)
    
    # feature sets for each stimulus type
    featsets = {}
    for stim in ['_prestim','_bluelight','_poststim']:
        featsets[stim] = [f for f in fset if stim in f]
