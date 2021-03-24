#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare control behaviour over time

Stats:
    ANOVA/Kruskal - for significant features in control across day/rig/plate/well
    t-test/ranksum - for significant features between each day/rig/plate/well and a control for each
    Linear mixed models - for significant features in control across days, accounting for 
                          rig/plate/well variation
    
Plots:
    Boxplots of significant features by ANONVA/Kruskal/LMM
    Boxplots of significant features by t-test/ranksum
    Heatmaps of control across day/rig/plate/well
    PCA/tSNE/UMAP of control across day/rig/plate/well

@author: saul.moore11@lms.mrc.ac.uk
@date: 09/02/2021
"""

#%% Imports

import argparse
from pathlib import Path
from matplotlib import pyplot as plt

# Custom imports
from filter_data.clean_feature_summaries import clean_summary_results
from visualisation.super_plots import superplot

from tierpsytools.read_data.hydra_metadata import read_hydra_metadata
from tierpsytools.hydra.platechecker import fix_dtypes

#%% Globals

EXAMPLE_METADATA_PATH = "/Volumes/hermes$/Keio_Tests_96WP/AuxiliaryFiles/metadata_annotated.csv"
EXAMPLE_RESULTS_DIR = "/Volumes/hermes$/Keio_Tests_96WP/Results"
EXAMPLE_FEATURE_LIST = ['speed_50th']

IMAGING_RUN = 3

# Mapping stimulus order for plotting
STIMULUS_DICT = {'prestim' : 0, 
                 'bluelight' : 1, 
                 'poststim' : 2}

CUSTOM_STYLE = 'visualisation/style_sheet_20210126.mplstyle'
DODGE = True

#%% Main

# TODO: Investigate control variation module of helper functions for plotting PCA by day, 
#       rig, well & temperature/humidity during/across runs
        
# TODO: Update timeseries plots for plotting windowed summaries
# Making more summaries and handling windows is a pain, no tierpsytools functions exist yet
# Luigi wrote something for Ida but is not ready for tierpsytools
# "Unless Andre is specifically telling you to look at windowed summaries, you should not 
#  go down the rabbit hole" - Luigi

if __name__ == '__main__':
    # Accept feature list from command line
    parser = argparse.ArgumentParser(description='Time-series analysis of selected features')
    parser.add_argument("--compiled_metadata_path", help="Path to compiled metadata file",
                        default=EXAMPLE_METADATA_PATH)
    parser.add_argument("--results_dir", help="Path to 'Results' directory containing full features\
                        and filenames summary files", default=EXAMPLE_RESULTS_DIR)
    parser.add_argument('--feature_list', help="List of selected features for time-series analysis", 
                        nargs='+', default=EXAMPLE_FEATURE_LIST)
    args = parser.parse_args()

    assert Path(args.compiled_metadata_path).exists()
    assert Path(args.results_dir).is_dir()
    assert type(args.feature_list) == list

    combined_feats_path = Path(args.results_dir) / "full_features.csv"
    combined_fnames_path = Path(args.results_dir) / "full_filenames.csv"
    
    # Ensure align bluelight is False
    # NB: leaves the df in a "long format" that seaborn likes    
    features, metadata = read_hydra_metadata(feat_file=combined_feats_path,
                                             fname_file=combined_fnames_path,
                                             meta_file=args.compiled_metadata_path,
                                             add_bluelight=True)

    # Convert metadata column dtypes, ie. stringsAsFactors, no floats, Δ, etc
    metadata = fix_dtypes(metadata)
    metadata['food_type'] = [f.replace("Δ","_") for f in metadata['food_type']]
    
    features, metadata = clean_summary_results(features, metadata)
    
    # Find masked HDF5 video files
    print("%d selected features loaded." % len(args.feature_list))

    # # Subset data for given imaging run
    # from filter_data.clean_feature_summaries import subset_results
    # run_feats, run_meta = subset_results(features, metadata, 'imaging_run_number', [IMAGING_RUN])
    
    # Time-series plots of day/run variation for selected features
    variable_list = ['date_yyyymmdd', 'imaging_plate_id', 'imaging_run_number', 'instrument_name']
    for variable in variable_list:
        
        for feat in args.feature_list:
            
            # plate ID vs run number
            superplot(features, metadata, feat, 
                      x1="imaging_plate_id", x2='imaging_run_number',
                      plot_type='box', #show_points=True,
                      sns_colour_palettes=["plasma","viridis"], 
                      dodge=True, saveDir=None)
            plt.show(); plt.pause(5)
        
        # instrument name
        
        # well name
        
        
        
        