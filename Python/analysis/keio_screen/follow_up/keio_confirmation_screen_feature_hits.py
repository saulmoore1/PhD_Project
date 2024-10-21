#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of confirmation screen data to identify gene-deletion mutant E. coli strains that elicit 
significant behavioural differences in N2 worms compared to BW control bacteria. This script 
performs a cluster analysis to identify the strains with the biggest differences in terms of each 
feature

Andre: It might make more sense to just do this for a reduced set of features 
       (e.g. one feature from each of the obvious clusters in the hits heatmap)

@author: sm5911
@date: 21/10/2024

"""

#%% Imports

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from matplotlib import transforms
from matplotlib import pyplot as plt
from scipy.stats import zscore

from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from visualisation.plotting_helper import sig_asterix
from clustering.hierarchical_clustering import plot_clustermap
from time_series.plot_timeseries import plot_timeseries, get_strain_timeseries
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from tierpsytools.plot.plot_plate_from_raw_video import plot_plates_from_metadata
from tierpsytools.preprocessing.filter_data import select_feat_set

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Screen_Confirmation"
SAVE_DIR = "/Users/sm5911/Documents/PhD_DLBG/4_Keio_Screen_Confirmation/feature_hits_for_Filipe"

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
MIN_NSKEL_PER_VIDEO = None
MIN_NSKEL_SUM = 500
P_VALUE_THRESHOLD = 0.05
FDR_METHOD = 'fdr_bh'

FEATURE = 'speed_50th'
N_WELLS = 6
DPI = 600
FPS = 25

BLUELIGHT_TIMEPOINTS_SECONDS = [(60, 70),(160, 170),(260, 270)]
WINDOW_DICT = {0:(290,300)}
WINDOW_NAME_DICT = {0:"20-30 seconds after blue light 3"}

#%% Functions

#%%