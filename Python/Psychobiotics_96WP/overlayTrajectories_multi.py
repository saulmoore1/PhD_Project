#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot 96-well Plate Trajectories - multiple plates, for the full list of 
filenames in a given Tierpsy filenames summary file.

@author: sm5911
@date: 24/06/2020
"""

import sys
from tqdm import tqdm
from pathlib import Path
import pandas as pd

# Path to Github helper functions (USER-DEFINED path to local copy of Github repo)
PATH_LIST = ['/Users/sm5911/Tierpsy_Versions/tierpsy-tools-python/',
             '/Users/sm5911/Documents/GitHub/PhD_Project/Python/Psychobiotics_96WP']
for sysPath in PATH_LIST:
    if sysPath not in sys.path:
        sys.path.insert(0, sysPath)
        
from overlayTrajectories import tile96well

#%%
# Directory to save results
outDir = Path('/Users/sm5911/OneDrive - Imperial College London/Psychobiotics/Experiments/Microbiome/plate_trajectories')

# Read Tierpsy feature/filenames summaries
filenames_path = Path('/Volumes/behavgenom$/Saul/MicrobiomeScreen96WP/Results/full_filenames.csv')
filenames_df = pd.read_csv(filenames_path, comment='#')
filenames_list = filenames_df[filenames_df['is_good']==True]['file_name']

#%%
filestem_list = []
featurefile_list = []
for fname in filenames_list:
    # obtain file stem
    filestem = Path(fname).parent.parent / Path(fname).parent.stem
    if filestem not in filestem_list:
        filestem_list.append(filestem)
        featurefile_list.append(fname)

for featurefilepath in tqdm(featurefile_list):
    tile96well(featurefilepath, saveDir=outDir)