#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT: ON/OFF FOOD

A script written to read in trajectory data for food choice assay videos, filter the data to remove 
noise, and evaluate whether worms are on or off food in each frame of the assay. A presence/absence 
truth matrix (ie. on food vs not on food) is appended to the trajectory data for each video and is 
saved to file.       

@author: sm5911
@date: 25/03/2019

"""

# TODO: Make this a function and if name == main it

# GENERAL IMPORTS / DEPENDENCIES
import os, time
import pandas as pd

# CUSTOM IMPORTS
from _deprecated.find import change_path_phenix
from read_data.read import get_trajectory_data
from filter_data.filter_trajectories import filter_worm_trajectories
from food_choice_assay.food_choice_helper import onfood

#%% Globals

PROJECT_ROOT_DIR = '/Volumes/behavgenom$/Saul/FoodChoiceAssay/' # Project working directory
DATA_DIR = PROJECT_ROOT_DIR.replace('Saul', 'Priota/Data') # Location of features files

# Filter parameters
threshold_time = 25 # threshold trajectory length (n frames) - Recorded at 25 frames per second => ~1 second, for 2hrs (~180,000 frames)
threshold_move = 10 # threshold movement (n pixels) - 1 pixel = 10 microns => ~100 microns minimum movement

# Conduct analysis on new videos only?
NEW = True

#%% PRE-AMBLE
# Read metadata
fullMetaData = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "fullmetadata.csv"), header=0, index_col=0)

if NEW:
    fullMetaData = fullMetaData[fullMetaData['worm number']==10]

n_files = len(fullMetaData['filename'])
if NEW:
    print("%d NEW video file entries found in metadata." % n_files)
else:
    print("%d video file entries found in metadata." % n_files)

#%% ON/OFF FOOD (TRUTH MATRIX)
# - Read in food region coordinates + features file trajectory data for each video
# - Threshold by time/movement to filter bad features file data ('fake worms')
# - Calculate the proportion of worms present/absent in each food region, in each frame

# Error-handling
errorlog = 'Errorlog_OnFood.txt'
FAIL = []

tic = time.time()
for i, maskedfilepath in enumerate(fullMetaData['filename']):
    toc = time.time()
    # Extract file information
    file_info = fullMetaData.iloc[i,:]
    date = file_info['date(YEARMODA)']
    conc = file_info['Food_Conc']
    assaytype = file_info['Food_Combination']
    prefed = file_info['Prefed_on']
    print("\nProcessing file: %d\n%s\nAssay:  %s\nConc:   %.3f\nPrefed: %s" % (i + 1,\
          maskedfilepath, assaytype, conc, prefed))
    
    # Specify file paths
    coordfilepath = change_path_phenix(maskedfilepath, returnpath='coords')
    featurefilepath = change_path_phenix(maskedfilepath, returnpath='features')
    onfoodpath = change_path_phenix(maskedfilepath, returnpath='onfood')
        
    # Read food coordinates
    f = open(coordfilepath, 'r').read()
    poly_dict = eval(f) # Use 'evaluate' to read as dictionary, not string
    
    # Calculate on/off food truth matrix for filtered trajectory data
    try:
        # Read trajectory data
        traj_df = get_trajectory_data(featurefilepath)
        
        # Perform filtering step based on thresholds of trajectory movement/persistence over time
        traj_df = filter_worm_trajectories(traj_df, threshold_move, threshold_time)
        
        # Compute on/off food + append to trajectory data
        onfood_df = onfood(poly_dict, traj_df)
        
        # Save on/off food results
        directory = os.path.dirname(onfoodpath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        onfood_df.to_csv(onfoodpath)
        print("ON/OFF food results saved to file.\n(Time taken: %d seconds.)\n" % (time.time() - toc))
    except Exception as e:
        FAIL.append(maskedfilepath)
        print(e)
print("Complete!\n(Total time taken: %d seconds.)\n" % (time.time() - tic))

# If errors, save error log to file
if FAIL:
    fid = open(os.path.join(PROJECT_ROOT_DIR, errorlog), 'w')
    print(FAIL, file=fid)
    fid.close()
    

