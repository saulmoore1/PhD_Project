#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT: ON/OFF FOOD

A script written to evaluate whether a each set of worm centroid x,y coordinates
in Tierpsy-generated trajectory data fall inside/outside of the coordinates of 
labelled food regions in the assay. Trajectory data is filtered to remove 'fake'
worms (by spatiotemporal thresholding), and a presence/absence truth matrix 
(ie. on food vs not on food) generated and saved to file.       

@author: sm5911
@date: 25/03/2019

"""

# GENERAL IMPORTS / DEPENDENCIES
import os, time#, sys
import pandas as pd

# CUSTOM IMPORTS
#sys.path.insert(0, '/Users/sm5911/Documents/GitHub/PhD_Project/Python') # OPTIONAL: Path to GitHub functions
from Read import gettrajdata
from Find import findworms, changepath
from Calculate import onfood

#%% PREAMBLE

# Global variables
PROJECT_ROOT_DIR = '/Volumes/behavgenom$/Saul/FoodChoiceAssay/' # Project working directory
DATA_DIR = PROJECT_ROOT_DIR.replace('Saul', 'Priota/Data') # Location of features files

# Filter parameters
threshold_time = 25 # threshold trajectory length (n frames) - Recorded at 25 frames per second => ~1 second, for 2hrs (~180,000 frames)
threshold_move = 10 # threshold movement (n pixels) - 1 pixel = 10 microns => ~100 microns minimum movement

# Conduct analysis on new videos only?
NEW = True

# Read metadata
fullMetaData = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "fullmetadata.csv"), header=0, index_col=0)

if NEW:
    fullMetaData = fullMetaData[fullMetaData['worm number']==10]

n_files = len(fullMetaData['filename'])
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
    coordfilepath = changepath(maskedfilepath, returnpath='coords')
    featurefilepath = changepath(maskedfilepath, returnpath='features')
    onfoodpath = changepath(maskedfilepath, returnpath='onfood')
        
    # Read food coordinates
    f = open(coordfilepath, 'r').read()
    poly_dict = eval(f) # Use 'evaluate' to read as dictionary, not string
    
    # Calculate on/off food truth matrix for filtered trajectory data
    try:
        # Read trajectory data
        traj_df = gettrajdata(featurefilepath)
        
        # Perform filtering step based on thresholds of trajectory movement/persistence over time
        traj_df = findworms(traj_df, threshold_move, threshold_time)
        
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
