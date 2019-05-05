#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT: LEAVING RATES

A script written to analyse the food choice assay videos and Tierpsy results, 
and invesitgate worm velocities and leaving rates across food patches.

@author: sm5911
@date: 29/04/2019

"""

# Imports
import os
import numpy as np
import pandas as pd

# Custom imports
from Find import changepath
from Read import getskeldata

#%% PRE-AMBLE
# GLOBAL VARIABLES
PROJECT_ROOT_DIR = '/Volumes/behavgenom$/Saul/FoodChoiceAssay/' # Project working directory
DATA_DIR = PROJECT_ROOT_DIR.replace('Saul', 'Priota/Data') # Location of features files

# Conduct analysis on new videos only?
NEW = True

# Read metadata
fullMetaData = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "fullmetadata.csv"), header=0, index_col=0)

if NEW:
    fullMetaData = fullMetaData[fullMetaData['worm number']==10]

n_files = len(fullMetaData['filename'])
print("%d video file entries found in metadata." % n_files)

# Extract assay information
pretreatments = list(np.unique(fullMetaData['Prefed_on']))
assaychoices = list(np.unique(fullMetaData['Food_Combination']))
treatments = list(np.unique([assay.split('/') for assay in assaychoices]))
concentrations = list(np.unique(fullMetaData['Food_Conc']))

#%% # TODO: Investigate velocity before/after leaving events

for i, maskedfilepath in enumerate(fullMetaData['filename']):
    # Extract file information
    info = fullMetaData.iloc[i,:]
    conc = info['Food_Conc']
    assaychoice = info['Food_Combination']
    prefed = info['Prefed_on']
    foods = info['Food_Combination'].split('/')
    if foods[0] == foods[1]:
        foods = ["{}_{}".format(food, i + 1) for i, food in enumerate(foods)]
    print("\nProcessing file: %d\n%s\nAssay:  %s\nConc:   %.3f\nPrefed: %s"\
          % (i + 1, maskedfilepath, assaychoice, conc, prefed))
    
    skeletonfilepath = changepath(maskedfilepath, returnpath="skeletons")
    
    skeleton_data = getskeldata(skeletonfilepath)

