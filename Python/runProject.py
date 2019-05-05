#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT: RUN FOOD CHOICE ASSAY

A script written to execute the food choice assay reproducible analysis workflow. 
- Collates video information and saves to file.
- Manual labelling
- On/off food
- Food choice
- Leaving events/rate
- Checks that all results files have been saved successfully + cleans up workflow 
  to remove erroneous files.

@author: sm5911
@date: 21/03/2019

"""

# GENERAL IMPORTS / DEPENDENCIES
import os, time, re#, sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, date

# CUSTOM IMPORTS
#sys.path.insert(0, '/Users/sm5911/Documents/GitHub/PhD_Project/Python') # OPTIONAL: Path to GitHub functions
from Find import lookforfiles
from Read import getauxinfo
from Plot import manuallabelling, wormtrajectories

# NB: Could create a config.yaml file to specify global variables for filter/crop/plot params such as thresholds, windows, bin sizes, etc?

#%% PREAMBLE

# GLOBAL VARIABLES
PROJECT_ROOT_DIR = '/Volumes/behavgenom$/Saul/FoodChoiceAssay/' # Project working directory
DATA_DIR = PROJECT_ROOT_DIR.replace('Saul', 'Priota/Data') # Location of data: RawVideos + Tierpsy results

# Find all masked video files in the data directory
maskedfilelist = lookforfiles(os.path.join(DATA_DIR, 'MaskedVideos'), '.*.hdf5$')
print("%d masked video files found." % len(maskedfilelist))

#%% PREPROCESSING - Compile metadata and auxiliary file info

# Read metadata file
metafilepath = os.path.join(DATA_DIR, "AuxiliaryFiles", "metadata.csv")
metaData = pd.read_csv(metafilepath)

# Retrieve filenames for missing entries in metadata
for i, filepath in enumerate(metaData['filename']):
    if isinstance(filepath, str):
        # If filepath exists, make sure it contains no spaces
        metaData.loc[i,'filename'] = filepath.replace(" ", "")
    else:
        # Extract filepath information for regex search to pair filenames with their metadata
        print("WARNING: Filename missing, searching for matching video in data directory..")
        file_info = metaData.iloc[i]
        Date = str(file_info['date(YEARMODA)'])
        Set = str(file_info['set_number'])
        Camera = str(file_info['channel'])
        
        # Re-format date + form string to query by regex from date + set + channel info
        d = datetime.datetime.strptime(Date, '%Y%m%d')
        Date = d.strftime('%d%m%Y')
        querystring = '/Set{0}'.format(Set) + '/Set' + Set + '_Ch' + Camera + '_' + Date + '_'
        
        # Find matching filename in masked video list
        for file in maskedfilelist:           
            if re.search(querystring, file):
                print("Match found! Filename added.")
                metaData.loc[i,'filename'] = file
                
# Subset metadata to remove entries with missing filenames that could not be retrieved
is_filename = [isinstance(path, str) for path in metaData['filename']]
if any(list(~np.array(is_filename))):
    print("WARNING: Filenames could not be found for %d entries in metadata.\nThese files will be omitted from further analyses!"\
          % sum(list(~np.array(is_filename))))
    metaData = metaData[list(np.array(is_filename))].reset_index(drop=True)

# Check for errors
if not len(np.unique(metaData.filename)) == len(metaData.filename):
    print("ERROR: Duplicate filenames found!")
    
# Pre-allocate datframe for storing auxiliary file info
out_columns = ['Food_Conc','Food_Combination','Food_Marker',\
               'Pick','Pick_time','Image_time','Pick_type']
out_df = pd.DataFrame(index=metaData.index, columns=out_columns)

# For each file, locate auxiliary file + grab info
for i, filepath in enumerate(metaData['filename']): 
    meta = metaData[metaData['filename']==filepath] 
    aux_info = getauxinfo(filepath, sheet=0)
    aux_info.index = meta.index # set indices to align row slicing
    out_df.iloc[meta.index,:] = aux_info.loc[:,out_columns].values # insert values into sliced row in out-df
        
# Append auxiliary information to metadata
fullMetaData = pd.concat([metaData, out_df], axis=1, ignore_index=False)

# Extract + append pre-feeding info
prefed_list = []
for i, filepath in enumerate(metaData['filename']):
    prefed_info = getauxinfo(filepath, sheet=1)
    prefed_info = str(prefed_info[0]).upper()
    prefed_info = re.split('PREFED ON: ', prefed_info)[1]
    prefed_list.append(prefed_info)
fullMetaData['Prefed_on'] = prefed_list

# Calculate acclimation period + append
acclimation_list = []
for i, pick_time in enumerate(fullMetaData['Pick_time']):
    picktime = datetime.strptime(str(pick_time), "%H:%M:%S").time()
    imagetime = datetime.strptime(str(fullMetaData['Image_time'][i]), "%H:%M:%S").time()
    acclimation_time = datetime.combine(date.min, imagetime) - datetime.combine(date.min, picktime)     
    acclimation_list.append(acclimation_time.total_seconds())
fullMetaData['Acclim_time_s'] = acclimation_list

# Save full metadata
print("Saving combined metadata/auxiliary dataframe..")
fullMetaData.to_csv(os.path.join(PROJECT_ROOT_DIR, "fullmetadata.csv"))
print("Done.")

# NB: Date '20181101' has no record in the 'metadata.csv' file. These 6 video files are omitted from subsequent analyses

#%% MANUAL LABELLING

# Find masked HDF5 video files (for labelling) 
print("Labelling and plotting trajectories for %d videos..." % len(maskedfilelist))

# Interactive plotting (for user input when labelling plots)
plt.ion()
tic = time.time()
for i in range(len(maskedfilelist)):    
    maskedfilepath = maskedfilelist[i]

    # Manually outline + assign labels to food regions
    # And save coordinates + trajectory overlay to file           
    manuallabelling(maskedfilepath, save=True, skip=True)
plt.ioff()
print("Manual labelling complete!\n(Time taken: %d seconds.)\n" % (time.time() - tic))

#%% Plot worm trajectory start/end points (unfiltered data)
    
tic = time.time()
for i, maskedfilepath in enumerate(fullMetaData['filename']):
    # Extract file information
    file_info = fullMetaData.iloc[i,:]
    date = file_info['date(YEARMODA)']
    conc = file_info['Food_Conc']
    assaytype = file_info['Food_Combination']
    prefed = file_info['Prefed_on']
    print("\nProcessing file: %d\n%s\nAssay:  %s\nConc:   %.3f\nPrefed: %s" % (i + 1,\
          maskedfilepath, assaytype, conc, prefed))
    
    # Plot + save worm trajectories (start/end points)
    wormtrajectories(maskedfilepath, downsample=1, save=True)
print("Plotting complete!\n(Time taken: %d seconds.)" % (time.time() - tic))

#%% ON/OFF FOOD

import OnFood

#%% FOOD CHOICE

import FoodChoice

#%% LEAVING RATE

import LeavingEvents

#%% CLEAN UP / REMOVE FILES

# Find files by regex to be removed
files_to_remove = lookforfiles(os.path.join(PROJECT_ROOT_DIR, 'Results', 'FoodChoice'), ".*_Summary.csv$")

tic = time.time()
print("Removing %d files.." % len(files_to_remove))
for file in files_to_remove:
    if os.path.exists(file):
        os.remove(file)
print("Done. (Time taken: %d seconds.)" % (time.time() - tic))

#%% A check that files exist & have saved properly:

# Filename list
regex_list = {"Plots": [".*_LabelledOverlayPlot.png$", ".*_FoodChoiceTS.png$", ".*_ViolinPlot.png$",\
                        ".*_PiePlot.png$", ".*_LeavingPlot.png$", ".*_WormTrajPlot.png$"],\
              "FoodCoords": [".*_FoodCoords.txt$"],\
              "FoodChoice": [".*_OnFood.csv$", ".*_FoodChoice_Mean.csv$",\
              ".*_FoodChoice_Count.csv$", ".*_FoodChoiceSummary_Mean.csv$",\
              ".*_FoodChoiceSummary_Count.csv$"],\
              "LeavingRate": [".*_LeavingEvents.csv$"]}

# Check files in filename list
for folder, items in regex_list.items():
    for item in items:
        files = []
        files = lookforfiles(os.path.join(PROJECT_ROOT_DIR, 'Results', folder), item)
        print("Number of %s files found: %d" % (item.split(".*_")[-1].split(".")[0], len(files)))
