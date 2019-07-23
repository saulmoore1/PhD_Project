#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT: RUN FOOD CHOICE ASSAY

A script written to execute the food choice assay reproducible analysis workflow,
and call all relevent scripts and import the necessary modules. 
The analysis workflow does the following:
- Collates video metadata + saves to file
- Read masked video file, accepts user input to manually label food regions + 
  saves coordinates to file
- Reads Tierpsy features file + plots worm trajectory data
- Computes whether worms are on or off food (using user-labelled coordinates)
- Calculates worm food choice preference in each video
- Calculates the number of times worms leave the food in each video
- # Investigates the rate of leaving over time + worm velocity
- # Determines worm locomotory state
- Checks that results files have been saved successfully + cleans up workflow

@author: sm5911
@date: 21/03/2019

"""

# GENERAL IMPORTS / DEPENDENCIES
import os, time, re#, sys
import pandas as pd
import numpy as np
import subprocess
from matplotlib import pyplot as plt
from datetime import datetime, date

# CUSTOM IMPORTS
#sys.path.insert(0, '/Users/sm5911/Documents/GitHub/PhD_Project/Python') # OPTIONAL: Path to GitHub functions
from Find import lookforfiles
from Read import getauxinfo
from Plot import manuallabelling, wormtrajectories

# NB: Could create a config file (.yaml?) to specify global variables for filter/crop/plot params such as thresholds, windows, bin sizes, etc?

#%% PREAMBLE

# GLOBAL VARIABLES
PROJECT_ROOT_DIR = '/Volumes/behavgenom$/Saul/FoodChoiceAssay/' # Project working directory
DATA_DIR = PROJECT_ROOT_DIR.replace('Saul', 'Priota/Data') # Location of data: RawVideos, MaskedVideos + Tierpsy results

# Find all masked video files in the data directory
maskedfilelist = lookforfiles(os.path.join(DATA_DIR, 'MaskedVideos'), '.*.hdf5$')
print("%d masked video files found." % len(maskedfilelist))

#%% PREPROCESSING - Compile metadata and auxiliary file info

print("\nPreprocessing video metadata:")
# Read metadata file
metafilepath = os.path.join(DATA_DIR, "AuxiliaryFiles", "metadata.csv")
metaData = pd.read_csv(metafilepath)

# Retrieve filenames for entries in metadata

# TODO: Make this a function?
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
        querystring = '/Set{0}'.format(Set) + '/Set' + Set + '_Ch' + Camera + '_' + Date + '_' # Put all in format
        
        # Find matching filename in masked video list + add to metadata
        for file in maskedfilelist:           
            if re.search(querystring, file):
                metaData.loc[i,'filename'] = file
                print("Match found! Filename added.")
                
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
# TODO: Automate food labelling using (U-Net) ML algorithm -- pre-trained already by Luigi

# Find masked HDF5 video files (for labelling) 
print("\nManual labelling:\n%d masked videos found..\n" % len(maskedfilelist))

# Interactive plotting (for user input when labelling plots)
plt.ion()
tic = time.time()
for i in range(len(maskedfilelist)):    
    maskedfilepath = maskedfilelist[i]
    # Manually outline + assign labels to food regions + save coordinates + trajectory overlay to file           
    manuallabelling(maskedfilepath, save=True, skip=True)
plt.ioff()
print("Manual labelling complete!\n(Time taken: %d seconds.)\n" % (time.time() - tic))

#%% VISUALISATION - Plot worm trajectory start/end points (unfiltered data)

print("\nPlotting tracked worm trajectories (start/end points):\n")  
tic = time.time()
for i, maskedfilepath in enumerate(fullMetaData['filename']):    
    # Plot + save worm trajectories (start/end points)
    wormtrajectories(maskedfilepath, downsample=10, save=True, skip=True)
print("Worm trajectory plotting complete!\n(Time taken: %d seconds.)" % (time.time() - tic))

#%% ON/OFF FOOD

print("\nComputing whether worms are on/off food:\n")
#subprocess.call(['python', 'OnFood.py', ""]) # TODO: Cleanest as a function 
os.system("python OnFood.py")

#%% FOOD CHOICE

print("\nCalculating worm food preference:\n")
#subprocess.call(['python', 'FoodChoice.py'])
os.system("python FoodChoice.py")

#%% LEAVING EVENTS

print("\nCalculating worm leaving events:\n")
#subprocess.call(['python', 'LeavingEvents.py'])
os.system("python LeavingEvents.py")

#%% CLEAN UP - Remove unwanted files

# Are you sure?
CLEAN_UP = True

# Find files by regex to be removed
tic = time.time()
files_to_remove = lookforfiles(os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots'), ".*_FoodChoiceTS.eps$")

# Remove files
if CLEAN_UP:
    print("Removing %d files.." % len(files_to_remove))
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
print("Done. (Time taken: %d seconds.)" % (time.time() - tic))

#%% CHECK - if files exist & have saved properly:

# Filename list
regex_list = {"Plots": [".*_LabelledOverlayPlot.png$", ".*_FoodChoiceTS.png$",\
                        ".*_PiePlot.png$", ".*_LeavingPlot.png$", ".*_WormTrajPlot.png$"],\
              "FoodCoords": [".*_FoodCoords.txt$"],\
              "FoodChoice": [".*_OnFood.csv$", ".*_FoodChoice_Mean.csv$",\
              ".*_FoodChoice_Count.csv$", ".*_FoodChoiceSummary_Mean.csv$",\
              ".*_FoodChoiceSummary_Count.csv$"],\
              "LeavingRate": [".*_LeavingEvents.csv$"]}

print("\nChecking all results files:\n")
# Check files in filename list
for folder, items in regex_list.items():
    for item in items:
        files = []
        files = lookforfiles(os.path.join(PROJECT_ROOT_DIR, 'Results', folder), item)
        print("Number of %s files found: %d" % (item.split(".*_")[-1].split(".")[0], len(files)))

#%% BACTERIAL FOOD SELECTION
filepath = "/Volumes/behavgenom$/Saul/Misc/Dirksen_2016_Caenorhabditis_Microbiome.xlsx"

excelfile = pd.ExcelFile(filepath)
worksheet = excelfile.sheet_names
Top100_elegans = excelfile.parse(worksheet[3], skiprows=1, header=0, index_col=None)
Top100_remanei = excelfile.parse(worksheet[4], skiprows=1, header=0, index_col=None)
Top100_briggsae = excelfile.parse(worksheet[5], skiprows=1, header=0, index_col=None)

SharedOTUs = set(Top100_elegans['OTU']).intersection(Top100_remanei['OTU'], Top100_briggsae['OTU'])

colnames = list(set(Top100_elegans.columns).intersection(Top100_remanei.columns, Top100_briggsae.columns))
SharedBiome_df = pd.DataFrame(index=SharedOTUs, columns=colnames)
for OTU in SharedOTUs:
    SharedBiome_df.loc[OTU] = Top100_elegans[Top100_elegans['OTU']==OTU][colnames].values
SharedBiome_df = SharedBiome_df.reset_index(drop=True)

# Save info for shared microbial species (Caenorhabditis gut flora)
SharedBiome_df.to_csv("/Volumes/behavgenom$/Saul/Misc/Dirksen_2016_Shared_Microbiome.csv")

#%% 
print("\nFood choice analysis complete!")
