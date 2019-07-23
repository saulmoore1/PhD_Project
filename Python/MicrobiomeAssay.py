#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bacterial affects on Caenorhabditis elegans Behaviour - A Microbiome Analysis

This script reads Tierpsy results files for experimental data collected during 
preliminary screening of Schulenberg Lab bacterial strains isolated from the C. 
elegans gut microbiome. 

The script does the following: 
    - Reads the project metadata file, and completes missing filepath info
    - Checks for results files (features/skeletons/intensities)
    - Extracts relevant features of interest for visualisation 
    - Comparison of features between N2 worms on different foods

@author: sm5911
@date: 2019-07-07

"""

#%% IMPORTS

# General dependencies
import os, sys, re, time
import pandas as pd
import numpy as np
#import datetime

# Custom imports
sys.path.insert(0, '/Users/sm5911/Documents/GitHub/PhD_Project/Python') # OPTIONAL: Path to GitHub functions
from Find import lookforfiles, changepath
from Read import gettrajdata, getfeatsums
from Plot import manuallabelling


#%% PRE-AMBLE

# Global variables
PROJECT_ROOT_DIR = '/Volumes/behavgenom$/Saul/MicrobiomeAssay/'

DATA_DIR = '/Volumes/behavgenom$/Priota/Data/FoodChoiceAssay/'

IMAGING_DATES = ['20190704','20190705'] #  # Select imaging date(s) for analysis

CAM2CH_DICT = {"22956818":'Ch1', # Hydra01
              "22956816":'Ch2',
              "22956813":'Ch3',
              "22956805":'Ch4',
              "22956807":'Ch5',
              "22956832":'Ch6',
              "22956839":'Ch1', # Hydra02
              "22956837":'Ch2',
              "22956836":'Ch3',
              "22956829":'Ch4',
              "22956822":'Ch5',
              "22956806":'Ch6',
              "22956814":'Ch1', # Hydra03
              "22956827":'Ch2',
              "22956819":'Ch3',
              "22956833":'Ch4',
              "22956823":'Ch5',
              "22956840":'Ch6',
              "22956812":'Ch1', # Hydra04
              "22956834":'Ch2',
              "22956817":'Ch3',
              "22956811":'Ch4',
              "22956831":'Ch5',
              "22956809":'Ch6',
              "22594559":'Ch1', # Hydra05
              "22594547":'Ch2',
              "22594546":'Ch3',
              "22436248":'Ch4',
              "22594549":'Ch5',
              "22594548":'Ch6'}

#%% READ METADATA

# Read metadata (CSV file)
metafilepath = os.path.join(DATA_DIR, "AuxiliaryFiles", "microbiome_metadata.csv")
metadata = pd.read_csv(metafilepath)
print("Metadata file loaded.")


#%% OBTAIN MASKED VIDEO FILEPATHS FOR METADATA

print("Processing project metadata...")
n_filepaths = sum([isinstance(path, str) for path in metadata.filename])
n_entries = len(metadata.filename)

print("%d/%d filepath entries found in metadata" % (n_filepaths, n_entries))
print("Fetching filepaths for %d entries..." % (n_entries - n_filepaths))

# Return list of pathnames for masked videos in the data directory under given imaging dates
maskedfilelist = []
date_total = []
for i, expDate in enumerate(IMAGING_DATES):
    tmplist = lookforfiles(os.path.join(DATA_DIR, "MaskedVideos", expDate), ".*.hdf5$")
    date_total.append(len(tmplist))
    maskedfilelist.extend(tmplist)
print("%d masked videos found for imaging dates: %s" % (len(maskedfilelist), date_total))

# Preacllocate column in metadata for storing camera ID info
metadata['cameraID'] = ''  

# Retrieve filenames for entries in metadata
for i, filepath in enumerate(metadata.filename):
    # If filepath is already present, make sure there is no spaces
    if isinstance(filepath, str):
        metadata.loc[i,'filename'] = filepath.replace(" ", "")  
    else:
        file_info = metadata.iloc[i]
        
        # Extract unique date/camera info for metadata entry
        Date = str(file_info['date(YEARMODA)']) # which experiment date?
        
        # Obtain unique ID for hydra camera using hydra number / channel number
        Set = str(file_info['set_number'])
        channel = int(file_info['channel']) # which camera channel
        hydra = int(str(file_info['instrument_name']).lower().split('hydra')[1]) # which Hydra?
        
        # TODO: Read dictionary properly
        #camera_df = pd.DataFrame(data=zip(CAM2CH_DICT.keys(),CAM2CH_DICT.values()),index=np.arange(len(CAM2CH_DICT.keys())))
        cameraID = list(CAM2CH_DICT.keys())[(hydra - 1) * 6 + (channel - 1)]
        
        # Update cameraID in metadata
        metadata.loc[i,'cameraID'] = cameraID
        
#        # Re-format date string (Phenix only)
#        d = datetime.datetime.strptime(Date, '%Y%m%d')
#        Date = d.strftime('%d%m%Y')
        
        # Query by regex using date/camera info
        querystring = '/food_behaviour_s{0}_'.format(Set) + Date + '_'
                
        # Search for 1st video segment + record filepath in metadata (ie. "000000.hdf5")   
        for file in maskedfilelist:
            if re.search(querystring, file) and re.search(('.' + cameraID + '/000000.hdf5'), file):
#                print(file, querystring, cameraID)
                # Record filepath
                metadata.loc[i,'filename'] = file
#                print("Match found! Filename added.")

matches = sum([isinstance(path, str) for path in metadata.filename]) - n_filepaths
    
print("\nComplete!\n%d filepaths were sucessfully retrieved (%d imaging dates)" % (matches, len(IMAGING_DATES)))

# Save full metadata
print("Saving updated metadata..")
metadata.to_csv(os.path.join(PROJECT_ROOT_DIR, "metadata.csv"))
print("Done.")

# Subset metadata to remove remaining incomplete entries
is_filename = [isinstance(path, str) for path in metadata['filename']]
if any(list(~np.array(is_filename))):
    print("WARNING: Could not find filepaths for %d entries in metadata." \
          % sum(list(~np.array(is_filename))))
    print("These files will be omitted from further analyses!")
    metadata = metadata[list(np.array(is_filename))].reset_index(drop=True)


##%% CHECK FOR RESULTS FILES 
#
#RESULTS_FILES = ['_featuresN.hdf5', '_skeletons.hdf5', '_intensities.hdf5']
#
#
## Return list of all results files present for given imaging date
#results_list = []
#for date in IMAGING_DATES:
#    for filetype in RESULTS_FILES:
#        results = lookforfiles(os.path.join(DATA_DIR, 'Results', date), filetype)
#        results_list.extend(results)
#
## Check that results files are present for each masked video
#incomplete_analysis = []
#for maskedvideo in maskedfilelist:
#    # Results filepaths
#    features = changepath(maskedvideo, returnpath = 'features')
#    skeletons = changepath(maskedvideo, returnpath = 'skeletons')
#    intensities = changepath(maskedvideo, returnpath = 'intensities')
#    
#    # Do features/skeletons/intensities files exist for masked videos?
#    for i, resultspath in enumerate([features, skeletons, intensities]):
#        if resultspath not in results_list:
#            print("Missing %s files for masked video: \n%s" \
#                  % (RESULTS_FILES[i].split('_')[1].split('.')[0], resultspath))
#            incomplete_analysis.append(resultspath)
#            
#print("%d results files missing!" % len(incomplete_analysis))
#
## TODO: Ignore empty video snippets at end of 2hr assay recording
#
#
##%% READ FEATURES SUMMARY (Tierpsy analysis results)
#
#for date in IMAGING_DATES:
#    results_dir = os.path.join(DATA_DIR, "Results", date)
#
#    files_df, feats_df = getfeatsums(results_dir)
#    
#
##%% READ TRAJECTORY DATA
#
#ERROR_LIST = []
#errorlogname = "Unprocessed_MaskedVideos.txt"
#for i, maskedvideo in enumerate(metadata.filename):
#    if i % 10 == 0:
#        print("Processing file: %d/%d" % (i, len(maskedfilelist)))
#    featuresfilepath = changepath(maskedvideo, returnpath='features')
#    try:
#        data = gettrajdata(featuresfilepath)
##        if data.shape[0] > 1:
##            print(data.head())
#    except Exception as EE:
#        print("ERROR:", EE)
#        ERROR_LIST.append(maskedvideo)
#
#if ERROR_LIST:
#    fid = open(os.path.join(PROJECT_ROOT_DIR, errorlogname), 'w')
#    print(ERROR_LIST, file=fid)
#    fid.close()
#
#
##%% MANUAL LABELLING OF FOOD REGIONS (To check time spent feeding throughout video)
#    
## TODO: Automate food labelling using (U-Net) ML algorithm -- pre-trained already by Luigi
#print("\nManual labelling:\nTotal masked videos found: %d\n" % len(maskedfilelist))
#
## Interactive plotting (for user input when labelling plots)
#tic = time.time()
#for i in range(len(maskedfilelist)):    
#    maskedfilepath = maskedfilelist[i]
#    # Manually outline + assign labels to food regions + save coordinates + trajectory overlay to file           
#    manuallabelling(maskedfilepath, n_poly=1, out_dir='MicrobiomeAssay', save=True, skip=True)
#print("Manual labelling complete!\n(Time taken: %d seconds.)\n" % (time.time() - tic))


#%% PLOT WORM TRAJECTORIES



#%% VISUALISE SUMMARY FEATURES



#%% PRINCIPLE COMPONENTS ANALYSIS
    