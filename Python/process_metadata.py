#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 23:46:53 2019

@author: sm5911

PROCESS METADATA

A script written to process microbiome assay project metadata CSV file. It
performs the following actions:
    1. Finds masked video files and adds filenames (paths) for missing entries in metadata
    2. Records the number of video segments (12min chunks) for each entry (2hr video/replicate) in metadata
    3. Records the number of featuresN results files for each entry
    4. Saves updated metadata file

"""
#%% IMPORTS

# General imports
import os, re, time, pdb
import pandas as pd

# Custom imports
import sys; sys.path.insert(0, '/Users/sm5911/Documents/GitHub/PhD_Project/Python') # Path to GitHub functions
from Find import lookforfiles


#%% PRE-AMBLE

# Record script start time
tic = time.time()

# Global variables
PROJECT_NAME = 'MicrobiomeAssay'
PROJECT_ROOT_DIR = '/Volumes/behavgenom$/Saul/' + PROJECT_NAME
DATA_DIR = '/Volumes/behavgenom$/Priota/Data/' + PROJECT_NAME

# Select imaging date(s) for analysis
IMAGING_DATES = ['20190704', '20190705']

# Hydra rig dictionary of unique camera IDs across channels
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

# Convert dictionary of unique cameraIDs to dataframe
CAM2CH_DF = pd.DataFrame([(k,v) for k,v in CAM2CH_DICT.items()], columns=['cameraID','channel'])

#%% READ METADATA

# Read metadata (CSV file)
metafilepath = os.path.join(DATA_DIR, "AuxiliaryFiles", "metadata.csv")
metadata = pd.read_csv(metafilepath)
print("'{0}' project metadata loaded.".format(PROJECT_NAME))


#%% OBTAIN MASKED VIDEO FILEPATHS FOR METADATA

print("\nProcessing metadata file...")
n_filepaths = sum([isinstance(path, str) for path in metadata.filename])
n_entries = len(metadata.filename)

print("%d/%d filename entries found in metadata" % (n_filepaths, n_entries))
print("\nFetching filenames for %d entries..." % (n_entries - n_filepaths))

# Return list of pathnames for masked videos in the data directory under given imaging dates
maskedfilelist = []
date_total = []
for i, expDate in enumerate(IMAGING_DATES):
    tmplist = lookforfiles(os.path.join(DATA_DIR, "MaskedVideos", expDate), ".*.hdf5$")
    date_total.append(len(tmplist))
    maskedfilelist.extend(tmplist)
print("%d masked videos found for imaging dates provided:\n%s" % (len(maskedfilelist), [*zip(IMAGING_DATES, date_total)]))

# Pre-allocate column in metadata for storing camera ID info
metadata['cameraID'] = ''  

# Retrieve filenames for entries in metadata
for i, filepath in enumerate(metadata.filename):
    
    # If filepath is already present, make sure there is no spaces
    if isinstance(filepath, str):
        metadata.loc[i,'filename'] = filepath.replace(" ", "")  
        
    else:
        # Extract date/set/camera info for metadata entry
        file_info = metadata.iloc[i]
        date = str(file_info['date_yyyymmdd'])                                  # which experiment date?
        set_number = str(file_info['set_number'])                               # which set/run?
        channel = int(file_info['channel'])                                     # which camera channel?
        hydra = int(str(file_info['instrument_name']).lower().split('hydra')[1])# which Hydra?
        
        # Obtain unique ID for hydra camera using hydra number / channel number
        # by indexing cameraID dataframe using hydra/channel combination
        camera_info = CAM2CH_DF.iloc[(hydra - 1) * 6 + (channel - 1)]
        
        # Quick (not so fool-proof) check that indexing worked successfully
        if camera_info['channel'] != 'Ch' + str(channel):
            print("ERROR: Incorrect camera channel!")
            pdb.set_trace()
        
        # Get cameraID for file
        cameraID = camera_info['cameraID']
        
        # Update cameraID in metadata
        metadata.loc[i,'cameraID'] = cameraID
                                        
        # Query by regex using date/camera info
        querystring = '/food_behaviour_s{0}_'.format(set_number) + date + '_'
        
        # Search for 1st video segment, if present, record filepath in metadata (ie. "000000.hdf5")   
        for file in maskedfilelist:
            if re.search(querystring, file) and re.search(('.' + cameraID + '/000000.hdf5'), file):
                
                # Record filepath to parent directory (containing all chunks for that video)
                metadata.loc[i,'filename'] = os.path.dirname(file)
matches = sum([isinstance(path, str) for path in metadata.filename]) - n_filepaths
print("Complete!\n%d filenames added." % matches)
              
# Return list of pathnames for featuresN files
print("\nSearching for results files..")
featuresNlist = []
for i, expDate in enumerate(IMAGING_DATES):
    tmplist = lookforfiles(os.path.join(DATA_DIR, "Results", expDate), ".*_featuresN.hdf5$")
    featuresNlist.extend(tmplist)

# Pre-allocate columns in metadata for storing n_video_chunks, n_featuresN_files
metadata['n_maskedvideo_chunks'] = ''
metadata['n_featuresN_files'] = ''

# Add n_video_chunks, n_featuresN_files as columns to metadata
extra_chunk = 0   
for i, dirpath in enumerate(metadata.filename):
    # If filepath is present, return the filepaths to the rest of the chunks for that video
    if isinstance(dirpath, str):
        file_info = metadata.iloc[i]
        set_number = int(file_info['set_number'])
        date = file_info['date_yyyymmdd']
        
        chunklist = [chunkpath for chunkpath in maskedfilelist if dirpath in chunkpath] 
        n_chunks = len(chunklist)
        if n_chunks != 10:
            if n_chunks == 11: 
                extra_chunk += 1
                #print("Extra chunk!", n_chunks, date, set_number)
            else:
                pdb.set_trace()
        #cameras = list(set([re.findall(r'(?<=\d{8}_\d{6}\.)\d{8}', chunkpath)[0] for chunkpath in chunklist]))
        #channels = {CAM2CH_DICT[k] for k in cameras}
        
        # Record number of video segments (chunks) in metadata
        metadata.loc[i, 'n_maskedvideo_chunks'] = n_chunks
        
        featlist = [featpath for featpath in featuresNlist if dirpath.replace("MaskedVideos", "Results") in featpath]
        n_featuresN = len(featlist)
        
        # Record the number of featuresN files
        metadata.loc[i, 'n_featuresN_files'] = n_featuresN
print("Complete!\n(Number of video chunks and featuresN files recorded in metadata)")

#print("%d extra chunks found (11th video segment)" % extra_chunk)

# Save full metadata
print("\nSaving updated metadata...")
metadata.to_csv(os.path.join(PROJECT_ROOT_DIR, "metadata.csv"))

# Record script end time
toc = time.time()
print("Done.\n(Time taken: %.1f seconds)" % (toc-tic))