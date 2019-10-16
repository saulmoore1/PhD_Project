#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 23:46:53 2019

@author: sm5911

PROCESS METADATA (96-well plate)

A script written to process microbiome assay project metadata CSV file. It
performs the following actions:
    1. Finds masked video files and adds filenames (paths) for missing entries in metadata
    2. Records the number of video segments (12min chunks) for each entry (2hr video/replicate) in metadata
    3. Records the number of featuresN results files for each entry
    4. Saves updated metadata file

"""
#%% IMPORTS

# General imports
import os, sys, re, time
import numpy as np
import pandas as pd

# Custom imports
sys.path.insert(0, '/Users/sm5911/Documents/GitHub/PhD_Project/Python')
from SM_find import lookforfiles

if __name__ == '__main__':
    # Record script start time
    tic = time.time()    
    
#%% INPUT HANDLING + READ METADATA
    
    if len(sys.argv) == 1:
        print("\nRunning script", sys.argv[0], "...")
    if len(sys.argv) > 1:
        metafilepath = sys.argv[1]   
    else: 
        print("ERROR: No metadata filepath provided!")

    PROJECT_ROOT_DIR = metafilepath.split("/AuxiliaryFiles/")[0]

    # Read metadata (CSV file)
    metadata = pd.read_csv(metafilepath)
    print("\nMetadata file loaded.")
        
    if len(sys.argv) > 2:
        IMAGING_DATES = list(sys.argv[2:])
        print("%d imaging dates provided:\n%s" % (len(IMAGING_DATES), IMAGING_DATES))
    else:
        try:
            IMAGING_DATES = list(metadata['date_recording_yyyymmdd'].dropna().astype(int).unique().astype(str))
            print("WARNING: No imaging dates provided. Using defaults:\n%s\n" % IMAGING_DATES)
        except:
            print("ERROR: Imaging date column 'date_recording_yyyymmdd' must be present in the metadata")
                
            
#%% Hydra rig dictionary of unique camera IDs across channels
        
    CAM2CH_LIST = [('22956818', 'Ch1', 'Hydra01'), # Hydra01
                   ('22956816', 'Ch2', 'Hydra01'),
                   ('22956813', 'Ch3', 'Hydra01'),
                   ('22956805', 'Ch4', 'Hydra01'),
                   ('22956807', 'Ch5', 'Hydra01'),
                   ('22956832', 'Ch6', 'Hydra01'),
                   ('22956839', 'Ch1', 'Hydra02'), # Hydra02
                   ('22956837', 'Ch2', 'Hydra02'),
                   ('22956836', 'Ch3', 'Hydra02'),
                   ('22956829', 'Ch4', 'Hydra02'),
                   ('22956822', 'Ch5', 'Hydra02'),
                   ('22956806', 'Ch6', 'Hydra02'),
                   ('22956814', 'Ch1', 'Hydra03'), # Hydra03
                   ('22956827', 'Ch2', 'Hydra03'),
                   ('22956819', 'Ch3', 'Hydra03'),
                   ('22956833', 'Ch4', 'Hydra03'),
                   ('22956823', 'Ch5', 'Hydra03'),
                   ('22956840', 'Ch6', 'Hydra03'),
                   ('22956812', 'Ch1', 'Hydra04'), # Hydra04
                   ('22956834', 'Ch2', 'Hydra04'),
                   ('22956817', 'Ch3', 'Hydra04'),
                   ('22956811', 'Ch4', 'Hydra04'),
                   ('22956831', 'Ch5', 'Hydra04'),
                   ('22956809', 'Ch6', 'Hydra04'),
                   ('22594559', 'Ch1', 'Hydra05'), # Hydra05
                   ('22594547', 'Ch2', 'Hydra05'),
                   ('22594546', 'Ch3', 'Hydra05'),
                   ('22436248', 'Ch4', 'Hydra05'),
                   ('22594549', 'Ch5', 'Hydra05'),
                   ('22594548', 'Ch6', 'Hydra05')]
    
    # Convert list of camera-channel-hydra triplets to a dictionary with 
    # hydra-channel unique keys, and camera serial numbers as values
    HYCH2CAM_DICT = {}
    for line in CAM2CH_LIST:
        HYCH2CAM_DICT[(line[2], line[1])] = line[0]

#    CAM2CH_DICT = {}
#    for line in CAM2CH_LIST:
#        CAM2CH_DICT[line[0]] = (line[1], line[2])
    
    # Camera to well number mappings
    UPRIGHT_96WP = pd.DataFrame.from_dict({('Ch1',0):[ 'A1', 'B1', 'C1', 'D1'],
                                           ('Ch1',1):[ 'A2', 'B2', 'C2', 'D2'],
                                           ('Ch1',2):[ 'A3', 'B3', 'C3', 'D3'],
                                           ('Ch1',3):[ 'A4', 'B4', 'C4', 'D4'],
                                           ('Ch2',0):[ 'E1', 'F1', 'G1', 'H1'],
                                           ('Ch2',1):[ 'E2', 'F2', 'G2', 'H2'],
                                           ('Ch2',2):[ 'E3', 'F3', 'G3', 'H3'],
                                           ('Ch2',3):[ 'E4', 'F4', 'G4', 'H4'],
                                           ('Ch3',0):[ 'A5', 'B5', 'C5', 'D5'],
                                           ('Ch3',1):[ 'A6', 'B6', 'C6', 'D6'],
                                           ('Ch3',2):[ 'A7', 'B7', 'C7', 'D7'],
                                           ('Ch3',3):[ 'A8', 'B8', 'C8', 'D8'],
                                           ('Ch4',0):[ 'E5', 'F5', 'G5', 'H5'],
                                           ('Ch4',1):[ 'E6', 'F6', 'G6', 'H6'],
                                           ('Ch4',2):[ 'E7', 'F7', 'G7', 'H7'],
                                           ('Ch4',3):[ 'E8', 'F8', 'G8', 'H8'],
                                           ('Ch5',0):[ 'A9', 'B9', 'C9', 'D9'],
                                           ('Ch5',1):['A10','B10','C10','D10'],
                                           ('Ch5',2):['A11','B11','C11','D11'],
                                           ('Ch5',3):['A12','B12','C12','D12'],
                                           ('Ch6',0):[ 'E9', 'F9', 'G9', 'H9'],
                                           ('Ch6',1):['E10','F10','G10','H10'],
                                           ('Ch6',2):['E11','F11','G11','H11'],
                                           ('Ch6',3):['E12','F12','G12','H12']})
    
#%% GLOBAL PARAMETERS (User-defined optional)
    
    metadata_outfile = metafilepath.replace(".csv", "_updated.csv")
    maskedvideo_dir = os.path.join(PROJECT_ROOT_DIR, "MaskedVideos")
    featuresN_dir = os.path.join(PROJECT_ROOT_DIR, "Results")
    rawvideo_dir = os.path.join(PROJECT_ROOT_DIR, "RawVideos")
    
    #%% OBTAIN MASKED VIDEO FILEPATHS FOR METADATA
    
    n_filepaths = sum([isinstance(path, str) for path in metadata.filename])
    n_entries = len(metadata.filename)
    print("%d/%d filename entries found in metadata" % (n_filepaths, n_entries))
    print("Attempting to fetch filenames for %d entries..." % (n_entries - n_filepaths))    
    
    # Return list of pathnames for masked videos in the data directory under given imaging dates
    maskedfilelist = []
    date_total = []
    print("Looking in '%s' for MaskedVideo files..." % maskedvideo_dir)
    for i, expDate in enumerate(IMAGING_DATES):
        tmplist = lookforfiles(os.path.join(maskedvideo_dir, expDate), ".*.hdf5$")
        date_total.append(len(tmplist))
        maskedfilelist.extend(tmplist)
    print("%d masked video snippets found for imaging dates provided:\n%s" % \
          (len(maskedfilelist), [*zip(IMAGING_DATES, date_total)]))    


#%% # Parse over metadata entries and use well number/run number/date/hydra rig 
    # information to locate and fill in missing filename entries
    
    for i, filepath in enumerate(metadata.filename):
        
        # If filepath is already present, make sure there is no spaces
        if isinstance(filepath, str):
            metadata.loc[i,'filename'] = filepath.replace(" ", "")  
        
        else:
            file_info = metadata.iloc[i]
            
            # Extract date/run/hydra/plate/well info
            date = str(file_info['date_recording_yyyymmdd'].astype(int))             # which experiment date?
            hydra = file_info['instrument_name']                                     # which Hydra rig?
            well_number = str(file_info['well_number'])                              # which well in 96-well plate?
            run_number = str(int(file_info['run_number']))                           # which run?
            
            # Obtain channel number from well-to-channel mapping dictionary: 'UPRIGHT_96WP'
            channel = UPRIGHT_96WP.iloc[np.where(UPRIGHT_96WP == well_number)].columns[0][0]
            
            # Obtain camera serial number unique ID using hydra/channel combination, using dictionary: HYCH2CAM_DICT
            cameraID = HYCH2CAM_DICT[(hydra,channel)]                                # which camera?
            
            # Update cameraID in metadata
            metadata.loc[i,'camera_number'] = cameraID
            
            # Use run/date/cameraID to construct regex query to find results filename                                            
            # Query by regex using run/date/camera info
            file_querystr1 = '_run{0}_'.format(run_number)
            file_querystr2 = '_' + date + '_\d{6}.' + cameraID
            
            # Retrieve filepath, using data recorded in metadata
            for file in maskedfilelist:
                # If folder name contains '_runX_' (WARNING: this is manually assigned/typed when recording)
                if re.search(file_querystr1, file.lower()): # or re.search(file_querystr1, file.lower()):
                    # If filepath contains: '_date_XXXXXX.cameraID'...
                    # NB: auto-generated to include date/time(exact time not known)/cameraID
                    if re.search(file_querystr2, file.lower()):
                        # Record filepath to MaskedVideo file: '*/metadata.hdf5'
                        metadata.loc[i,'filename'] = os.path.dirname(file)
                            
    matches = sum([isinstance(path, str) for path in metadata.filename]) - n_filepaths
    print("Complete!\n%d/%d filenames added.\n" % (matches, n_entries - n_filepaths))

#%% OBTAIN RAW VIDEO FILEPATHS FOR COUNTING SNIPPETS
    
    # Return list of pathnames for raw videos in the data directory for given imaging dates
    rawvideolist = []
    date_total = []
    print("Looking in '%s' for RawVideo files..." % rawvideo_dir)
    for i, expDate in enumerate(IMAGING_DATES):
        tmplist = lookforfiles(os.path.join(rawvideo_dir, expDate), ".*.mp4$")
        date_total.append(len(tmplist))
        rawvideolist.extend(tmplist)

    # Get list of pathnames for featuresN files for given imaging dates
    featuresNlist = []
    print("Looking in '%s' for featuresN files..." % featuresN_dir)
    for i, expDate in enumerate(IMAGING_DATES):
        tmplist = lookforfiles(os.path.join(featuresN_dir, str(expDate)), ".*_featuresN.hdf5$")
        featuresNlist.extend(tmplist)
    
    # Pre-allocate columns in metadata for storing n_video_chunks, n_featuresN_files
    metadata['rawvideo_snippets'] = ''
    metadata['featuresN_exists'] = ''
    
    # Add n_video_snippets, n_featuresN_files as columns to metadata
    for i, masked_dirpath in enumerate(metadata.filename):
        # If filepath is present, return the filepaths to the rest of the chunks for that video
        if isinstance(masked_dirpath, str):
            # Record number of video segments (chunks) in metadata
            raw_dirpath = masked_dirpath.replace("/MaskedVideos", "/RawVideos")
            snippetlist = [snippet for snippet in rawvideolist if raw_dirpath in snippet] 
            n_snippets = len(snippetlist)
            metadata.loc[i, 'rawvideo_snippets'] = int(n_snippets)
            
            # Record the number of featuresN files
            featuresN_dirpath = masked_dirpath.replace("/MaskedVideos", "/Results")
            featlist = [featpath for featpath in featuresNlist if featuresN_dirpath in featpath]
            n_featuresN = len(featlist)
            metadata.loc[i, 'featuresN_exists'] = (n_featuresN * n_snippets == n_snippets)
            
    print("(Metadata updated: Checked for featuresN files and tallied number of RawVideo snippets found.)")
    
    # Save full metadata
    print("Saving updated metadata to: '%s'" % metadata_outfile)
    metadata.to_csv(metadata_outfile, index=False)        

#%%   
    # Record script end time
    toc = time.time()
    print("Done.\n(Time taken: %.1f seconds)" % (toc-tic))

#%%
#    for maskedvideo in maskedfilelist:
#        video_str = maskedvideo.split('/')[-2]
#        
#        # Obtain run number and plate number from video path string (USER MUST PROVIDE THIS INFO WHEN RECORDING!)
#        run, plate = None, None
#        match_run = re.search(r"\Brun\d*", video_str.lower().replace(" ",""))
#        if match_run:
#            result = match_run.group()
#            if len([result]) == 1:
#                run = result.split('run')[-1]
#            else:
#                print("WARNING: Multiple instances of 'run' in filename.\
#                      Attempting to use run number associated with first instance.")
#                run = [result][0].split('run')[-1]
#        else:
#            print("ERROR: No run number found in filename!")
#            
#        match_plate = re.search(r"\Bplate\d*", video_str.lower().replace(" ",""))
#        if match_plate:
#            result = match_plate.group()
#            if len([result]) == 1:
#                plate = result.split('plate')[-1]
#            else:
#                print("WARNING: Multiple instances of 'plate' in filename.\
#                      Attempting to use plate number associated with first instance.")
#                plate = [result][0].split('plate')[-1]
#        else:
#            print("WARNING: No plate number found in filename.")     
#           
#        # Obtain camera serial ID from video path string
#        camID = video_str.split('.')[-1]
#        
#        # Obtain date and timestamp from video path (auto-generated and appended to the filename)
#        timestamp = video_str.split('.')[0].split('_')[-1]
#        date = video_str.split('_')[-2]
#        
#        
#        print(date, run, plate, camID)
#        
#        # Use extracted info to look up channel number
#        
#        # Use channel number to obtain well numbers
#        UPRIGHT_96WP['Ch1'].values
#        
#        # Use well number info in metadata associated with the correct date, run, plate