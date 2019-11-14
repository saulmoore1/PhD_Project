#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROCESS METADATA (96-well plate)

A script written to process microbiome assay project metadata CSV file. It
performs the following actions:
    1. Finds masked video files and adds filenames (paths) for missing entries in metadata
    2. Records the number of video segments (12min chunks) for each entry (2hr video/replicate) in metadata
    3. Records the number of featuresN results files for each entry
    4. Saves updated metadata file
    
Required fields in metadata: 
    ['filename','date_recording_yyyymmdd','instrument_name','well_number','run_number','camera_number','food_type']

@author: sm5911
@date: 13/10/2019

"""
#%% IMPORTS

# General imports
import os, sys, re, time, datetime#, json, glob
import numpy as np
import pandas as pd

# Path to Github/local helper functions (USER-DEFINED: Path to local copy of my Github repo)
PATHS = ['/Users/sm5911/Documents/GitHub/PhD_Project/Python/Psychobiotics_96WP/',\
         '/Users/sm5911/OneDrive - Imperial College London/Tierpsy/tierpsy-tracker-loopbioFOVsplitter_traj/tierpsy/analysis/']
for PATH in PATHS:
    if PATH not in sys.path:
        sys.path.append(PATH)
    
# Custom imports
from helper import lookforfiles
from split_fov.helper import CAM2CH_list, UPRIGHT_96WP # Dictionaries camera-channel-hydra mappings + camera-well mappings

#%% 
def compile_from_day_metadata(COMPILED_METADATA_FILEPATH, IMAGING_DATES):
    """ COMPILE FULL METADATA FROM EXPERIMENT DAY METADATA """
    
    METADATA_DIR = os.path.dirname(COMPILED_METADATA_FILEPATH)
    
    print("Compiling full metadata from day-metadata files in: '%s'" % METADATA_DIR)
    AuxFileList = os.listdir(METADATA_DIR)
    ExperimentDates = sorted([expdate for expdate in AuxFileList if re.match(r'\d{8}', expdate)])
    if IMAGING_DATES:
        ExperimentDates = [expdate for expdate in ExperimentDates if expdate in IMAGING_DATES]
    else:
        IMAGING_DATES = ExperimentDates
    
    day_metadata_df_list = []
    for expdate in IMAGING_DATES:
        expdate_metadata_path = os.path.join(METADATA_DIR, expdate, 'metadata_' + expdate + '.csv')
        try:
            expdate_metadata = pd.read_csv(expdate_metadata_path)
            day_metadata_df_list.append(expdate_metadata)   
        except Exception as EE:
            print("WARNING:", EE)
    
    # Concatenate into a single full metadata
    metadata = pd.concat(day_metadata_df_list, axis=0, ignore_index=True, sort=False)

    return(metadata)    
    
#%%
def find_metadata_filenames(metadata, PROJECT_ROOT_DIR, IMAGING_DATES):       
    """ OBTAIN MASKED VIDEO FILEPATHS FOR METADATA """
    
    if not IMAGING_DATES:
        IMAGING_DATES = sorted(metadata['date_recording_yyyymmdd'].astype(str).unique())
    
    # Convert list of camera-channel-hydra triplets to a dictionary with 
    # hydra-channel unique keys, and camera serial numbers as values
    HYCH2CAM_DICT = {}
    for line in CAM2CH_list:
        HYCH2CAM_DICT[(line[2], line[1])] = line[0]                       

    # PATHS TO RESULTS DIRECTORIES (User-defined, optional)    
    maskedvideo_dir = os.path.join(PROJECT_ROOT_DIR, "MaskedVideos")
    featuresN_dir = os.path.join(PROJECT_ROOT_DIR, "Results")
    rawvideo_dir = os.path.join(PROJECT_ROOT_DIR, "RawVideos")
    
    n_filepaths = sum([isinstance(path, str) for path in metadata.filename])
    n_entries = len(metadata.filename)
    print("%d/%d filename entries found in metadata" % (n_filepaths, n_entries))
    
    if not n_entries == n_filepaths:
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
            if isinstance(filepath, str):
                # If filepath is already present, make sure there are no whitespaces
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
        
#%% OPTIONAL EXTRAS
         
    # Ensure 'food_type' entries are grouped correctly by converting to uppercase
    metadata['food_type'] = metadata['food_type'].str.upper()   
 
    # Calculate L1 diapause duration (if possible) and append to results
    diapause_required_columns = ['date_bleaching_yyyymmdd','time_bleaching',\
                                 'date_L1_refed_yyyymmdd','time_L1_refed_OP50']
    
    if all(x in metadata.columns for x in diapause_required_columns):
        # Extract bleaching dates and times
        bleaching_datetime = [datetime.datetime.strptime(date_str + ' ' +\
                              time_str, '%Y%m%d %H:%M:%S') for date_str, time_str\
                              in zip(metadata['date_bleaching_yyyymmdd'].astype(str),\
                              metadata['time_bleaching'])]
        # Extract dispensing dates and times
        dispense_L1_datetime = [datetime.datetime.strptime(date_str + ' ' +\
                                time_str, '%Y%m%d %H:%M:%S') for date_str, time_str\
                                in zip(metadata['date_L1_refed_yyyymmdd'].astype(str),\
                                metadata['time_L1_refed_OP50'])]
        # Estimate duration of L1 diapause
        L1_diapause_duration = [dispense - bleach for bleach, dispense in \
                                zip(bleaching_datetime, dispense_L1_datetime)]
        
        # Add duration of L1 diapause to metadata
        metadata['L1_diapause_seconds'] = [int(timedelta.total_seconds()) \
                                           for timedelta in L1_diapause_duration]

    return metadata

#%% # TODO: Read JSON files, extract hydra imaging rig temperature and humidity info,
#       and append to metadata
#    rig_data_colnames = ['filename_JSON_snippet','frame_number','rig_internal_humidity_percent','rig_internal_temperature_C']
#    rig_data_full = pd.DataFrame(columns=rig_data_colnames)
#    for i, filepath in enumerate(metadata['filename']):
#        if i % 10 == 0:
#            print("Extracting hydra rig data from JSON snippets for file: %d/%d" % (i,len(metadata['filename'])))
#        raw_json_dir = filepath.replace("/MaskedVideos","/RawVideos")
#        extra_json_filelist = glob.glob(os.path.join(raw_json_dir, "*.extra_data.json"))
#        for json_snippet in extra_json_filelist:
#            with open(json_snippet) as fid:
#                extras = json.load(fid)
#                rig_data_snippet = pd.DataFrame(index=range(len(extras)), columns=rig_data_colnames)
#                for d, dictionary in enumerate(extras):
#                    rig_data_snippet.loc[d, rig_data_colnames] = json_snippet, dictionary['frame_index'], dictionary['humidity'], dictionary['tempo']
#                    rig_data_full = pd.concat([rig_data_full, rig_data_snippet], axis=0, sort=False).reset_index(drop=True)

#%% 
if __name__ == '__main__':
    # Record script start time
    tic = time.time()  
    
    # INPUT HANDLING    
    print("\nRunning script", sys.argv[0], "...")
    if len(sys.argv) > 1:
        COMPILED_METADATA_FILEPATH = sys.argv[1]  
        
    IMAGING_DATES = None
    if len(sys.argv) > 2:
        IMAGING_DATES = list(sys.argv[2:])
        print("Using %d imaging dates provided: %s" % (len(IMAGING_DATES), IMAGING_DATES))     

    # MAIN
    if os.path.exists(COMPILED_METADATA_FILEPATH):
        print("ERROR: Compiled metadata file already exists!")
    else:  
        PROJECT_ROOT_DIR = COMPILED_METADATA_FILEPATH.split("/AuxiliaryFiles/")[0]
                                     
        # COMPILE FULL METADATA FROM EXPERIMENT DAY METADATA
        metadata = compile_from_day_metadata(COMPILED_METADATA_FILEPATH, IMAGING_DATES)
        
        # OBTAIN MASKED VIDEO FILEPATHS FOR METADATA
        metadata = find_metadata_filenames(metadata, PROJECT_ROOT_DIR, IMAGING_DATES)
        
        print("Saving updated metadata to: '%s'" % COMPILED_METADATA_FILEPATH)
        metadata.to_csv(COMPILED_METADATA_FILEPATH, index=False)        

    # Record script end time
    toc = time.time()
    print("Done.\n(Time taken: %.1f seconds)" % (toc-tic))

  