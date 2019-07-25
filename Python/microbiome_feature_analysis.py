#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bacterial affects on Caenorhabditis elegans Behaviour - Microbiome Analysis

This script reads Tierpsy results files for experimental data collected during 
preliminary screening of Schulenberg Lab bacterial strains isolated from the C. 
elegans gut microbiome. 

The script does the following: 
    - Reads the project metadata file, and completes missing filepath info
    - Checks for results files (features/skeletons/intensities)
    - Extracts summary features of interest and preps for visualisation
The script WILL DO the following:
    - Principal components analysis (PCA) to extract most important features
    - Visualisation of extracted features
    - Comparison of these features between N2 worms on different foods

@author: sm5911
@date: 07/07/2019

"""

#TODO: Add 'date_poured' to metadata

#%% IMPORTS

# General imports
#import datetime
#sys.path.insert(0, '/Users/sm5911/Documents/GitHub/PhD_Project/Python') # OPTIONAL: Path to GitHub functions
import os, sys, time
import subprocess as sp
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import zscore
from sklearn.decomposition import PCA

# Custom imports
from Find import changepath
from Read import gettrajdata, getfeatsums
from Calculate import PCAinfo
#from Plot import manuallabelling
from Save import savefig


#%% PRE-AMBLE

# TODO: Use subprocess to call 'process_metadata.py'
process_metadata = sp.Popen([sys.executable, "process_metadata.py"])
process_metadata.communicate()

# Global variables
PROJECT_NAME = 'MicrobiomeAssay'
PROJECT_ROOT_DIR = '/Volumes/behavgenom$/Saul/' + PROJECT_NAME
DATA_DIR = '/Volumes/behavgenom$/Priota/Data/' + PROJECT_NAME

# Select imaging date(s) for analysis
IMAGING_DATES = ['20190704', '20190705']

#%% READ METADATA FOR IMAGING DATES

# Read metadata (CSV file)
metafilepath = os.path.join(PROJECT_ROOT_DIR, "metadata.csv")
metadata = pd.read_csv(metafilepath)
print("Metadata file loaded.")

# Subset metadata to remove remaining entries with missing filepaths
is_filename = [isinstance(path, str) for path in metadata['filename']]
if any(list(~np.array(is_filename))):
    print("WARNING: Could not find filepaths for %d entries in metadata.\n\t These files will be omitted from further analyses!" \
          % sum(list(~np.array(is_filename))))
    metadata = metadata[list(np.array(is_filename))].reset_index(drop=True)


#%% READ FEATURES SUMMARY (Tierpsy analysis results) + PLOT A FEW FEATURES + PERFORM PCA

features2plot = ['speed', 'angular_velocity']

tic = time.time()
for date in IMAGING_DATES:
    results_dir = os.path.join(DATA_DIR, "Results", date)
    
    ##### Get files summaries and features summaries #####
    # NB: Ignores empty video snippets at end of some assay recordings 
    #     2hrs = 10 x 12min video segments (+/- a few frames)
    files_df, feats_df = getfeatsums(results_dir)
    
    ##### Plot a few example features #####
    
    for featroot in features2plot:
        plt.close('all')
        feature_prctiles = [featroot + end for end in ['_10th', '_50th', '_90th']]
        feature_IQR = featroot + '_IQR'
        
        fig, ax = plt.subplots(1,2,figsize=[12,5])
        
        for feature in feature_prctiles:
            sns.boxplot(x=feature, data=feats_df, color='lightblue')
        plt.pause(1)  
        
    # TODO: Plot average - FINISH...
    
    ##### Prepare features summary data for PCA #####
    
    # Drop non-data column(s)
    data = feats_df.drop(columns='file_id')
    
    # Drop columns that are all zeroes
    data.drop(columns=data.columns[(data==0).all()], inplace=True)
    
    # Drop columns with too many nans
    nan_threshold = 0.75
    data.dropna(axis='columns', thresh=nan_threshold, inplace=True)
    
    # Impute data to fill in remaining nans with mean value
    data = data.apply(lambda x : x.fillna(x.mean(axis=0)))
    
    # Normalise the data
    zscores = data.apply(zscore)
    
    ##### Perform PCA on extracted features #####
    pca = PCA()
    pca.fit(zscores)
    
    # Plot summary data from PCA: explained variance (most important feats)
    important_feats, fig = PCAinfo(pca, zscores)
    
    # Save plot of PCA explained variance
    plotpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'PCA_explained_' + date + '.eps')
    directory = os.path.dirname(plotpath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    savefig(plotpath, tight_layout=True, tellme=True, saveFormat='eps')
    
    # Project zscores onto pc
    projected = pca.transform(zscores)  # produces a matrix

toc = time.time()
print("Time taken: %.1f seconds" % (toc - tic))

#%% READ TRAJECTORY DATA

ERROR_LIST = []
errorlogname = "Unprocessed_MaskedVideos.txt"
for i, maskedvideo in enumerate(metadata.filename):
    if i % 10 == 0:
        print("Processing file: %d/%d" % (i, len(metadata.filename)))
    featuresfilepath = changepath(maskedvideo, returnpath='features')
    try:
        data = gettrajdata(featuresfilepath)
#        if data.shape[0] > 1:
#            print(data.head())
    except Exception as EE:
        print("ERROR:", EE)
        ERROR_LIST.append(maskedvideo)

if ERROR_LIST:
    fid = open(os.path.join(PROJECT_ROOT_DIR, errorlogname), 'w')
    print(ERROR_LIST, file=fid)
    fid.close()


#%% MANUAL LABELLING OF FOOD REGIONS (To check time spent feeding throughout video)
    
# TODO: Automate food labelling using (U-Net) ML algorithm -- pre-trained already by Luigi
# =============================================================================
# # Return list of pathnames for masked videos in the data directory under given imaging dates
# maskedfilelist = []
# date_total = []
# for i, expDate in enumerate(IMAGING_DATES):
#     tmplist = lookforfiles(os.path.join(DATA_DIR, "MaskedVideos", expDate), ".*.hdf5$")
#     date_total.append(len(tmplist))
#     maskedfilelist.extend(tmplist)
# print("%d masked videos found for imaging dates provided:\n%s" % (len(maskedfilelist), [*zip(IMAGING_DATES, date_total)]))
#
# print("\nManual labelling:\nTotal masked videos found: %d\n" % len(maskedfilelist))
# 
# # Interactive plotting (for user input when labelling plots)
# tic = time.time()
# for i in range(len(maskedfilelist)):    
#     maskedfilepath = maskedfilelist[i]
#     # Manually outline + assign labels to food regions + save coordinates + trajectory overlay to file           
#     manuallabelling(maskedfilepath, n_poly=1, out_dir='MicrobiomeAssay', save=True, skip=True)
# print("Manual labelling complete!\n(Time taken: %d seconds.)\n" % (time.time() - tic))
# =============================================================================


#%% PLOT WORM TRAJECTORIES



#%% VISUALISE SUMMARY FEATURES



#%% PRINCIPLE COMPONENTS ANALYSIS

#%%
   
# =============================================================================
# # CHECK FOR RESULTS FILES 
# RESULTS_FILES = ['_featuresN.hdf5', '_skeletons.hdf5', '_intensities.hdf5']
# 
# 
# # Return list of all results files present for given imaging date
# results_list = []
# for date in IMAGING_DATES:
#     for filetype in RESULTS_FILES:
#         results = lookforfiles(os.path.join(DATA_DIR, 'Results', date), filetype)
#         results_list.extend(results)
# 
# # Check that results files are present for each masked video
# incomplete_analysis = []
# for maskedvideo in maskedfilelist:
#     # Results filepaths
#     features = changepath(maskedvideo, returnpath = 'features')
#     skeletons = changepath(maskedvideo, returnpath = 'skeletons')
#     intensities = changepath(maskedvideo, returnpath = 'intensities')
#     
#     # Do features/skeletons/intensities files exist for masked videos?
#     for i, resultspath in enumerate([features, skeletons, intensities]):
#         if resultspath not in results_list:
#             print("Missing %s files for masked video: \n%s" \
#                   % (RESULTS_FILES[i].split('_')[1].split('.')[0], resultspath))
#             incomplete_analysis.append(resultspath)
#             
# print("%d results files missing!" % len(incomplete_analysis))
# =============================================================================

# =============================================================================
#  # OBTAIN MASKED VIDEO FILEPATHS FOR METADATA
# 
# print("Processing project metadata...")
# n_filepaths = sum([isinstance(path, str) for path in metadata.filename])
# n_entries = len(metadata.filename)
# 
# print("%d/%d filepath entries found in metadata" % (n_filepaths, n_entries))
# print("Fetching filepaths for %d entries..." % (n_entries - n_filepaths))
# 
# # Return list of pathnames for masked videos in the data directory under given imaging dates
# maskedfilelist = []
# date_total = []
# for i, expDate in enumerate(IMAGING_DATES):
#     tmplist = lookforfiles(os.path.join(DATA_DIR, "MaskedVideos", expDate), ".*.hdf5$")
#     date_total.append(len(tmplist))
#     maskedfilelist.extend(tmplist)
# print("%d masked videos found for imaging dates: %s" % (len(maskedfilelist), date_total))
# 
# # Pre-allocate column in metadata for storing camera ID info
# metadata['cameraID'] = ''  
# 
# # Retrieve filenames for entries in metadata
# for i, filepath in enumerate(metadata.filename):
#     
#     # If filepath is already present, make sure there is no spaces
#     if isinstance(filepath, str):
#         metadata.loc[i,'filename'] = filepath.replace(" ", "")  
#         
#     else:
#         # Extract date/set/camera info for metadata entry
#         file_info = metadata.iloc[i]
#         date = str(file_info['date_yyyymmdd'])                                  # which experiment date?
#         set_number = str(file_info['set_number'])                               # which set/run?
#         channel = int(file_info['channel'])                                     # which camera channel?
#         hydra = int(str(file_info['instrument_name']).lower().split('hydra')[1])# which Hydra?
#         
#         # Obtain unique ID for hydra camera using hydra number / channel number
#         # by indexing cameraID dataframe using hydra/channel combination
#         camera_info = CAM2CH_DF.iloc[(hydra - 1) * 6 + (channel - 1)]
#         
#         # Quick (not so fool-proof) check that indexing worked successfully
#         if camera_info['channel'] != 'Ch' + str(channel):
#             print("ERROR: Incorrect camera channel!")
#             pdb.set_trace()
#         
#         # Get cameraID for file
#         cameraID = camera_info['cameraID']
#         
#         # Update cameraID in metadata
#         metadata.loc[i,'cameraID'] = cameraID
#                                 
# #        # Re-format date string (Phenix only)
# #        d = datetime.datetime.strptime(date, '%Y%m%d')
# #        date = d.strftime('%d%m%Y')
#         
#         # Query by regex using date/camera info
#         querystring = '/food_behaviour_s{0}_'.format(set_number) + date + '_'
#         
#         # Search for 1st video segment, if present, record filepath in metadata (ie. "000000.hdf5")   
#         for file in maskedfilelist:
#             if re.search(querystring, file) and re.search(('.' + cameraID + '/000000.hdf5'), file):
#                 
#                 # Record filepath to parent directory (containing all chunks for that video)
#                 metadata.loc[i,'filename'] = os.path.dirname(file)
#               
# # Return list of pathnames for featuresN files
# print("Searching for results files..")
# featuresNlist = []
# for i, expDate in enumerate(IMAGING_DATES):
#     tmplist = lookforfiles(os.path.join(DATA_DIR, "Results", expDate), ".*_featuresN.hdf5$")
#     featuresNlist.extend(tmplist)
# 
# # Pre-allocate columns in metadata for storing n_video_chunks, n_featuresN_files
# metadata['n_maskedvideo_chunks'] = ''
# metadata['n_featuresN_files'] = ''
# 
# # Add n_video_chunks, n_featuresN_files as columns to metadata
# extra_chunk = 0   
# for i, dirpath in enumerate(metadata.filename):
#     print("Processing entry %d/%d" % (i, len(metadata.filename)))
#     # If filepath is present, return the filepaths to the rest of the chunks for that video
#     if isinstance(dirpath, str):
#         file_info = metadata.iloc[i]
#         set_number = int(file_info['set_number'])
#         date = file_info['date_yyyymmdd']
#         
#         chunklist = [chunkpath for chunkpath in maskedfilelist if dirpath in chunkpath] 
#         n_chunks = len(chunklist)
#         if n_chunks != 10:
#             if n_chunks == 11: 
#                 extra_chunk += 1
#                 #print("Extra chunk!", n_chunks, date, set_number)
#             else:
#                 pdb.set_trace()
#         #cameras = list(set([re.findall(r'(?<=\d{8}_\d{6}\.)\d{8}', chunkpath)[0] for chunkpath in chunklist]))
#         #channels = {CAM2CH_DICT[k] for k in cameras}
#         
#         # Record number of video segments (chunks) in metadata
#         metadata.loc[i, 'n_maskedvideo_chunks'] = n_chunks
#         
#         featlist = [featpath for featpath in featuresNlist if dirpath.replace("MaskedVideos", "Results") in featpath]
#         n_featuresN = len(featlist)
#         
#         # Record the number of featuresN files
#         metadata.loc[i, 'n_featuresN_files'] = n_featuresN
#         
# #        print("Number of video chunks: %d\nNumber of featuresN files: %d\n" % (n_chunks, n_featuresN))          
# 
# matches = sum([isinstance(path, str) for path in metadata.filename]) - n_filepaths
# print("\nComplete!\n%d filepaths added." % matches)
# 
# #print("%d extra chunks found (11th video segment)" % extra_chunk)
# 
# # Save full metadata
# print("Saving updated metadata..")
# metadata.to_csv(os.path.join(PROJECT_ROOT_DIR, "metadata.csv"))
# print("Done.")
# =============================================================================

