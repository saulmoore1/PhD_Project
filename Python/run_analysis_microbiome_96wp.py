#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

PSYCHOBIOTICS: ANALYSIS OF MICROBIOME CORE SET (96WP ASSAY)

@author: sm5911
@date: Sun Oct 13 14:04:25 2019

A script written to visualise and interpret results for the quantitative 
behavioural analysis of freely moving N2 C. elegans raised monoxenically on 
various bacterial strains cultered from the C. elegans gut microbiome, and 
compared to N2 performance on standard laboratory strains of E. coli, with OP50 
as the control. 

"""

#%% IMPORTS & DEPENDENCIES

import os, sys, time, decimal, umap
import numpy as np
import pandas as pd
import subprocess as sp
from matplotlib import pyplot as plt
import seaborn as sns

# Record script start time
tic = time.time()

#%% GLOBAL PARAMETERS (USER-DEFINED)

PROJECT_ROOT_DIR = '/Volumes/behavgenom$/Saul/MicrobiomeAssay96WP'

IMAGING_DATES = ['20191003','20191010','20191011']

# Process the metadata prior to analysis?
PROCESS_METADATA = True

# Process feature summary files?
PROCESS_FEATURE_SUMMARY_RESULTS = True


#%% PROCESS & LOAD METADATA

# Use subprocess to call 'process_metadata_96WP.py'
# Optional: pass imaging dates as arguments to the script
if PROCESS_METADATA:
    print("\nProcessing metadata file...")
    SCRIPT_PATH = "/Users/sm5911/OneDrive - Imperial College London/Psychobiotics/Code/process_metadata_96WP.py"
    metafilepath = os.path.join(PROJECT_ROOT_DIR, "AuxiliaryFiles", "metadata.csv")
    process_metadata = sp.Popen([sys.executable, SCRIPT_PATH, metafilepath, *IMAGING_DATES])
    process_metadata.communicate()

metafilepath = os.path.join(PROJECT_ROOT_DIR, "AuxiliaryFiles", "metadata_updated.csv")
metadata = pd.read_csv(metafilepath)
print("\nMetadata file loaded.")

# Subset metadata to remove remaining entries with missing filepaths
is_filename = [isinstance(path, str) for path in metadata['filename']]
if any(list(~np.array(is_filename))):
    print("WARNING: Could not find filepaths for %d entries in metadata.\n\t Omitting these files from analysis..." \
          % sum(list(~np.array(is_filename))))
    metadata = metadata[list(np.array(is_filename))]
    # Reset index
    metadata.reset_index(drop=True, inplace=True)

  
#%% PROCESS FEATURE SUMMARY RESULTS

if PROCESS_FEATURE_SUMMARY_RESULTS:
    print("\nProcessing feature summary results...")
    process_feature_summary = sp.Popen([sys.executable, "process_feature_summary.py",\
                                        metafilepath, *IMAGING_DATES])
    process_feature_summary.communicate()


toc = time.time()
print("Total time taken: %.1f" % (toc - tic))  