#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rename Hydra Videos to replace a unique sub-string in filepaths 
(that is incompatible with Tierpsy Tools analysis functions) with the unique 
imaging run number (required for compiling metadata and results), in all
RawVideos/MaskedVideos/Results files for the given imaging dates provided.

@author: sm5911
@date: 21/11/2020

"""

#%% Imports
import os
from tqdm import tqdm
from pathlib import Path

#%% Globals

PROJECT_ROOT_DIR = "/Volumes/hermes$/Filipe_Tests_96WP"

IMAGING_DATES = ['20201028','20201029','20201105','20201106','20201109']

# Mapping of incorrect unique substring in filepaths to correct run number
# information for compatibility with Tierpsy Tools functions
REPLACE_SUBSTR_DICT = {"_2hrs_": "_run1_",
                       "_4hrs_": "_run2_",
                       "_6hrs_": "_run3_",
                       "_24hrs_": "_run4_"}

#%% Functions

def rename_tierpsy_filepaths(PROJECT_ROOT_DIR, REPLACE_SUBSTR_DICT, IMAGING_DATES):
    """ A function to rename Tierpsy filepaths """
    
    DIRNAME_LIST = ["RawVideos", "MaskedVideos", "Results"]
    old2new = []
    for DIRNAME in DIRNAME_LIST:
        for date in IMAGING_DATES:
            # Rename raw video directories for given imaging date    
            PARENT_DIR = Path(PROJECT_ROOT_DIR) / DIRNAME
            date_dir = PARENT_DIR / str(date)
            dirpath_list = os.listdir(date_dir)
            dirpath_list = [(date_dir / dirpath) for dirpath in dirpath_list]
            for dirpath in dirpath_list:
                for key in REPLACE_SUBSTR_DICT:
                    old_name = Path(dirpath).name
                    if key in old_name:
                        # rename directory by replacing substr key with value
                        new_name = old_name.replace(key, REPLACE_SUBSTR_DICT[key])
                        new_dirpath = Path(dirpath).parent / new_name
                        old2new.append((dirpath, new_dirpath))   
    return old2new
    
#%% Main   

old2new = rename_tierpsy_filepaths(PROJECT_ROOT_DIR,
                                   REPLACE_SUBSTR_DICT, 
                                   IMAGING_DATES)
   
# Rename directories according to match old/new filepaths
print("Renaming %d filepaths\n" % len(old2new))
for (old, new) in tqdm(old2new):
    os.rename(old, new)
print("Done! Please re-compile any features summaries files!")

