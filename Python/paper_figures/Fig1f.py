#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1f

@author: sm5911
@date: 10/06/2023

"""

#%% Imports

import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

#%% Globals

PROJECT_DIR = "/Users/sm5911/Documents/Keio_Screen_Confirmation"
# PROJECT_DIR = "/Volumes/hermes$/Saul/Keio_Screen/Output/Keio_Screen_Confirmation"

#%% Functions

def main():
    
    # load 'clean' metadata & feature summaries
    metadata_path = Path(PROJECT_DIR) / 'metadata.csv'
    features_path = Path(PROJECT_DIR) / 'features.csv'
    
    metadata = pd.read_csv(metadata_path, header=0, index_col=None, dtype={'comments':str})
    features = pd.read_csv(features_path, header=0, index_col=None)
    
    
    return


#%% Main

if __name__ == '__main__':
    main()