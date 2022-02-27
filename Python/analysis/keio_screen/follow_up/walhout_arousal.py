#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-referencing Walhout 2019 list of 244 hits with my initial Keio screen results to look for 
arousal phenotypes in any of these strains, which have been shown to slow worm development 
(through low iron). This developmental delay can be rescued by iron or antioxidant supplementation.
 
@author: sm5911
@date: 27/02/2022

"""

#%% Imports

import pandas as pd
from pathlib import Path

#%% Globals

PROJECT_DIR = "/Users/sm5911/Documents/Keio_Screen"

WALHOUT_SI_PATH = "/Users/sm5911/Documents/Keio_Screen/Walhout_2019_arousal_crossref/Walhout_2019_SI_Table1.xlsx"

FEATURE = 'motion_mode_paused_fraction'

#%% Functions

def load_walhout_244(supplementary_path=WALHOUT_SI_PATH):
    """ Function to load Walhout 2019 SI Table 1 and extract the full list of 244 strains that slow
        C. elegans development 
    """
    xl = pd.ExcelFile(supplementary_path)
    SI_Table1 = xl.parse(sheet_name=xl.sheet_names[0], header=0, index_col=None)
    walhout_244_gene_list = list(sorted(SI_Table1['Keio Gene Name'].unique()))
    
    return walhout_244_gene_list

def main():
    
    # load Walhout 2019 gene list
    walhout_244 = load_walhout_244(WALHOUT_SI_PATH)
    
    # load my Keio screen results
    metadata = pd.read_csv(Path(PROJECT_DIR) / "metadata.csv", dtype={"comments":str})
    features = pd.read_csv(Path(PROJECT_DIR) / "features.csv")
    
    # filter my results for Walhout low-iron (slow development) genes
    meta_walhout = metadata[metadata['gene_name'].isin(walhout_244)]
    feat_walhout = features.reindex(meta_walhout.index)
    
    return

#%% Main

if __name__ == "__main__":
    main()
    