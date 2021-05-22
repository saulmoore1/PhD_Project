#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load and append COG Category to Keio strain metadata (Supporting Information: Supplementary Table 7)

@author: sm5911
@date: 07/03/2021

"""

#%% Imports

import argparse
import pandas as pd
from pathlib import Path

#%% Globals 

EXAMPLE_METADATA_PATH = "/Volumes/hermes$/KeioScreen_96WP/AuxiliaryFiles/metadata_annotated.csv"
EXAMPLE_SUP_PATH = "/Volumes/hermes$/KeioScreen_96WP/AuxiliaryFiles/Baba_et_al_2006/Supporting_Information/Supplementary_Table_7.xls"

#%% Functions


def load_supplementary_7(path_sup_info):
    """ Load Supplementary Information 7 and return contents as dataframe with numbers stripped 
        from column names
    """
        # 
    supplementary_7 = pd.read_excel(path_sup_info, skiprows=1, header=0)
    
    # strip off numbers from column names    
    new_cols = [c.split('. ')[-1] for c in supplementary_7.columns]
    new_cols = [c.replace('number','num') for c in new_cols]
    new_cols = [c.replace(' ','_') for c in new_cols]
    
    supplementary_7.columns = new_cols
    
    return supplementary_7

def append_supplementary_7(metadata, supplementary_7, column_name='food_type'):
    """ Append Supplementary Information to metadata for genes in metadata 'food_type' column """
    
    
    assert column_name in metadata.columns and 'gene' in supplementary_7.columns

    metadata.loc[:,'column_order'] = range(metadata.shape[0])
    
    # TODO: Do not drop duplicate functionality genes with multiple COGs
    print("WARNING: Dropping duplicate COG entries for genes (using first only)")
    _idx = metadata.index # record index prior to merge
    updated_metadata = metadata.merge(supplementary_7.drop_duplicates('gene'), 
                                      how='left', 
                                      left_on=column_name, 
                                      right_on='gene')
    
    updated_metadata = updated_metadata.sort_values(by='column_order', axis=0, ascending=True)
    updated_metadata.index = _idx # restore index after merge
    
    return updated_metadata.drop(columns=['gene','column_order'])
    
#%% Main

if __name__ == "__main__":
    # Accept command-line inputs
    parser = argparse.ArgumentParser(description='Append supplementary information for COG category to metadata')
    parser.add_argument('--metadata_path', help="Path to metadata file", default=EXAMPLE_METADATA_PATH)
    parser.add_argument('--path_sup_info', help="Path to supplementary information",
                        default=EXAMPLE_SUP_PATH, type=str)
    parser.add_argument('--save_path', help="Path to save updated metadata",
                        default=None, type=str)
    args = parser.parse_args()  
    
    # Overwrite metadata is save path is None
    args.save_path = Path(args.metadata_path) if args.save_path is None else Path(args.save_path)

    # Load metadata
    assert Path(args.metadata_path).exists()
    metadata = pd.read_csv(args.metadata_path, dtype={"comments":str})
    
    supplementary_7 = load_supplementary_7(args.path_sup_info)
    
    updated_metadata = append_supplementary_7(metadata, supplementary_7)
    
    # save metadata to file
    args.save_path.parent.mkdir(exist_ok=True, parents=True)
    updated_metadata.to_csv(args.save_path, index=False)

    
    
    
    

     

