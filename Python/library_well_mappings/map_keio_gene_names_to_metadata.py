#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keio Screen - Map knockout strain gene name to metadata

@author: sm5911
@date: 11/05/2021

"""

import argparse
import numpy as np
import pandas as pd

PATH_GENE_MAPPING = "/Volumes/hermes$/KeioScreen_96WP/AuxiliaryFiles/keio_library_gene_well_mapping.csv"

def make_mapping_dict(path_gene_mapping=PATH_GENE_MAPPING):
    """ Map Keio gene name info to plate/well info in day metadata """
    # read csv
    mapping_df = pd.read_csv(path_gene_mapping)
    
    # replace column name whitespaces with underscore and make lowercase
    mapping_df.columns = [c.replace(" ", "_").lower() for c in mapping_df.columns]
    
    # replace missing gene names ' ' with NaNs
    mapping_df['gene_name'] = mapping_df['gene_name'].replace(r'^\s*$', np.nan, regex=True)
    
    # create 'well_name' column
    mapping_df['well_name'] = [row + str(col) for row, col in zip(mapping_df['row'], mapping_df['col'])]
    
    # create mapping dictionary from 'gene_name' and 'plate'/'well_name' info
    mapping_dict = {(plate, well) : gene for plate, well, gene in zip(mapping_df['plate'], 
                                                                      mapping_df['well_name'], 
                                                                      mapping_df['gene_name'])}
    return mapping_dict

def map_gene_name_to_metadata(metadata_path, mapping_dict, saveto=None):
    """ """
    
    metadata = pd.read_csv(metadata_path)
    
    # metadata_updated = metadata.map(mapping_dict)
    
    return metadata_updated

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Map Keio gene name data to plate/well data in\
                                     experiment day metadata')
    parser.add_argument('--metadata_path', help="Path to (day) metadata file to update with gene\
                        name info", type=str)
    args = parser.parse_args()

