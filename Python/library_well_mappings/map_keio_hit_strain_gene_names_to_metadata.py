#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keio Confirmational Screen - Map gene names for interesting 'hit' Keio mutants to metadata

@author: sm5911
@date: 11/05/2021

"""

#%% Imports

import argparse
import numpy as np
import pandas as pd

#%% Globals

METADATA_PATH = "/Volumes/hermes$/KeioScreen2_96WP/AuxiliaryFiles/20210719/20210719_day_metadata.csv"
PATH_GENE_MAPPING = "/Volumes/hermes$/KeioScreen2_96WP/AuxiliaryFiles/keio_hit_strains_plate_layout_gene_well_mapping.csv"

#%% Functions

def make_mapping_dict(path_gene_mapping=PATH_GENE_MAPPING):
    """ Make mapping dictionary of gene names for the Keio knockout strains in each plate / well """
    
    # read mapping csv
    mapping_df = pd.read_csv(path_gene_mapping, dtype={"comment":str})
    
    # make sure all 96 wells of the plate are recorded with location of hit strains
    assert mapping_df.shape[0] == 96
    
    # add 'plate' column for Keio hit strain confirmation screen plate 
    # (identical layout across technical replicates)
    mapping_df['plate'] = 1

    # replace 'strain' with 'gene_name'
    mapping_df.columns = [c if not c == 'strain' else 'gene_name' for c in mapping_df.columns]
    
    # create 'well_name' column
    mapping_df['well_name'] = [row + str(col).upper() for row, col in zip(mapping_df['row'], 
                                                                          mapping_df['col'])]
    
    # create mapping dictionary from 'gene_name' and 'plate'/'well_name' info
    mapping_dict = {(plate, well) : gene for plate, well, gene in zip(mapping_df['plate'], 
                                                                      mapping_df['well_name'], 
                                                                      mapping_df['gene_name'])
                    if plate % 2 != 0} # TODO: Allele set 'A' and 'B' for Keio library??
    
    # TODO: Add control BW 'gene_name' as 'BW-wt' or something
    
    return mapping_dict

def map_gene_name_to_metadata(metadata_path, mapping_dict, saveto=None):
    """ Map gene names to (day) metadata using 'well_name' and 'imaging_plate_id' data """
    
    metadata = pd.read_csv(metadata_path, dtype={"comments":str})
    
    # convert column named 'bacteria_strain' to 'gene_name'
    metadata = metadata.rename(columns={'bacteria_strain' : 'gene_name'})

    #TODO: Fill in 'source_plate_id' column info for hit strains
    
    assert not any(metadata['well_name'].isna())   
    
    for (plate, well), gene in mapping_dict.items():
        
        metadata.loc[metadata[metadata['well_name'] == well].index,
                     'gene_name'] = gene
        
    nan_gene_names = metadata['gene_name'].isna().sum()
    if nan_gene_names > 0:
        print("%d entries in metadata with missing gene names" % nan_gene_names)
    
    return metadata

#%% Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Map gene name data to plate/well data in\
                                     experiment day metadata')
    parser.add_argument('--metadata_path', help="Path to (day) metadata file to update with gene\
                        name info", type=str, default=METADATA_PATH)
    parser.add_argument('--save_path', help="Path to save as new metadata", default=None, type=str)
    args = parser.parse_args()
    
    # if args.save_path is None:
    #     args.save_path = str(args.metadata_path).replace('.csv', '_updated.csv')
    
    args.save_path = METADATA_PATH # confident that it works as intended, so will overwrite metadata
    
    mapping_dict = make_mapping_dict()

    metadata_updated = map_gene_name_to_metadata(args.metadata_path, 
                                                 mapping_dict, 
                                                 saveto=args.save_path)
    
    metadata_updated.to_csv(args.save_path, header=True, index=False)
    
    print("Metadata updated.")
    
