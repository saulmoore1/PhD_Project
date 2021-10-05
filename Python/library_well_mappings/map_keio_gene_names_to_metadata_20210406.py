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

#%%
def make_mapping_dict(path_gene_mapping=PATH_GENE_MAPPING):
    """ Make mapping dictionary of gene names for the Keio knockout strains in each plate / well """
    # read csv
    mapping_df = pd.read_csv(path_gene_mapping)
    
    # replace column name whitespaces with underscore and make lowercase
    mapping_df.columns = [c.replace(" ", "_").lower() for c in mapping_df.columns]
    
    # replace missing gene names ' ' with NaNs
    mapping_df['gene_name'] = mapping_df['gene_name'].replace(r'^\s*$', np.nan, regex=True)
    
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
    
    metadata = pd.read_csv(metadata_path)
    
    # convert column named 'bacteria_strain' to 'gene_name'
    metadata = metadata.rename(columns={'bacteria_strain' : 'gene_name'})

    # extract plate number from 'imaging_plate_id' and fill source_plate_id column
    metadata['source_plate_id'] = [i.split("_p")[-1].split("_")[0] for i in metadata['imaging_plate_id']]
    
    assert not any(metadata['well_name'].isna())
    
    for (plate, well), gene in mapping_dict.items():
        
        metadata.loc[metadata[np.logical_and(metadata['source_plate_id'] == str(plate), 
                                             metadata['well_name'] == well)].index,
                     'gene_name'] = gene
    
    return metadata

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Map Keio gene name data to plate/well data in\
                                     experiment day metadata')
    parser.add_argument('--metadata_path', help="Path to (day) metadata file to update with gene\
                        name info", type=str)
    parser.add_argument('--save_path', help="Path to save as new metadata", default=None, type=str)
    args = parser.parse_args()
    
    if args.save_path is None:
        args.save_path = str(args.metadata_path).replace('.csv', '_updated.csv')
    
    mapping_dict = make_mapping_dict()

    metadata_updated = map_gene_name_to_metadata(args.metadata_path, 
                                                 mapping_dict, 
                                                 saveto=args.save_path)
    
    metadata_updated.to_csv(args.save_path, header=True, index=False)
    
    print("Metadata updated.")
