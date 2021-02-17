#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script to record which strains are in which wells for shuffled microbiome 
frozen stock plates 
- To be recorded in experimental metadata

@author: sm5911
@date: 20/01/2020

"""

import pandas as pd

####
master_to_shuffled_stock_well_mapping_INpath = "/Volumes/behavgenom$/Saul/MicrobiomeAssay96WP/Frozen_Stock_Plate_Well_Mappings_Microbiome_Strains_to_96-well_MANUAL_20200114.csv"
master_well_to_strain_mapping_INpath = "/Volumes/behavgenom$/Saul/MicrobiomeAssay96WP/Schulenburg_master_plate_well_to_strain_mapping.csv"
shuffled_strain_locations_OUTpath = "/Volumes/behavgenom$/Saul/MicrobiomeAssay96WP/Frozen_Stock_Plate_Microbiome_Strain_Locations_20200114.csv"

shuffled_well_mapping_df = pd.read_csv(master_to_shuffled_stock_well_mapping_INpath)
well_to_strain_mapping_df = pd.read_csv(master_well_to_strain_mapping_INpath, skiprows=1)

shuffled_strain_locations_merged_df = pd.merge(left = shuffled_well_mapping_df, 
                                               right = well_to_strain_mapping_df, 
                                               how = 'left',
                                               left_on = 'source_well', 
                                               right_on = 'well_number')

shuffled_well_to_strain_mapping_df = shuffled_strain_locations_merged_df[['source_well','destination_plate','destination_well','strain_identifier']]

shuffled_well_to_strain_mapping_df.columns = ['master_well', 'shuffled_plate', 'shuffled_well', 'strain_ID']

shuffled_well_to_strain_mapping_df = shuffled_well_to_strain_mapping_df.sort_values(by=['shuffled_plate','shuffled_well'])

shuffled_well_to_strain_mapping_df.to_csv(shuffled_strain_locations_OUTpath, index=False)
