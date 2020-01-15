#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script written to create shuffled well mappings and save mapping to file to 
follow during manual pipetting to make randomised frozen stock plates of the 
Schulenburg microbiome bacterial strains. 

1 random column is omitted from the shuffled mapping (for OP50 control)


@author: sm5911
@date: 14/01/2020

"""

import numpy as np
import pandas as pd

####################### user intuitive parameters

# Path to save dataframe of shuffled well mappings
save_path = "/Volumes/behavgenom$/Saul/MicrobiomeAssay96WP/Frozen_Stock_Plate_Well_Mappings_Microbiome_Strains_to_96-well_MANUAL_20200114.csv"

# stock plates
n_frozen_stock_plates = 5
n_columns = 12

# seed for randomness
seed = 20200114
np.random.seed(seed)


############### create mapping

# hardcode library wells (they are fixed)
rows = 'ABCDEFGH'
library_wells = ([r+str(col+1) for r in rows for col in range(5)]
                 + ['A6'] + [r+'7' for r in rows] + ['D8'])
library_wells.sort()

# destination wells
# this is a 5 by 88 list of lists, taking out a column every time
all_dst_wells = []
for pc in range(n_frozen_stock_plates):
    col_to_reserve = np.random.randint(1, n_columns)
    print('Reserving col {} in stock plate {} for OP50'.format(col_to_reserve, pc))
    all_dst_wells.append([r+str(col+1)
                          for r in rows for col in range(12)
                          if col != col_to_reserve])

# loop on plates
mapping_dict = {}
for pc, stock_wells in enumerate(all_dst_wells):
    # get the deck slots
    lib_slot = pc
    stock_slot = pc
    lib_wells = library_wells.copy()
    # first shuffle the copy of library wells and assign it to the stock wells
    np.random.shuffle(lib_wells)
    # now extract a random selection of len(stock)-len(lib) and
    # attach them to shuffled library
    n_remaining_wells = len(stock_wells) - len(lib_wells)
    lib_wells.extend(np.random.choice(library_wells,
                                      n_remaining_wells,
                                      replace=False))
    # now these are not in order of library
    mapping = list(zip(lib_wells, stock_wells))
    mapping.sort(key=lambda x: x[0])
    print(len(mapping)) # SHOULD BE 88 with a column removed for OP50

    for lib_well, stock_well in mapping:
        key = (lib_slot, lib_well)
        value = (stock_slot, stock_well)
        if key not in mapping_dict.keys():
            mapping_dict[key] = [value]
        else:
            mapping_dict[key].append(value)

print(len(mapping_dict))
print(len([v for vv in mapping_dict.values() for v in vv]))

manual_reference_dataframe = pd.DataFrame(columns=['source_plate','source_well','destination_plate','destination_well'])
index = 0
for source_plate_location, frozen_stock_plate_location in mapping_dict.items():
    source_plate, source_well = source_plate_location
    for duplicate_well in range(len(frozen_stock_plate_location)):
        shuffled_frozen_stock_plate, shuffled_frozen_stock_well = frozen_stock_plate_location[duplicate_well]
        manual_reference_dataframe.loc[index, 'source_plate'] = source_plate
        manual_reference_dataframe.loc[index, 'source_well'] = source_well
        manual_reference_dataframe.loc[index, 'destination_plate'] = shuffled_frozen_stock_plate
        manual_reference_dataframe.loc[index, 'destination_well'] = shuffled_frozen_stock_well
        index += 1

# Save dataframe to file to be read manually
manual_reference_dataframe.to_csv(save_path, index=False)
