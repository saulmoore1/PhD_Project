#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shuffle wells for CeMbio frozen stock plates (96-well format)

@author: sm5911
@date: 8/10/2020

"""

# Imports
import numpy as np
import pandas as pd

# Globals
CEMBIO_LAYOUT_PATH = "/Volumes/behavgenom$/Saul/CeMbioScreen/AuxiliaryFiles/CeMbio_strain_plate_layout_Schulenburg_20200928.xlsx"
SAVE_PATH = "/Volumes/behavgenom$/Saul/CeMbioScreen/AuxiliaryFiles/Frozen_Stock_Plate_Well_Mappings_CeMbio_Strains_to_96-well_MANUAL_20201008.csv"

# stock plates
n_frozen_stock_plates = 3

# seed for randomness
seed = 20201009
np.random.seed(seed)

############### read CeMbio strain layout

cembio_plate_layout_df = pd.ExcelFile(CEMBIO_LAYOUT_PATH).parse(sheet_name='CeMbio_plate_layout', 
                                                                header=0,
                                                                index_col=0)

well_IDs = [(row,col+1) for row in 'ABCDEFGH' for col in range(6)] # 48-well plate

cembio_df = pd.DataFrame(index=range(len(well_IDs)), 
                         columns=['well_ID', 'strain_ID'], 
                         dtype=str)
for i, (row, col) in enumerate(well_IDs):
    strain = cembio_plate_layout_df.loc[row,col]
    well = row + str(col)
    cembio_df.loc[i,'well_ID'] = well
    cembio_df.loc[i,'strain_ID'] = strain
cembio_df = cembio_df.fillna('OP50') # fill in empty well (H6) with OP50
print(cembio_df.head(10))

############### create shuffled 96-well mappings

# stock plate stock wells (this is a 3 by 96 list of lists)
all_dst_wells = []
for pc in range(n_frozen_stock_plates):
    all_dst_wells.append([r+str(col+1) for r in 'ABCDEFGH' for col in range(12)])

# cembio library wells
library_wells = list(cembio_df['well_ID'])

# loop on the stock plates to fill
mapping_dict = {}
for pc, stock_wells in enumerate(all_dst_wells):
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
    
    # check that each CeMbio strain is represented twice per shuffled stock plate
    for i in np.unique(lib_wells):
        assert len(np.where(np.array(lib_wells) == i)[0]) == 2

    # now these are not in order of library
    mapping = list(zip(lib_wells, stock_wells))
    mapping.sort(key=lambda x: x[0])
    assert len(mapping) == 96

    for lib_well, stock_well in mapping:
        key = (lib_slot, lib_well)
        value = (stock_slot, stock_well)
        if key not in mapping_dict.keys():
            mapping_dict[key] = [value]
        else:
            mapping_dict[key].append(value)

assert len(mapping_dict) == len(library_wells) * n_frozen_stock_plates
assert len([v for vv in mapping_dict.values() for v in vv]) == 96 * n_frozen_stock_plates

# Swap round entries for these two wells due to mistake when making frozen plates
mapping_dict[(0, 'F2')][1] = (0, 'G10')
mapping_dict[(0, 'F4')][1] = (0, 'G11')

manual_reference_df = pd.DataFrame(columns=['source_well','strain_ID','stock_plate','stock_well'])
index = 0
for source_plate_location, frozen_stock_plate_location in mapping_dict.items():
    source_plate, source_well = source_plate_location
    strain_name = list(cembio_df.loc[cembio_df['well_ID']==source_well, 'strain_ID'])[0]
    for duplicate_well in range(len(frozen_stock_plate_location)):
        shuffled_frozen_stock_plate, shuffled_frozen_stock_well = frozen_stock_plate_location[duplicate_well]
        manual_reference_df.loc[index, 'source_well'] = source_well
        manual_reference_df.loc[index, 'strain_ID'] = strain_name
        manual_reference_df.loc[index, 'stock_plate'] = shuffled_frozen_stock_plate
        manual_reference_df.loc[index, 'stock_well'] = shuffled_frozen_stock_well
        index += 1

# Sort dataframe by columns 1-6 instead of rows A-H
manual_reference_df['id_key'] = manual_reference_df['source_well'] + '_' +\
                                manual_reference_df['stock_plate'].astype(str) + '_' +\
                                manual_reference_df['stock_well'].astype(str)
manual_reference_df = manual_reference_df.set_index('id_key')
manual_reference_df = manual_reference_df.reindex(sorted(manual_reference_df.index,\
                      key=lambda x: x.split('_')[0][::-1])).reset_index().drop(columns=['id_key'])
        
# change 'OP50' to 'empty' in strain_ID column - OP50 will be cultured separately
# and so will not be included in the frozen stocks - these wells will be empty
manual_reference_df.loc[manual_reference_df['strain_ID']=='OP50', 'strain_ID'] = "empty (reserved for OP50)"

print(manual_reference_df.head(10))

# # Sort dataframe by shuffled stock plate wells
# manual_reference_df = manual_reference_df.sort_values(by=['stock_plate','stock_well'])

# Save dataframe to file to be read manually
manual_reference_df.to_csv(SAVE_PATH, index=False)