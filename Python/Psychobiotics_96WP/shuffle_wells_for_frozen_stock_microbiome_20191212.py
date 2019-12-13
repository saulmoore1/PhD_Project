#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@date:   12/12/2019
@author: Saul Moore (sm5911)

SHUFLLE WELLS FOR FROZEN STOCK

A simple script to generate well mappings for creating randomised frozen stock 
plates for the microbiome strains that can be cultured directly in 96-well 
format, side-stepping the need for the OpenTrons robot for plate shuffling.
An option exists to repeat-sample strains as necessary when shuffling to fill 
up remaining wells of a 96-well plate, making sure every strain is represented 
at least once.

"""
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#%%
seed = 20191212
np.random.seed(seed)

n_shuffled_frozen_stock_plates = 5

# This is an option to shuffle just the well positions containing the Schulenburg
# strains in the layout recieved November 2019
Schulenburg_strain_layout = True 

in_filepath = "/Volumes/behavgenom$/Saul/MicrobiomeAssay96WP/AuxiliaryFiles/20191212/metadata_20191212.csv"
out_dirpath = "/Volumes/behavgenom$/Saul/MicrobiomeAssay96WP"

#%%
metadata = pd.read_csv(in_filepath)

master_plate_data = metadata[metadata['plate_number']==1][['food_type','well_number']]

Schulenburg_strain_data = master_plate_data[master_plate_data['food_type']!='OP50']

master_well_list = list(master_plate_data['well_number'])

Schulenburg_strain_well_list = list(Schulenburg_strain_data['well_number'])

#master_well_list = []
#import string
#for letter in string.ascii_uppercase[:8]:
#    for number in range(1,13):
#        master_well_list.append(letter + str(number))

#%% # Selected wells containing microbiome strains must be shuffled into 96-well plates

if Schulenburg_strain_layout:
    assert len(Schulenburg_strain_well_list) < len(master_well_list) # To ensure the every strain is represented at least once

    shuffled_wells_mapping_df = pd.DataFrame(index = master_well_list,\
                                             columns=['Plate_{0}'.format(i+1) for \
                                                      i in range(n_shuffled_frozen_stock_plates)])
    for shuffled_plate in shuffled_wells_mapping_df.columns:
        shuffled_well_list = Schulenburg_strain_well_list.copy()
        np.random.shuffle(shuffled_well_list) # this acts in-place
        
        n_remaining_wells = len(master_well_list) - len(shuffled_well_list)
        extra_strains_to_fill_96WP = np.random.choice(shuffled_well_list, n_remaining_wells, replace=False)
        shuffled_well_list.extend(extra_strains_to_fill_96WP) # this acts in place
        shuffled_wells_mapping_df[shuffled_plate] = shuffled_well_list
        
    save_path = os.path.join(out_dirpath, "Frozen_Stock_Plate_Well_Mappings_Microbiome_Strains_Only_to_96-well.csv")

#%% # Full plate shuffle
    
else:    
    shuffled_wells_mapping_df = pd.DataFrame(index = master_well_list,\
                                             columns=['Plate_{0}'.format(i+1) for \
                                                      i in range(n_shuffled_frozen_stock_plates)])
    for shuffled_plate in shuffled_wells_mapping_df.columns:
        shuffled_well_list = master_well_list.copy()
        np.random.shuffle(shuffled_well_list) # this acts in-place
        shuffled_wells_mapping_df[shuffled_plate] = shuffled_well_list 
    
    save_path = os.path.join(out_dirpath, "Frozen_Stock_Plate_Well_Mappings_96-well_to_96-well.csv")

#%% # Save shuffled well mappings to file
    
shuffled_wells_mapping_df.to_csv(save_path, index=True)

#%% # Plot of strain representations across shuffled frozen stock plates

#tally_strain_representation = [np.array([(shuffled_wells_mapping_df[plate]==i).sum(axis=0)\
#                                for i in Schulenburg_strain_well_list])\
#                                    for plate in shuffled_wells_mapping_df.columns]
#plt.close('all')
#plt.plot(tally_strain_representation)
##plt.xlim(4.75,5.25)
#plt.show()


