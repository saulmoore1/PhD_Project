#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Produce Keio Strain List from well mappings

@author: sm5911
@date: 11/03/2020

"""
#%% Imports

import os, sys, string
import numpy as np
import pandas as pd

#%% Functions

def compile_Keio_strain_mapping(Keio_library_well_mapping_dir):
    files_list = [i for i in os.listdir(Keio_library_well_mapping_dir) if "Keio_Plate" in i]
    print("Well mappings found for %d plates" % len(files_list))
    
    well_strain_df = pd.DataFrame(index = np.arange(384*len(files_list)), columns = ["plate_number_384wp","well_number_384wp","food_type"], dtype=str)
    for f, file in enumerate(files_list):
        plate_384_df = pd.read_csv(os.path.join(Keio_library_well_mapping_dir, file), index_col=0)
        well_strain_plate_dict = {(f + 1, i + str(j)): plate_384_df.loc[i,str(j)] for i in string.ascii_uppercase[:16] for j in np.arange(1,25)}
        
        plateID = [k[0] for k in well_strain_plate_dict.keys()]
        wellID = [k[1] for k in well_strain_plate_dict.keys()]
        strainID = [str(v) for v in well_strain_plate_dict.values()]
        well_strain_df.loc[f*384:f*384+383, "plate_number_384wp"] = plateID
        well_strain_df.loc[f*384:f*384+383, "well_number_384wp"] = wellID
        well_strain_df.loc[f*384:f*384+383, "food_type"] = strainID
        
    return well_strain_df


#%%
def Keio_from_384wp_to_96_wp(well_strain_df, quadrant, plateID=None):
    """ Quadrant layout example: 
              1   2   3   4
            
        A     A   B   A   B
        B     C   D   C   D
        C     A   B   A   B
        D     C   D   C   D
        
    """
    quadrants = np.array([["A","B"],["C","D"]])
    quadrants = pd.Series(np.tile(quadrants, (8, 12)).flatten())
    
    if type(quadrant) == str:
        quadrant = quadrant.upper()
    elif type(quadrant) == int and quadrant < 4:
        quadrant = string.ascii_uppercase[:4][quadrant]
        
    inds2keep = list(quadrants[quadrants == quadrant].index)
    
    if plateID:
        well_strain_df = well_strain_df[well_strain_df['plate_number_384wp'] == plateID]
    
    plate_96_df = well_strain_df.iloc[inds2keep]
    
    plate_96_wells = [i + str(j) for i in string.ascii_uppercase[:8] for j in np.arange(1,13)]
    plate_96_df['well_number_96wp'] = plate_96_wells
    
    plate_96_df['plate_number_96wp'] = [str(pID) + quadrant for pID in plate_96_df['plate_number_384wp']]
    
    return(plate_96_df)
    

#%% Main

if __name__ == "__main__":    
    print("Running script to compile Keio strain list..")
    if len(sys.argv) > 1:
        Keio_library_well_mapping_dir = sys.argv[1]
    else:
        Keio_library_well_mapping_dir = '/Volumes/hermes$/KeioScreen_96WP/AuxiliaryFiles/Keio_library_well_mappings'
        print("No path provided! Using default: %s" % Keio_library_well_mapping_dir)
        
    df = compile_Keio_strain_mapping(Keio_library_well_mapping_dir)
    print(df)
    df.to_csv(os.path.join(Keio_library_well_mapping_dir, "Keio_Well_Strain_Mappings.csv"), index=False)
    
    for pID in list(df['plate_number_384wp'].unique()):
        for Q in ['A','B','C','D']:
            df_pID_Q = Keio_from_384wp_to_96_wp(well_strain_df=df, plateID=pID, quadrant=Q)
            df_pID_Q.to_csv(os.path.join(Keio_library_well_mapping_dir, "Plate_{}_mapping.csv".format(str(pID) + Q)), index=False)
