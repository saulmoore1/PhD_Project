#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:51:16 2020

@author: sm5911
"""

import pandas as pd

mapping_plate_5 = pd.read_csv("/Volumes/behavgenom$/Saul/MicrobiomeAssay96WP/schulenburg_shuffled_plate_5.csv")
mapping_food = pd.read_csv("/Volumes/behavgenom$/Saul/MicrobiomeAssay96WP/Schulenburg_master_plate_well_to_strain_mapping.csv", skiprows=1)


foo = pd.merge(left=mapping_plate_5, right=mapping_food, how='left', left_on='source_well', right_on='well_number')
foo2 = foo[['destination_well','strain_identifier']]
foo2 = foo2.rename(columns={'destination_well':'well_name'})


bar = [l+str(n+1) for l in "ABCDEFGH" for n in range(12)]

bar = pd.DataFrame(bar, columns=['well_name'])

foobar = pd.merge(bar, foo2, how='left', on='well_name')

foobar.loc[foobar['strain_identifier'].isna(), 'strain_identifier'] = "OP50"

foo2.query('well_name == "H9"')
foobar.query('well_name == "H9"')

foobar.to_csv("/Volumes/behavgenom$/Saul/MicrobiomeAssay96WP/PLATE_5_well_to_strain_mapping.csv")
