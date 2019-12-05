#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@date:   27/11/2019
@author: Saul Moore (sm5911)

###############################################################################
## OpenTrons Protocol - Microbiome Assay
-  96-well, multichannel
-  Randomisation, column-wise 
-  1 source plate  ->  5 destination plates
-  1 tip rack required (each column of tips is re-used 5 times for each destination plate)
-  Dispense 5ul bacterial culture (solution in LB broth, overnight growth from single colony),
   using multichannel pipette (shuffling columns)
   
Tip rack position:            8
Source plate position:        2
Destination plate positions:  1,3,4,5,6 

###############################################################################
"""

#%% IMPORTS

import numpy as np
from opentrons import labware, instruments, robot

seed = 20191127 # Set seed for reproducibility (NB: Use the experimental date for the actual experiment)
np.random.seed(seed)

#%% FUNCTIONS

def count_used_tips():
    """ A simple function to tally the number of tips that have been used thus
        far in the protocol. """
    tc = 0
    for c in robot.commands():
        if 'Picking up tip wells' in c:
            tc += 8
        elif 'Picking up tip well' in c:
            tc += 1
    print('Tips used so far: %d' % tc)
    return

#%% USER-INTUITIVE PARAMETERS

# Pipette (multi-channel) parameters
multi_pipette_type = 'p10-Multi'
multi_pipette_mount = 'left'

# Tip rack parameters
tiprack_slots = ['8']
tiprack_type = 'opentrons-tiprack-10ul'
tiprack_startfrom = '1'

# Source plate (bacterial solution) parameters
source_slot = '2'
source_type = '96-flat' # TODO: not '96-well-plate-sqfb-whatman'/'96-well-plate-pcr-thermofisher' ??
frombottom_off = +0.3 # distance of bottom of source wells from bottom of source plate (in millimetres, mm)
dispense_volume = 5 # volume of solution to dispense from src plate into dst plates (in microlitres, ul)
n_columns = 12

# Destination plate parameters
destination_slots = ['1','3','4','5','6']
destination_type = '96-well-plate-sqfb-whatman'
# - (Normal NGM / No Peptone agar) # TODO: 150ul-200ul ?
agar_thickness = +3.7 # mm from the bottom of the well (for 200ul agar per well)

#%% LABWARE

# Define custom labware
if '48-well-plate-sarsted' not in labware.list():
    custom_plate = labware.create(
        '48-well-plate-sarsted',        # name of you labware
        grid=(8, 6),                    # specify amount of (columns, rows)
        spacing=(12.4, 12.4),           # distances (mm) between each (column, row)
        diameter=10,                    # diameter (mm) of each well on the plate
        depth=17.05,                    # depth (mm) of each well on the plate
        volume=500)                     # Sarsted had a "volume of work"

    print('Wells in 48WP Sarsted:')
    for well in custom_plate.wells():
        print(well)

if '96-well-plate-sqfb-whatman' not in labware.list():
    custom_plate = labware.create(
        '96-well-plate-sqfb-whatman',   # name of you labware
        grid=(12, 8),              # specify amount of (columns, rows)
        spacing=(8.99, 8.99),      # distances (mm) between each (column, row)
        diameter=7.57,             # diameter (mm) of each well on the plate (here width at bottom)
        depth=10.35,               # depth (mm) of each well on the plate
        volume=650)                # this is the actual volume as per specs, not a "volume of work"

    print('Wells in 96WP Whatman:')
    for well in custom_plate.wells():
        print(well)

if '96-well-plate-pcr-thermofisher' not in labware.list():
    custom_plate = labware.create(
        '96-well-plate-pcr-thermofisher',   # name of you labware
        grid=(12, 8),              # specify amount of (columns, rows)
        spacing=(9.00, 9.00),      # distances (mm) between each (column, row)
        diameter=5.50,             # diameter (mm) of each well on the plate (here width at top!!)
        depth=15.00,               # depth (mm) of each well on the plate
        volume=200)                # as per manufacturer's website

    print('Wells in 96WP PCR Thermo Fisher:')
    for well in custom_plate.wells():
        print(well)

#%% PIPETTES + TIPRACK

# Define multi-channel pipette + tipracks
if multi_pipette_type == 'p10-Multi': # safety check
    tipracks = [labware.load(tiprack_type, slot) \
                      for slot in tiprack_slots]
    pipette_multi = instruments.P10_Multi(
        mount=multi_pipette_mount,
        tip_racks=tipracks)
pipette_multi.start_at_tip(tipracks[0].well(tiprack_startfrom))
pipette_multi.plunger_positions['drop_tip'] = -6

#%% TRANSLATE MAPPINGS TO ROBOT LABWARE INSTRUCTIONS

# Mapping from source plate to destination plate:
mapping_dict = {}
src_cols = np.arange(n_columns) # Array of column numbers

for ds in destination_slots:
    dst_cols = src_cols.copy() # array of column numbers to be shuffled
    np.random.shuffle(dst_cols) # shuffle columns in destination. This acts in place!!
    mapping_dict[(source_slot,ds)] = (src_cols,dst_cols)

for key, value in mapping_dict.items():
    _src_slot, _dst_slot = key
    _src_cols, _dst_cols = value
    for _src_col, _dst_col in zip(_src_cols, _dst_cols):
        print('slot {0} col {1} --> slot {2} col {3}'.format(_src_slot, _src_col, _dst_slot, _dst_col))
        
#print(mapping_dict)

src_plate = labware.load(source_type, source_slot)
dst_plates = [labware.load(destination_type, dst_slot) for dst_slot in destination_slots]

# Translate well mappings to robot instructions for given labware
wells_mapping = {}
for i,(k,v) in enumerate(mapping_dict.items()):
    print(i,k,v)            

    # Unpack slots
    _src_slot, _dst_slot = k
    _src_cols, _dst_cols = v
        
    # Load labware for src and dst
    #src_plate = labware.load(source_type, _src_slot) # NOT REQUIRED: Single source plate needed
    dst_plate = dst_plates[i]
    print("Destination plate: %s\nSource plate: %s" % (dst_plate, src_plate))
    
    # Create list of wells with the right offset
    src_wells = [well.bottom(frombottom_off) for well in src_plate.rows('A')]
    dst_wells = [well.bottom(agar_thickness) for well in dst_plate.rows('A')]
    
    # Normal lists, so should be re-shuffleable
    src_wells = [src_wells[wi] for wi in _src_cols]
    dst_wells = [dst_wells[wi] for wi in _dst_cols]
    
    # Store wells_mapping
    wells_mapping[(src_plate, dst_plate)] = (src_wells, dst_wells)
    
#print(wells_mapping)

#%% COMMANDS

# Safety command to make sure the robot starts with no previous tips attached
pipette_multi.drop_tip()
count_used_tips() # this should be 0 to start with

# Dispense solution from source plate into destination plates
for plates_tuple, wells_tuple in wells_mapping.items():
    # Unpack well mappings
    src_plate, dst_plate = plates_tuple
    src_wells, dst_wells = wells_tuple

    # Transfer solution
    pipette_multi.transfer(dispense_volume,
                           src_wells,
                           dst_wells,
                           new_tip='always',
                           blow_out=True)

    for s,d in zip(src_wells, dst_wells):
        print('{} {} -> {} {}'.format(src_plate.parent, s[0], dst_plate.parent, d[0]))

    count_used_tips()
    robot.pause(60)
    pipette_multi.reset_tip_tracking()

# Save robot commands to file
if not robot.is_simulating():
    import datetime
    out_fname = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+'_runlog.txt'
    out_fname = '/data/user_storage/opentrons_data/protocols_logs/' + out_fname
    with open(out_fname,'w') as fid:
        for command in robot.commands():
            print(command,file=fid)
            
# TODO: Since dispensing from the same source column 5 times,
# could experiment with only changing tips when changing source columns
# NB: May have bad effects like the droplet of bacteria not detaching completely
