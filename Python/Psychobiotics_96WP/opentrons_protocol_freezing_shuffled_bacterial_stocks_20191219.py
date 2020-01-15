#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@date:   19/12/2019
@author: Saul Moore (sm5911)

###############################################################################
## OpenTrons Protocol - Microbiome Assay
-  96-well, multichannel
-  Randomisation, column-wise, leaving a random column free for OP50 control
-  1 source plate  ->  9 destination plates 
                       (shuffling columns)
-  OPTIOINAL: Eco friendly - only 1 tip rack required
               (each column of tips is re-used 9 times -> 9 destination plates)
-  Dispense 100ul bacterial culture using P-300 single-channel pipette
    (solution in LB broth, overnight growth from single colony)
   
Tip rack position(s):         3
Source plate position:        1,4,7,10,6
Destination plate positions:  2,5,8,11,9 

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

# Pipette (single-channel) parameters
single_pipette_type = 'p300-Single'
single_pipette_mount = 'right'

# Tip rack parameters
eco_friendly_tip_use = False
tiprack_type = 'opentrons-tiprack-200ul'
tiprack_startfrom = '1'
tiprack_slots = ['3']
    
# Source plate (bacterial solution) parameters
library_slots = ['6','10','7','4','1']
source_type = '96-flat' # TODO: not '96-well-plate-sqfb-whatman'/'96-well-plate-pcr-thermofisher' ??
n_columns = 12

# Destination plate parameters
stock_slots = ['9','11','8','5','2']
destination_type = '96-flat'

bacterial_volume = 100 # ul

#%% LABWARE

assert '96-flat' in labware.list()

# Define custom labware
if '48-well-plate-sarsted' not in labware.list():
    custom_plate = labware.create(
        '48-well-plate-sarsted',                                               # name of labware
        grid=(8, 6),                                                           # specify layout (columns, rows)
        spacing=(12.4, 12.4),                                                  # distances (mm) between each (column, row)
        diameter=10,                                                           # diameter (mm) of each well on the plate
        depth=17.05,                                                           # depth (mm) of each well on the plate
        volume=500)                                                            # Sarsted had a "volume of work"

    print('Wells in 48WP Sarsted:')
    for well in custom_plate.wells():
        print(well)

if '96-well-plate-sqfb-whatman' not in labware.list():
    custom_plate = labware.create(
        '96-well-plate-sqfb-whatman',                                          # name of labware
        grid=(12, 8),                                                          # specify layout (columns, rows)
        spacing=(8.99, 8.99),                                                  # distances (mm) between each (column, row)
        diameter=7.57,                                                         # diameter (mm) of each well on the plate (here width at bottom)
        depth=10.35,                                                           # depth (mm) of each well on the plate
        volume=650)                                                            # this is the actual volume as per specs, not a "volume of work"

    print('Wells in 96WP Whatman:')
    for well in custom_plate.wells():
        print(well)

if '96-well-plate-pcr-thermofisher' not in labware.list():
    custom_plate = labware.create(
        '96-well-plate-pcr-thermofisher',                                      # name of labware
        grid=(12, 8),                                                          # specify layout (columns, rows)
        spacing=(9.00, 9.00),                                                  # distances (mm) between each (column, row)
        diameter=5.50,                                                         # diameter (mm) of each well on the plate (here width at top!!)
        depth=15.00,                                                           # depth (mm) of each well on the plate
        volume=200)                                                            # actual specs, as per manufacturer's website

    print('Wells in 96WP PCR Thermo Fisher:')
    for well in custom_plate.wells():
        print(well)

#%% PIPETTES + TIPRACK

# Define multi-channel pipette + tipracks
if single_pipette_type == 'p300-Single': # safety check
    tipracks = [labware.load(tiprack_type, slot) \
                      for slot in tiprack_slots]
    pipette_single = instruments.P300_Single(
        mount=single_pipette_mount,
        tip_racks=tipracks)
pipette_single.start_at_tip(tipracks[0].well(tiprack_startfrom))
pipette_single.plunger_positions['drop_tip'] = -6

#%% SOURCE-DESTINATION PLATE MAPPINGS

# Hard-coded library wells
rows = 'ABCDEFGH'
library_wells = ([r+str(col+1) for r in rows for col in range(len(destination_slots))]\
                  + ['A6'] + [r+'7' for r in rows] + ['D8'])
library_wells.sort()

# Destination wells
all_dst_wells = [r+str(col+1) for r in rows for col in range(12)]

# Omit random column for OP50



src_cols = np.arange(n_columns) # Array of column numbers

# Shuffling of source plate columns for destination plate mappings
dst_shuffled_mapping_dict = {}
for ds in destination_slots:
    dst_cols = src_cols.copy() # array of column numbers to be shuffled
    np.random.shuffle(dst_cols) # shuffle columns in destination. This acts in place!!
    dst_shuffled_mapping_dict[ds] = dst_cols

# Mapping from source plate to destination plate:
mapping_dict = {}
for sc in src_cols:
    for ds in destination_slots:
        dst_col = dst_shuffled_mapping_dict[ds][sc]
        mapping_dict[(source_slot, sc, ds)] = dst_col
   
for key, value in mapping_dict.items():
    _src_slot, _src_col, _dst_slot = key
    _dst_col = value
    print('Source: slot {0} col {1} --> Destination: slot {2} col {3}'.format(_src_slot, _src_col, _dst_slot, _dst_col))

#%% TRANSLATE MAPPINGS TO ROBOT LABWARE INSTRUCTIONS
        
src_plate = labware.load(source_type, source_slot)
dst_plates = [labware.load(destination_type, dst_slot) for dst_slot in destination_slots]
   
wells_mapping = {}
for i,(k,v) in enumerate(mapping_dict.items()):
    source_slot, src_col, dst_slot = k
    dst_col = v       
    
    # Load labware for dst plate 
    dst_plate = dst_plates[int(np.where(np.array(destination_slots)==dst_slot)[0])]
    
    # Create list of wells with the right offset
    src_wells = src_plate.rows('A')[int(src_col)]
    dst_wells = dst_plate.rows('A')[int(dst_col)].bottom(agar_thickness)
    
#    src_wells = [well.bottom(frombottom_off) for well in src_plate.rows('A')]
#    dst_wells = [well.bottom(agar_thickness) for well in dst_plate.rows('A')]   
#    src_wells = src_wells[src_col]
#    dst_wells = dst_wells[dst_col]
        
    # Store wells_mapping
    wells_mapping[(src_plate, src_col, dst_plate)] = (src_wells, dst_wells)
    
#print(wells_mapping)

#%% COMMANDS

# Safety command to make sure the robot starts with no previous tips attached
pipette_multi.drop_tip()
count_used_tips()

# Dispense solution from source plate into destination plates
for src_col in src_cols:
    for dst_plate in dst_plates:
        
        dst_col = mapping_dict[(source_slot, src_col, dst_slot)]
        #print(dst_col)
                
        src_wells, dst_wells = wells_mapping[(src_plate, src_col, dst_plate)]
        
        # Pick up wells
        is_pick_up_tip_time = ((dst_plate == dst_plates[0]) or
                               (eco_friendly_tip_use == False))
        
        if is_pick_up_tip_time:
            pipette_multi.pick_up_tip()

        # Aspirate (suck up) bacteria in source plate + saturate the tips to remove air bubbles
        pipette_multi.aspirate(air_gap, src_wells.bottom(30), rate=4.0)
        pipette_multi.aspirate(aspirating_volume, src_wells.bottom(frombottom_off), rate=4.0)
        pipette_multi.dispense(dispensing_volume, dst_wells, rate=4.0)
        pipette_multi.blow_out()
        
        # Drop tips (eco-friendly option)
        is_drop_tip_time = ((dst_plate == dst_plates[-1]) or
                            (eco_friendly_tip_use == False))
        
        if is_drop_tip_time:
            print("Dropping tips")
            pipette_multi.drop_tip()
                    
        print('{} {} -> {} {}'.format(src_plate.parent, src_wells, dst_plate.parent, dst_wells))
        count_used_tips()

        #robot.pause(60)
        #pipette_multi.reset_tip_tracking() # This tells the robot to restart 
        # the tip counter if you're going to change tipracks manually. 
        # The robot otherwise thinks you've run out of tips.
        
# Save robot commands to file
if not robot.is_simulating():
    import datetime
    out_fname = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+'_runlog.txt'
    out_fname = '/data/user_storage/opentrons_data/protocols_logs/' + out_fname
    with open(out_fname,'w') as fid:
        for command in robot.commands():
            print(command,file=fid)
            
print(robot.commands())
            