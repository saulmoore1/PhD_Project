#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@date: 11/09/2019
@author: sm5911

###############################################################################
## OpenTrons Protocol Seed bacteria column-wise into separate 96-well plates ##
###############################################################################

Materials (what: where):
- 96WP flat, source plate of bacterial strains: 1
- 96WP square wells, bacteria destination plates: 2
- Tip racks for multi-channel pipette: 2

Protocol:
- Seed 10ul of bacteria from column(s) of wells containing bacteria in source
  plate into all wells in destination plate(s)
- Number of destination plates must equal the number of columns in the
  source plate that contain bacteria (max=5/12 due to space in OpenTrons robot)

"""

#%% IMPORTS

from opentrons import labware, instruments, robot

###############################################################################
#%% USER INTUITIVE PARAMETERS
###############################################################################

# Bacterial source plate params
source_slot = '4'                 # Position of source plate in OpenTrons robot
source_type = '96-flat'           # Type of 96WP used for source plate
source_active_columns = ['1','2'] # Columns in source 96-well plate that contain bacteria

# Destination plate params
agar_thickness = +2.5             # Adjustment for agar height in mm from bottom of well
destination_slots = ['1','2']     # Positions of destination plates in OpenTrons robot
destination_type = '96-well-plate-sqfb-whatman' # Type of 96WP use for destination plates

# Pipette params
multi_pipette_type = 'p10-Multi'  # Type of multi pipette used
multi_pipette_mount = 'left'      # Mount orientation

# Tip rack params
tiprack_type_multi = 'tiprack-10ul' # Type of tip rack used
# LF: I think we have a few boxes/refills of this: 'opentrons_96_tiprack_300ul'
tiprack_slots_multi = ['7','8']      # Positions of tip racks in OpenTrons robot
tip_start_from_multi = '1'           # Position of which well/column to start from

# Air gap params
aspirating_volume = 5            # Bacterial volume to pick up
dispensing_volume = 10            # Bacterial volume to dispense (greater to ensure all contents are dispensed)
air_gap = dispensing_volume - aspirating_volume # Resulting air gap

###############################################################################

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

#%% LABWARE

# Source plate
src_plate = labware.load(source_type, source_slot)

# Destination plates
if '96-well-plate-sqfb-whatman' not in labware.list():
    # Define custom 96WP Whatman plates
    custom_plate = labware.create(
            '96-well-plate-sqfb-whatman', # Name of labware
            grid     = (12,8),            # Specify number of (columns, rows)
            spacing  = (8.99, 8.99),      # Distance (mm) between each (column, row)
            diameter = 7.57,              # Diameter (mm) of each well on the plate
            depth    = 10.35,             # Depth (mm) of each well on the plate
            volume   = 650)               # Volume of well as per specs (not 'working volume')
    print('Wells in 96WP Whatman:')
    for well in custom_plate.wells():
        print(well)

dst_plates = [labware.load(destination_type, dst_slot) for dst_slot in\
              destination_slots]

#%% PIPETTES

# Multi-channel pipette + tip rack
if multi_pipette_type == 'p10-Multi':
    tipracks_multi = [labware.load(tiprack_type_multi, tiprack_slot)\
                      for tiprack_slot in tiprack_slots_multi]
    pipette_multi = instruments.P10_Multi(
            mount = multi_pipette_mount,
            tip_racks = tipracks_multi)

pipette_multi.start_at_tip(tipracks_multi[0][tip_start_from_multi])
pipette_multi.plunger_positions['drop_tip'] = -6

#%% COMMANDS

# Safety command - gives warning when no tip is attached
pipette_multi.drop_tip()

count_used_tips() # Should be 0 to begin with

# Each column in source plate is replicated (12 times) into separate destination plates
for src_column, dst_plate in zip(source_active_columns, dst_plates):
    print(src_column, dst_plate)
    # Iterate over top row only and 8-channel multi-pipette will fill entire plate
    src_well = src_plate.columns(src_column)[0]

    for dst_well in dst_plate.rows('A'): #dst_wells = [well.bottom(2.5) for well in dst_plate.rows('A')]
        # Adjust tip height to account for agar in destination plate
        dst_well = dst_well.bottom(agar_thickness) # with offset
        #print("SOURCE WELL:", src_well, "\nDST_WELL:", dst_well)

        # Pick up tips (8-channel)
        pipette_multi.pick_up_tip()

        # Aspirate (suck up) bacteria in source plate + saturate the tips to remove air bubbles
        pipette_multi.aspirate(air_gap, src_well.bottom(30), rate=4.0)
        pipette_multi.aspirate(aspirating_volume, src_well, rate=4.0)
        pipette_multi.dispense(dispensing_volume, dst_well, rate=4.0)
        pipette_multi.blow_out()
        pipette_multi.drop_tip()

        count_used_tips() # In total should be (n_dst_plates * 96)

#         This could be a one-liner equivalent
#         pipette_multi.transfer(dispensing_volume,
#                                src_well,
#                                dst_well,
#                                rate=4.0,
#                                blow_out=True)


