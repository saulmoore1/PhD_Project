#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Map plate layout list to metadata
- Will map on well name column only, so make sure each well has only one strain associated with it!

Format expected: CSV (rows=A1-H12, cols=[well_name,food_type])

@author: sm5911
@date: 9/2/21

"""

# Imports
import argparse
import pandas as pd
from pathlib import Path

# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fill food type data in metadata')
    parser.add_argument('--plate_layout_path', help="Path to CSV file containing 96-well plate \
                        layout as a matrix (rows=A-H, cols=1-12)", default=None, type=str)
    parser.add_argument('--metadata_path', help="Path to metadata to fill with extracted strain \
                        list for 'food_type'", default=None, type=str)
    parser.add_argument('--saveto', help="Path to save updated metadata (If None, will overwrite \
                        metadata)", default=None, type=str)
    args = parser.parse_args()
    
    assert Path(args.plate_layout_path).exists() and Path(args.metadata_path).exists()
    
    # Extract list of strains from the plate layout
    plate_layout = pd.read_csv(args.plate_layout_path, header=None, index_col=None, na_values='NA')
    
    strains = plate_layout.values.flatten() # concatenate row-wise (A1-A12, B1-B12, ..., H1-H12)
    wells = [row + str(col + 1) for row in 'ABCDEFGH' for col in range(12)]
    
    plate_layout_dict = {well : strain for well, strain in zip(wells, strains)}
    
    # Read metadata
    metadata = pd.read_csv(args.metadata_path, header=0, index_col=None, dtype={'comments':str})
    
    # Add food type data to metadata    
    # TODO: Ensure that plate layout CSV filename matches desired imaging_plate_id in metadata
    metadata['food_type'] = metadata['well_name'].map(plate_layout_dict)

    # Save updated metadata
    if args.saveto is not None:
        saveto = Path(args.saveto)
        saveto.parent.mkdir(exist_ok=True, parents=True)
    else:
        saveto = Path(args.metadata_path)
        #saveto = Path(str(args.metadata_path).replace('.csv', '_with_food.csv'))
    
    # layout_df = pd.DataFrame(data=plate_layout_dict.values(), index=plate_layout_dict.keys())
    # layout_df = layout_df.reset_index(drop=False)
    # layout_df.columns = ['well_name', 'food_type']
    # layout_df.to_csv(saveto, header=True, index=False, na_rep='NA')
    metadata.to_csv(saveto, index=None)
    print("Done! Food type column updated in metadata.")