#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Egg Laying - Keio Acute Effect Screen

Plots of number of eggs recorded on plates +1hr after picking 10 worms onto 60mm plates seeded 
with either BW background or fepD knockout mutant bacteria and recording without delay
(with bluelight stimulus delivered every 30 minutes)

@author: sm5911
@date: 20/01/2022

"""

#%% Imports
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt


#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Keio_Fast_Effect"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Fast_Effect/egg_counting"
IMAGING_DATES = [20211102, 20211109] # 20211116
CONTROL_STRAIN = "BW"

#%% Functions

def get_egg_counts(metadata, masked_dir, regex='metadata_eggs.csv', group_name='/full_data'):
    """ Search MaskedVideos directory for egg counter GUI save files 
        and update metadata with egg count data for matched files
    """

    # find egg_counter GUI 'metadata_eggs.csv' files
    egg_files = list(masked_dir.rglob('metadata_eggs.csv'))
    
    # assemble paths to egg files
    metadata['egg_filepath'] = [masked_dir / i / 'metadata_eggs.csv' for i in 
                                metadata['imgstore_name']]
    assert all(f in metadata['egg_filepath'].unique() for f in egg_files)
    
    n_missing_egg_files = sum([f not in egg_files for f in metadata['egg_filepath'].unique()])
    if n_missing_egg_files > 0:
        print("\nWARNING: %d entries in metadata are missing egg count data!" % n_missing_egg_files)
   
    # get egg counts + append to metadata
    for eggfile in egg_files:
        egg_data = pd.read_csv(eggfile)
        egg_counts = egg_data.groupby('frame_number').count()['group_name']
        
        # find matching entry in metadata
        meta_rowidx = np.where(metadata['egg_filepath']==eggfile)[0]
        assert len(meta_rowidx) == 1
        
        # append egg counts for each frame analysed
        for frame in list(egg_counts.index):
            metadata.loc[meta_rowidx[0],'n_eggs_frame_{}'.format(frame)] = egg_counts[frame]
    
    return metadata

#%% Main

if __name__ == "__main__":   
    
    # load metadata
    metadata_path = Path(PROJECT_DIR) / "AuxiliaryFiles" / "metadata.csv"
    metadata = pd.read_csv(metadata_path, dtype={"comments":str})
    
    # subset for imaging dates
    metadata = metadata[metadata['date_yyyymmdd'].isin(IMAGING_DATES)]
        
    # get egg data
    masked_dir = Path(PROJECT_DIR) / "MaskedVideos"
    metadata = get_egg_counts(metadata, masked_dir)
            
    # calculate number of eggs laid in first hour on-food
    metadata['number_eggs_1hr'] = metadata['n_eggs_frame_12'] - metadata['n_eggs_frame_0']
    
    # compute mean/std number of eggs on each food
    eggs = metadata[['gene_name','number_eggs_1hr']]
    
    # drop NaN entries
    eggs = eggs.dropna(subset=['gene_name','number_eggs_1hr'])
    
    strain_list = [CONTROL_STRAIN] + [s for s in eggs['gene_name'].unique() if s != CONTROL_STRAIN]
    
    # Plot bar chart + save
    plt.close()
    fig, ax = plt.subplots(figsize=(14, 8))
    ax = sns.barplot(x="gene_name", y="number_eggs_1hr", order=strain_list, data=eggs, 
                     estimator=np.mean, dodge=True, ci=95, capsize=.1, palette='plasma')
    save_path = Path(SAVE_DIR) / "eggs_after_1hr_on_food.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    
    # TODO: Use univariate_tests function with chi_sq test to compare n_eggs between fepD vs BW
     