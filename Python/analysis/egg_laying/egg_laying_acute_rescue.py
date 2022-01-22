#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Egg Laying - Keio Acute Antioxidant Screen

Plots of number of eggs recorded on plates +24hrs after picking 10 worms onto 60mm plates seeded 
with either BW background or fepD knockout mutant bacteria, in combination with exogenous delivery
of antioxidants (with bluelight stimulus delivered every 5 minutes)

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

PROJECT_DIR = "/Volumes/hermes$/Keio_Acute_Rescue"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Acute_Rescue/egg_counting"
CONTROL_STRAIN = "BW"
CONTROL_ANTIOXIDANT = "None"

#%% Main

if __name__ == "__main__":
    
    # Load metadata
    metadata_path = Path(PROJECT_DIR) / "AuxiliaryFiles" / "metadata.csv"
    metadata = pd.read_csv(metadata_path, dtype={"comments":str})
    
    # Extract egg count + compute average for each strain/antioxidant treatment combination
    eggs = metadata[['gene_name','antioxidant','number_eggs_24hrs']]
    # mean_eggs = eggs.groupby(['gene_name','antioxidant']).mean()
    # std_eggs = eggs.groupby(['gene_name','antioxidant']).std()
    
    # drop NaN entries
    eggs = eggs.dropna(subset=['gene_name','antioxidant','number_eggs_24hrs'])
    
    strain_list = [CONTROL_STRAIN] + [s for s in eggs['gene_name'].unique() if s != CONTROL_STRAIN]
    antioxidant_list = [CONTROL_ANTIOXIDANT] + [a for a in eggs['antioxidant'].unique() if
                                                a != CONTROL_ANTIOXIDANT]
    
    # Plot bar chart + save
    plt.close()
    fig, ax = plt.subplots(figsize=(14, 8))
    ax = sns.barplot(x="gene_name", y="number_eggs_24hrs", hue="antioxidant", 
                     order=strain_list, hue_order=antioxidant_list, data=eggs, 
                     estimator=np.mean, dodge=True, ci=95, capsize=.1, palette='plasma')
    save_path = Path(SAVE_DIR) / "eggs_after_24hrs_on_food.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)

    # TODO: Use univariate_tests function with chi_sq test to compare n_eggs between fepD vs BW