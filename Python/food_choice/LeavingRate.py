#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT: LEAVING RATES

A script written to analyse the food choice assay videos and Tierpsy results, 
and invesitgate worm velocities and leaving rates across food patches.

@author: sm5911
@date: 29/04/2019

"""

# GENERAL IMPORTS / DEPENDENCIES
import os, sys, time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches

# CUSTOM IMPORTS
sys.path.insert(0, '/Users/sm5911/Documents/GitHub/PhD_Project/Python')

from food_choice.SM_calculate import foodchoice
from food_choice.SM_plot import hexcolours, plottimeseries
from food_choice.SM_find import changepath
from food_choice.SM_read import getskeldata

#%% PRE-AMBLE
# GLOBAL VARIABLES
PROJECT_ROOT_DIR = '/Volumes/behavgenom$/Saul/FoodChoiceAssay/' # Project working directory
DATA_DIR = PROJECT_ROOT_DIR.replace('Saul', 'Priota/Data') # Location of features files

fps = 25 # frames per second

# Conduct analysis on new videos only?
NEW = True

# Read metadata
fullMetaData = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "fullmetadata.csv"), header=0, index_col=0)

if NEW:
    fullMetaData = fullMetaData[fullMetaData['worm number']==10]

n_files = len(fullMetaData['filename'])
if NEW:
    print("%d NEW video file entries found in metadata." % n_files)
else:
    print("%d video file entries found in metadata." % n_files)

# Extract assay information
pretreatments = list(np.unique(fullMetaData['Prefed_on']))
assaychoices = list(np.unique(fullMetaData['Food_Combination']))
treatments = list(np.unique([assay.split('/') for assay in assaychoices]))
concentrations = list(np.unique(fullMetaData['Food_Conc']))

# Plot parameters
colours = hexcolours(len(treatments)) # Create a dictionary of colours for each treatment (for plotting)
colour_dict = {key: value for (key, value) in zip(treatments, colours)}

#%% FIGURE 7 - Leaving Probability Time Series (GROUPED BY TREATMENT COMBINATION)
# TODO: Probability of leaving: Leaving rate per minute throughout video
# Following the Methods of Shtonda & Avery (2006) 
# Probability of Leaving, P(leaving), in each minute-intervals of the recording is the ratio of the number
# of worms that left the colony during the minute to the total number of worms in the colony at the start of that minute. 
# (Fig.2A, per-minute probability was averaged between 1h and 2h of the recording (i.e. for minutes 61–120))
# (Fig.3C, P(leaving) was averaged in several intervals as described in the legend to this figure.

binning_window = int(5 * 60 * fps)

# Read recorded leaving events
true_leaving_df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "Results", "LeavingRate",\
                              "leavingevents_true.csv"), header=0, index_col=0)

# Group leaving event data by prefed-assaychoice-concentration treatment combinations
groupedLeavingData = true_leaving_df.groupby(['Prefed_on','Food_Combination','Food_Conc'])

groupedfullMetaData = fullMetaData.groupby(['Prefed_on','Food_Combination','Food_Conc'])

# For each prefood-assaychoice-concentration treatment combination
tic = time.time()
for p, prefood in enumerate(pretreatments):
    # Initialise plot for prefed group
    plt.close("all")
    fig, axs = plt.subplots(nrows=len(assaychoices), ncols=len(concentrations),\
                            figsize=(16,8), sharex=True, sharey=True) # 15 subplots (3 assay types, 5 assay concentrations)
    plot_df = pd.DataFrame(columns=['filename','n_Events'])
    for a, assay in enumerate(assaychoices):
        for c, conc in enumerate(concentrations):
            
            # Get prefood-assaychoice-concentration group data
            metaData = groupedfullMetaData.get_group((prefood,assay,conc))
            leavingData = groupedLeavingData.get_group((prefood,assay,conc)).reset_index(drop=True)

            info = metaData.iloc[0]
            # Plot labels/colours
            colnames = leavingData.iloc[0]['Food_Combination'].split('/')
            if colnames[0] == colnames[1]:
                colnames = ['{}_{}'.format(col, i+1) for i, col in enumerate(colnames)]

            labels = [lab.split('_')[0] for lab in colnames]
            colours = [colour_dict[treatment] for treatment in labels]

            df_prob_leaving = pd.DataFrame(columns=colnames)
            for maskedfilepath in metaData['filename']:
                onfoodpath = changepath(maskedfilepath, returnpath='onfood')
                onfood_df = pd.read_csv(onfoodpath, header=0, index_col=0)
                                
                count_df = foodchoice(onfood_df, mean=False)
                xmax = count_df.index.max()
                
                # Frame binning
                bins = np.arange(count_df.index.min(), count_df.index.max(), binning_window)
                wormcount_binned_2mins = count_df.groupby(np.digitize(count_df.index, bins))
                
                df_count_binned = pd.DataFrame(index=range(0,len(bins)), columns=colnames)
                for food in colnames: 
                    for fc, frame in enumerate(bins):
                        start_frame_info = onfood_df[onfood_df['frame_number']==frame]
                        n_worms_start = start_frame_info[food].sum()
                        df_count_binned.loc[fc, food] = n_worms_start
                
                leavingData_video = leavingData[leavingData['filename']==maskedfilepath]
            
                # Insert leaving events into timeline order
                leaving_df = pd.DataFrame(0, index=np.arange(0,xmax).astype(int), columns=colnames)
                for i, frame in enumerate(leavingData_video['frame_number']):
                    if not int(frame) == int(xmax): # Omit last frame leaving event
                        food = leavingData_video.iloc[i]['Food_left_from']
                        leaving_df[food].iloc[int(frame)] = leaving_df[food].iloc[int(frame)] + 1
            
                wormsleaving_binned_2mins = leaving_df.groupby(np.digitize(leaving_df.index, bins)) # use same bins as above
                df_leaving_binned = wormsleaving_binned_2mins.sum()
                
                prob_leaving = df_leaving_binned / df_count_binned
                
                # Convert Inf to NaN
                prob_leaving = prob_leaving.replace([np.inf, -np.inf], np.nan)
                
                # Drop all frames where both foods contain NaN values 
                # prob_leaving = prob_leaving.dropna(subset=colnames, how="all") # Preserve time-binning index
                
                df_prob_leaving = df_prob_leaving.append(prob_leaving, sort=True)
                
            fundict = {x:['mean','std'] for x in colnames}
            plot_df = df_prob_leaving.groupby(df_prob_leaving.index).agg(fundict)
            
            for food in colnames:
                adjusted_values = np.where(plot_df[food, 'mean'].values > 1)[0]
                print("Limiting %d probability values to 1" % len(adjusted_values))
                plot_df[food, 'mean'].values[plot_df[food, 'mean'].values > 1] = 1
                
            
            plottimeseries(plot_df, colour_dict, window=False, acclimtime=False,\
                           annotate=False, legend=False, show=True, ax=axs[a,c])
            
            axs[a,c].set_ylim(-0.05, 1.05)
            
            # Add number of replicates (videos) for each treatment combination
            axs[a,c].text(0.79, 0.9, ("n={0}".format(len(metaData['filename']))),\
                          transform=axs[a,c].transAxes, fontsize=18)
            
            # Set column labels on first row
            if a == 0:
                axs[a,c].text(0.5, 1.15, ("conc={0}".format(conc)),\
                              horizontalalignment='center', fontsize=22,\
                              transform=axs[a,c].transAxes)
                
            # Set main y axis label + ticks along first column of plots
            if c == 0:
                yticks = list(np.round(np.linspace(0,1,num=6,endpoint=True),decimals=1))
                axs[a,c].set_yticks(yticks)
                axs[a,c].set_yticklabels(yticks)
                axs[a,c].set_ylabel("{0}".format(assay), labelpad=15, fontsize=18)
                if a == 1:
                    axs[a,c].text(-0.5, 0.5, "Probability of Leaving Food",\
                                  fontsize=24, rotation=90, transform=axs[a,c].transAxes,\
                                  verticalalignment='center')
            else:
                axs[a,c].set_yticklabels([])

            # Set main x axis label + ticks along final row of plots
            if a == len(assaychoices) - 1:
                xticks = np.arange(0, plot_df.index.max()+2, 7.5)
                axs[a,c].set_xticks(xticks)
                xticklabels = [str(int(lab*2)) for lab in xticks]
                axs[a,c].set_xticklabels(xticklabels)
                if c == 1:
                    axs[a,c].set_xlabel("Time (minutes)", labelpad=25, fontsize=24, horizontalalignment='left')
            else:
                axs[a,c].set_xticklabels([])
# TODO: Fix bins on OP50/OP50 row -- xticks/labels
# TODO: Add acclimation time + fix probability calculation errors                                       
#                acclim = int(leavingData['Acclim_time_s'][0] * fps)
#                leaving_df.index = leaving_df.index + acclim 
    
    # Add 'prefed on' to multiplot
#    plt.text(max(plot_df.index), -0.6, "Prefed on: {0}".format(prefood), horizontalalignment='right', fontsize=30)

    # Add legend
    patches = []
    for key, value in colour_dict.items():
        if key == "None":
            continue
        patch = mpatches.Patch(color=value, label=key)
        patches.append(patch)
    fig.legend(handles=patches, labels=treatments, loc="upper right", borderaxespad=0.4,\
               frameon=False, fontsize=15)
    
    # Tight-layout + adjustments
    fig.tight_layout(rect=[0.07, 0.02, 0.9, 0.93])
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.show(); plt.pause(0.0001)
    
    # Save figure 7
    fig_name = "LeavingProbabilityTS_prefed" + prefood + ".png"
    figure_out = os.path.join(PROJECT_ROOT_DIR, "Results", "Plots", fig_name)
    plt.savefig(figure_out, format='png', dpi=300)
print("Complete!\n(Time taken: %d seconds)" % (time.time() - tic))

#%% # TODO: Determine worm locomotory state: Roaming or Dwelling 
# - Investigate worm velocity before/after leaving

for i, maskedfilepath in enumerate(fullMetaData['filename']):
    # Extract file information
    info = fullMetaData.iloc[i,:]
    conc = info['Food_Conc']
    assaychoice = info['Food_Combination']
    prefed = info['Prefed_on']
    foods = info['Food_Combination'].split('/')
    if foods[0] == foods[1]:
        foods = ["{}_{}".format(food, i + 1) for i, food in enumerate(foods)]
    print("\nProcessing file: %d/%d\n%s\nAssay:  %s\nConc:   %.3f\nPrefed: %s" % (i + 1,\
          len(fullMetaData['filename']), maskedfilepath, assaychoice, conc, prefed))
    
    skeletonfilepath = changepath(maskedfilepath, returnpath="skeletons")
    
    skeleton_data = getskeldata(skeletonfilepath)

