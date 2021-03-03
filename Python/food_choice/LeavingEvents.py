#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT: LEAVING EVENTS

A script written to analyse the food choice assay videos and Tierpsy-generated 
feature summary data, to determine leaving events, in which individual worms
leave a food patch.

@author: sm5911
@date: 21/03/2019

"""

# GENERAL IMPORTS / DEPENDENCIES
import os, sys, time, copy
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches

# CUSTOM IMPORTS
sys.path.insert(0, '/Users/sm5911/Documents/GitHub/PhD_Project/Python')

from food_choice.SM_calculate import leavingevents, onfood
from food_choice.SM_plot import hexcolours, plotbrightfield, plotpoly, plottrajectory, plotpoints
from food_choice.SM_find import changepath
from read_data.get_trajectories import get_trajectory_data

#%% PRE-AMBLE
# GLOBAL VARIABLES
PROJECT_ROOT_DIR = '/Volumes/behavgenom$/Saul/FoodChoiceAssay/' # Project working directory
DATA_DIR = PROJECT_ROOT_DIR.replace('Saul', 'Priota/Data') # Location of features files

# Leaving event estimation parameters
fps = 25 # frames per second
threshold_leaving_time = 2 # in seconds
leaving_window = fps * threshold_leaving_time # 25fps, for n seconds
OpticalDensity600 = 1.8

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
colour_dict = {key: value for (key, value) in zip(treatments, colours)} # list comprehension

#%% Leaving Events
# - Analysis of leaving events in each video
# - Compiles a full results dataframe of leaving event info across all videos
tic = time.time()
colnames = ['filename', 'Food_Conc', 'Food_Combination', 'Prefed_on', 'nWormsAssay', 'worm_id',\
            'Food_left_from', 'frame_number', 'x', 'y', 'leaving_duration_nframes', 'Acclim_time_s']
total_leaving_events_df = pd.DataFrame(columns=colnames)
plt.ion()
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
    
    # Specify file paths
    coordfilepath = changepath(maskedfilepath, returnpath='coords')
    featurefilepath = changepath(maskedfilepath, returnpath='features')
    # plots_out = changepath(maskedfilepath, returnpath='plots', figname='leavingTSplot.eps')

    # Read coordinates file
    f = open(coordfilepath, 'r').read()
    poly_dict = eval(f) # Use 'evaluate' to read in as a dictionary, not as a string
    
    # Read trajectory data
    traj_df = get_trajectory_data(featurefilepath)

    # Compute on/off food (no filtering step)
    onfood_df = onfood(poly_dict, traj_df)
    
    # Find leaving events
    leaving_events_df = leavingevents(onfood_df, window=leaving_window, removeNone=True,\
                                      plot=False, savePath=None) # (OPTIONAL: plot=True, savePath=plots_out)
    print("Total number of leaving events: %d\nMean leaving duration (seconds): %d"\
          % (len(leaving_events_df), int(leaving_events_df['leaving_duration_nframes'].mean()/fps)))
        
    # Append file metadata info + Insert column for which food the leaving event occured
    food_left_from = []
    for i in range(len(leaving_events_df)):
        food_left_from.append(leaving_events_df.columns[np.where(leaving_events_df.iloc[i]==True)[0]][0])
    leaving_events_df['Food_left_from'] = food_left_from
    leaving_events_df['filename'] = maskedfilepath
    leaving_events_df['Food_Conc'] = conc
    leaving_events_df['Food_Combination'] = assaychoice
    leaving_events_df['Prefed_on'] = prefed
    leaving_events_df['nWormsAssay'] = info['worm number']
    leaving_events_df['Acclim_time_s'] = info['Acclim_time_s']
    
    # Drop file-specific column
    leaving_events_df = leaving_events_df.drop(foods, axis=1)
    
    # Append to full leaving events dataframe
    total_leaving_events_df = total_leaving_events_df.append(leaving_events_df[colnames])
plt.ioff()

 # We can drop the onfood index, as we are storing worm_id/frame_number for each leaving event
total_leaving_events_df = total_leaving_events_df.reset_index(drop=True)
print("\nTotal number of leaving events found (%d videos): %d" % (fullMetaData.shape[0], total_leaving_events_df.shape[0]))

# Filter by 'leaving window' threshold (trajectory duration n frames after leaving)
true_leaving_df = total_leaving_events_df[total_leaving_events_df['leaving_duration_nframes'] >= leaving_window]
filtered_leaving_df = total_leaving_events_df[total_leaving_events_df['leaving_duration_nframes'] < leaving_window]
print("Number of leaving events filtered (duration > %d seconds): %d" % (threshold_leaving_time,\
                                                                         filtered_leaving_df.shape[0]))
print("Number of true leaving events recorded: %d" % true_leaving_df.shape[0])
# TODO: Also filter by distance from food edge (spatial threshold)

# Save both the full (unfiltered) + true (filtered) leaving events to file
print("Saving leaving results..")
savepath = os.path.join(PROJECT_ROOT_DIR, "Results", "LeavingRate", "leavingevents_all.csv")
directory = os.path.dirname(savepath)
if not os.path.exists(directory):
    os.makedirs(directory)
total_leaving_events_df.to_csv(savepath)

savepath = savepath.replace("_all.csv", "_true.csv")
true_leaving_df.to_csv(savepath)
print("\nComplete!\nTime taken: %d seconds" % (time.time() - tic))

#%% Leaving event duration histograms
# - worms that did not leave the food are not counted here
tic = time.time()
print("Reading leaving event data..")
total_leaving_events_df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "Results", "LeavingRate", "leavingevents_all.csv"), header=0, index_col=0)

# Filter by 'leaving window' threshold (trajectory duration n frames after leaving)
print("Filtering leaving event data..")
true_leaving_df = total_leaving_events_df[total_leaving_events_df['leaving_duration_nframes'] >= leaving_window]
filtered_leaving_df = total_leaving_events_df[total_leaving_events_df['leaving_duration_nframes'] < leaving_window]
print("Number of leaving events filtered (duration < %d seconds): %d" % (threshold_leaving_time,\
                                                                         filtered_leaving_df.shape[0]))

# Get histogram bin positions prior to plotting
bins = np.histogram(np.hstack((true_leaving_df['leaving_duration_nframes'],\
                    filtered_leaving_df['leaving_duration_nframes'])),\
                    bins=36000)[1]

# Plot histogram of leaving event durations + threshold for leaving event identification
print("Plotting histogram of leaving durations..")
plt.close("all")
plt.figure(figsize=(12,8))
plt.hist(true_leaving_df['leaving_duration_nframes'].values.astype(int), bins=bins, color='skyblue')
plt.hist(filtered_leaving_df['leaving_duration_nframes'].values.astype(int), bins=bins, color='gray', hatch='/')
plt.rcParams['hatch.color'] = 'lightgray'
plt.xlabel("Duration after leaving food (n frames)", fontsize=15, labelpad=10)
plt.ylabel("Number of leaving events", fontsize=15, labelpad=10)

# Zoom-in on the very short leaving durations + plot threshold for leaving event selection
plt.xlim(0,(leaving_window*5+leaving_window/5))
plt.xticks(np.arange(0, leaving_window*5+1, leaving_window))
plt.tick_params(labelsize=12)
plt.axvline(leaving_window, ls='--', lw=2, color='k')
plt.text(leaving_window+1, 300, "Threshold Leaving Duration = {0} seconds".format(int(leaving_window/fps)),\
         ha='left', va='center', rotation=-90, color='k', fontsize=13)
plt.show()

# Save histogram
fig_name = "LeavingDurationHist" + ".png"
figure_out = os.path.join(PROJECT_ROOT_DIR, "Results", "Plots", fig_name)
plt.tight_layout()
plt.savefig(figure_out, format='png', dpi=300)
print("\nComplete!\n(Time taken: %d seconds)" % (time.time() - tic))

#%% Leaving event trajectory overlay plots
tic = time.time()
true_leaving_df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "Results", "LeavingRate", "leavingevents_true.csv"), header=0, index_col=0)

for i, maskedfilepath in enumerate(fullMetaData['filename']):
    # Extract file information
    info = fullMetaData.iloc[i,:]
    conc = info['Food_Conc']
    assaychoice = info['Food_Combination']
    prefed = info['Prefed_on']
    print("\nProcessing file: %d/%d\n%s\nAssay:  %s\nConc:   %.3f\nPrefed: %s" % (i + 1,\
          len(fullMetaData['filename']), maskedfilepath, assaychoice, conc, prefed))
    
    # Specify file paths
    coordfilepath = changepath(maskedfilepath, returnpath='coords')
    featurefilepath = changepath(maskedfilepath, returnpath='features')
    plots_out = changepath(maskedfilepath, returnpath='plots', figname='LeavingEventsOverlay.png')
    
    try:
        # Read coordinates file
        f = open(coordfilepath, 'r').read()
        poly_dict = eval(f) # Use 'evaluate' to read in as a dictionary, not as a string
        
        # Plot first brightfield image
        plt.close("all")
        fig, ax = plotbrightfield(maskedfilepath, frame=0, figsize=(10,10))
        
        # Overlay food regions
        fig, ax = plotpoly(fig, ax, poly_dict, colour=False)
        
        # Overlay worm trajectories
        fig, ax = plottrajectory(fig, ax, featurefilepath, downsample=10)
        
        # Plot leaving events for video
        fig, ax = plotpoints(fig, ax, true_leaving_df[true_leaving_df.filename==maskedfilepath]['x'],\
                             true_leaving_df[true_leaving_df.filename==maskedfilepath]['y'], color='r',\
                             marker='+', markersize=3, linestyle='')
        print("Number of leaving events found: %d" %\
              true_leaving_df[true_leaving_df.filename==maskedfilepath].shape[0])
        
        # Save plot
        directory = os.path.dirname(plots_out)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.tight_layout()
        plt.savefig(plots_out, format='png', dpi=300) # Save figure
        plt.show(); plt.pause(1)
    except Exception as e:
        print(e)
        continue
print("Done!\n(Time taken: %d seconds)" % (time.time() - tic))

#%% FIGURE 4 - Box plots of number of leaving events recorded (GROUPED BY ASSAY / CONCENTRATION)

# Read recorded leaving events
true_leaving_df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "Results", "LeavingRate",\
                              "leavingevents_true.csv"), header=0, index_col=0)

# Group leaving event data by prefed-assaychoice-concentration treatment combinations
groupedLeavingData = true_leaving_df.groupby(['Prefed_on','Food_Combination','Food_Conc'])

# For each prefood-assaychoice-concentration treatment combination
tic = time.time()
for p, prefood in enumerate(pretreatments):
    # Initialise plot for prefed group
    plt.close("all")
    fig, axs = plt.subplots(nrows=len(assaychoices), ncols=len(concentrations),\
                            figsize=(14,9), sharey=True) # 15 subplots (3 assay types, 5 assay concentrations)
    ymax = 150
    for a, assay in enumerate(assaychoices):
        for c, conc in enumerate(concentrations):
            try:
                # Get prefood-assaychoice-concentration group
                df_leaving = groupedLeavingData.get_group((prefood,assay,conc)).reset_index(drop=True)
                
                # Create dataframe from groupby - number of leaving events on each food in each file appended as column
                nleaving_df = pd.DataFrame({"nLeaving" : df_leaving.groupby(['filename',\
                                            'Food_left_from']).size()}).reset_index()
    
#                MAX_LEAVING = nleaving_df.iloc[np.where(nleaving_df['nLeaving'] == max(nleaving_df['nLeaving']))[0]]
#                print(MAX_LEAVING[['Food_left_from','nLeaving']])
                
                # Plot labels/colours
                colnames = df_leaving.iloc[0]['Food_Combination'].split('/')
                if colnames[0] == colnames[1]:
                    colnames = ['{}_{}'.format(col, i+1) for i, col in enumerate(colnames)]

                labels = [lab.split('_')[0] for lab in colnames]
                colours = [colour_dict[treatment] for treatment in labels]
                
                # Seaborn colour palette
                RGBAcolours = sns.color_palette(colours)
                palette = {key: val for key, val in zip(colnames, RGBAcolours)}
                # sns.palplot(sns.color_palette(RGBAcolours))
                
                # Seaborn Boxplots
                sns.boxplot(x='Food_left_from', y='nLeaving', hue='Food_left_from',\
                            data=nleaving_df, palette=palette, dodge=False, ax=axs[a,c])
                axs[a,c].get_legend().set_visible(False)
                
                # Set x & y axes ticks/labels/limits
                if max(nleaving_df['nLeaving']) > ymax:
                    ymax = max(nleaving_df['nLeaving'])
                axs[a,c].set_ylim(-2, ymax+10)
                axs[a,c].set_ylabel('')    
                axs[a,c].set_xlim(-0.5,len(colnames)-0.5)
                axs[a,c].set_xticks(np.arange(0,len(colnames)))
                xlabs = axs[a,c].get_xticklabels()
                xlabs = [lab.get_text().split('_')[0] for lab in xlabs[:]]
                axs[a,c].set_xticklabels(labels=xlabs, fontsize=12)
                axs[a,c].set_xlabel('')
                
                if c == 0:
                    axs[a,c].set_ylabel("{0}".format(assay), labelpad=15, fontsize=18)
                    if a == 1:
                        axs[a,c].text(-0.65, 0.5, "Number of leaving events", fontsize=25,\
                                      rotation=90, verticalalignment='center', transform=axs[a,c].transAxes)

                # Add number of replicates (videos) to plots
                axs[a,c].text(0.81, 0.92, ("n={0}".format(len(np.unique(nleaving_df['filename'])))),\
                              transform=axs[a,c].transAxes, fontsize=15)
                
                # Add concentration labels to plots
                if a == 0:
                    axs[a,c].text(0.5, 1.15, "$OD_{{{}}}={}$".format(600, conc*OpticalDensity600),\
                                  horizontalalignment='center', fontsize=18,\
                                  transform=axs[a,c].transAxes)  
            except Exception as e:
                print("No videos found for treatment combination: %s\n" % e)
                axs[a,c].axis('off')
                axs[a,c].text(0.81, 0.9, "n=0", fontsize=12, transform=axs[a,c].transAxes)
                if a == 0:
                    axs[a,c].text(0.5, 1.15, "$OD_{{{}}}={}$".format(600, conc*OpticalDensity600),\
                                  horizontalalignment='center', fontsize=18,\
                                  transform=axs[a,c].transAxes)
    # Add 'prefed on' to plot
#    plt.text(1, -ymax/2.5, "Prefed on: {0}".format(prefood), horizontalalignment='center', fontsize=25)

    # Add legend
    patches = []
    for i, (key, value) in enumerate(colour_dict.items()):
        if key == "None":
            continue
        patch = mpatches.Patch(color=value, label=key)
        patches.append(patch)
    fig.legend(handles=patches, labels=list(colour_dict.keys()), loc="upper right", borderaxespad=0.1,\
               frameon=False, fontsize=15)
    
    # Plot layout + adjustments
    fig.tight_layout(rect=[0.07, 0.02, 0.9, 0.95])
    fig.subplots_adjust(hspace=0.2, wspace=0.1)    
    plt.show(); plt.pause(1)
    
    # Save figure 4
    fig_name = "LeavingEventsBox_prefed" + prefood + ".eps"
    figure_out = os.path.join(PROJECT_ROOT_DIR, "Results", "Plots", fig_name)
    plt.savefig(figure_out, format='eps', dpi=300)
print("Complete!\n(Time taken: %d seconds)" % (time.time() - tic))

#%% FIGURE 5 - Time-series plots of total number of leaving events throughout the assay (GROUPED BY ASSAY/CONC)

smooth_window = int(2 * 60 * fps) # 2-minute binning window for smoothing

# Read recorded leaving events
true_leaving_df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "Results", "LeavingRate",\
                              "leavingevents_true.csv"), header=0, index_col=0)

# Group leaving event data by prefed-assaychoice-concentration treatment combinations
groupedLeavingData = true_leaving_df.groupby(['Prefed_on','Food_Combination','Food_Conc'])

# For each prefood-assaychoice-concentration treatment combination
tic = time.time()
for p, prefood in enumerate(pretreatments):
    # Initialise plot for prefed group
    plt.close("all")
    fig, axs = plt.subplots(nrows=len(assaychoices), ncols=len(concentrations),\
                            figsize=(16,8), sharey=True) # 15 subplots (3 assay types, 5 assay concentrations)
    plot_df = pd.DataFrame(columns=['filename','n_Events'])
    ymax = 25
    for a, assay in enumerate(assaychoices):
        for c, conc in enumerate(concentrations):
            try:
                # Get prefood-assaychoice-concentration group
                df_leaving = groupedLeavingData.get_group((prefood,assay,conc)).reset_index(drop=True)

                # Plot labels/colours
                colnames = df_leaving.iloc[0]['Food_Combination'].split('/')
                colnames = sorted(colnames, key=str.lower)
                labels = copy.deepcopy(colnames)
                if colnames[0] == colnames[1]:
                    colnames = ['{}_{}'.format(col, i+1) for i, col in enumerate(colnames)]
                colours = [colour_dict[treatment] for treatment in labels]
                
                xmax = df_leaving['frame_number'].max()
                df = pd.DataFrame(0, index=np.arange(0,xmax).astype(int), columns=colnames)
                for i, frame in enumerate(df_leaving['frame_number']):
                    if not int(frame) == int(xmax): # Omit last frame leaving event
                        info = df_leaving.iloc[i]
                        food = info['Food_left_from']
                        df[food].iloc[int(frame)] = df[food].iloc[int(frame)] + 1
                
                acclim = int(df_leaving['Acclim_time_s'][0] * fps)
                df.index = df.index + acclim 
                
                # Plot the number of leaving events on each food through time
                for i, food in enumerate(df.columns):                
                    # Smooth results for each food + overlay
                    moving_count = df[food].rolling(smooth_window,center=True).sum()
                    axs[a,c].plot(moving_count, color=colour_dict[labels[i]], ls='-') # x=np.arange(xlim)
                    
                    x = np.arange(0, acclim)
                    y = acclim
                    axs[a,c].fill_between(x, y, -0.05, color='grey', alpha='0.5', interpolate=True)
                    axs[a,c].axvline(0, ls='-', lw=1, color='k')
                    axs[a,c].axvline(acclim, ls='-', lw=1, color='k')
                    # plt.text(acclim/max(df.index)+0.01, 0.97, "Acclimation: {0} mins".format(int(acclim/25/60)),\
                    #          ha='left', va='center', transform=axs[a,c].transAxes, rotation=0, color='k')
                
                # X-axis limits/labels/ticks
                axs[a,c].set_xlim(0, np.round(xmax,-5))
                xticks = np.linspace(0,np.round(xmax,-5),num=5,endpoint=False).astype(int)
                axs[a,c].set_xticks(xticks)
                if a == len(assaychoices) - 1:
                    xticklabels = ["0", "30", "60", "90", "120"]
                    xticks = [int(int(lab)*fps*60) for lab in xticklabels]
                    axs[a,c].set_xticks(xticks)
                    axs[a,c].set_xticklabels(xticklabels)
                    axs[a,c].tick_params(axis='x', which='major', labelsize=12)
                    if c == 1:
                        axs[a,c].set_xlabel("Time (minutes)", labelpad=20, fontsize=24, horizontalalignment='left')
                else:
                    axs[a,c].set_xticklabels([])
                    
                # Y-axis limits/labels/ticks
                if moving_count.max(axis=0).max() > ymax:
                    ymax = moving_count.max(axis=0).max()
                axs[a,c].set_ylim(0, ymax + 0.5)
                yticks = list(np.arange(0,np.round(ymax,decimals=-1)+5,5).astype(int))
                axs[a,c].set_yticks(yticks)
                if c == 0:
                    axs[a,c].set_ylabel("{0}".format(assay), labelpad=15, fontsize=15)
                    if a == 1:
                        axs[a,c].text(-0.5, 0.5, "Number of Leaving Events",\
                                      fontsize=24, rotation=90, horizontalalignment='center',\
                                      verticalalignment='center', transform=axs[a,c].transAxes)
                    
                # Add text 'food concentration' to first row of plots
                if a == 0:
                    axs[a,c].text(0.5, 1.1, "$OD_{{{}}}={}$".format(600, conc*OpticalDensity600),\
                                  horizontalalignment='center', fontsize=22,\
                                  transform=axs[a,c].transAxes)

                # Add number of replicates (videos) to plot
                axs[a,c].text(0.88, 0.88, ("n={0}".format(len(np.unique(df_leaving['filename'])))),\
                              horizontalalignment='center', transform=axs[a,c].transAxes, fontsize=18)
            except Exception as e:
                print("No videos found for concentration: %s\n(Assay: %s, Prefed on: %s)\n" % (e, assay, prefood))
                
                axs[a,c].axis('off')
                axs[a,c].text(0.88, 0.88, "n=0", fontsize=15, transform=axs[a,c].transAxes)
                if a == 0:
                    axs[a,c].text(0.5, 1.1, "$OD_{{{}}}={}$".format(600, conc*OpticalDensity600),\
                                  horizontalalignment='center', fontsize=22,\
                                  transform=axs[a,c].transAxes)
                    
    # Add text 'prefed on' info to plot        
#    plt.text(xmax, -ymax/1.8, "Prefed on: {0}".format(prefood), horizontalalignment='right', fontsize=30)

    # Plot legend
    patches = []
    for key, value in colour_dict.items():
        if key == "None":
            continue
        patch = mpatches.Patch(color=value, label=key)
        patches.append(patch)
    fig.legend(handles=patches, labels=treatments, loc="upper right", borderaxespad=0.4,\
               frameon=False, fontsize=15)
    
    # Plot layout + adjustments
    fig.tight_layout(rect=[0.07, 0.02, 0.9, 0.95])
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.show(); plt.pause(2)
    
    # Save figure 6
    fig_name = "LeavingEventsTS_prefed" + prefood + ".eps"
    figure_out = os.path.join(PROJECT_ROOT_DIR, "Results", "Plots", fig_name)
    plt.savefig(figure_out, format='eps', dpi=300)
print("Complete!\n(Time taken: %d seconds)" % (time.time() - tic))

#%% FIGURE 6 - Box plots of leaving events durations (GROUPED BY ASSAY / CONCENTRATION)

tic = time.time()

# Read recorded leaving events
true_leaving_df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "Results", "LeavingRate",\
                              "leavingevents_true.csv"), header=0, index_col=0)

# Group leaving event data by prefed-assaychoice-concentration treatment combinations
groupedLeavingData = true_leaving_df.groupby(['Prefed_on','Food_Combination','Food_Conc'])

# For each prefood-assaychoice-concentration treatment combination
for p, prefood in enumerate(pretreatments):
    # Initialise plot for prefed group
    plt.close("all")
    fig, axs = plt.subplots(nrows=len(assaychoices), ncols=len(concentrations),\
                            figsize=(16,10), sharey=True) # 15 subplots (3 assay types, 5 assay concentrations)
    ymax = 0
    for a, assay in enumerate(assaychoices):
        for c, conc in enumerate(concentrations):
            try:
                # Get prefood-assaychoice-concentration group
                df_leaving = groupedLeavingData.get_group((prefood,assay,conc)).reset_index(drop=True)
                df_leaving['leaving_duration_s'] = df_leaving['leaving_duration_nframes']/fps
                
                # Plot labels/colours
                colnames = df_leaving.iloc[0]['Food_Combination'].split('/')
                if colnames[0] == colnames[1]:
                    colnames = ['{}_{}'.format(col, i+1) for i, col in enumerate(colnames)]

                labels = [lab.split('_')[0] for lab in colnames]
                colours = [colour_dict[treatment] for treatment in labels]
                
                # Seaborn colour palette
                RGBAcolours = sns.color_palette(colours)
                palette = {key: val for key, val in zip(colnames, RGBAcolours)}
                # sns.palplot(sns.color_palette(RGBAcolours))
                
                # Seaborn Boxplots
                sns.boxplot(x='Food_left_from', y='leaving_duration_s', hue='Food_left_from',\
                            data=df_leaving, palette=palette, dodge=False, ax=axs[a,c])
                axs[a,c].get_legend().set_visible(False)
                
                # Set x & y axes ticks/labels/limits
                if max(df_leaving['leaving_duration_s']) > ymax:
                    ymax = max(df_leaving['leaving_duration_s'])
                axs[a,c].set_yscale('log')
                axs[a,c].set_ylabel('')
                yticks = [1,10,100,1000]
                axs[a,c].set_yticks(yticks)
                axs[a,c].set_yticklabels(labels=[str(int(lab)) for lab in yticks])
                axs[a,c].set_xlim(-0.5,len(colnames)-0.5)
                xlabs = axs[a,c].get_xticklabels()
                xlabs = [lab.get_text().split('_')[0] for lab in xlabs[:]]
                axs[a,c].set_xticklabels(labels=xlabs, fontsize=12)
                axs[a,c].set_xlabel('')

                if c == 0:
                    axs[a,c].set_ylabel("{0}".format(assay), labelpad=15, fontsize=18)
                    if a == 1:
                        axs[a,c].text(-0.65, 0.5, "Leaving duration (log seconds)", fontsize=25,\
                                      rotation=90, verticalalignment='center', transform=axs[a,c].transAxes)
                
                # Add number of replicates (videos) to plots
                axs[a,c].text(0.81, 0.9, ("n={0}".format(len(np.unique(df_leaving['filename'])))),\
                              transform=axs[a,c].transAxes, fontsize=12)
                
                # Add concentration labels to plots
                if a == 0:
                    axs[a,c].text(0.5, 1.1, "$OD_{{{}}}={}$".format(600, conc*OpticalDensity600),\
                                  horizontalalignment='center', fontsize=22,\
                                  transform=axs[a,c].transAxes)  
            except Exception as e:
                print("No videos found for treatment combination: %s\n" % e)
                axs[a,c].axis('off')
                axs[a,c].text(0.81, 0.9, "n=0", fontsize=15, transform=axs[a,c].transAxes)
                if a == 0:
                    axs[a,c].text(0.5, 1.15, "$OD_{{{}}}={}$".format(600, conc*OpticalDensity600),\
                                  horizontalalignment='center', fontsize=18,\
                                  transform=axs[a,c].transAxes)
    # Add 'prefed on' to plot
#    plt.text(0.4, -0.4, "Prefed on: {0}".format(prefood), verticalalignment='bottom', fontsize=25, transform=axs[a,c].transAxes)

    # Add legend
    patches = []
    for i, (key, value) in enumerate(colour_dict.items()):
        if key == "None":
            continue
        patch = mpatches.Patch(color=value, label=key)
        patches.append(patch)
    fig.legend(handles=patches, labels=list(colour_dict.keys()), loc="upper right", borderaxespad=0.4,\
               frameon=False, fontsize=15)
    
    # Plot layout + adjustments
    fig.tight_layout(rect=[0.07, 0.02, 0.9, 0.95])
    fig.subplots_adjust(hspace=0.2, wspace=0.1)    
    plt.show(); plt.pause(1)
    
    # Save figure 5
    fig_name = "LeavingDurationBox_prefed" + prefood + ".eps"
    figure_out = os.path.join(PROJECT_ROOT_DIR, "Results", "Plots", fig_name)
    plt.savefig(figure_out, format='eps', dpi=300)
print("Complete!\n(Time taken: %d seconds)" % (time.time() - tic))

#%% 
# TODO: Leaving event duration histograms by treatment combination

#%%
# =============================================================================
# # Does prefeeding matter to leaving duration?
# true_leaving_prefedOP50 = true_leaving_df[true_leaving_df['Prefed_on']=='OP50']
# true_leaving_prefedHB101 = true_leaving_df[true_leaving_df['Prefed_on']=='HB101']
# bins = np.histogram(np.hstack((true_leaving_prefedOP50['leaving_duration_nframes'],\
#                     true_leaving_prefedHB101['leaving_duration_nframes'])),\
#                     bins=int(xlim/leaving_window*5))[1] # get the bin positions
# plt.close("all")
# plt.figure(figsize=(10,7))
# plt.hist(true_leaving_prefedOP50['leaving_duration_nframes'].values.astype(int), bins=bins, color='blue', alpha=0.75)
# plt.hist(true_leaving_prefedHB101['leaving_duration_nframes'].values.astype(int), bins=bins, color='green', alpha=0.75)
# plt.xlabel("Duration after leaving food (n frames)")
# plt.ylabel("Number of trajectories that leave food")
# 
# # Zoom-in on small leaving durations
# plt.xlim(0,(leaving_window*10+leaving_window/5))
# #plt.ylim(0,40)
# plt.xticks(np.arange(0,leaving_window*10+1, leaving_window))
# plt.axvline(leaving_window, ls='--', lw=2, color='k')
# plt.text(leaving_window+1, 100, "Threshold Leaving Duration ~ {0} seconds".format(leaving_window/fps), ha='left', va='center', rotation=-90, color='k')
# plt.show()
# 
# # Save histogram
# fig_name = "LeavingDurationPrefedHist" + ".png"
# figure_out = os.path.join(PROJECT_ROOT_DIR, "Results", "Plots", fig_name)
# save_fig(figure_out)
# print("\nHistogram complete!\n(Time taken: %d seconds)" % (time.time() - tic))
# =============================================================================

#%%
#    foods = assaychoice.split('/')
#    if foods[0] == foods[1]:
#        colnames = ["{}_{}".format(lab, i + 1) for i, lab in enumerate(foods)]
#    else:
##        colnames = foods
#    # Initialise a dataframe for storing how long the worms leave the food for
#    timesincerolling_leaving_df = pd.DataFrame(data=None, index=worm_ids, columns=colnames, dtype=int)
#    # Group trajectory data by worm ID
#    group_worm_leaving = leaving_events.groupby('worm_id')
#    worm_ids = list(np.unique(leaving_events['worm_id']))
#    # Investigate leaving events for each worm and compute duration since leaving
#    for w, worm in enumerate(worm_ids):
#        df_worm = group_worm_leaving.get_group(worm)
#        
#        event_data = leaving_events.iloc[row,:]
#        # Grab the trajectory data for that worm
#        df_worm = group_worm.get_group(event_data['worm_id'])
#        leavingframeindex = df_worm[df_worm['frame_number']==event_data['frame_number']].index
#        if not any(df_worm.index == list(leavingframeindex)[0]):
#            print("We have big problems")
#        n_frames_since_leaving = df_worm.loc[list(leavingframeindex)[0]:].shape[0]
##        colour = colours[w]
#        # we want n_frames_since_leaving
#        for fc, food in enumerate(colnames[-nfood:]):   
##    colours = hexcolours(len(worm_ids))
#    n_frames_traj = df_worm.shape[0]

#%%
# =============================================================================
# group_assaychoice = fullMetaData.groupby('Food_Combination')
# plt.close("all")
# fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(16,10)) # 15 subplots (3 assay types, 5 concentrations)
# for a, assay in enumerate(assaychoices):
#     df_assay = group_assaychoice.get_group(assay) # Group by assay type
#     group_conc = df_assay.groupby('Food_Conc')
#     concs = np.unique(df_assay['Food_Conc'])
#     for c, conc in enumerate(concs):
#         # PLOT NUMBER OF LEAVING EVENTS BY CONCENTRATION + ASSAY TYPE (15x subplots)
#         df_conc = group_conc.get_group(conc)
#         info = df_conc.iloc[0,:]
#         xlabs = info['Food_Combination'].split('/')
#         if xlabs[0] == xlabs[1]:
#             colnames = ["{}_{}".format(lab, i + 1) for i, lab in enumerate(xlabs)]
#         else:
#             colnames = xlabs
#         df = pd.DataFrame(index=range(df_conc.shape[0]), columns=colnames)
#         if df_conc.shape[0] < 3:
#             for row in range(df_conc.shape[0]):
#                 info = df_conc.iloc[row,:]
#                 leaving_path = info['filename'].replace(".hdf5", "_LeavingEvents.csv")
#                 leaving_path = leaving_path.replace("Priota/Data/FoodChoiceAssay/MaskedVideos/",\
#                                                     "Saul/FoodChoiceAssay/Results/LeavingRate/")
#                 # Read data for each frame in video after which a leaving event occurred
#                 df_leaving = pd.read_csv(leaving_path, header=0, index_col=0)
#                 # Calculate sum total number of leaving events for each food
#                 sum_leaving = df_leaving[colnames].sum()
#                 df.loc[0,:] = sum_leaving.values
#                 # Plot means as points not boxplots in cases where n=1 videos
#                 for i, col in enumerate(df.columns):
#                     axs[a,c].scatter(x=np.repeat((i + 1), df.shape[0]), y=df[col],\
#                                      marker='.', s=150, color=colour_dict[xlabs[i]],\
#                                      linewidths=1, edgecolors='k')
#             axs[a,c].set_xticks(ticks=list(np.round(np.arange(1,3,1))))
#             axs[a,c].set_xlim(0.5,2.5)
#         else:
#             for row in range(df_conc.shape[0]):
#                 info = df_conc.iloc[row,:]
#                 # Read leaving events csv file
#                 leaving_path = info['filename'].replace(".hdf5", "_LeavingEvents.csv")
#                 leaving_path = leaving_path.replace("Priota/Data/FoodChoiceAssay/MaskedVideos/",\
#                                                     "Saul/FoodChoiceAssay/Results/LeavingRate/")
#                 # Read data for each frame in video after which a leaving event occurred
#                 df_leaving = pd.read_csv(leaving_path, header=0, index_col=0)
#                 sum_leaving = df_leaving[colnames].sum()
#                 df.loc[row,:] = sum_leaving.values
#             # Plot boxplots of replicate means (each video) for prop on/off food, by assay + concentration
#             bp_dict = df.plot(y=list(df.columns), kind='box', ax=axs[a,c],\
#                               patch_artist=True, return_type='dict')
#             # COLOUR BOXPLOTS BY FOOD
#             for i, box in enumerate(bp_dict['boxes']):
#                 box.set_edgecolor('black')
#                 box.set_facecolor(colour_dict[xlabs[i]])
#             for item in ['whiskers', 'fliers', 'medians', 'caps']:
#                 plt.setp(bp_dict[item], color='black')
#         axs[a,c].set_xticklabels(labels=xlabs, fontsize=13)
#         axs[a,c].set_yticks(ticks=list(np.round(np.arange(0,101,20), decimals=0)))
#         axs[a,c].set_yticklabels(labels=list(np.round(np.arange(0,101,20), decimals=0)), fontsize=13)
#         axs[a,c].set_ylim(-5,105)
#         axs[a,c].text(len(xlabs)+0.25, 93, "n={0}".format(df_conc.shape[0]),\
#                       horizontalalignment='center', fontsize=15)
#         if a == 0:
#             axs[a,c].text((len(xlabs)+1)/2, 115, ("conc={0}".format(info['Food_Conc'])),\
#                           horizontalalignment='center', fontsize=18)
#         if c == 0 and a == 1:
#             axs[a,c].set_ylabel("Total Number of Leaving Events", labelpad=20, fontsize=20)
# patches = []
# for i, (key, value) in enumerate(colour_dict.items()):
#     if i == len(xlabs):
#         break
#     patch = mpatches.Patch(color=value, label=key)
#     patches.append(patch)
# fig.legend(handles=patches, labels=treatments[:-1], loc="upper right", borderaxespad=0.4,\
#            frameon=False, fontsize=15)
# fig.tight_layout(rect=[0.02, 0, 0.9, 0.95])
# fig.subplots_adjust(hspace=0.25, wspace=0.25)
# plt.show()
# # SAVE FIGURE 4
# figure_out = os.path.join(PROJECT_ROOT_DIR, "Results", "Plots", "LeavingEventsBox.png")
# save_fig(figure_out, tight_layout=False)
# =============================================================================

# =============================================================================
# group_assaychoice = fullMetaData.groupby('Food_Combination')
# plt.close("all")
# fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(16,10)) # 15 subplots (3 assay types, 5 concentrations)
# for a, assay in enumerate(assaychoices):
#     df_assay = group_assaychoice.get_group(assay) # Group by assay type
#     group_conc = df_assay.groupby('Food_Conc')
#     concs = np.unique(df_assay['Food_Conc'])
#     for c, conc in enumerate(concs):
#         df_conc = group_conc.get_group(conc)
#         info = df_conc.iloc[0,:]
#         xlabs = info['Food_Combination'].split('/')
#         if xlabs[0] == xlabs[1]:
#             colnames = ["{}_{}".format(lab, i + 1) for i, lab in enumerate(xlabs)]
#         else:
#             colnames = xlabs
#         # Pre-allocate out dataframe with zeroes, since it is already known that leaving events did not occur in those frames
#         df = pd.DataFrame(0, index=np.zeros((1,xlim), dtype=int).squeeze(), columns=colnames)
#         for row in range(df_conc.shape[0]):
#             info = df_conc.iloc[row,:]
#             # Specify filepath to leaving events results CSV file
#             leaving_path = info['filename'].replace(".hdf5", "_LeavingEvents.csv")
#             leaving_path = leaving_path.replace("Priota/Data/FoodChoiceAssay/MaskedVideos/",\
#                                                 "Saul/FoodChoiceAssay/Results/LeavingRate/")
#             # READ LEAVING EVENTS DATA
#             df_leaving = pd.read_csv(leaving_path, header=0, index_col=0)
#             for frame in df_leaving['frame_number']:
#                 row_data = df_leaving[df_leaving['frame_number']==frame] # Locate frame data
#                 try:
#                     df.iloc[frame,:] = df.iloc[frame,:].values + row_data[colnames].values.squeeze()
#                 except Exception as e:
#                     print("Frame %d out of bounds (>%d) in file:\n%s" % (frame, xlim, info['filename']))
#                     print("%s\n" % e)
#         df = df / df_conc.shape[0] # Divide by n replicate videos. Should again sum to 1
#         # Plot the number of leaving events on each food through time
#         for food in df.columns:
#             colourkey = food.split('_')[0]
#             moving_window = df[food].rolling(window, center=True)
#             moving_count = moving_window.sum().reset_index(drop=True)
#             # Overlay results for each food onto time series plot
#             axs[a,c].plot(moving_count, color=colour_dict[colourkey], linestyle='-') # x=np.arange(xlim)
#         if a == 0:
#             axs[a,c].text(0.5, 1.1, ("conc={0}".format(info['Food_Conc'])),\
#                            horizontalalignment='center', transform=axs[a,c].transAxes, fontsize=18)
#         if c == 0 and a == 1:
#             axs[a,c].set_ylabel("Number of Leaving Events", labelpad=20, fontsize=20)
# #        axs[a,c].set_ylim(-0.05, 20.05)
#         axs[a,c].set_xlim(0, xlim)
#         axs[a,c].set_xlabel("Frame Number", fontsize=12) # OPTIONAL: "Frame Number (window = {})".format(window)
#         axs[a,c].tick_params(axis='x', which='major', labelsize=9)
#         axs[a,c].tick_params(axis='y', which='major', labelsize=12)
#         axs[a,c].text(0.87, 0.9, ("n={0}".format(df_conc.shape[0])),\
#                       horizontalalignment='center', transform=axs[a,c].transAxes, fontsize=15)
# patches = []
# for i, (key, value) in enumerate(colour_dict.items()):
#     if i == len(xlabs):
#         break
#     patch = mpatches.Patch(color=value, label=key)
#     patches.append(patch)
# fig.legend(handles=patches, labels=treatments[:-1], loc="upper right", borderaxespad=0.4,\
#            frameon=False, fontsize=15)
# fig.tight_layout(rect=[0.02, 0, 0.9, 0.95])
# fig.subplots_adjust(hspace=0.25, wspace=0.25)
# =============================================================================

# =============================================================================
# #                for i, frame_idx in enumerate(leaving_info['frame_number']):
# #                    out_list.append(leaving_info.iloc[i][colnames].values.flatten().tolist())
#     
# #            diff = np.diff(df_worm[food].astype(int))
# #            if np.any(diff != 0):
# #                if np.any(diff < 0):
# #                    leaving_info = df_worm.iloc[np.where(diff < 0)]
# #                    for i, frame_idx in enumerate(leaving_info['frame_number']):
# #                        query_df = df.iloc[np.arange(frame_idx-window, frame_idx+window)]
# #                        if sum(np.diff(query_df[food].astype(int)) < 0) > 1: # IGNORE FENCE-SITTERS
# #                            print("Worm %d is fence-sitting! Frame: %d" % (worm, frame_idx))
# #                        elif sum(np.diff(query_df[food].astype(int)) < 0) == 1:
# #                            print(leaving_info.iloc[i][colnames].values.flatten().tolist())
#     
# #    df_group_worm = df.groupby(['worm_id'])
# #    unique_worm_ids = np.unique(df['worm_id'])
# #    for worm in unique_worm_ids:
# #        # Tierpsy-generated worm_ids always comprise of sequentially consecutive frames
# #        df_worm = df_group_worm.get_group(worm)
# #        for fc, food in enumerate(colnames[-foodpatches:]):
# #            diff = np.diff(df_worm[food].astype(int))
# #            # length is -1 due to no preceeding value to compare against the 1st value, so result for 1st frame is omitted
# #            if np.any(diff != 0):
# #                if np.any(diff < 0):
# #                    leaving_info = df_worm.iloc[np.where(diff < 0)]
# #                    # Returns info for the frame just before each leaving event (thus records which food it left from)
# #                    for i, frame_idx in enumerate(leaving_info['frame_number']):
# #                        # Needs smoothing to obtain only TRUE leaving events - fence-sitters, edge thickness, threshold time
# #                        # Check neighbouring frames to verify leaving event, using a threshold window for smoothing leaving event estimation
# #                        query_df = df.iloc[np.arange(frame_idx-window, frame_idx+window)]
# #                        if sum(np.diff(query_df[food].astype(int)) < 0) > 1:
# #                            print("Worm %d is fence-sitting! Frame: %d" % (worm, frame_idx))
# #                        # print("Worm %d left %s after frame %d" % (worm, food, leaving_info.iloc[event]['frame_number']))
# #                        # Append leaving event info to out_list
# #                        # else:
# #                        # append to out_df only if other frame entry doesnt exist in df within window
# #                        out_list.append(leaving_info.iloc[i][colnames].values.flatten().tolist())
# 
# #plt.close("all")
# #x = np.linspace(0,1000,1001)
# #onfood = x < 500
# #onfood[496:500:2] = 0
# #onfood[502:506:2] = 1
# #onfood_roll = pd.Series(onfood).rolling(window=10, center=True).mean()
# #true_leaving = (onfood_roll < 0.5).astype(int).diff() == 1
# ## foo only if you do it on the # of worms on food vs frame (grouping by frame, looking at all worms in frame)
# #foo = onfood_roll.values[20:] - onfood_roll.values[:-20] 
# #leaving = (pd.Series(onfood).astype(int).diff()<0).astype(int)
# #counts_roll = leaving.rolling(window=10, center=True).sum()
# #plt.plot(x,onfood,'-')
# #plt.plot(x, onfood_roll, '-r')
# #plt.plot(x, true_leaving, '-r')
# #plt.plot(foo)
# #plt.show()
# #plt.figure()
# #plt.plot(x, leaving)
# #plt.plot(x, counts_roll)
# =============================================================================

# =============================================================================
# food_roll = pd.Series(df_worm[food],index=df_worm.index).rolling(window=window, center=True).mean()
# # Crop to remove NaNs (false positives in diff computation when converted to type int)
# food_roll = food_roll[window//2:-window//2+1]#.reset_index(drop=True) -- DO NOT RESET INDEX. PRESERVE INDICES...
# true_leaving = (food_roll < 0.5).astype(int).diff() == 1
# if any(true_leaving):
#     leaving_frames = list(true_leaving.index[np.where(true_leaving==True)[0]])
#      # Last leaving event is compared to end of vidoe number of frames (minus the NaNs cropped by rolling window)
#     leaving_frames.insert(len(leaving_frames), true_leaving.index[-1])
#     for i, frame in enumerate(leaving_frames):
#         if i == len(leaving_frames)-1:
#             break
#         t_since_leaving = leaving_frames[i+1] - frame # duration since leaving (n frames)
#         leaving_data = df_worm.loc[frame]
#         if removeNone:
#             leaving_data = leaving_data.drop('None')
#         leaving_data = leaving_data.append(pd.Series(data=t_since_leaving))
#         out_df.loc[worm] = leaving_data.values
# out_df = out_df.dropna(axis=0, how='any')
# =============================================================================

# =============================================================================
# # Flattening a list/array
# # 1.
# # Flatten list and plot leaving durations across all videos
# # total_leaving_events_df = [item for sublist in total_leaving_events_df for item in sublist]
# # 2. 
# # flat_list = []
# # for sublist in total_leaving_events_df:
# #     for item in sublist:
# #         flat_list.append(item)
# # 3.
# # flatten = lambda total_leaving_events_df: [item for sublist in total_leaving_events_df for item in sublist]
# eg.     total_leaving_events_df.append(leaving_events_df['leaving_duration_nframes'].values.tolist())
# # Flatten list of leaving durations across all videos
# total_leaving_events_df = [item for sublist in total_leaving_events_df for item in sublist]
# =============================================================================

# =============================================================================
# # LEAVING EVENTS (FOR EACH VIDEO SEPARATELY) - OLD VERSION
# # - Analysis of leaving events
# # - Infer leaving events using rolling mean to avoid overestimation due to 'fence-sitting' in some videos 
# errorlog = 'ErrorLog_Leaving.txt'
# FAIL = []
# tic = time.time()
# for i, maskedfilepath in enumerate(fullMetaData['filename']):
#     toc = time.time()
#     file_info = fullMetaData.iloc[i,:]
#     date = file_info['date(YEARMODA)']
#     conc = file_info['Food_Conc']
#     assaychoice = file_info['Food_Combination']
#     prefed = file_info['Prefed_on']
#     print("\nProcessing file: %d\n%s\nAssay:  %s\nConc:   %.3f\nPrefed: %s" % (i + 1,\
#           maskedfilepath, assaychoice, conc, prefed))
#     # Corresponding features file path
#     featurefilepath = maskedfilepath.replace(".hdf5", "_featuresN.hdf5")
#     featurefilepath = featurefilepath.replace("MaskedVideos/", "Results/")
#     # Path to ON/OFF food csv file
#     onfoodpath = maskedfilepath.replace("Priota/Data/FoodChoiceAssay/MaskedVideos/",\
#                                  "Saul/FoodChoiceAssay/Results/FoodChoice/")
#     onfoodpath = onfoodpath.replace(".hdf5", "_OnFood.csv")
#     try:
#         # READ ON/OFF FOOD TRUTH MATRIX
#         onfood_df = pd.read_csv(onfoodpath, header=0, index_col=0)
#         # RECORD LEAVING EVENTS
#         rolling_leaving_df = leavingeventsroll(onfood_df, nfood=2, window=leaving_window, removeNone=True) 
#         print("Number of leaving events found: %d" % rolling_leaving_df.shape[0])
#         # SAVE LEAVING EVENTS
#         leavingpath = onfoodpath.replace("_OnFood.csv", "_LeavingEvents.csv")
#         leavingpath = leavingpath.replace("FoodChoice/", "LeavingRate/")
#         directory = os.path.dirname(leavingpath)
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#         rolling_leaving_df.to_csv(leavingpath)
#         print("Leaving events saved to file.\n(Time taken: %d seconds.)\n" % (time.time() - toc))
#     except Exception as e:
#         FAIL.append(maskedfilepath)
#         print("ERROR! Failed to process file: \n%s\n%s" % (maskedfilepath, e))
# print("Done!\n(Total time taken: %d seconds)" % (time.time() - tic))
# # Save error log to file
# fid = open(os.path.join(PROJECT_ROOT_DIR, errorlog), 'w')
# print(FAIL, file=fid)
# fid.close()
# =============================================================================

# =============================================================================
# # OVERLAY LEAVING EVENTS ONTO TRAJECTORY PLOTS (FOR EACH VIDEO SEPARATELY) - OLD VERSION
# tic = time.time()
# for i, maskedfilepath in enumerate(fullMetaData['filename']):
#     file_info = fullMetaData.iloc[i,:]
#     date = file_info['date(YEARMODA)']
#     conc = file_info['Food_Conc']
#     assaychoice = file_info['Food_Combination']
#     prefed = file_info['Prefed_on']
#     print("\nProcessing file: %d\n%s\nAssay:  %s\nConc:   %.3f\nPrefed: %s" % (i + 1,\
#           maskedfilepath, assaychoice, conc, prefed))
#     # Corresponding features file path
#     featurefilepath = maskedfilepath.replace(".hdf5", "_featuresN.hdf5")
#     featurefilepath = featurefilepath.replace("MaskedVideos/", "Results/")
#     # File path to leaving event results
#     leavingpath = maskedfilepath.replace(".hdf5", "_LeavingEvents.csv")
#     leavingpath = leavingpath.replace("Priota/Data/FoodChoiceAssay/MaskedVideos/",\
#                                       "Saul/FoodChoiceAssay/Results/LeavingRate/")
#     # File path to polygon coordinates
#     coordfilepath = leavingpath.replace("_LeavingEvents.csv", "_FoodCoords.txt")
#     coordfilepath = coordfilepath.replace("LeavingRate/", "FoodCoords/")
#     # Directory to save overlay plots
#     plots_out = leavingpath.replace("_LeavingEvents.csv", "_LeavingPlot.png")
#     plots_out = plots_out.replace("LeavingRate/", "Plots/")
#     try:
#         # READ LEAVING RESULTS
#         rolling_leaving_df = pd.read_csv(leavingpath, header=0, index_col=0)
#         # READ POLYGON COORDINATES
#         f = open(coordfilepath, 'r').read()
#         poly_dict = eval(f)
#         # PLOT BRIGHTFIELD
#         plt.close("all")
#         fig, ax = plotbrightfield(maskedfilepath, figsize=(10,8))
#         # OVERLAY FOOD REGIONS
#         fig, ax = plotpoly(fig, ax, poly_dict)
#         # OVERLAY WORM TRAJECTORIES
#         fig, ax = plottrajectory(fig, ax, featurefilepath, downsample=10)
#         # OVERLAY LEAVING EVENTS
#         fig, ax = plotpoints(fig, ax, rolling_leaving_df['x'], rolling_leaving_df['y'], color='r',\
#                              marker='+', markersize=3, linestyle='')
#         # SAVE PLOT
#         directory = os.path.dirname(plots_out)
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#         save_fig(plots_out) # Save figure
#         plt.show(); plt.pause(0.0001) # Close plot
#     except Exception as e:
#         print(e)
#         continue
# print("Done!\n(Time taken: %d seconds)" % (time.time() - tic))
# =============================================================================
   
# =============================================================================
# # Difference between rolling window leaving event estimation and filtering after
# len(np.unique(rolling_leaving_df.worm_id))
# len(np.unique(leaving_events_df.worm_id))
# np.setdiff1d(np.unique(leaving_events_df.worm_id), np.unique(rolling_leaving_df.worm_id), assume_unique=True)
# =============================================================================

# =============================================================================
# # Read total leaving events file
# total_leaving_events_df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "Results","LeavingRate",\
#                                                    "leavingevents_fulldata.csv"), header=0, index_col=0)
# =============================================================================

# =============================================================================
# #tic = time.time()
# #group_prefed = fullMetaData.groupby('Prefed_on')
# #for p, prefood in enumerate(pretreatments):
# #    df_prefed = group_prefed.get_group(prefood)
# #    plt.close("all")
# #    fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(16,10), sharey=True) # 15 subplots (3 assay types, 5 concentrations)
# #    ymax = 0
# #    group_assaychoice = df_prefed.groupby('Food_Combination')
# #    for a, assay in enumerate(assaychoices):
# #        try:
# #            df_assay = group_assaychoice.get_group(assay) # Group by assay type
# #            group_conc = df_assay.groupby('Food_Conc')
# #            for c, conc in enumerate(concentrations):
# #                try:
# #                    df_conc = group_conc.get_group(conc)
# #                    info = df_conc.iloc[0,:]
# #                    xlabs = info['Food_Combination'].split('/')
# #                    if xlabs[0] == xlabs[1]:
# #                        colnames = ["{}_{}".format(lab, i + 1) for i, lab in enumerate(xlabs)]
# #                    else:
# #                        colnames = xlabs
# #                    # Pre-allocate out dataframe with zeroes, since it is already known that leaving events did not occur in those frames
# #                    df = pd.DataFrame(0, index=np.zeros((1,xlim), dtype=int).squeeze(), columns=colnames)
# #                    if df_conc.shape[0] == 1:
# #                        leaving_path = info['filename'].replace(".hdf5", "_LeavingEvents.csv")
# #                        leaving_path = leaving_path.replace("Priota/Data/FoodChoiceAssay/MaskedVideos/",\
# #                                                            "Saul/FoodChoiceAssay/Results/LeavingRate/")
# #                        df_leaving = pd.read_csv(leaving_path, header=0, index_col=0)
# #                        for frame in df_leaving['frame_number']:
# #                            row_data = df_leaving[df_leaving['frame_number']==frame] # Locate frame data
# #                            try:
# #                                df.iloc[frame,:] = df.iloc[frame,:].values + row_data[colnames].values.squeeze()
# #                            except Exception as e:
# #                                print("Frame %d out of bounds (>%d) in file:\n%s" % (frame, xlim, info['filename']))
# #                                print("%s\n" % e)
# #                    elif df_conc.shape[0] > 1:
# #                        for row in range(df_conc.shape[0]):
# #                            info = df_conc.iloc[row,:]
# #                            leaving_path = info['filename'].replace(".hdf5", "_LeavingEvents.csv")
# #                            leaving_path = leaving_path.replace("Priota/Data/FoodChoiceAssay/MaskedVideos/",\
# #                                                                "Saul/FoodChoiceAssay/Results/LeavingRate/")
# #                            df_leaving = pd.read_csv(leaving_path, header=0, index_col=0)
# #                            for frame in df_leaving['frame_number']:
# #                                row_data = df_leaving[df_leaving['frame_number']==frame] # Locate frame data
# #                                try:
# #                                    df.iloc[frame,:] = df.iloc[frame,:].values + row_data[colnames].values.squeeze()
# #                                except Exception as e:
# #                                    print("Frame %d out of bounds (>%d) in file:\n%s" % (frame, xlim, info['filename']))
# #                                    print("%s\n" % e)
# #                        df = df / df_conc.shape[0] # Divide by n replicate videos. Should again sum to 1
# =============================================================================

