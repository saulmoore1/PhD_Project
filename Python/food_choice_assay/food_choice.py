#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT: FOOD CHOICE

A script written to analyse the food choice assay videos and Tierpsy-generated 
feature summary data. It calculates, plots and saves results for worm food preference
(for each video separately).

@author: sm5911
@date: 21/03/2019

"""

#%% Imports
import os, time
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches

# Custom imports
from time_series.time_series_helper import plot_timeseries_phenix
from visualisation.plotting_helper import hexcolours, plot_pie
from food_choice_assay.food_choice_helper import foodchoice, summarystats
from _deprecated.find import change_path_phenix

#%% Globals

PROJECT_ROOT_DIR = '/Volumes/behavgenom$/Saul/FoodChoiceAssay/' # Project working directory
DATA_DIR = PROJECT_ROOT_DIR.replace('Saul', 'Priota/Data') # Location of features files

# Plot parameters
fps = 25 # frames per second
smooth_window = fps*60*2 # 2-minute moving average window for time-series plot smoothing
OpticalDensity600 = 1.8 # E. coli average OD600

NEW = True # Conduct analysis on new videos only?

#%% Preamble
 
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
treatments.insert(len(treatments),"None") # treatments = [OP50, HB101, None]
concentrations = list(np.unique(fullMetaData['Food_Conc']))

# Plot parameters
colours = hexcolours(len(treatments)) # Create a dictionary of colours for each treatment (for plotting)
colour_dict = {key: value for (key, value) in zip(treatments, colours)}

#%% CALCULATE MEAN NUMBER OF WORMS ON/OFF FOOD IN EACH FRAME (FOR EACH VIDEO SEPARATELY)
# - PROPORTION of total worms in each frame

errorlog = 'ErrorLog_FoodChoice.txt'
FAIL = []
tic = time.time()
for i, maskedfilepath in enumerate(fullMetaData['filename']):
    toc = time.time()
    # Extract file information
    file_info = fullMetaData.iloc[i,:]
    date = file_info['date(YEARMODA)']
    conc = file_info['Food_Conc']
    assaychoice = file_info['Food_Combination']
    prefed = file_info['Prefed_on']
    print("\nProcessing file: %d/%d\n%s\nAssay:  %s\nConc:   %.3f\nPrefed: %s" % (i + 1,\
          len(fullMetaData['filename']), maskedfilepath, assaychoice, conc, prefed))
    try:
        # Specify file paths
        onfoodpath = change_path_phenix(maskedfilepath, returnpath='onfood')
        foodchoicepath = change_path_phenix(maskedfilepath, returnpath='foodchoice')

        # Read on/off food results
        onfood_df = pd.read_csv(onfoodpath, header=0, index_col=0)

        # Calculate mean + count number of worms on/off food in each frame
        # NB: Store proportions, along with total nworms, ie. mean (worms per frame) and later calculate mean (per frame across videos)
        choice_df = foodchoice(onfood_df, mean=True, tellme=True)
        
        # Save food choice results
        directory = os.path.dirname(foodchoicepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        choice_df.to_csv(foodchoicepath)
        print("Food choice results saved. \n(Time taken: %d seconds)\n" % (time.time() - toc))
    except:
        FAIL.append(maskedfilepath)
        print("ERROR! Failed to calculate food preference in file:\n %s\n" % maskedfilepath)
print("Complete!\n(Total time taken: %d seconds.)\n" % (time.time() - tic))

# If errors, save error log to file
if FAIL:
    fid = open(os.path.join(PROJECT_ROOT_DIR, errorlog), 'w')
    print(FAIL, file=fid)
    fid.close()

#%% FOOD CHOICE SUMMARY STATS + PIE/BOX PLOTS (FOR EACH VIDEO SEPARATELY)
# - Calculate summary statistics for mean proportion worms feeding in each video
# - Plot and save box plots + pie charts of mean proportion of worms on food

# =============================================================================
# # NB: Cannot pre-allocate full results dataframe to store food choice mean 
# #     proportion feeding per frame across all videos due to file size = 23GB
# colnames = ['filename','worm_number','Food_Conc','Food_Combination','Prefed_on',\
#             'Acclim_time_s','frame_number','Food','Mean']
# results_df = pd.DataFrame(columns=colnames)
# =============================================================================
    
tic = time.time()
for i, maskedfilepath in enumerate(fullMetaData['filename']):
    # Extract file information
    file_info = fullMetaData.iloc[i,:]
    date = file_info['date(YEARMODA)']
    conc = file_info['Food_Conc']
    assaychoice = file_info['Food_Combination']
    prefed = file_info['Prefed_on']
    print("\nProcessing file: %d/%d\n%s\nAssay:  %s\nConc:   %.3f\nPrefed: %s" % (i + 1,\
          len(fullMetaData['filename']), maskedfilepath, assaychoice, conc, prefed))
    
    # Specify file paths
    foodchoicepath = change_path_phenix(maskedfilepath, returnpath='foodchoice')
    statspath = change_path_phenix(maskedfilepath, returnpath='summary')
    pieplotpath = change_path_phenix(maskedfilepath, returnpath='plots', figname='PiePlot.eps')
    boxplotpath = change_path_phenix(maskedfilepath, returnpath='plots', figname='BoxPlot.eps')
    try:
        # READ FOOD CHOICE RESULTS (csv)
        choice_df = pd.read_csv(foodchoicepath, header=0, index_col=0)
        
        # SUMMARY STATISTICS
        feeding_stats = summarystats(choice_df)
        
        # Save summary stats
        feeding_stats.to_csv(statspath) # Save to CSV
       
        # Define plot labels + colours
        colnames = list(choice_df.columns)
        labels = [lab.split('_')[0] for lab in colnames]
        colours = [colour_dict[treatment] for treatment in labels]
        # Specify seaborn colour palette
        RGBAcolours = sns.color_palette(colours)
        palette = {key: val for key, val in zip(colnames, RGBAcolours)}
        # sns.palplot(sns.color_palette(values))
       
        # PIE CHARTS - mean proportion on food
        df_pie = feeding_stats.loc['mean']
        df_pie.index = df_pie.index.get_level_values(0)
        df_pie = df_pie.loc[df_pie!=0] # Remove any empty rows
        plt.close("all")
        fig = plot_pie(df_pie, rm_empty=False, show=True, labels=df_pie.index,\
                       colors=colours, textprops={'fontsize': 15}, startangle=90,\
                       wedgeprops={'edgecolor': 'k', 'linewidth': 1,\
                                   'linestyle': 'solid', 'antialiased': True})
        # Save pie charts
        directory = os.path.dirname(pieplotpath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.tight_layout()
        plt.savefig(pieplotpath, dpi=300)      
     
        # Convert to long format
        choice_df['frame_number'] = choice_df.index
        choice_df_long = choice_df.melt(id_vars='frame_number', value_vars=choice_df.columns[:-1],\
                                        var_name='Food', value_name='Mean')
                 
        # BOX PLOTS (Seaborn) - Mean proportion of worms on each food
        plt.close("all")
        fig, ax = plt.subplots(figsize=(9,7))
        ax = sns.boxplot(x='Food', y='Mean', hue='Food', data=choice_df_long, palette=palette, dodge=False)
        # NB: Could also produce violinplots, but why not swarmplots? Too many points?
        # ax = sns.violinplot(x='Food', y='Mean', hue='Food', data=choice_df_long, palette=palette, dodge=False)
        ax.set_ylim(-0.1,1.1)
        ax.set_xlim(-1,len(treatments)+0.25)
        ax.set_xlabel("Food",fontsize=20)
        ax.set_ylabel("Mean Proportion Feeding",fontsize=20)
        ax.xaxis.labelpad = 15; ax.yaxis.labelpad = 15
        ax.tick_params(labelsize=13, pad=5)
        fig.tight_layout(rect=[0.02, 0.07, 0.95, 0.95])
        plt.text(0.03, 0.93, "{0} worms".format(file_info['worm number']), transform=ax.transAxes, fontsize=20)
        plt.text(len(treatments)+0.25, -0.35, "Prefed on: {0}".format(prefed),\
                 horizontalalignment='right', fontsize=25)
        plt.legend(loc="upper right", borderaxespad=0.4, frameon=False, fontsize=15)
        plt.show(); plt.pause(0.0001)
            
        # Save box plots
        plt.tight_layout()
        plt.savefig(boxplotpath, format='eps', dpi=300)
        print("Plots saved.\n")
       
# =============================================================================
#        # Append file info
#        choice_df_long['filename'] = maskedfilepath
#        choice_df_long['worm_number'] = file_info['worm number']
#        choice_df_long['Food_Conc'] = conc
#        choice_df_long['Food_Combination'] = assaychoice
#        choice_df_long['Prefed_on'] = prefed
#        choice_df_long['Acclim_time_s'] = file_info['Acclim_time_s']
#
#        # Append to full results dataframe
#        results_df = results_df.append(choice_df_long[colnames])
# =============================================================================
    except:
        print("Error processing file:\n%s" % maskedfilepath)
        continue
print("Done.\n(Time taken: %d seconds.)" % (time.time() - tic))

# =============================================================================
# size = sys.getsizeof(results_df)
# # File size is too big! Not a good idea to save as full results file
# =============================================================================


#%% Time-series plots of proportion feeding through time (FOR EACH VIDEO SEPARATELY)

tic = time.time()
for i, maskedfilepath in enumerate(fullMetaData['filename']):
    toc = time.time()
    # Extract file information
    file_info = fullMetaData.iloc[i,:]
    conc = file_info['Food_Conc']
    assaychoice = file_info['Food_Combination']
    prefed = file_info['Prefed_on']
    print("\nProcessing file: %d/%d\n%s\nAssay:  %s\nConc:   %.3f\nPrefed: %s" % (i + 1,\
          len(fullMetaData['filename']), maskedfilepath, assaychoice, conc, prefed))
    
    # Specify file paths
    onfoodpath = change_path_phenix(maskedfilepath, returnpath='onfood')
    foodchoicepath = change_path_phenix(maskedfilepath, returnpath='foodchoice')
    plotpath = change_path_phenix(maskedfilepath, returnpath='plots', figname='FoodChoiceTS.png') # Path to save time series plots

    onfood_df = pd.read_csv(onfoodpath, header=0, index_col=0)

    # READ FOOD CHOICE RESULTS
#    df = pd.read_csv(foodchoicepath, header=0, index_col=0)
    df = foodchoice(onfood_df, mean=True, std=True, tellme=True)
    
    # Shift plot to include acclimation time prior to assay recording (ie. t(0) = pick time)
    acclim = int(file_info['Acclim_time_s'] * fps)
    df.index = df.index + acclim 
    
    # Caclculate mean + standard deviation per frame across videos
    colnames = list(df.columns.levels[0])

    # Remove erroneous frames where on/off food does not sum to 1
    frames_to_rm = np.where(np.sum([df[x]['mean'] for x in colnames], axis=0).round(decimals=5)!=1)[0]
    assert frames_to_rm.size == 0,\
        "{:d} frames found in which feeding proportions do not sum to 1.".format(len(frames_to_rm))

    
    # PLOT TIME-SERIES ON/OFF FOOD (count)
    plt.close("all")
    fig = plot_timeseries_phenix(df=df, colour_dict=colour_dict, window=smooth_window,\
                                 acclimtime=acclim, annotate=True, legend=True, ls='-')
    
    # SAVE TIME SERIES PLOTS
    directory = os.path.dirname(plotpath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.tight_layout()
    plt.savefig(plotpath, format='png', dpi=300)
    print("Time series plots saved.\n(Time taken: %d seconds.)\n" % (time.time() - toc))
print("Complete!\n(Total time taken: %d seconds.)\n" % (time.time() - tic))


#%% FIGURE 1 - Box plots of food choice (Grouped by treatment combination: prefed on (HB101/OP50), food combination (control/choice), and concentration (0.125,0.25,0.5,1))
# - Subset results by grouping files by assay type (control/choice experiment) and by food concentration

tic = time.time()
# Group files in metadata by prefed, assaychoice and concentration treatment combinations
groupedMetaData = fullMetaData.groupby(['Prefed_on','Food_Combination','Food_Conc'])

# For each prefood-assaychoice-concentration treatment combination
for p, prefood in enumerate(pretreatments):
    # Initialise plot for prefed group (12 subplots - 3 food combinations, 4 concentrations)
    plt.close("all")
    fig, axs = plt.subplots(nrows=len(assaychoices), ncols=len(concentrations),\
                            figsize=(14,10), sharey=True)
    for a, assay in enumerate(assaychoices):
        for c, conc in enumerate(concentrations):
            try:
                # Get prefood-assaychoice-concentration group
                df_conc = groupedMetaData.get_group((prefood,assay,conc))
                
                # Get group info
                info = df_conc.iloc[0,:]
                colnames = info['Food_Combination'].split('/')
                if colnames[0] == colnames[1]:
                    colnames = ["{}_{}".format(food, f + 1) for f, food in enumerate(colnames)]
                colnames.insert(len(colnames), "None")
                
                # Pre-allocate dataframe for boxplots
                df = pd.DataFrame(index=range(df_conc.shape[0]), columns=colnames)

                # If single file, read full food choice data (mean proportion feeding)
                if df_conc.shape[0] == 1:
                    foodchoicepath = change_path_phenix(info['filename'], returnpath='foodchoice')
                    df = pd.read_csv(foodchoicepath, header=0, index_col=0)   
                
                # Read summary stats for mean proportion feeding in each video
                elif df_conc.shape[0] > 1:
                    for i in range(df_conc.shape[0]):
                        info = df_conc.iloc[i]
                        # Read in food choice summary stats (mean proportion feeding)
                        statspath = change_path_phenix(info['filename'], returnpath='summary')
                        df.iloc[i] = pd.read_csv(statspath, header=0, index_col=0).loc['mean']                    
                    
# =============================================================================
#                     # Read food choice data for each file and compile into df for plotting
#                     df = pd.DataFrame()
#                     for row in range(df_conc.shape[0]):
#                         info = df_conc.iloc[row,:]
#                         foodchoicepath = changepath(info['filename'], returnpath='foodchoice')
#                         tmp_df = pd.read_csv(foodchoicepath, header=0, index_col=0)
#                         if df.empty:
#                             df = tmp_df
#                         else:
#                             df = df.append(tmp_df, sort=True)
# =============================================================================
                    
                # Plot labels/colours
                labels = [lab.split('_')[0] for lab in colnames]
                colours = [colour_dict[treatment] for treatment in labels]
                
                # Seaborn colour palette
                RGBAcolours = sns.color_palette(colours)
                palette = {key: val for key, val in zip(colnames, RGBAcolours)}
                # sns.palplot(sns.color_palette(values))
                
                # Convert to long format
                df['videoID'] = df.index
                df_long = df.melt(id_vars='videoID', value_vars=df.columns[:-1],\
                                  var_name='Food', value_name='Mean')
                df_long['Mean'] = df_long['Mean'].astype(float)
        
# =============================================================================
#                 # Convert to long format
#                 df['frame_number'] = df.index
#                 df_long = df.melt(id_vars='frame_number', value_vars=df.columns[:-1],\
#                                   var_name='Food', value_name='Mean')
# =============================================================================
                
                # Plot Seaborn boxplots
                sns.boxplot(data=df_long, x='Food', y='Mean', hue='Food', ax=axs[a,c], palette=palette, dodge=False)
                axs[a,c].get_legend().set_visible(False)
                axs[a,c].set_ylabel('')    
                axs[a,c].set_xlabel('')
                xlabs = axs[a,c].get_xticklabels()
                xlabs = [lab.get_text().split('_')[0] for lab in xlabs[:]]
                axs[a,c].set_xticklabels(labels=xlabs, fontsize=12)
                axs[a,c].set_ylim(-0.05, 1.05)
                axs[a,c].set_xlim(-0.75,len(np.unique(df_long['Food']))-0.25)
                axs[a,c].text(0.81, 0.9, ("n={0}".format(df_conc.shape[0])),\
                              transform=axs[a,c].transAxes, fontsize=12)
                if a == 0:
                    axs[a,c].text(0.5, 1.1, "$OD_{{{}}}={}$".format(600, conc*OpticalDensity600),\
                                  horizontalalignment='center', fontsize=18,\
                                  transform=axs[a,c].transAxes)
                if c == 0:
                    axs[a,c].set_ylabel("{0}".format(assay), labelpad=15, fontsize=18)
                    if a == 1:
                        axs[a,c].text(-2.75, 0.5, "Mean Proportion Feeding",\
                                      fontsize=25, rotation=90, horizontalalignment='center',\
                                      verticalalignment='center')
                    
            except Exception as e:
                print("No videos found for concentration: %s\n(Assay: %s, Prefed on: %s)\n" % (e, assay, prefood))
                axs[a,c].axis('off')
                axs[a,c].text(0.81, 0.9, "n=0", fontsize=12, transform=axs[a,c].transAxes)
                if a == 0:
                    axs[a,c].text(0.5, 1.1, "$OD_{{{}}}={}$".format(600, conc*OpticalDensity600),\
                                  horizontalalignment='center', fontsize=18,\
                                  transform=axs[a,c].transAxes)
                if c == 0:
                    axs[a,c].set_ylabel("{0}".format(assay), labelpad=15, fontsize=18)
                    if a == 1:
                        axs[a,c].text(-3.2, 0.5, "Mean Proportion Feeding",\
                                      fontsize=25, rotation=90, horizontalalignment='center',\
                                      verticalalignment='center')
                    
#    plt.text(3, -0.7, "Prefed on: {0}".format(prefood), horizontalalignment='center', fontsize=25)
    patches = []
    for i, (key, value) in enumerate(colour_dict.items()):
        patch = mpatches.Patch(color=value, label=key)
        patches.append(patch)
    fig.legend(handles=patches, labels=list(colour_dict.keys()), loc="upper right", borderaxespad=0.4,\
               frameon=False, fontsize=15)
    fig.tight_layout(rect=[0.07, 0.02, 0.88, 0.95])
    fig.subplots_adjust(hspace=0.2, wspace=0.1)    
    plt.show(); plt.pause(2)
    
    # Save figure 1
    fig_name = "FoodChoiceBox_prefed" + prefood + ".eps"
    figure_out = os.path.join(PROJECT_ROOT_DIR, "Results", "Plots", fig_name)
    plt.savefig(figure_out, format='eps', dpi=300)
print("Complete!\n(Time taken: %d seconds)" % (time.time() - tic))

#%% FIGURE 2 
# - OPTIONAL: Plot as fraction of a constant total?

#%% FIGURE 3 - Time series plots of food choice by concentration and by assay type (GROUPED BY ASSAY/CONC)
# Plot time series plots - proportion on-food through time  

tic = time.time()
# Group files in metadata by prefed, assaychoice and concentration treatment combinations
groupedMetaData = fullMetaData.groupby(['Prefed_on','Food_Combination','Food_Conc'])

# For each prefood-assaychoice-concentration treatment combination
for p, prefood in enumerate(pretreatments):
    # Initialise plot for prefed group
    plt.close("all")
    xmax = 180000
    fig, axs = plt.subplots(nrows=len(assaychoices), ncols=len(concentrations),\
                            figsize=(16,7), sharex=True) # 12 subplots (3 food combinations, 4 food concentrations)
    for a, assay in enumerate(assaychoices):
        for c, conc in enumerate(concentrations):
            try:
                # Get prefood-assaychoice-concentration group
                df_conc = groupedMetaData.get_group((prefood,assay,conc))
                
                # Get acclim time
                info = df_conc.iloc[0,:]
                acclim = int(info['Acclim_time_s'] * fps)
                
                # If single file, read food choice data (mean proportion feeding)
                if df_conc.shape[0] == 1:
                    foodchoicepath = change_path_phenix(info['filename'], returnpath='foodchoice')
                    df = pd.read_csv(foodchoicepath, header=0, index_col=0)   
                    
                    # Shift df indices to account for acclimation (t0 = pick time)
                    acclim = int(info['Acclim_time_s'] * fps)
                    df.index = df.index + acclim
                    
                # If multiple files, read food choice data for each file and compile into df for plotting
                elif df_conc.shape[0] > 1:
                    df = pd.DataFrame()
                    for row in range(df_conc.shape[0]):
                        info = df_conc.iloc[row,:]
                        foodchoicepath = change_path_phenix(info['filename'], returnpath='foodchoice')
                        tmp_df = pd.read_csv(foodchoicepath, header=0, index_col=0)  
                        
                        # Shift df indices to account for acclimation (t0 = pick time)
                        acclim = int(info['Acclim_time_s'] * fps)
                        tmp_df.index = tmp_df.index + acclim
                        
                        if df.empty:
                            df = tmp_df
                        else:
                            df = df.append(tmp_df, sort=True)
                            
                # Caclculate mean + standard deviation per frame across videos
                colnames = list(df.columns)
                df['frame'] = df.index
                fundict = {x:['mean','std'] for x in colnames}
                df_plot = df.groupby('frame').agg(fundict)

                # Remove erroneous frames where on/off food does not sum to 1
                frames_to_rm = np.where(np.sum([df_plot[x]['mean'] for x in colnames], axis=0).round(decimals=5)!=1)[0]
                assert frames_to_rm.size == 0,\
                    "{:d} frames found in which feeding proportions do not sum to 1.".format(len(frames_to_rm))
                
                # Time series plots
                plot_timeseries_phenix(df_plot, colour_dict, window=smooth_window,
                                       legend=False, annotate=False, acclimtime=acclim, ax=axs[a,c])
                
                # Add number of replicates (videos) for each treatment combination
                axs[a,c].text(0.79, 0.9, ("n={0}".format(df_conc.shape[0])),\
                              transform=axs[a,c].transAxes, fontsize=13)
                
                # Set axis limits
                if max(df_plot.index) > xmax:
                    xmax = max(df_plot.index)
                axs[a,c].set_xlim(0, np.round(xmax,-5))
                axs[a,c].set_ylim(-0.05, 1.05)
                                
                # Set column labels on first row
                if a == 0:
                    axs[a,c].text(0.5, 1.15, "$OD_{{{}}}={}$".format(600, conc*OpticalDensity600),\
                                  horizontalalignment='center', fontsize=18,\
                                  transform=axs[a,c].transAxes)
                    
                # Set main y axis label + ticks along first column of plots
                if c == 0:
                    yticks = list(np.round(np.linspace(0,1,num=6,endpoint=True),decimals=1))
                    axs[a,c].set_yticks(yticks)
                    axs[a,c].set_yticklabels(yticks)
                    axs[a,c].set_ylabel("{0}".format(assay), labelpad=15, fontsize=15)
                    if a == 1:
                        axs[a,c].text(-np.round(xmax,-5)/2, 0.5, "Mean Proportion Feeding",\
                                      fontsize=22, rotation=90, horizontalalignment='center',\
                                      verticalalignment='center')
                else:
                    axs[a,c].set_yticklabels([])
                    
                # Set main x axis label + ticks along final row of plots
                if a == len(assaychoices) - 1:
                    xticklabels = ["0", "30", "60", "90", "120"]
                    xticks = [int(int(lab)*fps*60) for lab in xticklabels]
                    axs[a,c].set_xticks(xticks)
                    axs[a,c].set_xticklabels(xticklabels)
                    if c == 1:
                        axs[a,c].set_xlabel("Time (minutes)", labelpad=25, fontsize=20, horizontalalignment='left')
                else:
                    axs[a,c].set_xticklabels([])

            except Exception as e:
                # Empty plots
                print("No videos found for concentration: %s\n(Assay: %s, Prefed on: %s)\n" % (e, assay, prefood))
                
                # Add number of replicates (videos) for each treatment combination
                axs[a,c].text(0.79, 0.9, "n=0", fontsize=13, transform=axs[a,c].transAxes)
                                
                # Set column labels on first row
                if a == 0:
                    axs[a,c].text(0.5, 1.15, ("conc={0}".format(conc)),\
                                  horizontalalignment='center', fontsize=18,\
                                  transform=axs[a,c].transAxes)
                axs[a,c].axis('off')
    
    # Add 'prefed on' to multiplot
    # plt.text(max(df_plot.index), -0.7, "Prefed on: {0}".format(prefood), horizontalalignment='right', fontsize=30)
    
    # Add legend
    patches = []
    for key, value in colour_dict.items():
        patch = mpatches.Patch(color=value, label=key)
        patches.append(patch)
    fig.legend(handles=patches, labels=treatments, loc="upper right", borderaxespad=0.4,\
               frameon=False, fontsize=15)
    
    # Tight-layout + adjustments
    fig.tight_layout(rect=[0.07, 0.02, 0.9, 0.93])
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.show(); plt.pause(1)
    
    # Save figure 3
    fig_name = "FoodChoiceTS_prefed" + prefood + ".png"
    figure_out = os.path.join(PROJECT_ROOT_DIR, "Results", "Plots", fig_name)
    plt.savefig(figure_out, saveFormat='png', dpi=300)
    
print("Complete!\n(Time taken: %d seconds)" % (time.time() - tic))
