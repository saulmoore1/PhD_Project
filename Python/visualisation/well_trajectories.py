#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot worm trajectories for a given well

@author: sm5911
@date: 01/03/2021

"""

#%% Imports 

import os
import sys
import time
import argparse
import numpy as np
#from pathlib import Path
from matplotlib import pyplot as plt

sys.path.insert(0, '/Users/sm5911/Documents/GitHub/PhD_Project/Python') # Path to GitHub functions

from read_data.get_trajectories import get_trajectory_data
from manual_labelling.label_lawns import plotbrightfield, hexcolours

# TODO: Check compatibility with Hydra camera 16-well videos + maintain backwards compatibility

#%% Functions

def plotpoly(fig, ax, poly_dict, colour=True):
    """ A function for plotting polygons onto an image from a dictionary of 
        coordinates
    """
    from matplotlib import patches

    labels = list(poly_dict.keys())
    if colour:
        colours = hexcolours(len(labels))
        for i, key in enumerate(labels):
            polygon = patches.Polygon(poly_dict[key], closed=True, color=colours[i], alpha=0.5)
            ax.add_patch(polygon)
    else:
        for i, key in enumerate(labels):
            polygon = patches.Polygon(poly_dict[key], closed=True, color='k', alpha=0.2)
            ax.add_patch(polygon)
    plt.show(); plt.pause(0.0001)
    
    return(fig, ax)

def plot_worm_trajectories(maskedfilepath, downsample=1, save=True, skip=True):
    """ A function to plot tracked trajectories for individual worms in a given 
        assay video recording
        
        Parameters
        ----------
        maskedfilepath :  str
            Path to MaskedVideo file for plotting worm trajectories
        downsample : int
            Downsample video to plot trajectories for every nth frame
        save : bool
            Save trajectory plots to file
        skip : bool
            Ignore videos that have already had trajectories plotted
    """
    
    # Specify file paths
    featurefilepath = maskedfilepath.replace(".hdf5", "_featuresN.hdf5")
    featurefilepath = featurefilepath.replace("MaskedVideos/", "Results/")
    coordfilepath = maskedfilepath.replace(".hdf5", "_FoodCoords.txt")
    coordfilepath = coordfilepath.replace("Priota/Data/FoodChoiceAssay/MaskedVideos/",\
                                          "Saul/FoodChoiceAssay/Results/FoodCoords/")
    plotpath = coordfilepath.replace("_FoodCoords.txt", "_WormTrajPlot.png")
    plotpath = plotpath.replace("FoodCoords/", "Plots/")

    # Avoid re-labelling videos
    if os.path.exists(coordfilepath) and skip: 
        print("Skipping file. Food coordinates already saved.")
    else:
        print("Plotting worm trajectories for file: %s" % maskedfilepath)
        # Read food coordinates
        f = open(coordfilepath, 'r').read()
        poly_dict = eval(f)
        
        # Plot brightfield
        plt.close("all")
        fig, ax = plotbrightfield(maskedfilepath, 0, figsize=(10,8))
        
        # Overlay food regions
        fig, ax = plotpoly(fig, ax, poly_dict, colour=False)
        
        # Read trajectory data
        df = get_trajectory_data(featurefilepath)
        
        # Overlay worm trajectories
        worm_ids = list(np.unique(df['worm_id']))
        colours = hexcolours(len(worm_ids))
        group_worm = df.groupby('worm_id')
        for w, worm in enumerate(worm_ids):
            colour = colours[w]
            df_worm = group_worm.get_group(worm)
            ax.scatter(x=df_worm['x'][::downsample], y=df_worm['y'][::downsample],\
                                  c=colour, s=3)
            ax.plot(df_worm['x'].iloc[0], df_worm['y'].iloc[0], color='r', marker='+',\
                    markersize=5, linestyle='', label="Start")
            ax.plot(df_worm['x'].iloc[-1], df_worm['y'].iloc[-1], color='b', marker='+',\
                    markersize=5, linestyle='', label="End")
        plt.legend(["Start", "End"], loc='upper right')
        plt.autoscale(enable=True, axis='x', tight=True) # re-scaling axes
        plt.autoscale(enable=True, axis='y', tight=True)
        plt.show(); plt.pause(0.0001)
        
        # Save plot
        if save:
            directory = os.path.dirname(plotpath)
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(plotpath, format='png', dpi=300)  

#%% Main

if __name__ == '__main__':
    # Accept MaskedVideo list from command line
    parser = argparse.ArgumentParser(description='Worm trajectory plots from list of MaskedVideos')
    parser.add_argument('--maskedvideo_list', help="List of MaskedVideo filepaths for videos to \
                        annotate lawns", nargs='+', default=['/Volumes/behavgenom$/Priota/Data/FoodChoiceAssay/MaskedVideos/20190404/PC2/Set2/Set2_Ch4_04042019_113928.hdf5'])
    args = parser.parse_args()

    assert type(args.maskedvideo_list) == list
    maskedfilelist = args.maskedvideo_list
        
    # Find masked HDF5 video files
    print("\nPlotting worm trajectories:\n%d masked videos found..\n" % len(maskedfilelist))
    
    # Plotting worm trajectories in well
    tic = time.time()
    for i in range(len(maskedfilelist)):    
        maskedfilepath = maskedfilelist[i]
        
        plot_worm_trajectories(maskedfilepath, downsample=1, save=False, skip=False)
        
    print("Trajectory plotting complete!\n(Time taken: %d seconds.)\n" % (time.time() - tic))
