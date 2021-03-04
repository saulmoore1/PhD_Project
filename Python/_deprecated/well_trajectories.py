#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot worm trajectories for a given well

@author: sm5911
@date: 01/03/2021

"""

#%% Imports 

import os
import time
import argparse
import numpy as np
from matplotlib import pyplot as plt

from read_data.read import get_trajectory_data
from visualisation.plotting_helper import hexcolours, plot_brightfield, plot_polygon

# TODO: Check compatibility with Hydra camera 16-well videos + maintain backwards compatibility

#%% Globals

EXAMPLE_FILE = "/Volumes/behavgenom$/Priota/Data/FoodChoiceAssay/MaskedVideos/20181101/PC1/Set1/Set1_Ch1_01112018_103218.hdf5"

#%% Functions

def overlay_trajectory_endpoints(maskedfilepath, downsample=1, save=True, skip=True):
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
        plt.ioff() if save else plt.ion()
        fig, ax = plot_brightfield(maskedfilepath, 0, figsize=(10,8))
        
        # Overlay food regions
        fig, ax = plot_polygon(fig, ax, poly_dict, colour=False)
        
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
        plt.tight_layout()
        
        # Save plot
        if save:
            directory = os.path.dirname(plotpath)
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(plotpath, format='png', dpi=300)
        else:
            plt.show()

#%% Main

if __name__ == '__main__':
    # Accept MaskedVideo list from command line
    parser = argparse.ArgumentParser(description='Worm trajectory plots from list of MaskedVideos')
    parser.add_argument('--maskedvideo_list', help="List of MaskedVideo filepaths for videos to \
                        annotate lawns", nargs='+', default=[EXAMPLE_FILE])
    args = parser.parse_args()

    assert type(args.maskedvideo_list) == list
    maskedfilelist = args.maskedvideo_list
        
    # Find masked HDF5 video files
    print("\nPlotting trajectories:\n%d masked videos found..\n" % len(maskedfilelist))
    
    # Plotting worm trajectories in well
    tic = time.time()
    for i in range(len(maskedfilelist)):    
        maskedfilepath = maskedfilelist[i]
        
        overlay_worm_trajectories(maskedfilepath, downsample=1, save=False, skip=False)
        
    print("Trajectory plotting complete!\n(Time taken: %d seconds.)\n" % (time.time() - tic))
