#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: sm5911
@date: 19/02/2019

"""

# IMPORTS
import os
import argparse
import numpy as np
from matplotlib import pyplot as plt

# CUSTOM IMPORTS
from read_data.read import get_trajectory_data
from read_data.paths import change_path
from visualisation.plate_trajectories import plot_trajectory
from filter_data.filter_trajectories import filter_worm_trajectories
from visualisation.plotting_helper import hexcolours
from manual_labelling.label_lawns import plot_brightfield, plot_polygon

#%% Globals

EXAMPLE_FILE = "/Volumes/behavgenom$/Priota/Data/MicrobiomeAssay/MaskedVideos/20190614/food_choice_set1_op50_20190614_140234.22956805/metadata.hdf5"

#%% FUNCTIONS
    
def plot_trajectory_phenix(featurefilepath, downsample=None, save=True, skip=True):
    """ A function to plot tracked trajectories for individual worms in a given 
        assay video recording
        Optional arguments:
        - downsample=1 -- set step size for frame downsampling
        - save=True -- if true, plots are saved to file
    """
    
    featurefilepath = change_path(featurefilepath, to='features')
    maskedfilepath = change_path(featurefilepath, to='masked')
    
    # Specify file paths
    coordfilepath = featurefilepath.replace(".hdf5", "_FoodCoords.txt")
    coordfilepath = coordfilepath.replace("Priota/Data/FoodChoiceAssay/MaskedVideos/",\
                                          "Saul/FoodChoiceAssay/Results/FoodCoords/")
    plotpath = coordfilepath.replace("_FoodCoords.txt", "_WormTrajPlot.png")
    plotpath = plotpath.replace("FoodCoords/", "Plots/")

    # Avoid re-labelling videos
    if os.path.exists(coordfilepath) and skip: 
        print("Skipping file. Food coordinates already saved.")
    else:
        print("Plotting worm trajectories for file: %s" % featurefilepath)

        # Read trajectory data
        df = get_trajectory_data(featurefilepath)
        
        df = filter_worm_trajectories(df)

        # Plot brightfield
        plt.close("all")
        plt.ioff() if save else plt.ion()
        fig, ax = plot_brightfield(maskedfilepath, 0, figsize=(10,8))
                
        # Plot food region (from coords)
        try:
            # Read food coordinates
            f = open(coordfilepath, 'r').read()
            poly_dict = eval(f)
                    
            # Overlay food regions
            ax = plot_polygon(poly_dict, ax, colour=False)
        except:
            print("WARNING: Could not read/plot food coords:\n'%s'" % coordfilepath)
        
        plot_trajectory(featurefilepath)
               
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
        
        # Save plot
        if save:
            directory = os.path.dirname(plotpath)
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(plotpath, format='png', dpi=300)  
        else:
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot worm trajecetory start/end points")
    
    # FeaturesN filepath
    parser.add_argument("--features_file", help="Input a single featuresN HDF5 \
                        filepath to plot worm trajecotries start/end for that video", 
                        default=EXAMPLE_FILE)  
    # Full filenames filepath
    parser.add_argument("--downsample", help="Downsample to plot trajectory xy coords for every \
                        nth frame", default=None)
    # Save dirpath
    parser.add_argument("--save_dir", help="Path to directory to save plate \
                        trajectories", default=None)
    # Downsample
    parser.add_argument("--downsample", help="Downsample trajectory data by \
                        plotting the worm centroid for every 'nth' frame",
                        default=None)
    args = parser.parse_args()
    
    featurefilepath = args.features_file
    
    plot_trajectory_phenix(str(args.features_file), 
                           downsample=int(args.downsample), 
                           saveDir=args.save_dir, 
                           skip_ok=args.skip_ok)

    
    