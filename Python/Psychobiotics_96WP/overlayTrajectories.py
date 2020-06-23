#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
View Tierpsy Trajectories

A script to plot trajectories for worms tracked in the 16 wells imaged under a 
given camera on the rig (unique filename containing camera_serial). 
Just provide a filename from tierpsy filenames summaries and a plot will be 
produced of the first frame from the video with worm trajectories overlaid.

@author: sm5911
@date: 23/06/2020

"""

import sys
import h5py
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path

def gettrajdata(featuresfilepath):
    """ A function to read Tierpsy-generated featuresN file trajectories data
        and extract the following info as a dataframe:
        ['coord_x', 'coord_y', 'frame_number', 'worm_index_joined'] """
    # Read HDF5 file + extract info
    with h5py.File(featuresfilepath, 'r') as f:
        df = pd.DataFrame({'x': f['trajectories_data']['coord_x'],\
                           'y': f['trajectories_data']['coord_y'],\
                           'frame_number': f['trajectories_data']['frame_number'],\
                           'worm_id': f['trajectories_data']['worm_index_joined']})
    # {'midbody_speed': f['timeseries_data']['speed_midbody']}
    return(df)

def plotbrightfield(maskedfilepath, frame, **kwargs):
    """ Plot the first bright-field image of a given masked HDF5 video. """
    
    with h5py.File(maskedfilepath, 'r') as f:
        if frame == 'all':
            # Extract all frames and take the brightest pixels over time
            brightfield = f['full_data'][:]
            brightfield = brightfield.max(axis=0)
        else:
            # Extract a given frame (index)
            brightfield = f['full_data'][frame]
    fig, ax = plt.subplots(**kwargs)
    plt.imshow(brightfield, cmap='gray', vmin=0, vmax=255)
    return(fig, ax)

def plottrajectory(fig, ax, featurefilepath, downsample=10):
    """ Overlay feature file trajectory data onto existing figure. 
        NB: Plot figure and axes objects must both be provided on function call. """
        
    df = gettrajdata(featurefilepath)
    # Downsample frames for plotting
    if downsample < 1 or downsample == None: # Input error handling
        downsample = 1
    plt.scatter(x=df['x'][::downsample], y=df['y'][::downsample],\
                s=0.25, c=df['frame_number'][::downsample])
    plt.tick_params(labelsize=5)
    legend = plt.colorbar(pad=0.01)
    legend.ax.get_yaxis().labelpad = 10                # legend spacing
    legend.ax.set_ylabel('Frame Number', rotation=270) # legend label #fontsize
    legend.ax.tick_params(labelsize=5)
    plt.autoscale(enable=True, axis='x', tight=True)   # re-scaling axes
    plt.autoscale(enable=True, axis='y', tight=True)
    plt.show(); plt.pause(0.0001)
    return(fig, ax)

if __name__ == "__main__":
    print("Running: ", sys.argv[0])
    
    if len(sys.argv) == 1:
        featuresN_filepath = Path("/Volumes/behavgenom$/Saul/MicrobiomeScreen96WP/Results/20200222/microbiome_screen2_run7_p1_20200222_122858.22956805/metadata_featuresN.hdf5")
        print("WARNING: No filepath provided!\nUsing example file instead: '%s'" % featuresN_filepath)
    else: 
        featuresN_filepath = Path(sys.argv[1])
    fig, ax = plt.subplots()
    maskedvideo_filepath = Path(str(featuresN_filepath).replace("Results/", 
                           "MaskedVideos/").replace("_featuresN.hdf5", ".hdf5"))
    fig, ax = plotbrightfield(maskedvideo_filepath, frame=0)
    fig, ax = plottrajectory(fig, ax, featuresN_filepath, downsample=25)
    

    