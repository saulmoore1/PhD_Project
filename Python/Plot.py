#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULE: PLOT

@author: sm5911
@date: 19/02/2019

"""

# IMPORTS
import os, h5py, colorsys
import numpy as np
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt

# CUSTOM IMPORTS
from Read import gettrajdata
from Save import savefig

# FUNCTIONS
def hexcolours(n):
    """ A function for generating a list of n hexadecimal colours for plotting. """
    hex_list = []
    HSV = [(x*1/n,0.5,0.5) for x in range(n)]
    # Generate RGB hex code
    for RGB in HSV:
        RGB = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*RGB))
        hex_list.append('#%02x%02x%02x' % tuple(RGB)) 
    return(hex_list)

def hex2rgb(hex):
    hex = hex.lstrip('#')
    hlen = len(hex)
    return tuple(int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))

def plotbrightfield(maskedfilepath, frame, **kwargs):
    """ Plot the first bright-field image of a given masked HDF5 video (file path). """
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
                c=df['frame_number'][::downsample], s=2)
    legend = plt.colorbar()
    legend.ax.get_yaxis().labelpad = 15                # legend spacing
    legend.ax.set_ylabel('Frame Number', rotation=270) # legend label
    plt.autoscale(enable=True, axis='x', tight=True)   # re-scaling axes
    plt.autoscale(enable=True, axis='y', tight=True)
    plt.show(); plt.pause(0.0001)
    return(fig, ax)
        
def drawpoly(plt, n_poly=2):
    """ A function to record coordinates from user mouse input when drawing a 
        number of polygons (eg. oulining food patches) on an image and return 
        them as a dictionary (keys=polygons,values=xycoords). 
        NB: For assigning labels to polygons, see 'labelpoly'. """
    poly_dict = {}
    for polygon in range(n_poly):
        print("Draw polygon %d" % (polygon + 1))
        poly_dict['{0}'.format('Poly_' + str(polygon))] = plt.ginput(n=-1,\
                  timeout=0,mouse_add=1,mouse_pop=3,mouse_stop=2)
        print("Done.")
    return(poly_dict)
    
def labelpoly(fig, ax, poly_dict):
    """ A function that accepts keyboard input from the user to assign labels 
        (stored as dictionary keys) to each polygon (set of x,y coords). """
    labels = list(poly_dict.keys())
    colours = hexcolours(len(labels))
    for i, key in enumerate(labels):
        polygon = mpatches.Polygon(poly_dict[key], closed=True, color=colours[i], alpha=0.5)
        ax.add_patch(polygon); plt.show()
        label = input("Assign name to {0}: ".format(key))
        poly_dict['{0}'.format(label.upper())] = poly_dict.pop(key)
    return(poly_dict)

def plotpoly(fig, ax, poly_dict, colour=True):
    """ A function for plotting polygons onto an image from a dictionary of 
        coordinates. """
    labels = list(poly_dict.keys())
    if colour:
        colours = hexcolours(len(labels))
        for i, key in enumerate(labels):
            polygon = mpatches.Polygon(poly_dict[key], closed=True, color=colours[i], alpha=0.5)
            ax.add_patch(polygon)
    else:
        for i, key in enumerate(labels):
            polygon = mpatches.Polygon(poly_dict[key], closed=True, color='k', alpha=0.2)
            ax.add_patch(polygon)
    plt.show(); plt.pause(0.0001)
    return(fig, ax)
    
def plotpoints(fig, ax, x, y, **kwargs):
    """ A function for plotting points onto an image. """
    ax.plot(x, y, **kwargs)
    plt.show(); plt.pause(0.0001)
    return(fig, ax)
            
def plotpie(df, rm_empty=True, show=True, **kwargs):
    """ A function to plot a pie chart from a labelled vector of values 
        that sum to 1. """
    if rm_empty: # Remove any empty rows
        df = df.loc[df!=0]
    fig = plt.pie(df, autopct='%1.1f%%', **kwargs)
    plt.axis('equal')
    plt.tight_layout()
    if show:
        plt.show(); plt.pause(0.0001)
    return(fig)
    
def plottimeseries(df, colour_dict, window=1000, acclimtime=None, count=False,\
                   legend=False, show=True, annotate=True, **kwargs):
    """ Function to plot a time series plot.
        Arguments: 
        - window = 500 (default) Number frames for moving average smoothing)
        - orderby = None (default) If provided, first groups df by variable and 
          calculates either mean/sum
        - count = False (default) 
    """
    # List of food labels + dictionary keys for plot colours
    food_labels = list(df.columns)
    colour_keys = [food.split('_')[0] for food in food_labels]
    
    # Calculate moving window + plot time series
    for i, food in enumerate(food_labels):
        if window:
            moving_window = df[food].rolling(window=window, center=True).mean()
            moving_window_of_std = yerr[food].rolling(window=window, center=True).mean()
#            ax = moving_window.plot(color=colour_dict[colour_keys[i]], yerr=moving_window_of_std[food], **kwargs)
            ax = plt.errorbar(moving_window.index(), moving_window, yerr=moving_window_of_std)
        else:
            ax = df[food].plot(color=colour_dict[colour_keys[i]], yerr=yerr[food], **kwargs)
    if annotate:
        if count:
            plt.ylim(-0.05, df.max(axis=1).max() + 0.5)
            plt.ylabel("Number of Worms", fontsize=15, labelpad=10)
            #plt.axhline(y=3, color='k', linestyle='--') # plot line at n_worms = 3
        else:
            plt.ylim(-0.05, 1.15)
            plt.ylabel("Proportion Feeding", fontsize=15, labelpad=10)
        plt.xlim(0, max(df.index))
        plt.xlabel("Frame Number", fontsize=15, labelpad=10)
    else:
        # Turn off tick/axes labels
        x_axis = ax.axes.get_xaxis()
        x_axis.set_label_text("")
    if acclimtime:
        x = np.arange(0, acclimtime)
        y = acclimtime
        ax.fill_between(x, y, -0.05, color='grey', alpha='0.5', interpolate=True)
        ax.axvline(0, ls='-', lw=1, color='k')
        ax.axvline(acclimtime, ls='-', lw=1, color='k')
        plt.text(acclimtime/max(df.index)+0.01, 0.97, "Acclimation: {0} mins".format(int(acclimtime/25/60)),\
                 ha='left', va='center', transform=ax.transAxes, rotation=0, color='k')
        # ax.axvline(acclimtime + window/2, ls='-', lw=2, color='r')
    if legend:
        patches = []
        legend_keys = list(np.unique(colour_keys))
        for key in legend_keys:
            patch = mpatches.Patch(color=colour_dict[key], label=key)
            patches.append(patch)
        plt.tight_layout(rect=[0, 0, 0.88, 0.98])
        plt.legend(handles=patches, labels=legend_keys, loc=(1.02, 0.8),\
                   borderaxespad=0.4, frameon=False, fontsize=15)
    if show:
        plt.show(); plt.pause(0.0001)
    return(plt)
    
def manuallabelling(maskedfilepath, save=True, skip=True):
    """ A function written to assist with the manual labelling of food regions in the worm
        food choice assay video recordings. Food regions are given labels and coordinates
        are saved to file. If a coordinate file already exists for the video, the file
        will be skipped.
        - Plot brightfield image
        - Accept user-input to outline and assign labels to food regions in image + save coordinates
        - Plot labelled trajectory overlay 
        - Save plot (default, save=True)
        - Skips files which have already been labelled (default, skip=True)
    """
    featurefilepath = maskedfilepath.replace(".hdf5", "_featuresN.hdf5")
    featurefilepath = featurefilepath.replace("MaskedVideos/", "Results/")
    coordfilepath = maskedfilepath.replace("Priota/Data/FoodChoiceAssay/MaskedVideos/",\
                                           "Saul/FoodChoiceAssay/Results/FoodCoords/")
    coordfilepath = coordfilepath.replace(".hdf5", "_FoodCoords.txt")
    plotpath = coordfilepath.replace("FoodCoords/", "Plots/")
    plotpath = plotpath.replace("_FoodCoords.txt", "_LabelledOverlayPlot.png")
    # Avoid re-labelling videos
    if os.path.exists(coordfilepath) and skip: 
        print("Skipping file. Food coordinates already saved.")
    else:
        try: # Manually label food regions
            # Plot the first brightfield image
            print("\n\nProcessing file:\n%s" % maskedfilepath)                    
            plt.close("all")
            fig, ax = plotbrightfield(maskedfilepath, 0, figsize=(10,10))
            plt.show()
            # USER INPUT: Draw polygons around food regions in the assay
            print("Manually outline food regions.\nLeft click - add a point.\n\
                   Right click - remove a point.\nMiddle click - next")
            poly_dict = drawpoly(plt, n_poly=2)
            plt.show()
            # USER INPUT: Assign labels to food regions
            poly_dict = labelpoly(fig, ax, poly_dict)
            if save:
                # Save coordinates to file
                directory = os.path.dirname(coordfilepath)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                fid = open(coordfilepath, 'w')
                print(poly_dict, file=fid)
                fid.close()
                print("Coords successfully saved!")
            # Plot trajectory overlay
            fig, ax = plottrajectory(fig, ax, featurefilepath, downsample=10)
            plt.show()
            # Save plot overlay
            if save:
                directory = os.path.dirname(plotpath)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                savefig(plotpath, saveFormat='png')
            # Return food coordinates
            return(poly_dict)
        except Exception as e:
            print("ERROR! Failed to process file: \n%s\n%s" % (maskedfilepath, e))

def wormtrajectories(maskedfilepath, downsample=1, save=True):
    """ A function to plot tracked trajectories for individual worms in a given 
        assay video recording. 
        Optional arguments:
        - downsample=1 -- set step size for frame downsampling
        - save=True -- if true, plots are saved to file
    """
    # Specify file paths
    featurefilepath = maskedfilepath.replace(".hdf5", "_featuresN.hdf5")
    featurefilepath = featurefilepath.replace("MaskedVideos/", "Results/")
    coordfilepath = maskedfilepath.replace(".hdf5", "_FoodCoords.txt")
    coordfilepath = coordfilepath.replace("Priota/Data/FoodChoiceAssay/MaskedVideos/",\
                                      "Saul/FoodChoiceAssay/Results/FoodCoords/")
    # Read food coordinates
    f = open(coordfilepath, 'r').read()
    poly_dict = eval(f)
    # Plot brightfield
    plt.close("all")
    fig, ax = plotbrightfield(maskedfilepath, 0, figsize=(10,8))
    # Overlay food regions
    fig, ax = plotpoly(fig, ax, poly_dict, colour=False)
    # Read trajectory data
    df = gettrajdata(featurefilepath)
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
    plt.show(); plt.pause(2)
    # Save plot
    if save:
        plotpath = coordfilepath.replace("_FoodCoords.txt", "_WormTrajPlot.png")
        plotpath = plotpath.replace("FoodCoords/", "Plots/")
        directory = os.path.dirname(plotpath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        savefig(plotpath, saveFormat='png', tellme=True)  


