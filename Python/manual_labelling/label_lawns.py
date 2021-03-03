#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual Labelling of Food Regions for On/Off Food Calculation

@author: sm5911
@date: 01/03/2021

"""

# TODO: Check compatibility with Hydra camera 16-well videos + maintain backwards compatibility

#%% Imports 

import os
import sys
import time
import argparse

sys.path.insert(0, '/Users/sm5911/Documents/GitHub/PhD_Project/Python') # Path to GitHub functions

from read_data.get_trajectories import get_trajectory_data

#%% Functions

def hexcolours(n):
    """ A function for generating a list of n hexadecimal colours for plotting. """
    
    import colorsys

    hex_list = []
    HSV = [(x*1/n,0.5,0.5) for x in range(n)]
    # Generate RGB hex code
    for RGB in HSV:
        RGB = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*RGB))
        hex_list.append('#%02x%02x%02x' % tuple(RGB))
        
    return(hex_list)


def hex2rgb(hex):
    """ A function for converting from hexadecimal to RGB colour format for plotting. """
    
    hex = hex.lstrip('#')
    hlen = len(hex)
    
    return tuple(int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))


def plotbrightfield(maskedfilepath, frame, **kwargs):
    """ Plot the first bright-field image of a given masked HDF5 video. """
    
    import h5py
    from matplotlib import pyplot as plt

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
    
    from matplotlib import pyplot as plt

    df = get_trajectory_data(featurefilepath)
    
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
    
    from matplotlib import pyplot as plt
    from matplotlib import patches
    
    labels = list(poly_dict.keys())
    colours = hexcolours(len(labels))
    
    for i, key in enumerate(labels):
        polygon = patches.Polygon(poly_dict[key], closed=True, color=colours[i], alpha=0.5)
        ax.add_patch(polygon); plt.show()
        label = input("Assign name to {0}: ".format(key))
        poly_dict['{0}'.format(label.upper())] = poly_dict.pop(key)
        
    return(poly_dict)


def manuallabelling(maskedfilepath, n_poly=2, save=True, skip=True, out_dir="FoodChoiceAssay"):
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
    
    from matplotlib import pyplot as plt
    plt.ion()

    featurefilepath = maskedfilepath.replace(".hdf5", "_featuresN.hdf5")
    featurefilepath = featurefilepath.replace("MaskedVideos/", "Results/")
    coordfilepath = maskedfilepath.replace("Priota/Data/FoodChoiceAssay/MaskedVideos/",\
                                           ("Saul/"+out_dir+"/Results/FoodCoords/"))
    coordfilepath = coordfilepath.replace(".hdf5", "_FoodCoords.txt")
    plotpath = coordfilepath.replace("FoodCoords/", "Plots/")
    plotpath = plotpath.replace("_FoodCoords.txt", "_LabelledOverlayPlot.png")
    
    # Avoid re-labelling videos
    if os.path.exists(coordfilepath) and skip: 
        print("Skipping file. Food coordinates already saved.")
    else:
        # Manually label food regions
        try:
            # Plot the first brightfield image
            print("\n\nProcessing file:\n%s" % maskedfilepath)                    
            plt.close("all")
            fig, ax = plotbrightfield(maskedfilepath, 0, figsize=(10,10))
            plt.show()
            
            # USER INPUT: Draw polygons around food regions in the assay
            print("Manually outline food regions.\nLeft click - add a point.\n\
                   Right click - remove a point.\nMiddle click - next")
            poly_dict = drawpoly(plt, n_poly=n_poly)
            plt.show()
            # TODO: Check that assay not bumped by using last frames for rest of analysis
            
            # USER INPUT: Assign labels to food regions
            poly_dict = labelpoly(fig, ax, poly_dict)
            
            # Save coordinates to file
            if save:
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
                plt.savefig(plotpath, format='png', dpi=300)
            
            # Return food coordinates
            return(poly_dict)
        except Exception as e:
            print("ERROR! Failed to process file: \n%s\n%s" % (maskedfilepath, e))

#%% Main

if __name__ == '__main__':
    # Accept MaskedVideo list from command line for labelling food regions in multiple videos
    parser = argparse.ArgumentParser(description='Manual labelling of lawns in MaskedVideos')
    parser.add_argument('--maskedvideo_list', help="List of MaskedVideo filepaths for videos to \
                        annotate lawns", nargs='+', default=None)
    args = parser.parse_args()

    assert type(args.maskedvideo_list) == list
    maskedfilelist = args.maskedvideo_list
    
    # MANUAL LABELLING
    # TODO: Automate food labelling using (U-Net) ML algorithm -- pre-trained already by Luigi
    
    # Find masked HDF5 video files (for labelling) 
    print("\nManual labelling:\n%d masked videos found..\n" % len(maskedfilelist))
    
    # Interactive plotting (for user input when labelling plots)
    tic = time.time()
    for i in range(len(maskedfilelist)):    
        maskedfilepath = maskedfilelist[i]
        # Manually outline + assign labels to food regions + save coordinates + trajectory overlay to file           
        manuallabelling(maskedfilepath, save=True, skip=True)
        
    print("Manual labelling complete!\n(Time taken: %d seconds.)\n" % (time.time() - tic))
