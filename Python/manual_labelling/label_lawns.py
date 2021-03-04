#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual Labelling of Food Regions for On/Off Food Calculation

@author: sm5911
@date: 01/03/2021

"""

# TODO: Check compatibility with Hydra camera 16-well videos + maintain backwards compatibility
# TODO: Automate food labelling using (U-Net) ML algorithm -- pre-trained already by Luigi

#%% Imports 

import os
import time
import argparse

from visualisation.plotting_helper import hexcolours
from visualisation.plate_trajectories import plot_trajectory                                          

#%% Globals

EXAMPLE_FILE = "/Volumes/behavgenom$/Priota/Data/FoodChoiceAssay/MaskedVideos/20181101/PC1/Set1/Set1_Ch1_01112018_103218.hdf5"

SKIP = False # Skip file if coordinates of food lawn are already saved

#%% Functions

def plot_brightfield(maskedfilepath, frame, **kwargs):
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

def draw_polygon(plt, n_poly=2):
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

def plot_polygon(poly_dict, ax=None, colour=True, alpha=0.5, **kwargs):
    """ A function for plotting polygons onto an image from a dictionary of coordinates """
    
    from matplotlib import pyplot as plt
    from matplotlib import patches

    if not ax:
        fig, ax = plt.subplots(**kwargs)

    labels = list(poly_dict.keys())
    colours = hexcolours(len(labels))

    for i, key in enumerate(labels):
        polygon = patches.Polygon(poly_dict[key], 
                                  closed=True, 
                                  color=colours[i] if colour else 'k', 
                                  alpha=alpha)
        ax.add_patch(polygon)
            
    return(ax)

def label_polygon(poly_dict, ax):
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

def annotate_lawns(maskedfilepath, n_poly=2, save=True, skip=True, out_dir="FoodChoiceAssay"):
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
    
    if type(maskedfilepath) is not str:
        maskedfilepath = str(maskedfilepath)
        
    plt.ion()

    # Define paths
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
            fig, ax = plot_brightfield(maskedfilepath, 0, figsize=(10,10))
            plt.show()
            
            # USER INPUT: Draw polygons around food regions in the assay
            print("Manually outline food regions.\nLeft click - add a point.\n\
                   Right click - remove a point.\nMiddle click - next")
            poly_dict = draw_polygon(plt, n_poly=n_poly)
            plt.show()
            # TODO: Check that assay not bumped by using last frames for rest of analysis
            
            # USER INPUT: Assign labels to food regions
            poly_dict = label_polygon(poly_dict, ax)
            
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
            fig, ax = plot_trajectory(fig, ax, featurefilepath, downsample=10)
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
                        annotate lawns", nargs='+', default=[EXAMPLE_FILE])
    args = parser.parse_args()

    assert type(args.maskedvideo_list) == list
    maskedfilelist = args.maskedvideo_list
        
    # Find masked HDF5 video files (for labelling) 
    print("\nManual labelling:\n%d masked videos found..\n" % len(maskedfilelist))
    
    # Interactive plotting (for user input when labelling plots)
    tic = time.time()
    for i in range(len(maskedfilelist)):    
        maskedfilepath = maskedfilelist[i]
        # Manually outline + assign labels to food regions + save coordinates + trajectory overlay to file           
        annotate_lawns(maskedfilepath, save=True, skip=SKIP)
        
    print("Manual labelling complete!\n(Time taken: %d seconds.)\n" % (time.time() - tic))
