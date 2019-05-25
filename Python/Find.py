#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULE: FIND

@author: sm5911
@date: 19/02/2019

"""

# IMPORTS
import os, re

#%% FUNCTIONS
def lookforfiles(root_dir, regex, depth=None, exact=False):
    """ A function to looks for files in a given starting directory 
        that match a given regular expression pattern. 
        eg. lookforfiles("~/Documents", ".*.csv$") """
    filelist = []
    # Iterate over all files within sub-directories contained within the starting root directory
    for root, subdir, files in os.walk(root_dir, topdown=True):
        if depth:
            start_depth = root_dir.count(os.sep)
            if root.count(os.sep) - start_depth < depth:
                for file in files:
                    if re.search(pattern=regex, string=file):
                        if exact:
                            if os.path.join(root, file).count(os.sep) - start_depth == depth:
                                filelist.append(os.path.join(root, file))
                        else: # if exact depth is not specified, return all matches to specified depth
                            filelist.append(os.path.join(root, file))
        else: # if depth argument is not provided, return all matches
            for file in files:
                if re.search(pattern=regex, string=file):
                    filelist.append(os.path.join(root, file))
    return(filelist)

#%%    
def lookfordirs(root_dir, regex, depth=None, exact=False):
    """ A function to look for sub-directories within a given starting directory 
        that match a given regular expression pattern. """
    dirlist = []
    # Iterate over all sub-directories contained within the starting root directory
    for root, subdir, files in os.walk(root_dir, topdown=True):
        if depth:
            start_depth = root_dir.count(os.sep)
            if root.count(os.sep) - start_depth < depth:                
                for dir in subdir:
                    if re.search(pattern=regex, string=dir):
                        if exact: 
                            if os.path.join(root, dir).count(os.sep) - start_depth == depth:
                                dirlist.append(os.path.join(root, dir))
                        else: # if exact depth is not specified, return all matches to specified depth
                            dirlist.append(os.path.join(root, dir))
        else: # if depth argument is not provided, return all matches
            for dir in subdir:
                if re.search(pattern=regex, string=dir):
                    dirlist.append(os.path.join(root, dir))        
    return(dirlist)   

#%%
def findworms(trajectory_df, threshold_move, threshold_time, tellme=True):
    """ A function to filter Tierpsy trajectory data worm IDs to remove unwanted
        tracked entities (ie. not worms), based on threshold parameters for the 
        duration (in frames) and amount of movement (in pixels) of their trajectories."""
    group_worm = trajectory_df.groupby('worm_id')
    n_worms = group_worm.count().shape[0]
    # Filter by TRAJECTORY LENGTH
    filterTime_df = group_worm.filter(lambda x: x['frame_number'].count() > threshold_time)
    # Re-group by worm id
    group_worm = filterTime_df.groupby('worm_id')
    n_worms_time = group_worm.count().shape[0]
    if tellme:
        print("%d worm IDs filtered that existed for less than %d frames (%.1f seconds)." %\
              (n_worms - n_worms_time, threshold_time, threshold_time/25))
    # Filter by MOVEMENT
    filterMove_df = group_worm.filter(lambda x: x['x'].ptp() > threshold_move or x['y'].ptp() > threshold_move)
    # Re-group by worm id
    group_worm = filterMove_df.groupby('worm_id')
    n_worms_alive = group_worm.count().shape[0]
    if tellme:
        print("%d worm IDs filtered that moved less than %d pixels (%d microns)." %\
              (n_worms_time - n_worms_alive, threshold_move, threshold_move*10))
        print("%d/%d worm IDs filtered in total." % (n_worms - n_worms_alive, n_worms))
    return(filterMove_df)

#%%
def changepath(maskedfilepath, returnpath=None, figname=None):
    """ A function written to change the filepath of a given masked video to one
        of the following file paths: 
        returnpath = ['features','skeletons','intensities','coords','onfood',
                      'foodchoice','summary','plots'] 
    """
    outfilepath=False
    if returnpath:
        if returnpath == 'features':
            outfilepath = maskedfilepath.replace("MaskedVideos/", "Results/")
            outfilepath = outfilepath.replace(".hdf5", "_featuresN.hdf5")
        elif returnpath == 'skeletons':
            outfilepath = maskedfilepath.replace("MaskedVideos/", "Results/")
            outfilepath = outfilepath.replace(".hdf5", "_skeletons.hdf5")
        elif returnpath == 'intensities':
            outfilepath = maskedfilepath.replace("MaskedVideos/", "Results/")
            outfilepath = outfilepath.replace(".hdf5", "_intensities.hdf5")
        elif returnpath == 'coords':
            outfilepath = maskedfilepath.replace("Priota/Data/FoodChoiceAssay/MaskedVideos/",\
                                                 "Saul/FoodChoiceAssay/Results/FoodCoords/")
            outfilepath = outfilepath.replace(".hdf5", "_FoodCoords.txt")
        elif returnpath == 'onfood':
            outfilepath = maskedfilepath.replace("Priota/Data/FoodChoiceAssay/MaskedVideos/",\
                                                 "Saul/FoodChoiceAssay/Results/FoodChoice/")
            outfilepath = outfilepath.replace(".hdf5", "_OnFood.csv")
        elif returnpath == 'foodchoice':
            outfilepath = maskedfilepath.replace("Priota/Data/FoodChoiceAssay/MaskedVideos/",\
                                                 "Saul/FoodChoiceAssay/Results/FoodChoice/")
            outfilepath = outfilepath.replace(".hdf5", "_FoodChoice.csv")
        elif returnpath == 'summary':
            outfilepath = maskedfilepath.replace("Priota/Data/FoodChoiceAssay/MaskedVideos/",\
                                                 "Saul/FoodChoiceAssay/Results/FoodChoice/")
            outfilepath = outfilepath.replace(".hdf5", "_Summary.csv")
#        elif returnpath == 'leavingevents':
#            outfilepath = maskedfilepath.replace("Priota/Data/FoodChoiceAssay/MaskedVideos/",\
#                                                 "Saul/FoodChoiceAssay/Results/LeavingRate/")
#            outfilepath = outfilepath.replace(".hdf5", "_leavingevents_true.csv")
        elif returnpath == 'plots':
            if figname:
                outfilepath = maskedfilepath.replace("Priota/Data/FoodChoiceAssay/MaskedVideos/",\
                                                     "Saul/FoodChoiceAssay/Results/Plots/")
                outfilepath = outfilepath.replace(".hdf5", "_" + figname)
            else:
                print("Please provide figname for plot!")
    else:
        print("Please select from the following options for returnpath:\
              \n['features', 'coords', 'onfood', 'foodchoice', 'summary', 'plots']")
    
    if outfilepath:
        return(outfilepath)
    else:
        print("ERROR!")
    
#%%    
    
    
    
    
    
    
    
    
    
    
    
    

