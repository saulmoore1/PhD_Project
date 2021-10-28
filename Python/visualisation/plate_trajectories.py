#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot 96-well Plate Trajectories

A script to plot trajectories for worms tracked in all the wells of 96-well plates under Hydra.
Just provide a featuresN filepath from Tierpsy filenames summaries and a plot will be produced of 
tracked worm trajectories throughout the video, for the entire 96-well plate 
(imaged under 6 cameras simultaneously)

@author: sm5911
@date: 23/06/2020

"""

#%% Imports 

import sys
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt

from read_data.read import get_trajectory_data
from filter_data.filter_trajectories import filter_worm_trajectories

sys.path.insert(0, "/Users/sm5911/Tierpsy_Versions/tierpsy-tracker") # path to tierpsy tracker repo
from tierpsy.analysis.split_fov.FOVMultiWellsSplitter import FOVMultiWellsSplitter
from tierpsy.analysis.split_fov.helper import CAM2CH_df, serial2channel, parse_camera_serial

#%% Globals

# Channel-to-plate mapping dictionary
# {'channel' : ((ax array location), rotate)}
CH2PLATE_dict = {'Ch1':((0,0),True),
                 'Ch2':((1,0),False),
                 'Ch3':((0,1),True),
                 'Ch4':((1,1),False),
                 'Ch5':((0,2),True),
                 'Ch6':((1,2),False)}

EXAMPLE_FILE = "/Volumes/hermes$/KeioScreen_96WP/Results/20210126/keio_plate3_run1_bluelight_20210126_124541.22956809/metadata_featuresN.hdf5"
EXAMPLE_FILE_OLD = "/Volumes/behavgenom$/Priota/Data/FoodChoiceAssay/Results/20181109/PC1/Set1/Set1_Ch1_09112018_101552_featuresN.hdf5"

THRESHOLD_NUMBER_PIXELS = 10
THRESHOLD_NUMBER_FRAMES = 25
FPS = 25
MICRONS_PER_PIXEL = 12.4

#%% Functions

def plot_trajectory(featurefilepath, 
                    well_name=None,
                    downsample=10,
                    filter_trajectories=False,
                    mark_endpoints=False,
                    annotate_lawns=False,
                    rotate=False, 
                    img_shape=None,
                    legend=True, 
                    ax=None,
                    verbose=True,
                    **kwargs):
    """ Overlay feature file trajectory data onto existing figure """

    from matplotlib import pyplot as plt
    
    df = get_trajectory_data(featurefilepath)        
    
    # Optional - filter trajectories using global movement/time threshold parameters
    if filter_trajectories:
        filter_worm_trajectories(df, 
                                 threshold_move=THRESHOLD_NUMBER_PIXELS, 
                                 threshold_time=THRESHOLD_NUMBER_FRAMES,
                                 fps=FPS,
                                 microns_per_pixel=MICRONS_PER_PIXEL,
                                 verbose=verbose)
    if not ax:
        fig, ax = plt.subplots(**kwargs)
        # # plot first frame of video + annotate wells
        # FOVsplitter = FOVMultiWellsSplitter(maskedfilepath)
        # FOVsplitter.plot_wells(is_rotate180=rotate, ax=ax, line_thickness=10)

    if annotate_lawns:
        # TODO: Update for compatibility with Tierpsy videos - new food coords path
        from manual_labelling.label_lawns import plot_polygon
        
        # Plot food region (from coords)
        coordfilepath = featurefilepath.replace("_featuresN.hdf5", "_FoodCoords.txt")
        coordfilepath = coordfilepath.replace("Priota/Data/FoodChoiceAssay/Results/",\
                                              "Saul/FoodChoiceAssay/Results/FoodCoords/") 
        print(coordfilepath)
        if Path(coordfilepath).exists():
            # Read food coordinates
            f = open(coordfilepath, 'r').read()
            poly_dict = eval(f)
                    
            # Overlay food regions
            ax = plot_polygon(poly_dict, ax, colour=False)
        else:
            print("WARNING: Could not find lawn annotations:\n\t%s\n" % coordfilepath) 
            
    # Rotate trajectories if necessary (for tiling 96-well plate)
    if rotate:
        # TODO: Update for compatibility with Phenix videos - rotate food coords polygon
        if not img_shape:
            raise ValueError('Image shape missing for rotation.')
        else:
            height, width = img_shape[0], img_shape[1]
            df['x'] = width - df['x']
            df['y'] = height - df['y']
            
    # Plot trajectory
    if downsample is not None:
        # Downsample frames for plotting
        downsample = 1 if downsample < 1 else downsample
        
        ax.scatter(x=df['x'][::downsample], y=df['y'][::downsample], 
                   c=df['frame_number'][::downsample], cmap='plasma', s=10)
    else:
        ax.scatter(x=df['x'], y=df['y'], c=df['frame_number'], cmap='plasma', s=10)
        
    if mark_endpoints:
        ax.plot(df['x'].iloc[0], df['y'].iloc[0], color='r', marker='+', 
                markersize=7, linestyle='', label="Start")
        ax.plot(df['x'].iloc[-1], df['y'].iloc[-1], color='b', marker='+', 
                markersize=7, linestyle='', label="End")
    
    if legend and mark_endpoints:
            plt.legend(["Start", "End"], loc='upper right')
            # ax = plt.gca() # get the current axes
            # PCM = ax.get_children()[2] # get the mappable, the 1st and the 2nd are the x and y axes
            # _legend = plt.colorbar(pad=0.01)
            # _legend.ax.get_yaxis().labelpad = 10 # legend spacing
            # _legend.ax.set_ylabel('Frame Number', rotation=270, size=7) # legend label
            # _legend.ax.tick_params(labelsize=5)

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)    
    ax.autoscale(enable=True, axis='x', tight=True) # re-scaling axes
    ax.autoscale(enable=True, axis='y', tight=True)
    
    return

def get_video_set(featurefilepath):
    """ Get the set of filenames of the featuresN results files that belong to
        the same 96-well plate that was imaged under that rig """
        
    dirpath = Path(featurefilepath).parent
    maskedfilepath = Path(str(dirpath).replace("Results/","MaskedVideos/"))
    
    # get camera serial from filename
    camera_serial = parse_camera_serial(featurefilepath)
    
    # get list of camera serials for that hydra rig
    hydra_rig = CAM2CH_df.loc[CAM2CH_df['camera_serial']==camera_serial,'rig']
    rig_df = CAM2CH_df[CAM2CH_df['rig']==hydra_rig.values[0]]
    camera_serial_list = list(rig_df['camera_serial'])
   
    # extract filename stem 
    file_stem = str(maskedfilepath).split('.' + camera_serial)[0]
    
    file_dict = {}
    for camera_serial in camera_serial_list:
        channel = serial2channel(camera_serial)
        _loc, rotate = CH2PLATE_dict[channel]
        
        # get path to masked video file
        maskedfilepath = Path(file_stem + '.' + camera_serial) / "metadata.hdf5"
        featurefilepath = Path(str(maskedfilepath.parent).replace("MaskedVideos/",\
                               "Results/")) / 'metadata_featuresN.hdf5'
        
        file_dict[channel] = (maskedfilepath, featurefilepath)
        
    return file_dict
    
def plot_plate_trajectories(featurefilepath, 
                            saveDir=None, 
                            downsample=10,
                            filter_trajectories=False,
                            mark_endpoints=False,
                            del_if_exists=False):
    """ Tile plots and merge into a single plot for the 
        entire 96-well plate, correcting for camera orientation. """

    file_dict = get_video_set(featurefilepath)
    
    # define multi-panel figure
    columns = 3
    rows = 2
    x = 25.5
    y = 16
    plt.ioff() if saveDir else plt.ion()
    plt.close('all')
    fig, axs = plt.subplots(rows,columns,figsize=[x,y])
    
    x_offset = 1.5 / x  # for bottom left image
    width = 0.3137      # for all but top left image
    width_tl = 0.3725   # for top left image
    height = 0.5        # for all images

    errlog = []    
    for channel, (maskedfilepath, featurefilepath) in file_dict.items():

        if saveDir:
            saveName = maskedfilepath.parent.stem + ('_filtered.png' if filter_trajectories else '.png')
            savePath = Path(saveDir) / saveName
            if savePath.exists():
                if del_if_exists:
                    os.remove(savePath)
                else:
                    print("Skipping file '%s' (already exists)" % savePath.name)
                    continue
        
        _loc, rotate = CH2PLATE_dict[channel]
        _ri, _ci = _loc

        # create bbox for image layout in figure
        if (_ri == 0) and (_ci == 0):
            # first image (with well names), bbox slightly shifted
            bbox = [0, height, width_tl, height]
        else:
            # other images
            bbox = [x_offset + width * _ci, height * (rows - (_ri + 1)), width, height]   
        
        # get location of subplot for camera
        ax = axs[_loc]
        
        try:
            # plot first frame of video + annotate wells
            FOVsplitter = FOVMultiWellsSplitter(maskedfilepath)
            FOVsplitter.plot_wells(is_rotate180=rotate, ax=ax, line_thickness=10)
            
            # plot worm trajectories
            plot_trajectory(featurefilepath, 
                            downsample=downsample,
                            filter_trajectories=filter_trajectories,
                            mark_endpoints=mark_endpoints,
                            rotate=rotate,
                            img_shape=FOVsplitter.img_shape,
                            legend=False, 
                            ax=ax)
        except Exception as e:
            print("WARNING: Could not plot video file: '%s'\n%s" % (maskedfilepath, e))
            errlog.append(maskedfilepath)
        
        # set image position in figure
        ax.set_position(bbox)
    
    if saveDir:
        if savePath.exists():
            print("Skipping file '%s' (already exists)" % savePath.name)
        else:
            Path(saveDir).mkdir(exist_ok=True, parents=True)
            fig.savefig(savePath,
                        bbox_inches='tight',
                        dpi=300,
                        pad_inches=0,
                        transparent=True)
    
    print(errlog) # TODO: Save errlog to file?
    return

def plot_plate_trajectories_from_filenames_summary(filenames_path, 
                                                   saveDir=None, 
                                                   downsample=10,
                                                   filter_trajectories=False,
                                                   mark_endpoints=False,
                                                   del_if_exists=False):
    """ Plot plate trajectories for all files in Tierpsy filenames summaries
        'filenames_path', and save results to 'saveDir' """

    filenames_df = pd.read_csv(filenames_path, comment='#')
    try:
        filenames_list = filenames_df[filenames_df['is_good']==True]['filename']
    except:
        filenames_list = filenames_df[filenames_df['is_good']==True]['file_name']        
    
    filestem_list = []
    featurefile_list = []  
    for fname in filenames_list:
        # obtain file stem
        filestem = Path(fname).parent.parent / Path(fname).parent.stem
        
        # only record featuresN filepaths with a different file stem as we only 
        # need 1 camera's video per plate to find the others
        if filestem not in filestem_list:
            filestem_list.append(filestem)
            featurefile_list.append(fname)
    
    # overlay trajectories and combine plots for each plate that was imaged
    for featurefilepath in tqdm(featurefile_list):
        print("\nPlotting plate trajectories for: %s" % Path(featurefilepath).parent.name)
        plot_plate_trajectories(featurefilepath, 
                                saveDir=saveDir, 
                                downsample=downsample,
                                filter_trajectories=filter_trajectories,
                                mark_endpoints=mark_endpoints,
                                del_if_exists=del_if_exists)

    return
        
#%% Main
    
if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_file", help="Input a single featuresN HDF5 \
                        filepath to plot all trajectories for that plate", 
                        default=EXAMPLE_FILE)
        
    parser.add_argument("--full_filenames", help="Input a full_filenames.csv \
                        filepath to plot plate trajectories for all videos", 
                        default=None)
        
    parser.add_argument("--save_dir", help="Path to directory to save plate \
                        trajectories", default=None)
                        
    parser.add_argument("--downsample", help="Downsample trajectory data by \
                        plotting the worm centroid for every 'nth' frame",
                        default=20)
        
    parser.add_argument("--filter_trajectories", help="Filter trsjectory data by global threshold \
                        parameters for movement and duration", default=False)
                        
    parser.add_argument("--mark_endpoints", help="Show trajectory start and end points on plot",
                        default=False)
    
    parser.add_argument("--annotate_lawns", help="Plot polygon outlining bacterial lawns from \
                        saved food coordinates", default=False)
                        
    parser.add_argument("--del_if_exists", help="Overwrite plate trajectory plots that have \
                        already been saved?", default=False)             
    args = parser.parse_args()
    
    FEAT_FILE_PATH = Path(args.features_file)
    FULL_FILES_PATH = Path(args.full_filenames) if args.full_filenames is not None else None
    SAVE_DIR = Path(args.save_dir) if args.save_dir else None
    
    plt.close('all')
    plt.ioff() if SAVE_DIR is not None else plt.ion()
    
    if FULL_FILES_PATH is not None:
        print("Plotting plate trajectories from full filenames summaries:\n\t%s\n" %FULL_FILES_PATH)
        plot_plate_trajectories_from_filenames_summary(FULL_FILES_PATH, 
                                                       saveDir=SAVE_DIR, 
                                                       downsample=int(args.downsample),
                                                       filter_trajectories=args.filter_trajectories,
                                                       mark_endpoints=args.mark_endpoints,
                                                       del_if_exists=args.del_if_exists)
    elif FEAT_FILE_PATH is not None:
        print("\nPlotting plate trajectories for:\n\t%s\n" % str(FEAT_FILE_PATH))
        plot_plate_trajectories(FEAT_FILE_PATH, 
                                saveDir=SAVE_DIR, 
                                downsample=int(args.downsample),
                                filter_trajectories=args.filter_trajectories,
                                mark_endpoints=args.mark_endpoints)
    else:
        print("\nNo file path provided!")
  