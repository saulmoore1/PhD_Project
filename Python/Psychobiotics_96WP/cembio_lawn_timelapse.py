#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Investigating CeMbio lawn growth rate 
- Make timelapse video from many small videos

@author: sm5911
@date: 23/10/2020

"""

#%% Imports 

import sys
import cv2
import re
import numpy as np
from tqdm import tqdm
from pathlib import Path
import moviepy.video.io.ImageSequenceClip
from matplotlib import pyplot as plt

# Path to Tierpsy Github functions
PATH_LIST = ['/Users/sm5911/Tierpsy_Versions/tierpsy-tracker',
             '/Users/sm5911/Documents/GitHub/PhD_Project/Python/Psychobiotics_96WP']
for sysPath in PATH_LIST:
    if sysPath not in sys.path:
        sys.path.insert(0, sysPath)

from tierpsy.analysis.compress.selectVideoReader import selectVideoReader
from plot_plate_trajectories_with_raw_video_background import get_video_set
from tierpsy.analysis.split_fov.FOVMultiWellsSplitter import FOVMultiWellsSplitter
from tierpsy.analysis.split_fov.helper import serial2channel, parse_camera_serial

#%% Globals

RAWVIDEO_DIR = Path("/Volumes/behavgenom$/Saul/CeMbioScreen/RawVideos")
EXP_DATES = ['20201022','20201023']

SAVE_DIR = Path("/Users/sm5911/Desktop/CeMbio_lawn_timelapse") # "/Volumes/behavgenom$/Saul/CeMbioScreen/Results/20201022/"
video_name = "CeMbio_lawn_growth_timelapse"

CH2PLATE_dict = {'Ch1':((0,0),True),
                 'Ch2':((1,0),False),
                 'Ch3':((0,1),True),
                 'Ch4':((1,1),False),
                 'Ch5':((0,2),True),
                 'Ch6':((1,2),False)}

#%% Functions

def write_list_to_file(list_to_save, save_path):
    """ Write a list to text file """
    
    with open(str(save_path), 'w') as fid:
        for line in list_to_save:
            fid.write("%s\n" % line)
                    
def read_list_from_file(filepath):
    """ Read a multi-line list from text file """   
    
    list_from_file = []
    with open(filepath, 'r') as fid:
        for line in fid:
            info = line[:-1]
            list_from_file.append(info)
    
    return list_from_file      

def get_video_list(RAWVIDEO_DIR, EXP_DATES=None, video_list_save_path=None):
    """ Search directory for 'metadata.yaml' video files and return as a list """
    
    if not EXP_DATES:
        video_list = list(RAWVIDEO_DIR.rglob("*metadata.yaml"))
    else:
        video_list = []
        for date in EXP_DATES:
            vid_date_dir = RAWVIDEO_DIR / date
            vids = list(vid_date_dir.rglob("*metadata.yaml"))
            video_list.extend(vids)
    
    if video_list_save_path:
        write_list_to_file(list_to_save=video_list, save_path=video_list_save_path)

    return video_list

def average_frame_yaml(metadata_yaml_path):
    """ Return the average of the frames in a given 'metadata.yaml' video """
    
    vid = selectVideoReader(str(metadata_yaml_path))
    frames = vid.read()
     
    avg_frame = np.mean(frames, axis=0)
        
    return avg_frame

def save_avg_frames_for_timelapse(video_list, SAVE_DIR):
    """ Take the average frame from each video and save to file """
    
    print('\nSaving average frame in %d videos' % len(video_list))
    for i, metadata_yaml_path in tqdm(enumerate(video_list)):
        
        metadata_yaml_path = Path(metadata_yaml_path)
        fstem = metadata_yaml_path.parent.name
        fname = fstem.replace('.','_') + '.tif'
        
        savepath = SAVE_DIR / fname
        savepath.parent.mkdir(exist_ok=True)
        
        if not savepath.exists():         
            avg_frame = average_frame_yaml(metadata_yaml_path) 
            cv2.imwrite(str(savepath), avg_frame)

def parse_frame_serial(filepath):
    """ Regex search of filestem for 4-digit number separated by underscores 
        denoting frame index """

    fname = str(Path(filepath).name)    
    regex = r"(\d{4})(?=\_\d{8}\_\d{6}$)" # \.\d{8}
    frame_idx = re.findall(regex, str(fname).lower())[0]
    
    return frame_idx

def match_plate_frame_filenames(raw_video_path_list):
    """ For each video frame timestamp, pair the video filenames for that 
        frame and return dictionary of filenames for each plate/frame"""
    
    video_list_no_camera_serial = []
    for fname in raw_video_path_list:
        
        # get camera serial from filename
        camera_serial = parse_camera_serial(fname)
        
        # append filestem to video list (no serial)
        fstem = str(fname).replace(('.' + camera_serial + '/metadata.yaml'),'')
        video_list_no_camera_serial.append(Path(fstem).name)
    
    hydra_timestamp_list = np.unique(video_list_no_camera_serial)
    
    plate_frame_filename_dict = {}
    for fname in hydra_timestamp_list:
        hydra_camera_video_set = [f for f in raw_video_path_list if fname in f]
        
        frame_idx = parse_frame_serial(fname)
        plate_frame_filename_dict[frame_idx] = hydra_camera_video_set
                
    #frames = list(plate_frame_filename_dict.keys())
    
    return plate_frame_filename_dict

def plate_frame_from_camera_frames(filepath, SAVE_DIR):
    """ Compile plate view by tiling images from each camera for a given frame
        and merging into a single plot of the entire 96-well plate, correcting 
        for camera orientation. """
        
    featurefilepath = '/Volumes/behavgenom$/Saul/MicrobiomeScreen96WP/Results/20200924/microbiome_run1_bluelight_20200924_140215.22956811/metadata_featuresN.hdf5'
    file_dict = get_video_set(featurefilepath)
    
    # define multi-panel figure
    columns = 3
    rows = 2
    h_in = 6
    x_off_abs = (3600-3036) / 3036 * h_in
    x = columns * h_in + x_off_abs
    y = rows * h_in
    fig, axs = plt.subplots(rows,columns,figsize=[x,y])

    x_offset = x_off_abs / x  # for bottom left image
    width = (1-x_offset) / columns  # for all but top left image
    width_tl = width + x_offset   # for top left image
    height = 1/rows        # for all images
    
    plt.ioff()
    for channel, (maskedfilepath, featurefilepath) in file_dict.items():
        
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
        
        # plot first frame of video + annotate wells
        FOVsplitter = FOVMultiWellsSplitter(maskedfilepath)
        FOVsplitter.plot_wells(is_rotate180=rotate, ax=ax, line_thickness=10)
        
        # set image position in figure
        ax.set_position(bbox)
        
    if SAVE_DIR:
        saveName = maskedfilepath.parent.stem + '.png'
        savePath = Path(SAVE_DIR) / saveName
        if not savePath.is_file():
            fig.savefig(savePath,
                        bbox_inches='tight',
                        dpi=300,
                        pad_inches=0,
                        transparent=True)   
    return(fig)      
   


def make_video_from_frames(IMAGES_DIR, video_name, fps):
    """ Create a video from the images (frames) in a given directory """
    
    image_list = list(IMAGES_DIR.rglob("*frame*.tif"))
    
    image0 = cv2.imread(str(image_list[0]))
    height, width, layers = image0.shape
    
    outpath_video = IMAGES_DIR / "{}.mp4".format(video_name)

    video = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_list, 
                                                                 fps=fps)
    video.write_videofile(outpath_video)
    
    # video = cv2.VideoWriter(outpath_video, 0, 1, (width, height))   
    # for image in image_path_list:
    #     video.write(cv2.imread(str(image))) 
    # cv2.destroyAllWindows()
    # video.release()
    
    
#%% Main

if __name__ == "__main__":
    
    video_list_save_path = SAVE_DIR / "cembio_lawn_video_list.txt"
    
    if not Path(video_list_save_path).exists():
        # get video list
        video_list = get_video_list(RAWVIDEO_DIR, EXP_DATES, video_list_save_path)
    else: 
        # read video list
        video_list = read_list_from_file(filepath=video_list_save_path)
    
    save_avg_frames_for_timelapse(video_list, SAVE_DIR)
    
    plate_frame_filename_dict = match_plate_frame_filenames(video_list)
    
    for (frame_idx, rig_video_list) in plate_frame_filename_dict.items():
        print(frame_idx, len(rig_video_list))
        
        filenames = plate_frame_filename_dict[frame_idx]
        #plate_frame_from_camera_frames(frame_idx, SAVE_DIR)
    
    #make_video_from_frames(SAVE_DIR, video_name, fps=25)
