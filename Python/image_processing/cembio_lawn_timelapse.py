#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Investigating CeMbio lawn growth rate 
- Make timelapse video from many small videos

@author: sm5911
@date: 23/10/2020

"""

#%% Imports 

import cv2
import argparse
import re
import numpy as np
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt

from read_data.read import read_list_from_file
from write_data.write import write_list_to_file
from visualisation.plate_trajectories import CH2PLATE_dict

from tierpsy.analysis.compress.selectVideoReader import selectVideoReader
from tierpsy.analysis.split_fov.helper import CAM2CH_df, serial2channel, parse_camera_serial

#%% Globals

VIDEO_NAME = 'timelapse'

FPS_TIMELAPSE = 25 # frames per second for timelapse video

#%% Functions
                    
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
    
    video2frame_dict = {}

    print('\nSaving average frame in %d videos' % len(video_list))
    for i, metadata_yaml_path in tqdm(enumerate(video_list)):
        
        metadata_yaml_path = Path(metadata_yaml_path)
        fstem = metadata_yaml_path.parent.name
        fname = fstem.replace('.','_') + '.tif'
        
        savepath = Path(SAVE_DIR) / "average_frames" / fname
        savepath.parent.mkdir(exist_ok=True)
        
        if not savepath.exists():         
            avg_frame = average_frame_yaml(metadata_yaml_path) 
            cv2.imwrite(str(savepath), avg_frame)
        
        video2frame_dict[str(metadata_yaml_path)] = str(savepath)
    
    return video2frame_dict

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
    for rawvideopath in raw_video_path_list:   
        # get camera serial from filename
        camera_serial = parse_camera_serial(rawvideopath)
        
        # append filestem to video list (no serial)
        fstem = str(rawvideopath).replace(('.' + camera_serial + '/metadata.yaml'),'')
        video_list_no_camera_serial.append(Path(fstem).name)
        
    video_list_no_camera_serial = list(np.unique(video_list_no_camera_serial))
    
    plate_frame_filename_dict = {}
    for fname in video_list_no_camera_serial:
        rig_video_set = [f for f in raw_video_path_list if fname in f]
        
        frame_idx = parse_frame_serial(fname)
        plate_frame_filename_dict[frame_idx] = rig_video_set
    print("Matched rig camera filenames for %d 96-well plate frame view\n" % len(video_list_no_camera_serial))   
         
    return plate_frame_filename_dict

def convert_filepath_to_rawvideo(filepath):
    """ Helper function to convert featuresN filepath or MaskedVideo filepath 
        into RawVideo filepath """
    
    parentdir = str(Path(filepath).parent)
    
    dirlist = ["Results/", "MaskedVideos/"]
    if "RawVideos/" in parentdir:
        rawvideopath = filepath
        return rawvideopath
    else:
        for dirname in dirlist:
            if dirname in parentdir:
                # featuresN filepath
                rawvideopath = Path(str(parentdir).replace(dirname, "RawVideos/")) / 'metadata.yaml'
                return rawvideopath
            
def get_rig_video_set(filepath):
    """ Get the set of filenames of the featuresN results files that belong to
        the same 96-well plate that was imaged under that rig """
        
    rawvideopath = convert_filepath_to_rawvideo(filepath)
    
    # get camera serial from filename
    camera_serial = parse_camera_serial(rawvideopath)
    
    # get list of camera serials for that hydra rig
    hydra_rig = CAM2CH_df.loc[CAM2CH_df['camera_serial']==camera_serial,'rig']
    rig_df = CAM2CH_df[CAM2CH_df['rig']==hydra_rig.values[0]]
    camera_serial_list = list(rig_df['camera_serial'])
   
    # extract filename stem 
    file_stem = str(rawvideopath).split('.' + camera_serial)[0]
    
    file_dict = {}
    for camera_serial in camera_serial_list:
        channel = serial2channel(camera_serial)
        _loc, rotate = CH2PLATE_dict[channel]
        
        # get rawvideopath for camera serial
        rawvideopath =  Path(file_stem + '.' + camera_serial) / "metadata.yaml"
        file_dict[channel] = rawvideopath
        
    return file_dict

def plate_frames_from_camera_frames(plate_frame_filename_dict, video2frame_dict, saveDir):
    """ Compile plate view by tiling images from each camera for a given frame
        and merging into a single plot of the entire 96-well plate, correcting 
        for camera orientation. """
    
    for (frame_idx, rig_video_set) in tqdm(plate_frame_filename_dict.items()):
        
        file_dict = get_rig_video_set(rig_video_set[0]) # gives channels as well 
        assert sorted(file_dict.values()) == sorted([Path(i) for i in rig_video_set])
                
        # define multi-panel figure
        columns = 3
        rows = 2
        h_in = 4
        x_off_abs = (3600-3036) / 3036 * h_in
        x = columns * h_in + x_off_abs
        y = rows * h_in
        fig, axs = plt.subplots(rows,columns,figsize=[x,y])
    
        x_offset = x_off_abs / x  # for bottom left image
        width = (1-x_offset) / columns  # for all but top left image
        width_tl = width + x_offset   # for top left image
        height = 1/rows        # for all images
        
        plt.ioff()
        for channel, rawvideopath in file_dict.items():
            
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
            
            # read average frame for rawvideopath
            av_frame_path = video2frame_dict[str(rawvideopath)]
            img = cv2.imread(av_frame_path)
            
            # rotate image 180° to align camera FOV if necessary
            if rotate:
                img = np.rot90(img, 2)   
                        
            # plot image without axes/labels
            ax.imshow(img)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            # set image position in figure
            ax.set_position(bbox)
            
        if saveDir:
            saveName = rawvideopath.parent.stem + '.png'
            savePath = Path(saveDir) / "plate_frame_timelapse" / saveName
            savePath.parent.mkdir(exist_ok=True)
            if not savePath.exists():
                fig.savefig(savePath,
                            bbox_inches='tight',
                            dpi=300,
                            pad_inches=0,
                            transparent=True)

def make_video_from_frames(IMAGES_DIR, video_name, fps=25):
    """ Create a video from the images (frames) in a given directory """
    
    image_path_list = sorted(list(IMAGES_DIR.rglob("*.png")))
    
    image0 = cv2.imread(str(image_path_list[0]))
    height, width, layers = image0.shape
    
    outpath_video = IMAGES_DIR / "{}.mp4".format(video_name)

    # video = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_list, fps=fps)
    # video.write_videofile(outpath_video)
    
    video = cv2.VideoWriter(str(outpath_video), cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))   
    for imPath in tqdm(image_path_list):
        video.write(cv2.imread(str(imPath))) 
    cv2.destroyAllWindows()
    video.release()
    
    
#%% Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create timelapse video from a series of RawVideos')
    parser.add_argument('--rawvideo_dir', help='Path to RawVideo directory', default=None)    
    parser.add_argument('--dates', help='List of experiment dates to analyse', default=None)
    parser.add_argument('--save_dir', help='Path to save directory', default=None)
    args = parser.parse_args()
    
    if not args.rawvideo_dir:
        raise IOError("Please provide RawVideo directory")
    else:
        args.rawvideo_dir = Path(args.rawvideo_dir)
    if not args.save_dir:
        args.save_dir = Path(args.rawvideo_dir) / "timelapse_results"
    else:
        args.save_dir = Path(args.save_dir)
                    
    video_list_save_path = args.save_dir / "cembio_lawn_video_list.txt"

    if not Path(video_list_save_path).exists():
        # get video list
        video_list = get_video_list(args.rawvideo_dir, args.dates, video_list_save_path)
    else: 
        # read video list
        video_list = read_list_from_file(filepath=video_list_save_path)    
    print("%d videos found." % len(video_list))
    
    # save average frames for timelapse
    video2frame_dict = save_avg_frames_for_timelapse(video_list, args.save_dir)
        
    plate_frame_filename_dict = match_plate_frame_filenames(video_list)
    
    plate_frames_from_camera_frames(plate_frame_filename_dict, video2frame_dict, args.save_dir)

    timelapse_dir = args.save_dir / "plate_frame_timelapse"
    make_video_from_frames(timelapse_dir, video_name=VIDEO_NAME, fps=FPS_TIMELAPSE)
