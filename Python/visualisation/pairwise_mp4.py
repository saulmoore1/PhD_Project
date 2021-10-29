#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Side-by-side RawVideo mp4 of sampled wells for strain vs control

@author: sm5911
@date: 29/10/2021

"""

#%% Imports 

import sys
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from pathlib import Path

sys.path.insert(0, "/Users/sm5911/Tierpsy_Versions/tierpsy-tracker") # path to tierpsy tracker repo
from tierpsy.helper.params.read_attrs import read_fps
from read_data.read_well import well_reader

#%% Globals

METADATA_PATH = '/Users/sm5911/Documents/Keio_Conf_Screen/metadata.csv'
FEATURES_PATH = '/Users/sm5911/Documents/Keio_Conf_Screen/features.csv'
SAVE_DIR = '/Users/sm5911/Documents/Keio_Conf_Screen/pairwise_mp4'

STRAIN_LIST = ['fepB','fepD','fes','atpB','nuoC','sdhD','entA'] # missing: 'trpA','trpD'
MAX_N_PAIRS = 5

CODEC = 'avc1' #'H264','MPEG','mp4v'

#%% Functions

def plot_pairwise_mp4(metadata, strain_colname, strain, control, saveDir, downsample=10, 
                      stim_type='bluelight', max_n_pairs=5, del_if_exists=False, verbose=True):
    """ Plot pairwise trajectory plots of strain vs control """
    
    assert strain in metadata[strain_colname].unique() and control in metadata[strain_colname].unique()
    
    # subset for strain and control
    strain_meta = metadata[metadata[strain_colname]==strain]
    control_meta = metadata[metadata[strain_colname]==control]
    
    assert strain_meta.shape[0] == len(strain_meta['featuresN_filename'].unique())

    # subset for bluelight condition
    assert stim_type in ['prestim','bluelight','poststim']
            
    # for each strain well, pair with random control well of the same plate
    strain_info = []
    control_info = []
    for i in strain_meta.index:
        s_imgstore = strain_meta.loc[i, 'imgstore_name_{}'.format(stim_type)]
        s_well = strain_meta.loc[i, 'well_name']

        # get path to strain featuresN results for a given stimulus type
        s_file = strain_meta.loc[i, 'featuresN_filename']
        _day = s_imgstore.split('/')[0]
        _res_dir = s_file.split(_day)[0]
        s_file = Path(_res_dir) / s_imgstore / 'metadata_featuresN.hdf5'
        assert s_file.exists()
        
        matched_control = control_meta[np.logical_and(
            control_meta['date_yyyymmdd'] == strain_meta.loc[i, 'date_yyyymmdd'],
            control_meta['imaging_plate_id'] == strain_meta.loc[i, 'imaging_plate_id'])]
        
        assert matched_control.shape[0] != 0
            
        # grab random control well data
        rand_idx = np.random.choice(matched_control.index, size=1, replace=True)[0]
        c_imgstore = matched_control.loc[rand_idx, 'imgstore_name_{}'.format(stim_type)]
        c_well = matched_control.loc[rand_idx, 'well_name']   
        
        c_file = Path(_res_dir) / c_imgstore / 'metadata_featuresN.hdf5'
        assert c_file.exists()
        
        strain_info.append([s_file, s_well])
        control_info.append([c_file, c_well])
   
    if max_n_pairs is not None:
        strain_info = strain_info[:max_n_pairs]
        control_info = control_info[:max_n_pairs]
        
    for i, (s, c) in enumerate(tqdm(zip(strain_info, control_info), total=len(strain_info))):
        s_file, s_well = s
        c_file, c_well = c
        
        # store time
        tic = time()
        
        # read original video frame rate
        fps = read_fps(s_file)

        # create the generator for the sample + control
        s_well_img_gen = well_reader(s_file, s_well)
        c_well_img_gen = well_reader(c_file, c_well)
        
                
        for i, (s_img, c_img) in enumerate(zip(s_well_img_gen, c_well_img_gen)):
            
            # create black frame with width = sum of two widths, height = max of two heights
            height = max(s_img.shape[0], c_img.shape[0])
            width = s_img.shape[1] + c_img.shape[1]
            img = np.zeros(shape=(height, width), dtype=np.uint8)
 
            # set the sample frame as the left image, control frame as the right image
            img[0:s_img.shape[0], 0:s_img.shape[1]] = s_img
            img[0:c_img.shape[0], s_img.shape[1]:] = c_img 
            
            if i == 0:
                # open video writer
                Path(saveDir).mkdir(parents=True, exist_ok=True)
                videoname = Path(saveDir) / (saveDir.name + '_(left)_vs_control_(right)_{}.mp4'.format(i + 1))
                vid_writer = cv2.VideoWriter(str(videoname), cv2.VideoWriter_fourcc(*CODEC), 
                                             fps/downsample, (img.shape[1], img.shape[0]), 
                                             isColor=False)
                assert vid_writer.isOpened()

            # write out with video writer
            vid_writer.write(img)
        
        # close video writer
        vid_writer.release()
        print("Done in %.1f seconds" % (time() - tic))
        
#%% Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot pairwise trajectories for each strain vs \
                                     control (aligned bluelight only)")
    parser.add_argument('--metadata_path', help="Path to metadata file containing columns: \
                        ['featuresN_filename','well_name','date_yyyymmdd','imaging_plate_id']",
                        default=METADATA_PATH)
    parser.add_argument('--features_path', help="Path to compiled features summaries file",
                        default=FEATURES_PATH)
    parser.add_argument('--control', help="Control strain to compare with", default='wild_type')
    parser.add_argument('--strain_list', help="List of strains to compare against control", 
                        nargs='+', default=STRAIN_LIST)
    parser.add_argument('-strain_colname', help="Column name containing strain and control variables",
                        default='gene_name')
    parser.add_argument('--save_dir', help="Directory path to save pairwise trajectory plots",
                        default=SAVE_DIR)
    parser.add_argument('--downsample', help="Downsample video by selecting only every nth frame",
                        default=10)
    parser.add_argument('--filter_trajectories', help="Filter trajectories by global distance and \
                        duration parameters", default=False)
    args = parser.parse_args()
        
    # load metadata and features
    metadata = pd.read_csv(args.metadata_path, dtype={"comments":str})
    features = pd.read_csv(args.features_path)
    
    assert all(s in metadata[args.strain_colname].unique() for s in args.strain_list)
    
    # plot pairwise trajectory plots
    for strain in args.strain_list:
        print("Plotting pairwise mp4 videos for '%s' vs '%s'" % (strain, args.control))
        plot_pairwise_mp4(metadata,
                          strain=strain,
                          strain_colname=args.strain_colname,
                          control=args.control,
                          stim_type='bluelight',
                          downsample=args.downsample,
                          max_n_pairs=MAX_N_PAIRS,
                          saveDir=Path(args.save_dir) / strain,
                          del_if_exists=False,
                          verbose=True)
        
