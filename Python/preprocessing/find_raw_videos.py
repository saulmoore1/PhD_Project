#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find raw videos from metadata

@author: sm5911
@date: 19/07/2023

"""

#%% Imports 

import shutil
import numpy as np
from time import time
from tqdm import tqdm
from pathlib import Path


#%% Globals

SEED = 101
np.random.seed(SEED)

#%% Functions

def find_raw_videos_from_metadata(metadata, 
                                  treatment_col='treatment', 
                                  treatment_list=['BW','fepD'], 
                                  n_videos=1):
    
    _treatment_list = metadata[treatment_col].unique()
    assert all(i in _treatment_list for i in treatment_list)
    
    video_dict = {}
    grouped = metadata.groupby(treatment_col)
    for treatment in treatment_list:
        meta_treatment = grouped.get_group(treatment)
        video_filenames = np.random.choice(meta_treatment['filename'], size=n_videos, replace=False)
        video_dict[treatment] = [i + '/000000.mp4' for i in sorted(video_filenames)]
    
    return video_dict

def copy_raw_videos_to_folder(video_dict, destination_folder):
    
    tic = time()
    for treatment in tqdm(video_dict.keys()):

        save_dir = Path(destination_folder) / treatment
        files_to_copy = video_dict[treatment]
        
        for file in tqdm(files_to_copy):
            filename = Path(Path(file).parent.name) / Path(file).name
            save_path = save_dir / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # copy file
            if not save_path.exists():
                shutil.copy2(src=file, dst=save_path, follow_symlinks=True)
                
    toc = time()
    print('Done in %.1f minutes' % ((toc-tic) / 60))
    
    return
    

