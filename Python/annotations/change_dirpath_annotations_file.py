#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:19:52 2021

@author: sm5911
"""

wells_annotations_file = "/Volumes/hermes$/KeioScreen_96WP/KeioScreen_96WP_20210208_155647_wells_annotations.hdf5"
new_dirpath = "/Volumes/hermes$/Test_dir"

def change_annotations_dirpath(wells_annotations_file, new_masked_video_dirpath):
    """ Change working directory attribute in annotations file to new working directory
        
        Parameters
        ----------
        wells_annotations_file : str, pathlib.Path
        new_dirpath : str, pathlib.Path
    
    """
    
    import h5py

    with h5py.File(str(wells_annotations_file), 'r+') as fid:
        print("Changing MaskedVideo dirpath from\n'%s' to\n'%s'"
              % (fid["/filenames_df"].attrs["working_dir"], str(new_dirpath)))
        
        fid["/filenames_df"].attrs["working_dir"] = str(new_dirpath)
        