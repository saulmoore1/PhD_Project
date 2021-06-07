#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:19:52 2021

@author: sm5911
"""

wells_annotations_file = "/Volumes/hermes$/KeioScreen_96WP/AuxiliaryFiles/20210420/KeioScreen_96WP_20210605_012418_wells_annotations.hdf5"
new_masked_video_dirpath = "/Volumes/hermes$/KeioScreen_96WP/MaskedVideos/20210420"

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
              % (fid["/filenames_df"].attrs["working_dir"], str(new_masked_video_dirpath)))
        
        fid["/filenames_df"].attrs["working_dir"] = str(new_masked_video_dirpath)

    return

def create_annotations_file_from_csv(wells_annotations_csv_path, maskedvideo_root=None, saveto=None):
    """ Create HDF5 dataset for annotations from CSV """
    
    import h5py
    import numpy as np
    import pandas as pd
    from pathlib import Path
    
    wells_annotations_csv_path = "/Volumes/behavgenom$/Saul/KeioScreen_96WP/AuxiliaryFiles/20210420/Keio Library_20210605_012418_wells_annotations.csv"
    
    # read annotations csv
    annotations_df = pd.read_csv(wells_annotations_csv_path, header=0, index_col=None)
    
    # change filepaths of videos annotated
    if maskedvideo_root is not None:
        if not all(maskedvideo_root in i for i in annotations_df['filename']):
            annotations_df['filename'] = [str(i).replace(str(Path(i).parent.parent), maskedvideo_root) 
                                          for i in annotations_df['filename']]
            
    # make h5py dataset
    #annotations_df.create_dataset

    # f = h5py.File(str(wells_annotations_file), 'r+')
    # f.visit(print)
    
    # axis0 = np.array(f.get('filenames_df/axis0'))
    # axis1 = np.array(f.get('filenames_df/axis1'))
    # block0 = np.array(f.get('filenames_df/block0_items'))

    # file_id = f['filenames_df/axis0'][b'file_id']
    # axis0 = wells_annots['axis0']


if __name__ == "__main__":
    change_annotations_dirpath(wells_annotations_file=wells_annotations_file,
                               new_masked_video_dirpath=new_masked_video_dirpath)
        