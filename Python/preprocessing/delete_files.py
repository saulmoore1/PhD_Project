#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script to delete files containing a given string recursively within a given folder

@author: sm5911
@date: 20/04/2022

"""

import os
import argparse
from tqdm import tqdm
from pathlib import Path


FOLDER = "/Volumes/hermes$/Keio_Worm_Stress_Mutants/MaskedVideos/20220730"
STRING = "000000"


def delete_files(folder_path, string):
    """ Recursively search parent folder for files containing given string in their filename and 
        ask to confirm before deleting them 
    """
    
    print("Finding files containing %s in their filename..." % string)
    files = list(folder_path.rglob('*'+args.string+'*'))
    print(files)
    print("Found %d files containing '%s' in their filename" % (len(files), string))
    print("\nWould you like to delete these files?")
    input("Press Enter to continue...")
    print("\nDeleting %d files...\n" % len(files))
    
    for f in tqdm(files):
        os.remove(f)
        
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--folder', help="Path to parent folder containing files to delete", 
                        type=str, default=FOLDER)
    parser.add_argument('-s','--string', help="String that files to delete contain in their filename",
                        type=str, default=STRING)
    args = parser.parse_args()
      
    delete_files(Path(args.folder), args.string)

