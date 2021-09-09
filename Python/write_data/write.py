#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Write data to file

@author: sm5911
@date: 03/03/2021

"""

from pathlib import Path

def write_list_to_file(list_to_save, save_path):
    """ Write a list to text file """
    
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    
    with open(str(save_path), 'w') as fid:
        for line in list_to_save:
            fid.write("%s\n" % line)
