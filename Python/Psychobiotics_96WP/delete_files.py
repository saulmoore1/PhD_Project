#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 18:09:26 2020

@author: sm5911
"""


import os
from helper import lookforfiles

dirpath = "/Users/sm5911/Documents/PanGenomeGFP/ilastik/ilastik_training_images/object_h5"

files2delete = lookforfiles(dirpath, "_table.csv")

for file in files2delete:
    os.remove(file)
    print("Removed: %s" % file)
