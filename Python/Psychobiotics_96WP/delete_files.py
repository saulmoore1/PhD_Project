#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 18:09:26 2020

@author: sm5911
"""


import os
from helper import lookforfiles

path = ""

files2delete = lookforfiles(path, "")

for file in files2delete:
    os.remove(file)
    print("Removed: %s" % file)
