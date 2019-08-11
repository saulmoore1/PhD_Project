#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULE: SAVE

@author: sm5911
@date: 24/04/2019

"""

# IMPORTS
import os
from matplotlib import pyplot as plt

#%% FUNCTIONS
def savefig(Path, tight_layout=True, tellme=True, saveFormat='eps', **kwargs):
    """ Helper function for easy plot saving. A simple wrapper for plt.savefig """
    if tellme:
        print("Saving figure:", os.path.basename(Path))
    if tight_layout:
        plt.tight_layout()
    if saveFormat == 'eps':
        plt.savefig(Path, format=saveFormat, dpi=600, **kwargs)
    else:
        plt.savefig(Path, format=saveFormat, dpi=600, **kwargs)
    if tellme:
        print("Done.")

#%%