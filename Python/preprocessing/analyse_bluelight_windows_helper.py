#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tierpsy window summaries helper

@author: sm5911
@DATE: 22/11/2021

"""

import argparse

BLUELIGHT_TIMEPOINTS = [5,10,15,20,25,30,35,40]

#[30,60,90,120,150,180,210,240,270]
#[5,10,15,20,25]
#[5,10,15,20,25,30,35,40]
#[30,31,32]

def optimal_window_ziwei_seconds(x):
    """ x (in minutes) converted to window (in seconds) """
    return print("%d:%d, %d:%d, %d:%d" % (x*60-10,x*60,x*60+5,x*60+15,x*60+15,x*60+25)) 

def keio_window_30seconds(x):
    """ x (in minutes) converted to window (in seconds) """
    return print("%d:%d" % (x*60+30,x*60+60))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate optimal bluelight window timepoints \
                                     (seconds) based on 'Ziwei's optimal windows'")
    parser.add_argument('-l', '--bluelight_timepoints', default=BLUELIGHT_TIMEPOINTS, 
                        help='List of timepoints (minutes) when 10s bluelight stimulus was delivered', 
                        nargs='+', type=int)
    args = parser.parse_args()
    
        
    for i in args.bluelight_timepoints:
        #optimal_window_ziwei_seconds(i)
        keio_window_30seconds(i)
        
        
#[30,60,90,120,150,180,210,240,270] --> 1830:1860, 3630:3660, 5430:5460, 7230:7260, 9030:9060, 10830:10860, 12630:12660, 14430:14460, 16230:16260
#[5,10,15,20,25] --> 330:360, 630:660, 930:960, 1230:1260, 1530:1560
#[5,10,15,20,25,30,35,40] --> 330:360, 630:660, 930:960, 1230:1260, 1530:1560, 1830:1860, 2130:2160, 2430:2460
#[30,31,32] --> 1830:1860, 1890:1920, 1950:1980
