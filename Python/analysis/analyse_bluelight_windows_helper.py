#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tierpsy window summaries helper

@author: sm5911
@DATE: 22/11/2021

"""

import argparse

BLUELIGHT_TIMEPOINTS = [5,10,15,20,25,30] #[30,60,90,120,150,180,210,240,270]


def optimal_window_ziwei_seconds(x):
    """ x (in minutes) converted to window (in seconds) """
    return print("%d:%d, %d:%d, %d:%d" % (x*60-10,x*60,x*60+5,x*60+15,x*60+15,x*60+25)) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate optimal bluelight window timepoints \
                                     (minutes) based on 'Ziwei's optimal windows'")
    parser.add_argument('-l', '--bluelight_timepoints', default=BLUELIGHT_TIMEPOINTS, 
                        help='List of timepoints (minutes) when 10s bluelight stimulus was delivered', 
                        nargs='+', type=int)
    args = parser.parse_args()
    
    for i in args.bluelight_timepoints:
        optimal_window_ziwei_seconds(i)