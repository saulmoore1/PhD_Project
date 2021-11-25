#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tierpsy window summaries helper

@author: sm5911
@DATE: 22/11/2021

"""

def optimal_window_ziwei(x):
    return print("%d:%d, %d:%d, %d:%d" % (x*60-10,x*60,x*60+5,x*60+15,x*60+15,x*60+25)) 

if __name__ == "__main__":
    for i in [30,60,90,120,150,180,210,240,270]:
        optimal_window_ziwei(i)