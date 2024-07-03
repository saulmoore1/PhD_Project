#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse results of FepD_supplementation data (collected by Alan and Riju) in which Shikimic acid,
Gentisic acid and 2,3-dihydroxybenzoic acid (or none - control) was added to the following bacteria:
    - BW, fepD, entA, entE, fepD_entA, and fepD_entE

This script:
    - compiles project metadata and feature summaries
    - cleans the summary results
    - calculates statistics for speed_50th across treatment groups (t-tests and ANOVA)
    - plots box plots of speed_50th (all treatments together)
    - plots time-series of speed throughout bluelight video (each treatment overlayed with BW and fepD)

@author: Saul Moore (sm5911)
@date: 03/07/2024
    
"""

