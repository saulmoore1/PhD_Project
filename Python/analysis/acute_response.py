#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse fast-effect (acute response) videos
- window feature summaries for Ziwei's optimal windows around each bluelight stimulus
- Bluelight delivered for 10 seconds every 30 minutes, for a total of 5 hours

When do we start to see an effect on worm behaviour? At which timepoint/window? 
Do we still see arousal of worms on siderophore mutants, even after a short period of time?

@author: sm5911
@date: 24/11/2021

"""

#%% Imports

import argparse
import pandas as pd
from pathlib import Path
from read_data.read import load_json

#%% Globals

JSON_PARAMETERS_PATH = 'analysis/20211102_parameters_keio_fast_effect.json'

FEATURE = 'motion_mode_paused_fraction'

# windows summary window number to corresponding frame number mapping dictionary
WINDOW_FRAME_DICT = {2:(1805,1815), 
                     5:(3605,3615), 
                     8:(5405,5415), 
                     11:(7205,7215),
                     14:(9005,9015), 
                     17:(10805,10815), 
                     20:(12605,12615), 
                     23:(14405,14415), 
                     26:(16205,16215)}

#%% Functions


def analyse_fast_effect(compiled_filenames, compiled_features):
    """ """
    
    return

#%% Main

if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description="Analyse acute response videos to investigate how \
    fast the food takes to influence worm behaviour")
    parser.add_argument('-j','--json', help="Path to JSON parameters file", default=JSON_PARAMETERS_PATH)
    args = parser.parse_args()
    args = load_json(args.json)

    aux_dir = Path(args.project_dir) / 'AuxiliaryFiles'
    results_dir =  Path(args.project_dir) / 'Results'
    
    # load compiled window summaries results
    compiled_filenames = pd.read_csv(results_dir / 'compiled_filenames_summaries.csv', index_col=False)
    compiled_features = pd.read_csv(results_dir / 'compiled_features_summaries.csv', index_col=False)

    # subset summaries for specific windows
    windows_list = list(WINDOW_FRAME_DICT.keys())
    compiled_filenames = compiled_filenames[compiled_filenames['window'].isin(windows_list)]
    compiled_features = compiled_features[compiled_features['window'].isin(windows_list)]   
    
    # load metadata (for gene name info)
    metadata = pd.read_csv(aux_dir / 'metadata.csv')

    analyse_fast_effect(compiled_filenames, compiled_features)