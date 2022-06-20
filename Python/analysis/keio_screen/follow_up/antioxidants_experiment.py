#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keio Antioxidants - Experiments adding antioxidants to E. coli BW25113 (control) and fepD mutant
bacteria of the Keio Collection

@author: sm5911
@date: 18/05/2022

"""

#%% Imports

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from time_series.time_series_helper import get_strain_timeseries
from time_series.plot_timeseries import plot_timeseries_motion_mode
from analysis.keio_screen.check_keio_screen_worm_trajectories import check_tracked_objects

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Keio_Antioxidants_6WP"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Antioxidants"

BLUELIGHT_WINDOWS_ONLY_TS = True
BLUELIGHT_TIMEPOINTS_MINUTES = [30,31,32]
FPS = 25
VIDEO_LENGTH_SECONDS = 38*60
SMOOTH_WINDOW_SECONDS = 5

motion_modes = ['forwards']

#%% Functions

def antioxidants_timeseries(metadata, project_dir=PROJECT_DIR, save_dir=SAVE_DIR):
    """ Timeseries plots for worm motion mode on BW and fepD bacteria with and without the 
        addition of antioxidants: Trolox (vitamin E), trans-resveratrol, vitamin C and NAC
    """
    
    # Each treatment vs BW control
    metadata['imaging_plate_drug_conc'] = metadata['imaging_plate_drug_conc'].astype(str)
    metadata['treatment'] = metadata[['food_type','drug_type','imaging_plate_drug_conc','solvent']
                                     ].agg('-'.join, axis=1)  
    
    control_list = list(np.concatenate([np.array(['fepD-none-nan-H2O']),
                                        np.repeat('BW-none-nan-H2O', 12)]))
    treatment_list = ['fepD-none-nan-EtOH','fepD-none-nan-H2O','BW-none-nan-EtOH',
                      'BW-vitC-500.0-H2O','fepD-vitC-500.0-H2O',
                      'BW-NAC-500.0-H2O','BW-NAC-1000.0-H2O',
                      'fepD-NAC-500.0-H2O','fepD-NAC-1000.0-H2O',
                      'BW-trolox-500.0-EtOH','fepD-trolox-500.0-EtOH',
                      'BW-trans-resveratrol-500.0-EtOH','fepD-trans-resveratrol-500.0-EtOH']
    title_list = ['fepD: H2O vs EtOH','BW vs fepD','BW: H2O vs EtOH',
                  'BW vs BW + vitC','BW vs fepD + vitC',
                  'BW vs BW + NAC','BW vs BW + NAC',
                  'BW vs fepD + NAC','BW vs fepD + NAC',
                  'BW vs BW + trolox','BW vs fepD + trolox',
                  'BW vs BW + trans-resveratrol','BW vs fepD + trans-resveratrol']
    labs = [('fepD + H2O', 'fepD + EtOH'),('BW', 'fepD'),('BW + H2O', 'BW + EtOH'),
            ('BW', 'BW + vitC (500ug/mL)'),('BW', 'fepD + vitC (500ug/mL)'),
            ('BW', 'BW + NAC (500ug/mL)'),('BW', 'BW + NAC (1000ug/mL)'),
            ('BW', 'fepD + NAC (500ug/mL)'),('BW', 'fepD + NAC (1000ug/mL)'),
            ('BW', 'BW + trolox (500ug/mL in EtOH)'),('BW', 'fepD + trolox (500ug/mL in EtOH)'),
            ('BW', 'BW + trans-resveratrol (500ug/mL in EtOH)'),
            ('BW', 'fepD + trans-resveratrol (500ug/mL in EtOH)')]
    
    for control, treatment, title, lab in tqdm(zip(control_list, treatment_list, title_list, labs)):
        
        # get timeseries for control data
        control_ts = get_strain_timeseries(metadata[metadata['treatment']==control], 
                                           project_dir=project_dir, 
                                           strain=control,
                                           group_by='treatment',
                                           n_wells=6,
                                           save_dir=Path(save_dir) / 'Data' / control,
                                           verbose=False,
                                           return_error_log=False)
        
        # get timeseries for treatment data
        treatment_ts = get_strain_timeseries(metadata[metadata['treatment']==treatment], 
                                             project_dir=project_dir, 
                                             strain=treatment,
                                             group_by='treatment',
                                             n_wells=6,
                                             save_dir=Path(save_dir) / 'Data' / treatment,
                                             verbose=False)
 
        colour_dict = dict(zip([control, treatment], sns.color_palette("pastel", 2)))
        bluelight_frames = [(i*60*FPS, i*60*FPS+10*FPS) for i in BLUELIGHT_TIMEPOINTS_MINUTES]

        for mode in motion_modes:
                    
            print("Plotting timeseries '%s' fraction for '%s' vs '%s'..." %\
                  (mode, treatment, control))
    
            plt.close('all')
            fig, ax = plt.subplots(figsize=(15,5), dpi=200)
    
            ax = plot_timeseries_motion_mode(df=control_ts,
                                             window=SMOOTH_WINDOW_SECONDS*FPS,
                                             error=True,
                                             mode=mode,
                                             max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
                                             title=None,
                                             saveAs=None,
                                             ax=ax,
                                             bluelight_frames=bluelight_frames,
                                             colour=colour_dict[control],
                                             alpha=0.25)
            
            ax = plot_timeseries_motion_mode(df=treatment_ts,
                                             window=SMOOTH_WINDOW_SECONDS*FPS,
                                             error=True,
                                             mode=mode,
                                             max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
                                             title=None,
                                             saveAs=None,
                                             ax=ax,
                                             bluelight_frames=bluelight_frames,
                                             colour=colour_dict[treatment],
                                             alpha=0.25)
        
            xticks = np.linspace(0, VIDEO_LENGTH_SECONDS*FPS, int(VIDEO_LENGTH_SECONDS/60)+1)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
            ax.set_xlabel('Time (minutes)', fontsize=12, labelpad=10)
            ax.set_ylabel('Fraction {}'.format(mode), fontsize=12, labelpad=10)
            ax.set_title(title, fontsize=12, pad=10)
            ax.legend([lab[0], lab[1]], fontsize=12, frameon=False, loc='best')
    
            if BLUELIGHT_WINDOWS_ONLY_TS:
                ts_plot_dir = Path(save_dir) / 'Plots' / 'timeseries_bluelight' / treatment
                ax.set_xlim([min(BLUELIGHT_TIMEPOINTS_MINUTES)*60*FPS-60*FPS, 
                             max(BLUELIGHT_TIMEPOINTS_MINUTES)*60*FPS+70*FPS])
            else:   
                ts_plot_dir = Path(save_dir) / 'Plots' / 'timeseries' / treatment
    
            #plt.tight_layout()
            ts_plot_dir.mkdir(exist_ok=True, parents=True)
            save_path = ts_plot_dir / '{0}_{1}.pdf'.format(treatment, mode)
            print("Saving to: %s" % save_path)
            plt.savefig(save_path)  

    return


#%% Main
if __name__ == '__main__':
    
    aux_dir = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    
    metadata_path = aux_dir / 'metadata.csv'
    metadata = pd.read_csv(metadata_path, dtype={'comments':str})
    
    # plot timeseries for each treatment vs control
    antioxidants_timeseries(metadata, 
                            project_dir=Path(PROJECT_DIR),
                            save_dir=Path(SAVE_DIR))
    
    # # Check length/area of tracked objects - prop bad skeletons
    # results_df = check_tracked_objects(metadata, 
    #                                    length_minmax=(200, 2000), 
    #                                    width_minmax=(20, 500),
    #                                    save_to=Path(SAVE_DIR) / 'tracking_checks.csv')
