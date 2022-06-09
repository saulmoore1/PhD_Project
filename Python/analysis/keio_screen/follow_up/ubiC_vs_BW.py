#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keio ubiC vs BW

@author: sm5911
@date: 11/05/2022

"""

#%% Imports

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from matplotlib import transforms
from matplotlib import pyplot as plt
# from matplotlib.lines import Line2D
# from matplotlib import patches as mpatches
from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from visualisation.plotting_helper import sig_asterix
from statistical_testing.stats_helper import do_stats
from time_series.time_series_helper import get_strain_timeseries
from time_series.plot_timeseries import plot_timeseries_motion_mode

#%% Globals

PROJECT_DIR = '/Volumes/hermes$/Keio_ubiC_6WP'
SAVE_DIR = '/Users/sm5911/Documents/Keio_ubiC'
IMAGING_DATES = ['20220418']
N_WELLS = 6

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
MIN_NSKEL_PER_VIDEO = None
MIN_NSKEL_SUM = 500

FEATURE = 'motion_mode_forward_fraction'
FEATURE_LIST = ['motion_mode_forward_fraction',
                'motion_mode_paused_fraction',
                'motion_mode_backward_fraction']

WINDOW_DICT_SECONDS = {0:(1790,1800), 1:(1805,1815), 2:(1815,1825),
                       3:(1850,1860), 4:(1865,1875), 5:(1875,1885),
                       6:(1910,1920), 7:(1925,1935), 8:(1935,1945)}

WINDOW_DICT_STIM_TYPE = {0:'prestim\n(30min)',1:'bluelight\n(30min)',2:'poststim\n(30min)',
                         3:'prestim\n(31min)',4:'bluelight\n(31min)',5:'poststim\n(31min)',
                         6:'prestim\n(32min)',7:'bluelight\n(32min)',8:'poststim\n(32min)'}

WINDOW_NUMBER = 2

food_type_list = ['BW','fepD','ubiC']
BLUELIGHT_TIMEPOINTS_MINUTES = [30,31,32]
FPS = 25
VIDEO_LENGTH_SECONDS = 38*60
SMOOTH_WINDOW_SECONDS = 5
BLUELIGHT_WINDOWS_ONLY_TS = True
  
#%% Functions

def ubiC_stats(metadata, 
               features,                       
               save_dir,
               window=WINDOW_NUMBER,
               feature_list=[FEATURE],
               pvalue_threshold=0.05,
               fdr_method='fdr_by'):
    """ T-tests comparing worm motion mode on ubiC and fepD vs BW control """
    
    # subset for window of interest
    window_meta = metadata.query("window==@window")
    
    # testing difference in motion mode forwards on ubiC or fepD vs BW
    do_stats(metadata=window_meta,
             features=features.reindex(window_meta.index),
             group_by='food_type',
             control='BW',
             save_dir=save_dir / 'ubiC_vs_BW',
             feat=feature_list,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method,
             ttest_if_nonsig=True)

    return
    
def ubiC_plots(metadata,
               features,
               plot_dir,
               stats_dir,
               window=WINDOW_NUMBER,
               feature_list=[FEATURE]):
    """ Plots of worm motion mode on ubiC and fepD vs BW control """
    
    assert metadata.shape[0] == features.shape[0]
        
    window_meta = metadata.query("window==@window")
    plot_df = window_meta.join(features.reindex(window_meta.index))

    for feature in tqdm(feature_list):
        plt.close('all')
        fig, ax = plt.subplots(figsize=(12,8))
        sns.boxplot(x='food_type', y=feature, order=food_type_list, ax=ax, data=plot_df,
                    palette='tab10', showfliers=False) 
                    #hue='is_dead', hue_order=is_dead_list, dodge=True
        sns.stripplot(x='food_type', y=feature, order=food_type_list, ax=ax, data=plot_df,
                      s=5, marker='D', color='k')
        ax.set_xlabel('')
        ax.set_ylabel(feature.replace('_',' '), fontsize=12, labelpad=10)
        ax.set_title('BW control vs fepD and ubiC', pad=30, fontsize=18)
        # annotate p-values - load t-test results for each treatment vs BW control
        ttest_path = stats_dir / 'ubiC_vs_BW' / 'food_type_ttest_results.csv'
        ttest_df = pd.read_csv(ttest_path, index_col=0)
        for i, food in enumerate(food_type_list):
            if food == 'BW':
                continue
            assert ax.get_xticklabels()[i].get_text() == food
            p = ttest_df.loc[feature, 'pvals_' + food]
            p_text = '***\nP < 0.001' if p < 0.001 else sig_asterix([p])[0] + '\nP = %.3f' % p
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(i, 1.01, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
        save_path = Path(plot_dir) / 'ubiC_vs_BW' / '{}.pdf'.format(feature)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)

    return

def ubiC_timeseries(metadata, project_dir=PROJECT_DIR, save_dir=SAVE_DIR, window=WINDOW_NUMBER):
    """ Timeseries plots for addition of enterobactin, paraquat, and iron to BW and fepD """
    
    metadata = metadata.query("window==@window")
        
    control = 'BW'        
    treatment_order = [t for t in sorted(metadata['food_type'].unique()) if t != control]

    # get timeseries for control data
    control_timeseries = get_strain_timeseries(metadata[metadata['food_type']==control], 
                                               project_dir=project_dir, 
                                               strain=control,
                                               group_by='food_type',
                                               n_wells=6,
                                               save_dir=Path(save_dir) / 'Data' / control,
                                               verbose=False)

    for treatment in tqdm(treatment_order):
        
        test_treatments = [control, treatment]
        motion_modes = ['forwards','backwards','stationary']

        for mode in motion_modes:
                    
            # get timeseries data for treatment data
            strain_metadata = metadata[metadata['food_type']==treatment]
            strain_timeseries = get_strain_timeseries(strain_metadata, 
                                                      project_dir=project_dir, 
                                                      strain=treatment,
                                                      group_by='food_type',
                                                      n_wells=6,
                                                      save_dir=Path(save_dir) / 'Data' / treatment,
                                                      verbose=False)

            print("Plotting timeseries '%s' fraction for '%s' vs '%s'..." %\
                  (mode, treatment, control))

            plt.close('all')
            fig, ax = plt.subplots(figsize=(15,5), dpi=200)
            col_dict = dict(zip(test_treatments, sns.color_palette("pastel", len(test_treatments))))
            bluelight_frames = [(i*60*FPS, i*60*FPS+10*FPS) for i in BLUELIGHT_TIMEPOINTS_MINUTES]

            ax = plot_timeseries_motion_mode(df=control_timeseries,
                                             window=SMOOTH_WINDOW_SECONDS*FPS,
                                             error=True,
                                             mode=mode,
                                             max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
                                             title=None,
                                             saveAs=None,
                                             ax=ax,
                                             bluelight_frames=bluelight_frames,
                                             colour=col_dict[control],
                                             alpha=0.25)
            
            ax = plot_timeseries_motion_mode(df=strain_timeseries,
                                             window=SMOOTH_WINDOW_SECONDS*FPS,
                                             error=True,
                                             mode=mode,
                                             max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
                                             title=None,
                                             saveAs=None,
                                             ax=ax,
                                             bluelight_frames=bluelight_frames,
                                             colour=col_dict[treatment],
                                             alpha=0.25)
        
            xticks = np.linspace(0, VIDEO_LENGTH_SECONDS*FPS, int(VIDEO_LENGTH_SECONDS/60)+1)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
            ax.set_xlabel('Time (minutes)', fontsize=12, labelpad=10)
            ax.set_ylabel('Fraction {}'.format(mode), fontsize=12, labelpad=10)
            ax.legend(test_treatments, fontsize=12, frameon=False, loc='best')
    
            if BLUELIGHT_WINDOWS_ONLY_TS:
                ts_plot_dir = Path(save_dir) / 'Plots' / 'timeseries_bluelight' / treatment
                ax.set_xlim([min(BLUELIGHT_TIMEPOINTS_MINUTES)*60*FPS-60*FPS, 
                             max(BLUELIGHT_TIMEPOINTS_MINUTES)*60*FPS+70*FPS])
            else:
                ts_plot_dir = Path(save_dir) / 'Plots' / 'timeseries' / treatment
    
            plt.tight_layout()
            ts_plot_dir.mkdir(exist_ok=True, parents=True)
            plt.savefig(ts_plot_dir / '{0}_{1}.pdf'.format(treatment, mode))  
    
    return
    
#%% Main

if __name__ == '__main__':
    
    AUX_DIR = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    RES_DIR = Path(PROJECT_DIR) / 'Results'

    META_PATH = Path(SAVE_DIR) / 'metadata.csv'
    FEAT_PATH = Path(SAVE_DIR) / 'features.csv'
    
    if not META_PATH.exists() and not FEAT_PATH.exists():
    
        # compile metadata
        metadata, metadata_path = compile_metadata(aux_dir=AUX_DIR, 
                                                   imaging_dates=IMAGING_DATES, 
                                                   n_wells=N_WELLS,
                                                   add_well_annotations=N_WELLS==96,
                                                   from_source_plate=True)
        
        # compile window summaries
        features, metadata = process_feature_summaries(metadata_path=metadata_path, 
                                                       results_dir=RES_DIR, 
                                                       compile_day_summaries=True,
                                                       imaging_dates=IMAGING_DATES, 
                                                       align_bluelight=False,
                                                       window_summaries=True,
                                                       n_wells=N_WELLS)
        
        # clean results
        features, metadata = clean_summary_results(features, 
                                                   metadata,
                                                   feature_columns=None,
                                                   nan_threshold_row=NAN_THRESHOLD_ROW,
                                                   nan_threshold_col=NAN_THRESHOLD_COL,
                                                   max_value_cap=1e15,
                                                   imputeNaN=True,
                                                   min_nskel_per_video=MIN_NSKEL_PER_VIDEO,
                                                   min_nskel_sum=MIN_NSKEL_SUM,
                                                   drop_size_related_feats=False,
                                                   norm_feats_only=False,
                                                   percentile_to_use=None)
    
        assert not features.isna().sum(axis=1).any()
        assert not (features.std(axis=1) == 0).any()
    
        # save features
        Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)
        metadata.to_csv(META_PATH, index=False)
        features.to_csv(FEAT_PATH, index=False) 
    
    else:
        # load clean metadata and features
        metadata = pd.read_csv(META_PATH, dtype={'comments':str, 'source_plate_id':str})
        features = pd.read_csv(FEAT_PATH, index_col=None)

    ubiC_stats(metadata,
               features,
               save_dir=Path(SAVE_DIR) / "Stats",
               window=WINDOW_NUMBER,
               feature_list=FEATURE_LIST)
    
    ubiC_plots(metadata,
               features,
               plot_dir=Path(SAVE_DIR) / "Plots",
               stats_dir=Path(SAVE_DIR) / "Stats",
               window=WINDOW_NUMBER,
               feature_list=FEATURE_LIST)
    
    ubiC_timeseries(metadata,
                    project_dir=Path(PROJECT_DIR),
                    save_dir=Path(SAVE_DIR),
                    window=WINDOW_NUMBER)
