#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot time-series plots of selected features during bluelight video recording
@author: sm5911
@date: 04/10/2021
"""

#%% Imports

import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import patches

from read_data.read import get_skeleton_data, read_list_from_file
from time_series.time_series_helper import get_strain_timeseries

#%% Globals

EXAMPLE_METADATA_PATH = '/Users/sm5911/Documents/Keio_Conf_Screen/metadata.csv'
EXAMPLE_RESULTS_DIR = '/Volumes/hermes$/KeioScreen2_96WP/Results'
EXAMPLE_SAVE_DIR = '/Users/sm5911/Documents/Keio_Conf_Screen'
EXAMPLE_FEATURE_SET_PATH = '/Users/sm5911/Documents/Keio_Screen2/selected_features_timeseries.txt'
EXAMPLE_STRAIN_LIST_PATH = None #'/Users/sm5911/Documents/Keio_Screen/52_selected_strains_from_initial_keio_top100_lowest_pval_tierpsy16.txt'
EXAMPLE_STRAIN_LIST = ['wild_type','fepB','fepD','fes','atpB','nuoC','sdhD','entA'] # missing: 'trpA','trpD'
EXAMPLE_CONTROL = 'wild_type'

BLUELIGHT_FRAMES = [(1500,1751),(4000,4251),(6500,6751)]


RENAME_DICT = {"BW" : "wild_type",
               "FECE" : "fecE",
               "AroP" : "aroP",
               "TnaB" : "tnaB"}
WINDOW = 100
MAX_N_FRAMES = 9000
MULTI_STRAIN = True # plot multiple strains on the same plot?
VIDEO_FPS = 25 # frrame rate per second

#%% Functions

def add_bluelight_to_plot(ax, bluelight_frames=BLUELIGHT_FRAMES, alpha=0.5):
    """ Add lines to plot to indicate video frames where bluelight stimulus was delivered 
    
        Inputs
        ------
        fig, ax : figure and axes from plt.subplots()
        bluelight_frames : list of tuples (start, end) 
    """
    assert type(bluelight_frames) == list or type(bluelight_frames) == tuple
    if not type(bluelight_frames) == list:
        bluelight_frames = [bluelight_frames]
    
    for (start, stop) in bluelight_frames:
        ax.axvspan(start, stop, facecolor='lightblue', alpha=alpha)
     
    return ax

def get_motion_mode_timestamp_stats(frac_mode, mode='stationary'):
           
        # average fraction in given motion mode
        mode_df = frac_mode[frac_mode['motion_name']==mode]
        
        total_frame_count = frac_mode.groupby(['timestamp'])['fraction'].agg(['count'])
        
        # group by timestamp (average fraction across videos for each timestamp)
        mode_frame_stats = mode_df.groupby(['timestamp'])['fraction'].agg(['sum','mean','std'])
        
        # total count may include frames where no recording was made for this motion mode,
        # so pad out results with 0 for this motion mode
        frame_mode_df = pd.merge(mode_frame_stats.reset_index(), total_frame_count.reset_index(), 
                                 on='timestamp', how='outer').fillna(0)
                
        return frame_mode_df

def _bootstrapped_ci(x, function=np.mean, n_boot=100, which_ci=95, axis=None):
    """ Wrapper for tierpsytools bootstrapped_ci function, which encounters name space / 
        variable scope conflicts when used in combination with pandas apply function 
    """
    from tierpsytools.analysis.statistical_tests import bootstrapped_ci
    
    lower, upper = bootstrapped_ci(x, func=function, n_boot=n_boot, which_ci=which_ci, axis=axis)
    
    return lower, upper

def bootstrapped_ci(x, n_boot=100, alpha=0.95):
    """ Wrapper for applying bootstrap function to sample array """

    from sklearn.utils import resample
    
    means = []
    for i in range(n_boot):
        s = resample(x, n_samples=int(len(x)))
        m = np.mean(s)
        means.append(m)
    # plt.hist(means); plt.show()
    
    # confidence intervals
    p_lower = ((1.0 - alpha) / 2.0) * 100
    lower = np.percentile(means, p_lower)
    p_upper = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = np.percentile(means, p_upper)
                        
    return lower, upper


def plot_timeseries(df, 
                    feature='speed', 
                    error=True, 
                    max_n_frames=None, 
                    smoothing=1, 
                    ax=None,
                    bluelight_frames=None, 
                    title=None, 
                    saveAs=None, 
                    colour=None):
    """ Plot timeseries for any feature in HDF5 timeseries data EXCEPT from motion mode or turn 
        features. For motion mode, please use 'plot_timeseries_motion_mode' """
    
    from time_series.plot_timeseries import add_bluelight_to_plot #, _bootstrapped_ci
    
    grouped_timestamp = df.groupby(['timestamp'])[feature]

    plot_df = grouped_timestamp.mean().reset_index()

    # mean and bootstrap CI error for each timestamp
    if error:            
                                
        conf_ints = grouped_timestamp.apply(bootstrapped_ci, n_boot=100)
        conf_ints = pd.concat([pd.Series([x[0] for x in conf_ints], index=conf_ints.index), 
                               pd.Series([x[1] for x in conf_ints], index=conf_ints.index)], 
                              axis=1)
        conf_ints = conf_ints.rename(columns={0:'lower',1:'upper'}).reset_index()
                            
        plot_df = pd.merge(plot_df, conf_ints, on='timestamp')
        #plot_df = plot_df.dropna(axis=0, how='any')

    plot_df = plot_df.set_index('timestamp').rolling(window=smoothing, 
                                                     center=True).mean().reset_index()
        
    # crop timeseries data to standard video length (optional)
    if max_n_frames:
        plot_df = plot_df[plot_df['timestamp'] <= max_n_frames]

    if ax is None:
        fig, ax = plt.subplots(figsize=(15,6))

    sns.lineplot(data=plot_df,
                 x='timestamp',
                 y=feature,
                 ax=ax,
                 ls='-',
                 hue=None,
                 palette=None,
                 color=colour)
    if error:
        ax.fill_between(plot_df['timestamp'], plot_df['lower'], plot_df['upper'], 
                        color=colour, edgecolor=None, alpha=0.25)
    
    # add decorations
    if bluelight_frames is not None:
        ax = add_bluelight_to_plot(ax, bluelight_frames=bluelight_frames, alpha=0.5)

    if title:
        plt.title(title, pad=10)

    if saveAs is not None:
        Path(saveAs).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(saveAs)
    
    if ax is None:
        return fig, ax
    else:
        return ax
   
def plot_timeseries_feature(metadata,
                            project_dir,
                            save_dir,
                            feature='speed',
                            group_by='treatment',
                            control='BW-nan-nan-N',
                            groups_list=None,
                            n_wells=6,
                            bluelight_stim_type='bluelight',
                            bluelight_timepoints_seconds=[(60, 70),(160, 170),(260, 270)],
                            video_length_seconds=360,
                            smoothing=10,
                            fps=25,
                            ylim_minmax=None,
                            palette='tab10',
                            col_dict=None):
        
    if groups_list is not None:
        assert isinstance(groups_list, list) 
        assert all(g in metadata[group_by].unique() for g in groups_list)
    else:
        groups_list = sorted(metadata[group_by].unique())
    groups_list = [g for g in groups_list if g != control]
    assert control in metadata[group_by].unique()
    
    if bluelight_stim_type is not None and 'window' not in metadata.columns:
        metadata['imgstore_name'] = metadata['imgstore_name_{}'.format(bluelight_stim_type)]
        
    if 'window' in metadata.columns:
        assert bluelight_stim_type is not None
        stimtype_videos = [i for i in metadata['imgstore_name'] if bluelight_stim_type in i]
        metadata = metadata[metadata['imgstore_name'].isin(stimtype_videos)]
    
    if bluelight_timepoints_seconds is not None:
        bluelight_frames = [(i*fps, j*fps) for (i, j) in bluelight_timepoints_seconds]
    
    # get control timeseries
    control_ts = get_strain_timeseries(metadata,
                                       project_dir=project_dir,
                                       strain=control,
                                       group_by=group_by,
                                       feature_list=[feature],#['motion_mode','speed']
                                       save_dir=save_dir,
                                       n_wells=n_wells,
                                       verbose=True)

    for group in tqdm(groups_list):
        ts_plot_dir = save_dir / 'Plots' / '{0}'.format(group)
        ts_plot_dir.mkdir(exist_ok=True, parents=True)
        save_path = ts_plot_dir / '{0}_{1}.pdf'.format(feature, bluelight_stim_type)
        
        if not save_path.exists():
            group_ts = get_strain_timeseries(metadata,
                                             project_dir=project_dir,
                                             strain=group,
                                             group_by=group_by,
                                             feature_list=[feature],
                                             save_dir=save_dir,
                                             n_wells=n_wells,
                                             verbose=True)
            
            print("Plotting '%s' timeseries for %s vs %s" % (feature, group, control))
            if col_dict is None:
                col_dict = dict(zip([control, group], sns.color_palette(palette, 2)))
            
            plt.close('all')
            fig, ax = plt.subplots(figsize=(15,6), dpi=300)
            ax = plot_timeseries(df=control_ts,
                                 feature=feature,
                                 error=True, 
                                 max_n_frames=video_length_seconds*fps, 
                                 smoothing=smoothing*fps, 
                                 ax=ax,
                                 bluelight_frames=(bluelight_frames if 
                                                   bluelight_stim_type == 'bluelight' else None),
                                 colour=col_dict[control])
            
            ax = plot_timeseries(df=group_ts,
                                 feature=feature,
                                 error=True, 
                                 max_n_frames=video_length_seconds*fps, 
                                 smoothing=smoothing*fps, 
                                 ax=ax,
                                 bluelight_frames=(bluelight_frames if 
                                                   bluelight_stim_type == 'bluelight' else None),
                                 colour=col_dict[group])
            
            if ylim_minmax is not None:
                assert isinstance(ylim_minmax, tuple)
                plt.ylim(ylim_minmax[0], ylim_minmax[1])
                    
            xticks = np.linspace(0, video_length_seconds*fps, int(video_length_seconds/60)+1)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(x/fps/60)) for x in xticks])   
            ax.set_xlabel('Time (minutes)', fontsize=12, labelpad=10)
            ylab = feature.replace("_", " (µm s$^{-1}$)") if feature == 'speed' else feature
            ax.set_ylabel(ylab, fontsize=12, labelpad=10)
            ax.legend([control, group], fontsize=12, frameon=False, loc='best', handletextpad=1)
            plt.subplots_adjust(left=0.1, top=0.95, bottom=0.1, right=0.95)
    
            # save plot
            print("Saving to: %s" % save_path)
            plt.savefig(save_path)

    return

def plot_window_timeseries_feature(metadata,
                                   project_dir,
                                   save_dir,
                                   group_by='gene_name',
                                   control='BW',
                                   groups_list=None,
                                   feature='speed',
                                   n_wells=6,
                                   bluelight_timepoints_seconds=None,
                                   bluelight_windows_separately=True,
                                   xlim_crop_around_bluelight_seconds=(30,120),
                                   smoothing=10,
                                   fps=25,
                                   figsize=(15,5),
                                   ylim_minmax=(-20,330),
                                   video_length_seconds=None):
        
    if groups_list is not None:
        assert isinstance(groups_list, list) 
        assert all(g in metadata[group_by].unique() for g in groups_list)
    else:
        groups_list = sorted(metadata[group_by].unique())
    groups_list = [g for g in groups_list if g != control]
    assert control in metadata[group_by].unique()
        
    if bluelight_timepoints_seconds is not None:
        bluelight_frames = [(i*fps, j*fps) for (i, j) in bluelight_timepoints_seconds]
    
    # get control timeseries
    control_ts = get_strain_timeseries(metadata,
                                       project_dir=project_dir,
                                       strain=control,
                                       group_by=group_by,
                                       feature_list=[feature],#['motion_mode','speed']
                                       save_dir=save_dir,
                                       n_wells=n_wells,
                                       verbose=True)
    
    if video_length_seconds is None:
        video_length_seconds = control_ts['timestamp'].max() / fps

    for group in tqdm(groups_list):
        ts_plot_dir = save_dir / 'Plots' / '{0}'.format(group)
        ts_plot_dir.mkdir(exist_ok=True, parents=True)
        save_path = ts_plot_dir / '{}.pdf'.format(feature)
        
        if not save_path.exists():
            group_ts = get_strain_timeseries(metadata,
                                             project_dir=project_dir,
                                             strain=group,
                                             group_by=group_by,
                                             feature_list=[feature],
                                             save_dir=save_dir,
                                             n_wells=n_wells,
                                             verbose=True)
            
            print("Plotting '%s' timeseries for %s vs %s" % (feature, group, control))
            col_dict = dict(zip([control, group], sns.color_palette('tab10', 2)))

            if bluelight_windows_separately:
                for pulse, frame in enumerate(tqdm(bluelight_frames), start=1):
                    
                    # crop -30sec before to +2mins after pulse
                    timestamp_range = (frame[0] - xlim_crop_around_bluelight_seconds[0] * fps, 
                                       frame[1] + xlim_crop_around_bluelight_seconds[1] * fps)
                    _control_ts = control_ts[np.logical_and(control_ts['timestamp']>=timestamp_range[0], 
                                                            control_ts['timestamp']<=timestamp_range[1])]
                    _group_ts = group_ts[np.logical_and(group_ts['timestamp']>=timestamp_range[0], 
                                                        group_ts['timestamp']<=timestamp_range[1])]                  
                    
                    plt.close('all')
                    fig, ax = plt.subplots(figsize=figsize, dpi=300)
                    ax = plot_timeseries(df=_control_ts,
                                         feature=feature,
                                         error=True, 
                                         max_n_frames=None, 
                                         smoothing=smoothing*fps, 
                                         ax=ax,
                                         bluelight_frames=bluelight_frames,
                                         colour=col_dict[control])
            
                    ax = plot_timeseries(df=_group_ts,
                                         feature=feature,
                                         error=True, 
                                         max_n_frames=None, 
                                         smoothing=smoothing*fps, 
                                         ax=ax,
                                         bluelight_frames=bluelight_frames,
                                         colour=col_dict[group])

                    xticks = np.linspace(0, video_length_seconds*fps, int(video_length_seconds/60)+1)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels([str(int(x/fps/60)) for x in xticks])   
                    ax.set_xlim([timestamp_range[0], timestamp_range[1]])
                    ax.set_xlabel('Time (minutes)', fontsize=12, labelpad=10)
                    if ylim_minmax is not None:
                        assert isinstance(ylim_minmax, tuple)
                        plt.ylim(ylim_minmax[0], ylim_minmax[1])
                    ylab = feature.replace("_", " (µm s$^{-1}$)") if feature == 'speed' else feature
                    ax.set_ylabel(ylab, fontsize=12, labelpad=10)
                    ax.set_title('{0} vs {1} (bluelight pulse {2} = {3} min)'.format(
                        group, control, pulse, int(frame[0]/fps/60)), fontsize=12, pad=10)
                    ax.legend([control, group], fontsize=12, frameon=False, loc='best',
                              handletextpad=1)
                    plt.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.95)
            
                    # save plot
                    ts_plot_dir = save_dir / 'Plots' / group
                    ts_plot_dir.mkdir(exist_ok=True, parents=True)
                    save_path = ts_plot_dir / '{0}_pulse{1}_{2}min.pdf'.format(
                        feature, pulse, int(frame[0]/fps/60))
                    print("Saving to: %s" % save_path)
                    plt.savefig(save_path)
                    
            else:
                plt.close('all')
                fig, ax = plt.subplots(figsize=figsize, dpi=300)
                ax = plot_timeseries(df=control_ts,
                                     feature=feature,
                                     error=True, 
                                     max_n_frames=(video_length_seconds*fps if video_length_seconds
                                                   is not None else None), 
                                     smoothing=smoothing*fps, 
                                     ax=ax,
                                     bluelight_frames=bluelight_frames,
                                     colour=col_dict[control])
        
                ax = plot_timeseries(df=group_ts,
                                     feature=feature,
                                     error=True, 
                                     max_n_frames=(video_length_seconds*fps if video_length_seconds
                                                   is not None else None), 
                                     smoothing=smoothing*fps, 
                                     ax=ax,
                                     bluelight_frames=bluelight_frames,
                                     colour=col_dict[group])
                xticks = np.linspace(0, video_length_seconds*fps, int(video_length_seconds/60)+1)
                ax.set_xticks(xticks)
                ax.set_xticklabels([str(int(x/fps/60)) for x in xticks])   
                ax.set_xlabel('Time (minutes)', fontsize=12, labelpad=10)
                if ylim_minmax is not None:
                    assert isinstance(ylim_minmax, tuple)
                    plt.ylim(ylim_minmax[0], ylim_minmax[1])

                ylab = feature.replace("_", " (µm s$^{-1}$)") if feature == 'speed' else feature
                ax.set_ylabel(ylab, fontsize=12, labelpad=10)
                ax.set_title('%s vs %s' % (group, control), fontsize=12, pad=10)
                ax.legend([control, group], fontsize=12, frameon=False, loc='best', handletextpad=1)
                plt.subplots_adjust(left=0.1, top=0.95, bottom=0.1, right=0.95)
        
                # save plot
                ts_plot_dir = save_dir / 'Plots' / group
                ts_plot_dir.mkdir(exist_ok=True, parents=True)
                save_path = ts_plot_dir / '{}.pdf'.format(feature)
                print("Saving to: %s" % save_path)
                plt.savefig(save_path)

    return

def plot_timeseries_motion_mode(df,
                                window=None, error=False, mode=None, max_n_frames=None,
                                title=None, figsize=(12,6), ax=None, saveAs=None,
                                sns_colour_palette='pastel', colour=None, 
                                bluelight_frames=None,
                                cols = ['motion_mode','filename','well_name','timestamp'], 
                                alpha=0.5):
    """ Plot motion mode timeseries from 'timeseries_data' for a given treatment (eg. strain) 
    
        Inputs
        ------
        df : pd.DataFrame
            Compiled dataframe of 'timeseries_data' from all featuresN HDF5 files for a given 
            treatment (eg. strain) 
        window : int
            Moving average window of n frames
        error : bool
            Add error to timeseries plots
        mode : str
            The motion mode you would like to plot (choose from: ['stationary','forwards','backwards'])
        max_n_frames : int
            Maximum number of frames in video (x axis limit)
        title : str
            Title of figure (optional, ax is returned so title and other plot params can be added later)
        figsize : tuple
            Size of figure to be passed to plt.subplots figsize param
        ax : matplotlib AxesSubplot, None
            Axis of figure subplot
        saveAs : str
            Path to save directory
        sns_colour_palette : str
            Name of seaborn colour palette
        colour : str, None
            Plot single colour for plot (if plotting a single strain or a single motion mode)
        bluelight_frames : list
            List of tuples for (start, end) frame numbers of each bluelight stimulus (optional)
        cols : list
            List of cols to group_by
        alpha : float
            Bluelight window transparency
            
        Returns
        -------
        fig : matplotlib Figure 
            If ax is None, so the figure may be saved
            
        ax : matplotlib AxesSubplot
            For iterative plotting   
    """

    # discrete data mapping
    motion_modes = ['stationary','forwards','backwards']
    motion_dict = dict(zip([0,1,-1], motion_modes))

    if mode is not None:
        if type(mode) == int or type(mode) == float:
            mode = motion_dict[mode]     
        else:
            assert type(mode) == str 
            mode = 'stationary' if mode == 'paused' else mode
            assert mode in motion_modes
    
    assert all(c in df.columns for c in cols)

    # # drop NaN data
    df = df.loc[~df['motion_mode'].isna(), cols]
     
    # map whether forwards, backwards or stationary motion in each frame
    df['motion_name'] = df['motion_mode'].map(motion_dict)
    assert not df['motion_name'].isna().any()
             
    # total number of worms in each motion mode in each timestamp of each video
    video_mode_count = df.groupby(['filename','timestamp','motion_name'])['well_name'].count()
    video_mode_count = video_mode_count.reset_index().rename(columns={'well_name':'mode_count'})
        
    # total number of worms in each timestamp in each video
    total_video_count = df.groupby(['filename','timestamp'])['well_name'].agg(['count']).reset_index()
    
    frac_video = pd.merge(video_mode_count, total_video_count, 
                          on=['filename','timestamp'], how='outer')
    
    frac_video['fraction'] = frac_video['mode_count'] / frac_video['count']
    frac_video = frac_video.drop(columns=['count','mode_count'])

    assert all(frac_video.groupby(['filename','timestamp'])['fraction'].sum().round(9) == 1)
    
    # where missing data for motion mode in any video/timestamp, add row with zero value for fraction
    mux = pd.MultiIndex.from_product([frac_video['filename'].unique(), 
                                      frac_video['timestamp'].unique(), 
                                      motion_modes])
    
    frac_video = frac_video.set_index(['filename','timestamp','motion_name']
                                      ).reindex(mux).reset_index().set_axis(
                                          frac_video.columns, axis=1).sort_values(
                                              by=['filename','timestamp'], ascending=True)
    
    NaN_mask = frac_video.groupby(['filename','timestamp']).mean()['fraction'].isna().reset_index()
    NaN_mask = NaN_mask.rename(columns={'fraction':'drop'})
    frac_video = pd.merge(frac_video, NaN_mask, on=['filename','timestamp'], how='outer')    
    frac_video = frac_video[~frac_video['drop']][frac_video.columns[:-1]].fillna(0, axis=1)

    assert all(frac_video.groupby(['filename','timestamp'])['fraction'].sum().round(9) == 1)

    # subset for motion mode to plot
    frac_video_mode = frac_video[frac_video['motion_name']==mode]

    grouped_timestamp_mode = frac_video_mode.groupby(['timestamp'])['fraction']
    
    plot_df = grouped_timestamp_mode.mean().reset_index()

    # mean and bootstrap CI error for each timestamp
    if error:
        conf_ints = grouped_timestamp_mode.apply(_bootstrapped_ci, 
                                                 function=np.mean, 
                                                 n_boot=100)
         
        conf_ints = pd.concat([pd.Series([x[0] for x in conf_ints], index=conf_ints.index), 
                               pd.Series([x[1] for x in conf_ints], index=conf_ints.index)], axis=1)
        conf_ints = conf_ints.rename(columns={0:'lower',1:'upper'}).reset_index()
         
        plot_df = pd.merge(plot_df, conf_ints, on='timestamp')
        #plot_df = plot_df.dropna(axis=0, how='any')

    # crop timeseries data to standard video length (optional)
    if max_n_frames:
        plot_df = plot_df[plot_df['timestamp'] <= max_n_frames]
    
    # moving average (optional)
    if window:
        plot_df = plot_df.set_index('timestamp').rolling(window=window, 
                                                         center=True).mean().reset_index()

    if ax is None:
        fig, ax = plt.subplots(figsize=(15,6))

    # motion_ls_dict = dict(zip(motion_modes, ['-','--','-.']))                
    sns.lineplot(data=plot_df, 
                 x='timestamp',
                 y='fraction',
                 ax=ax,
                 ls='-', # motion_ls_dict[mode] if len(mode_list) > 1 else '-',
                 hue=None, #'motion_name' if colour is None else None, 
                 palette=None, #palette if colour is None else None,
                 color=colour)
    if error:
        ax.fill_between(plot_df['timestamp'], plot_df['lower'], plot_df['upper'], 
                        color=colour, edgecolor=None, alpha=0.25)
    
    # add decorations
    if bluelight_frames is not None:
        ax = add_bluelight_to_plot(ax, bluelight_frames=bluelight_frames, alpha=alpha)

    if title:
        plt.title(title, pad=10)

    if saveAs is not None:
        Path(saveAs).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(saveAs)
    
    if ax is None:
        return fig, ax
    else:
        return ax
      

def selected_strains_timeseries(metadata, 
                                project_dir, 
                                save_dir, 
                                group_by='gene_name',
                                control='wild_type',
                                strain_list=['fepD'],
                                n_wells=96,
                                bluelight_stim_type='bluelight',
                                video_length_seconds=6*60,
                                bluelight_timepoints_seconds=[(60,70),(160,170),(260,270)],
                                motion_modes=['forwards','stationary','backwards'],
                                smoothing=10,
                                fps=25):
    """ Timeseries plots for standard imaging and bluelight delivery protocol for the initial and 
        confirmation screening of Keio Collection. Bluelight stimulation is delivered after 5 mins
        pre-stimulus, 10 secs stimulus every 60 secs, repeated 3 times (6 mins total), 
        followed by 5 mins post-stimulus (16 minutes total)
    """
            
    if strain_list is None:
        strain_list = list(metadata[group_by].unique())
    else:
        assert isinstance(strain_list, list)
        assert all(s in metadata[group_by].unique() for s in strain_list)
    strain_list = [s for s in strain_list if s != control]
    
    if 'window' not in metadata.columns:
        metadata['imgstore_name'] = metadata['imgstore_name_{}'.format(bluelight_stim_type)]
    
    if 'window' in metadata.columns:
        stimtype_videos = [i for i in metadata['imgstore_name'] if bluelight_stim_type in i]
        metadata = metadata[metadata['imgstore_name'].isin(stimtype_videos)]

    # remove entries with missing video filename info
    n_nan = len([s for s in metadata['imgstore_name'].unique() if not isinstance(s, str)])
    if n_nan > 1:
        print("WARNING: Ignoring {} entries with missing ingstore_name_{} info".format(
            n_nan, bluelight_stim_type))
        metadata = metadata[~metadata['imgstore_name'].isna()]
    
    bluelight_frames = [(i*fps, j*fps) for (i, j) in bluelight_timepoints_seconds]
    
    # get timeseries for BW
    control_ts = get_strain_timeseries(metadata[metadata[group_by]==control], 
                                       project_dir=project_dir, 
                                       strain=control,
                                       group_by=group_by,
                                       n_wells=n_wells,
                                       save_dir=Path(save_dir) / 'Data' / bluelight_stim_type /\
                                           control)
    
    for strain in tqdm(strain_list):
        col_dict = dict(zip([control, strain], sns.color_palette("pastel", 2)))

        # get timeseries for strain
        strain_ts = get_strain_timeseries(metadata[metadata[group_by]==strain], 
                                          project_dir=project_dir, 
                                          strain=strain,
                                          group_by=group_by,
                                          n_wells=n_wells,
                                          save_dir=Path(save_dir) / 'Data' /\
                                              bluelight_stim_type / strain)
    
        for mode in motion_modes:
            print("Plotting timeseries for motion mode %s fraction for %s vs BW.." % (mode, strain))

            plt.close('all')
            fig, ax = plt.subplots(figsize=(12,5), dpi=200)
    
            ax = plot_timeseries_motion_mode(df=control_ts,
                                             window=smoothing*fps,
                                             error=True,
                                             mode=mode,
                                             max_n_frames=video_length_seconds*fps,
                                             title=None,
                                             saveAs=None,
                                             ax=ax,
                                             bluelight_frames=(bluelight_frames if 
                                                               bluelight_stim_type == 'bluelight'
                                                               else None),
                                             colour=col_dict[control],
                                             alpha=0.25)
            
            ax = plot_timeseries_motion_mode(df=strain_ts,
                                             window=smoothing*fps,
                                             error=True,
                                             mode=mode,
                                             max_n_frames=video_length_seconds*fps,
                                             title=None,
                                             saveAs=None,
                                             ax=ax,
                                             bluelight_frames=(bluelight_frames if 
                                                               bluelight_stim_type == 'bluelight'
                                                               else None),
                                             colour=col_dict[strain],
                                             alpha=0.25)
        
            xticks = np.linspace(0, video_length_seconds*fps, int(video_length_seconds/60)+1)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(x/fps/60)) for x in xticks])   
            ax.set_xlabel('Time (minutes)', fontsize=12, labelpad=10)
            ax.set_ylabel('Fraction {}'.format(mode), fontsize=12, labelpad=10)
            ax.set_title('{0} vs {1}'.format(control, strain), fontsize=12, pad=10)
            ax.legend([control, strain], fontsize=12, frameon=False, loc='best')
            #TODO: plt.subplots_adjust(left=0.01,top=0.9,bottom=0.1,left=0.2)
    
            # save plot
            ts_plot_dir = save_dir / 'Plots' / '{0}'.format(strain)
            ts_plot_dir.mkdir(exist_ok=True, parents=True)
            save_path = ts_plot_dir / 'motion_mode_{0}_{1}.pdf'.format(mode, bluelight_stim_type)
            print("Saving to: %s" % save_path)
            plt.savefig(save_path)

    return

def plot_timeseries_from_metadata(metadata_path,
                                  results_dir,
                                  group_by='gene_name', 
                                  strain_list=None,
                                  control='wild_type',
                                  fset=['motion_mode'],
                                  motion_mode='all', # 'forwards', 'backwards', 'stationary', 
                                  multi_strain=MULTI_STRAIN,
                                  window=WINDOW,
                                  error=False,
                                  max_n_frames=MAX_N_FRAMES,
                                  save_dir=None,
                                  sns_colour_palette='Greens'):
    """ Plot timieseries data for specific strains in metadata
    
        Inputs
        ------
        metadata_path : str
            Path to metadata file
        group_by : str
            Metadata column name of variable to group timeseries data
        strain_list : list
            List of selected strains for timeseries analysis
        fset : list
            List of Tierpsy featuresN 'timeseries_data' features to plot
        save_dir : str
            Path to directory to save timeseries results 
    """

    metadata = pd.read_csv(metadata_path, dtype={"comments":str, "source_plate_id":str})

    # drop bad well samples
    n = metadata.shape[0]
    metadata = metadata[~metadata['is_bad_well']]
    if (n - metadata.shape[0]) > 0:
        print("%d bad well entries removed from metadata" % (n - metadata.shape[0]))
            
    # Subset results (rows) to remove entries for wells with unknown strain data for 'gene_name'
    n = metadata.shape[0]
    metadata = metadata.loc[~metadata[group_by].isna(),:]
    if (n - metadata.shape[0]) > 0:
        print("%d entries removed with no gene name data" % (n - metadata.shape[0]))

    # Rename gene names in metadata
    for k, v in RENAME_DICT.items():
        metadata.loc[metadata['gene_name'] == k, 'gene_name'] = v

    if strain_list is None:
        strain_list = metadata[group_by].unique()
    else:
        assert type(strain_list)==list
        seta = set(strain_list)
        setb = set([s for s in strain_list if s in metadata[group_by].unique()])
        missing_strains = list(seta - setb)
        if len(missing_strains) > 0:
            print("\nNo results found for %d strains: %s" % (len(missing_strains), missing_strains))
            strain_list = [s for s in strain_list if s in metadata[group_by].unique()]
            
    if not control in strain_list:
        strain_list.insert(0, control)

    results_dir = (Path(results_dir) if results_dir is not None else 
                   Path(str(Path(metadata_path).parent).replace('/AuxiliaryFiles','/Results')))
    save_dir = results_dir if save_dir is None else Path(save_dir)
                            
    grouped = metadata.groupby(group_by)
    
    ### Compiling timeseries results for each strain + saving
    
    for strain in tqdm(strain_list):
        
        ts_save_dir = save_dir / 'timeseries_data' / 'timeseries_{}.csv'.format(strain)
        
        if not ts_save_dir.exists():
            print("\nCompiling timeseries results for: %s" % strain)
            strain_df = grouped.get_group(strain)
            
            # if full metadata before 'align_bluelight' 
            if 'imgstore_name' in strain_df.columns:
                is_bluelight = ['bluelight' in f for f in strain_df['imgstore_name']]
                strain_df = strain_df[is_bluelight]
                
                # find paths to bluelight featuresN files
                featuresN_path = [str(results_dir / i / 'metadata_featuresN.hdf5') 
                                  for i in strain_df['imgstore_name']]
            
            # After aligning metadata for all bluelight stimulus videos per row
            elif 'imgstore_name_bluelight' in strain_df.columns:
                featuresN_path = [str(results_dir / i / 'metadata_featuresN.hdf5') 
                                  for i in strain_df['imgstore_name_bluelight']]
                
            strain_df['featuresN_path'] = featuresN_path
            
            # read all bluelight video timeseries data for strain
            timeseries_strain_data = []
            for i, (file, well) in enumerate(zip(strain_df['featuresN_path'], strain_df['well_name'])):
                
                # load video timeseries data
                df = get_skeleton_data(file, rig='Hydra', dataset='timeseries_data')
                
                # subset for well data
                df = df[df['well_name'] == well]
                
                # append filename + well_name info
                df['filename'] = file
                df['well_name'] = well
                
                assert all(f in df.columns for f in fset)
                #df = df[['filename','timestamp','well_name',*fset]] # append data for fset only
    
                # store video timeseries data
                timeseries_strain_data.append(df)
            
            # collate strain data across wells/plates/runs/days
            timeseries_strain = pd.concat(timeseries_strain_data, axis=0)
            
            # save timeseries data for strain
            ts_save_dir.parent.mkdir(parents=True, exist_ok=True)
            timeseries_strain.to_csv(ts_save_dir, header=True, index=False)

    ### Plotting timeseries for each feature, for only selected strains
    for feature in fset:
        if feature == 'motion_mode':
            
            # discrete data
            motion_mode_list = ['forwards','backwards','stationary']
            motion_dict = dict(zip([0,1,-1], motion_mode_list))

            if motion_mode is None or motion_mode == 'all':
                mode_list = motion_mode_list
            elif type(motion_mode) == float or type(motion_mode) == int:
                # get mode name
                motion_mode = motion_dict[int(motion_mode)]
                mode_list = [motion_mode]
            elif type(motion_mode) == list:
                assert all(m in motion_mode_list for m in motion_mode)
                mode_list = motion_mode
            else:
                assert type(motion_mode) == str and motion_mode in motion_mode_list
                mode_list = [motion_mode]

            # plot multiple strains on the same plot for each motion mode
            if multi_strain:
                            
                # plot timeseries for each mode separately
                for mode in tqdm(mode_list):
                    print("\nPlotting motion mode: '%s'" % mode)

                    # make colour palette for strains to plot
                    colour_dict = dict(zip(strain_list, 
                                           sns.color_palette(sns_colour_palette, len(strain_list))))
                    
                    # initialise figure for plotting multiple strains
                    plt.close('all')
                    fig, ax = plt.subplots(figsize=(15,8))
                    
                    legend_handles = []
                    for strain in strain_list:
                        # read timeseries data for strain
                        ts_save_dir = save_dir / 'timeseries_data' / 'timeseries_{}.csv'.format(strain)
                        timeseries_strain = pd.read_csv(ts_save_dir)
                      
                        ax = plot_timeseries_motion_mode(df=timeseries_strain,
                                                         mode=mode,
                                                         window=window,
                                                         error=error,
                                                         max_n_frames=max_n_frames,
                                                         title=None,
                                                         ax=ax,
                                                         saveAs=None,
                                                         sns_colour_palette=sns_colour_palette,
                                                         colour=colour_dict[strain])
                        
                        legend_handles.append(patches.Patch(color=colour_dict[strain], label=strain))

                    ax.set_xticks(np.linspace(0, max_n_frames, num=7))
                    ax.set_xticklabels([str(int(l)) for l in np.linspace(0, max_n_frames, 
                                                                    num=7) / VIDEO_FPS])
                    ax.set_xlabel('Time (seconds)')
                    ax.set_ylabel('Fraction of worms %s' % mode)
                    plt.title("%s (%s)" % (feature, str(mode)))
                    plt.legend(handles=legend_handles, labels=colour_dict.keys(), loc='best',\
                               borderaxespad=0.4, frameon=False, fontsize=15)
                    if save_dir:
                        '_'.join(strain_list)
                        saveAs = save_dir / 'timeseries_plots' / '_'.join(strain_list) /\
                                 ('%s_%s.pdf' % (feature, mode))
                        saveAs.parent.mkdir(parents=True, exist_ok=True)
                        plt.savefig(saveAs, dpi=300)
           
            # plot timeseries for each strain separately
            elif not multi_strain:
                
                for strain in tqdm(strain_list):
                    print("\nPlotting motion modes for: '%s'" % strain)
                
                    # read timeseries data for strain
                    ts_save_dir = save_dir / 'timeseries_data' / 'timeseries_{}.csv'.format(strain)
                    timeseries_strain = pd.read_csv(ts_save_dir)
     
                    # make colour palette for motion modes to plot
                    colour_dict = dict(zip(motion_mode_list, 
                                           sns.color_palette(sns_colour_palette, len(motion_mode_list))))

                    # initialise new figure for each strain
                    plt.close('all')
                    fig, ax = plt.subplots(figsize=(15,8))    
                       
                    legend_handles = []
                    for mode in mode_list:
                        
                        ax = plot_timeseries_motion_mode(df=timeseries_strain,
                                                         mode=mode,
                                                         error=error,
                                                         window=window,
                                                         max_n_frames=MAX_N_FRAMES,
                                                         title=None,
                                                         ax=ax,
                                                         saveAs=None,
                                                         sns_colour_palette=sns_colour_palette,
                                                         colour=colour_dict[mode])
                        
                        legend_handles.append(patches.Patch(color=colour_dict[mode], label=mode))
                        
                    ax.set_xticks(np.linspace(0, max_n_frames, num=7))
                    ax.set_xticklabels([str(int(l)) for l in np.linspace(0, max_n_frames, 
                                                                    num=7) / VIDEO_FPS])
                    
                    # save plot for strain
                    ax.set_ylabel('Fraction of worms ({})'.format(feature), labelpad=10, fontsize=15)
                    ax.set_xlabel("Time (seconds)")
                    plt.title("%s timeseries for '%s'" % (feature, strain))
                    plt.legend(handles=legend_handles, labels=colour_dict.keys(), loc='upper left',\
                           borderaxespad=0.4, frameon=False, fontsize=15)
                    if save_dir is not None:
                        saveAs = save_dir / 'timeseries_plots' / strain / '{}.pdf'.format(feature)
                        saveAs.parent.mkdir(parents=True, exist_ok=True)
                        plt.savefig(saveAs, dpi=300)
            

#%% Main
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot worm motion mode timeseries for videos in \
                                     filenames summaries file")
    # parser.add_argument('-f', '--filenames_summaries_path', help="Tierpsy filenames summaries path", 
    #                     default=FILENAMES_SUMMARIES_PATH, type=str)
    parser.add_argument('--metadata_path', 
                        help="Path to metadata file", type=str, default=EXAMPLE_METADATA_PATH)
    parser.add_argument('--strain_list_path', 
                        help="Path to text file with list of strains to plot", type=str,
                        default=EXAMPLE_STRAIN_LIST_PATH)
    parser.add_argument('--fset_path', 
                        help="Path to text file with list of features to plot (currently only \
                        'motion_mode' is supported!)", type=str, default=EXAMPLE_FEATURE_SET_PATH)
    parser.add_argument('--save_dir', 
                        help="Path to save timeseries plots", type=str, default=EXAMPLE_SAVE_DIR)
    args = parser.parse_args()
    
    if args.save_dir is None:
        args.save_dir = EXAMPLE_SAVE_DIR #Path(args.filenames_summaries_path).parent
    
    strain_list = EXAMPLE_STRAIN_LIST if args.strain_list_path is None else read_list_from_file(args.strain_list_path)
    fset = None if args.fset_path is None else read_list_from_file(args.fset_path)

    # plot timeseries of all motion modes together, for each strain separately
    print("Plotting timeseries for each strain:")
    plot_timeseries_from_metadata(metadata_path=args.metadata_path, 
                                  results_dir=EXAMPLE_RESULTS_DIR,
                                  group_by='gene_name',
                                  strain_list=strain_list, 
                                  control=EXAMPLE_CONTROL,
                                  fset=fset,
                                  save_dir=Path(args.save_dir) / 'timeseries',
                                  motion_mode='all', # 'forwards', 'backwards', 'stationary', 
                                  multi_strain=False,
                                  window=WINDOW,
                                  error=True,
                                  max_n_frames=MAX_N_FRAMES,
                                  sns_colour_palette='Greens')
    
    # plot timeseries of strain vs control, for each motion mode separately
    strain_list = ['fepB','fepD','fes','atpB','nuoC','sdhD','entA']
    for strain in strain_list:
        print("\nPlotting timeseries for: %s vs %s.." % (strain, EXAMPLE_CONTROL))
        plot_timeseries_from_metadata(metadata_path=args.metadata_path, 
                                      results_dir=EXAMPLE_RESULTS_DIR,
                                      group_by='gene_name',
                                      strain_list=[EXAMPLE_CONTROL, strain], 
                                      control=EXAMPLE_CONTROL,
                                      fset=fset,
                                      save_dir=Path(args.save_dir) / 'timeseries',
                                      motion_mode='all', # 'all','forwards', 'backwards', 'stationary'
                                      multi_strain=True,
                                      window=WINDOW,
                                      error=True,
                                      max_n_frames=MAX_N_FRAMES,
                                      sns_colour_palette='Greens')

