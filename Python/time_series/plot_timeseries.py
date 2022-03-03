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
from preprocessing.compile_keio_results import RENAME_DICT

#%% Globals

METADATA_PATH = '/Users/sm5911/Documents/Keio_Conf_Screen/metadata.csv'

RESULTS_DIR = '/Volumes/hermes$/KeioScreen2_96WP/Results'
#FILENAMES_SUMMARIES_PATH = '/Volumes/hermes$/KeioScreen2_96WP/Results/full_filenames.csv'

FEATURE_SET_PATH = '/Users/sm5911/Documents/Keio_Screen2/selected_features_timeseries.txt'

STRAIN_LIST_PATH = None #'/Users/sm5911/Documents/Keio_Screen/52_selected_strains_from_initial_keio_top100_lowest_pval_tierpsy16.txt'
STRAIN_LIST = ['wild_type','fepB','fepD','fes','atpB','nuoC','sdhD','entA'] # missing: 'trpA','trpD'

SAVE_DIR = '/Users/sm5911/Documents/Keio_Conf_Screen'

BLUELIGHT_FRAMES = [(1500,1751),(4000,4251),(6500,6751)]

CONTROL = 'wild_type'

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

def _bootstrapped_ci(x, function=np.mean, n_boot=100, which_ci=95, axis=None):
    """ Wrapper for tierpsytools bootstrapped_ci function, which encounters name space / 
        variable scope conflicts when used in combination with pandas apply function 
    """
    from tierpsytools.analysis.statistical_tests import bootstrapped_ci
    
    lower, upper = bootstrapped_ci(x, func=function, n_boot=n_boot, which_ci=which_ci, axis=axis)
    
    return lower, upper

# =============================================================================
# def plot_timeseries(filename, x, y, save_dir=None, window=100, n_frames_video=9000, **kwargs):
#     """ Timeseries plot of feature (y) throughout video """
#     
#     timeseries_data = get_skeleton_data(filename, rig='Hydra', dataset='timeseries_data')
#         
#     wells_list = list(timeseries_data['well_name'].unique())
# 
#     if not len(wells_list) == 16:
#         stem = Path(filename).parent.name
#         print("WARNING: Missing results for %d well(s): '%s'" % (16 - len(wells_list), stem))
#      
#     # get data for each well in turn
#     grouped_well = timeseries_data.groupby('well_name')
#     for well in wells_list:
#         well_data = grouped_well.get_group(well)
# 
#         xmax = max(n_frames_video, well_data[x].max())
# 
#         # frame average
#         grouped_frame = well_data.groupby(x)
#         well_mean = grouped_frame[y].mean()
#         well_std = grouped_frame[y].std()
#         
#         # moving average (optional)
#         if window:
#             well_mean = well_mean.rolling(window=window, center=True).mean()
#             well_std = well_std.rolling(window=window, center=True).std()
# 
#         colours = []
#         for mm in np.array(well_mean):
#             if np.isnan(mm):
#                 #colours.append('white')
#                 colours.append([255,255,255]) # white
#             elif int(mm) == 1:
#                 #colours.append('blue')
#                 colours.append([0,0,255]) # blue
#             elif int(mm) == -1:
#                 #colours.append('red')
#                 colours.append([255,0,0]) # red
#             else:
#                 #colours.append('grey')
#                 colours.append([128,128,128]) # gray
#         colours = np.array(colours) / 255.0
#                 
#         # cmap = plt.get_cmap('Greys', 3)
#         # cmap.set_under(color='red', alpha=0)
#         # cmap.set_over(color='blue', alpha=0)
#         
#         # Plot time series                
#         plt.close('all')
#         fig, ax = plt.subplots(figsize=(12,6))
# 
#         #sns.scatterplot(x=well_mean.index, y=well_mean.values, ax=ax) # hue=well_mean.index
#         ax.scatter(x=well_mean.index, y=well_mean.values, c=colours, ls='-', marker='.', **kwargs)
#         ax.set_xlim(0, xmax)
#         ax.axhline(0, 0, xmax, ls='--', marker='o') 
#         ax.fill_between(well_mean.index, well_mean-well_std, well_mean+well_std, 
#                         where=(well_mean > 0), facecolor='blue', alpha=0.5) # egdecolor=None
#         ax.fill_between(well_mean.index, well_mean-well_std, well_mean+well_std, 
#                         where=(well_mean < 0), facecolor='red', alpha=0.5) # egdecolor=None        
# 
#         ax = add_bluelight_to_plot(ax, alpha=0.5)
#         
#         # sns.scatterplot(x=x, y=y, data=timeseries_data, **kwargs)
#         if save_dir is not None:
#             Path(save_dir).mkdir(exist_ok=True, parents=True)
#             plt.savefig(Path(save_dir) / 'roaming_state_{}.png'.format(well))
#             plt.close()
#         else:
#             plt.show()
#         
#     return
# =============================================================================

# def plot_timeseries_turn(df, window=None, title=None, figsize=(12,6), ax=None):
#     """ """
#     turn_dict = {0:'straight', 1:'turn'}
#     df['turn_type'] = ['NA' if pd.isna(t) else turn_dict[int(t)] for t in df['turn']]


def plot_timeseries_motion_mode(df, window=None, error=False, mode=None, max_n_frames=None,
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
            assert type(mode) == str and mode in motion_modes
    
    assert all(c in df.columns for c in cols)

    # drop NaN data
    df = df.loc[~df['motion_mode'].isna(), cols]
     
    # map whether forwards, backwards or stationary motion in each frame
    df['motion_name'] = df['motion_mode'].map(motion_dict)
    assert not df['motion_name'].isna().any()

    # average number of worms (wormIDs) in each motion mode for each video/well/timestamp
    grouped_vid_frame = df.groupby([c for c in cols if c != 'motion_mode'])
    total_count = grouped_vid_frame['motion_mode'].count()
    motion_count = grouped_vid_frame['motion_name'].value_counts()
    frac_mode = motion_count / total_count
    frac_mode = frac_mode.reset_index(drop=None).rename(columns={0:'fraction'})
    
    # mean and bootstrap CI error for each timestamp
    mode_df = frac_mode[frac_mode['motion_name']==mode]
    mode_grouped_frame = mode_df.groupby('timestamp')
     
    # mean mode df
    df = mode_grouped_frame.mean().reset_index(drop=None)

    if error:           
        conf_ints = mode_grouped_frame['fraction'].apply(_bootstrapped_ci, function=np.mean, n_boot=100)
        lower_ci = [x[0] for x in conf_ints]   
        upper_ci = [x[1] for x in conf_ints]
        df['lower'] = lower_ci
        df['upper'] = upper_ci
    
    # crop timeseries data to standard video length (optional)
    if max_n_frames:
        df = df[df['timestamp'] <= max_n_frames]
    
    # moving average (optional)
    if window:
        df = df.set_index('timestamp').rolling(window=window, center=True).mean().reset_index()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # motion_ls_dict = dict(zip(motion_modes, ['-','--','-.']))                
    sns.lineplot(data=df, x='timestamp', y='fraction', ax=ax, 
                 ls='-', # motion_ls_dict[mode] if len(mode_list) > 1 else '-',
                 hue=None, #'motion_name' if colour is None else None, 
                 palette=None, #palette if colour is None else None,
                 color=colour)
    if error:
        ax.fill_between(df.index, df['lower'], df['upper'], color=colour, alpha=alpha, edgecolor=None)
    
    xmax = df['timestamp'].max()
    ax.set_xlim(0, xmax)
    #ax.set_ylim(0, 1)

    # add decorations
    if bluelight_frames is not None:
        ax = add_bluelight_to_plot(ax, bluelight_frames=bluelight_frames, alpha=alpha)
    # ax.axhline(0, 0, xmax, ls='--', marker='o')    
    if title:
        plt.title(title, pad=10)

    if saveAs is not None:
        Path(saveAs).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(saveAs)
    
    if ax is None:
        return fig, ax
    else:
        return ax
    
#def _timeseries_feature():
    
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
  
        elif feature != 'motion_mode':
            # TODO
            raise IOError("Only motion mode timeseries is supported!")
# =============================================================================
#         elif feature == 'turn':
#             # discrete data
#             #plot_timeseries_turn()
#         else:
#             # continuous data
#             ax = plot_timeseries(df=timeseries_strain,
#                                  x='timestamp',
#                                  y=feature,
#                                  window=WINDOW,
#                                  max_n_frames=MAX_N_FRAMES,
#                                  title=feature,
#                                  ax=ax,
#                                  saveAs=None,
#                                  sns_colour_palette='Greens',
#                                  colour=colour_dict[strain] if multi_strain else None)   
# =============================================================================

#%% Main
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot worm motion mode timeseries for videos in \
                                     filenames summaries file")
    # parser.add_argument('-f', '--filenames_summaries_path', help="Tierpsy filenames summaries path", 
    #                     default=FILENAMES_SUMMARIES_PATH, type=str)
    parser.add_argument('--metadata_path', 
                        help="Path to metadata file", type=str, default=METADATA_PATH)
    parser.add_argument('--strain_list_path', 
                        help="Path to text file with list of strains to plot", type=str,
                        default=STRAIN_LIST_PATH)
    parser.add_argument('--fset_path', 
                        help="Path to text file with list of features to plot (currently only \
                        'motion_mode' is supported!)", type=str, default=FEATURE_SET_PATH)
    parser.add_argument('--save_dir', 
                        help="Path to save timeseries plots", type=str, default=SAVE_DIR)
    args = parser.parse_args()
    
    if args.save_dir is None:
        args.save_dir = SAVE_DIR #Path(args.filenames_summaries_path).parent
    
    strain_list = STRAIN_LIST if args.strain_list_path is None else read_list_from_file(args.strain_list_path)
    fset = None if args.fset_path is None else read_list_from_file(args.fset_path)

    # plot timeseries of all motion modes together, for each strain separately
    print("Plotting timeseries for each strain:")
    plot_timeseries_from_metadata(metadata_path=args.metadata_path, 
                                  results_dir=RESULTS_DIR,
                                  group_by='gene_name',
                                  strain_list=strain_list, 
                                  control=CONTROL,
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
        print("\nPlotting timeseries for: %s vs %s.." % (strain, CONTROL))
        plot_timeseries_from_metadata(metadata_path=args.metadata_path, 
                                      results_dir=RESULTS_DIR,
                                      group_by='gene_name',
                                      strain_list=[CONTROL, strain], 
                                      control=CONTROL,
                                      fset=fset,
                                      save_dir=Path(args.save_dir) / 'timeseries',
                                      motion_mode='all', # 'all','forwards', 'backwards', 'stationary'
                                      multi_strain=True,
                                      window=WINDOW,
                                      error=True,
                                      max_n_frames=MAX_N_FRAMES,
                                      sns_colour_palette='Greens')
