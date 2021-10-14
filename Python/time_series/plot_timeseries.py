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

from read_data.read import get_skeleton_data, read_list_from_file

#%% Globals

FILENAMES_SUMMARIES_PATH = ('/Volumes/hermes$/KeioScreen2_96WP/Results/20210928/' +
                            'filenames_summary_tierpsy_plate_20211004_140945.csv')
METADATA_PATH = '/Users/sm5911/Documents/Keio_Conf_Screen/metadata.csv'
FEATURE_SET_PATH = '/Volumes/hermes$/KeioScreen2_96WP/AuxiliaryFiles/selected_features_timeseries.txt'
STRAIN_LIST_PATH = '/Volumes/hermes$/KeioScreen2_96WP/AuxiliaryFiles/selected_strains_timeseries.txt'
SAVE_DIR = '/Users/sm5911/Documents/Keio_Conf_Screen'

BLUELIGHT_FRAMES = [(1500,1751),(4000,4251),(6500,6751)]

# plot multiple strains on the same plot?
MULTI_STRAIN = True

#%% Function

def add_bluelight_to_plot(ax, bluelight_frames=BLUELIGHT_FRAMES, alpha=0.75):
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

def plot_timeseries(filename, x, y, save_dir=None, window=100, n_frames_video=9000, **kwargs):
    """ Timeseries plot of feature (y) throughout video """
    
    timeseries_data = get_skeleton_data(filename, rig='Hydra', dataset='timeseries_data')
        
    wells_list = list(timeseries_data['well_name'].unique())

    if not len(wells_list) == 16:
        stem = Path(filename).parent.name
        print("WARNING: Missing results for %d well(s): '%s'" % (16 - len(wells_list), stem))
     
    # get data for each well in turn
    grouped_well = timeseries_data.groupby('well_name')
    for well in wells_list:
        well_data = grouped_well.get_group(well)

        xmax = max(n_frames_video, well_data[x].max())

        # frame average
        grouped_frame = well_data.groupby(x)
        well_mean = grouped_frame[y].mean()
        well_std = grouped_frame[y].std()
        
        # moving average (optional)
        if window:
            well_mean = well_mean.rolling(window=window, center=True).mean()
            well_std = well_std.rolling(window=window, center=True).std()

        colours = []
        for mm in np.array(well_mean):
            if np.isnan(mm):
                #colours.append('white')
                colours.append([255,255,255]) # white
            elif int(mm) == 1:
                #colours.append('blue')
                colours.append([0,0,255]) # blue
            elif int(mm) == -1:
                #colours.append('red')
                colours.append([255,0,0]) # red
            else:
                #colours.append('grey')
                colours.append([128,128,128]) # gray
        colours = np.array(colours) / 255.0
                
        # cmap = plt.get_cmap('Greys', 3)
        # cmap.set_under(color='red', alpha=0)
        # cmap.set_over(color='blue', alpha=0)
        
        # Plot time series                
        plt.close('all')
        fig, ax = plt.subplots(figsize=(12,6))

        #sns.scatterplot(x=well_mean.index, y=well_mean.values, ax=ax) # hue=well_mean.index
        ax.scatter(x=well_mean.index, y=well_mean.values, c=colours, ls='-', marker='.', **kwargs)
        ax.set_xlim(0, xmax)
        ax.axhline(0, 0, xmax, ls='--', marker='o') 
        ax.fill_between(well_mean.index, well_mean-well_std, well_mean+well_std, 
                        where=(well_mean > 0), facecolor='blue', alpha=0.5) # egdecolor=None
        ax.fill_between(well_mean.index, well_mean-well_std, well_mean+well_std, 
                        where=(well_mean < 0), facecolor='red', alpha=0.5) # egdecolor=None        

        ax = add_bluelight_to_plot(ax)
        
        # sns.scatterplot(x=x, y=y, data=timeseries_data, **kwargs)
        if save_dir is not None:
            Path(save_dir).mkdir(exist_ok=True, parents=True)
            plt.savefig(Path(save_dir) / 'roaming_state_{}.png'.format(well))
            plt.close()
        else:
            plt.show()
        
    return

def plot_timeseries_from_filenames_summaries(filenames_summaries_path, metadata, strain_list=None, 
                                             fset=None, saveDir=None):
    """ """

    assert Path(args.filenames_summaries_path).exists()
    filenames_summaries = pd.read_csv(args.filenames_summaries_path, comment="#")
    filenames_list = filenames_summaries.loc[filenames_summaries['is_good'],'filename'].to_list()

    for filename in tqdm(filenames_list):
        stem = Path(filename).parent.name
        print("\nPlotting timeseries for '%s'" % stem)
        
        for feature in fset:
            print("\t%s" % feature)
            plot_timeseries(filename=filename, 
                            x='timestamp', 
                            y=feature, 
                            saveDir=args.save_dir / feature / stem, 
                            window=None)      
        break
      
def plot_timeseries_turn(df, window=None, title=None, figsize=(12,6), ax=None):
    """ """
    turn_dict = {0:'straight', 1:'turn'}
    df['turn_type'] = ['NA' if pd.isna(t) else turn_dict[int(t)] for t in df['turn']]

#%%
def plot_timeseries_motion_mode(df, window=None, mode=None, n_frames_video=None,
                                title=None, figsize=(12,6), ax=None, saveAs=None):
    """ """
    
    # discrete data mapping 
    motion_modes = ['stationary','forwards','backwards']
    motion_dict = dict(zip([0,1,-1], motion_modes))
    
    assert all(c in df.columns for c in ['timestamp','motion_mode'])

    # drop NaN data
    df = df.loc[~df['motion_mode'].isna(), ['timestamp','motion_mode']]
     
    # record whether forwards, backwards or stationary motion in each frame
    df['motion_name'] = df['motion_mode'].map(motion_dict)

    grouped_frame = df.groupby(['timestamp'])

    # timestamp average number of worms (wormIDs) in each motion mode
    mean_df = df.groupby(['timestamp','motion_name']).std()
    mean_df = grouped_frame['motion_mode'].mean()
    std_df = grouped_frame['motion_mode'].std()
    total_count = grouped_frame['motion_mode'].count()
    motion_count = grouped_frame['motion_name'].value_counts()
    frac_mode = motion_count / total_count
    frac_mode = frac_mode.reset_index(drop=None)
    frac_mode = frac_mode.rename(columns={0 :'fraction'})

    # crop timeseries data to standard video length (optional)
    if n_frames_video:
        # mean_df = mean_df[mean_df['timestamp'] <= n_frames_video]
        # std_df = std_df[std_df['timestamp'] <= n_frames_video]
        frac_mode = frac_mode[frac_mode['timestamp'].isin(frac_mode.index[:n_frames_video+1])]

    if mode is None or mode == 'all':
        mode_list = motion_modes
    elif type(mode) == float or type(mode) == int:
        # get mode name
        mode = motion_dict[int(mode)]
        mode_list = [mode]
    elif type(mode) == list:
        assert all(m in motion_modes for m in mode)
        mode_list = mode
    else:
        assert type(mode) == str and mode in motion_modes
        mode_list = [mode]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # plot timeseries for each mode
    grouped_mode = frac_mode.groupby('motion_name')
    for mode in mode_list:
        df = grouped_mode.get_group(mode)
        
        
        # drop motion_name column
        df = df.reset_index(drop=True)
        
        # moving average (optional)
        if window:
            df = df.set_index(['timestamp','motion_name']).rolling(window=window, 
                                                                   center=True).mean().reset_index()
            std_df = std_df.rolling(window=window, center=True).std()
        
        # create colour palette for plot
        palette = dict(zip(['stationary','forwards','backwards'], 
                           sns.color_palette(palette='pastel', n_colors=3)))
                    
        sns.lineplot(data=df, x='timestamp', y='fraction', ls='-', ax=ax, 
                     hue='motion_name', palette=palette) #color=colour_dict[strain]
        
        xmax = df['timestamp'].max()
        ax.set_xlim(0, xmax)

        # add decorations
        ax.axhline(0, 0, xmax, ls='--', marker='o')    
        ax = add_bluelight_to_plot(ax)
        ax.fill_between(mean_df.index, mean_df-std_df, mean_df+std_df, 
                        where=(mean_df > 0), facecolor='lightblue', alpha=0.5) # egdecolor=None
        ax.fill_between(mean_df.index, mean_df-std_df, mean_df+std_df, 
                        where=(mean_df < 0), facecolor='pink', alpha=0.5) # egdecolor=None
        plt.title(title, pad=10)

    # TODO: plot each strain as separate linetype? or colour_dict to specify line colour for each strain plot
    # TODO: map plot colours for points <, = and > 0 as red, grey, and blue respectively
    
    if saveAs is not None:
        Path(saveAs).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(saveAs)
    
    if ax is None:
        return fig, ax
    else:
        return ax
    
def plot_timeseries_from_metadata(metadata_path, 
                                  group_by='gene_name', 
                                  strain_list=None, 
                                  fset=['motion_mode'],
                                  motion_mode='all', # 'forwards', 'backwards', 'stationary', 
                                  save_dir=None):
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

    metadata = pd.read_csv(args.metadata_path, dtype={"comments":str, "source_plate_id":str})
 
    # Subset results (rows) to remove entries for wells with unknown strain data for 'gene_name'
    n = metadata.shape[0]
    metadata = metadata.loc[~metadata[group_by].isna(),:]
    if (n - metadata.shape[0]) > 0:
        print("%d entries removed with no gene name metadata" % (n - metadata.shape[0]))

    if strain_list is None:
        strain_list = metadata[group_by].unique()
    else:
        assert type(strain_list)==list and all(s in metadata[group_by].unique() for s in strain_list)

    aux_dir = metadata.loc[0,'filename'].split('/RawVideos/')[0] + '/AuxiliaryFiles'
    res_dir = Path(str(aux_dir).replace('/AuxiliaryFiles','/Results'))
    save_dir = res_dir if save_dir is None else Path(save_dir)
                            
    grouped = metadata.groupby(group_by)
    
    ### Compiling timeseries results for each strain + saving
    
    for strain in tqdm(strain_list):
        
        ts_save_dir = save_dir / 'timeseries_data' / 'timeseries_{}.csv'.format(strain)
        
        if not ts_save_dir.exists():
            strain_df = grouped.get_group(strain)
            
            # drop bad well samples
            strain_df = strain_df[~strain_df['is_bad_well']]
            
            # process bluelight videos only
            is_bluelight = ['bluelight' in i for i in strain_df.imgstore]
            strain_df = strain_df.reindex(strain_df.index[is_bluelight])
            
            # find paths to featuresN files
            featuresN_list = [str(res_dir / str(d) / i / 'metadata_featuresN.hdf5') for d, i in 
                              zip(strain_df['date_yyyymmdd'], strain_df['imgstore'])] 
            strain_df['featuresN_path'] = featuresN_list
            
            # read all bluelight video timeseries data for strain
            timeseries_strain_data = []
            for i, (file, well) in enumerate(zip(strain_df['featuresN_path'], strain_df['well_name'])):
                
                # load video timeseries data
                df = get_skeleton_data(file, rig='Hydra', dataset='timeseries_data')
                
                # subset for well data
                df = df[df['well_name'] == well]
                
                # append filename info
                df['filename'] = file
                
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
        if MULTI_STRAIN:
            plt.close('all')
            fig, ax = plt.subplots(figsize=(15,8))

        for strain in tqdm(strain_list):
            print("\nPlotting %s for %s" % (feature, strain))
            
            # read timeseries data for strain
            ts_save_dir = save_dir / 'timeseries_data' / 'timeseries_{}.csv'.format(strain)
            timeseries_strain = pd.read_csv(ts_save_dir)
 
            if not MULTI_STRAIN:
                plt.close('all')
                fig, ax = plt.subplots(figsize=(15,8))
                       
            if feature == 'motion_mode':
                ax = plot_timeseries_motion_mode(df=timeseries_strain,
                                                 mode=motion_mode,
                                                 window=100,
                                                 n_frames_video=9000,
                                                 title="Motion mode: '%s'" % str(motion_mode),
                                                 ax=ax,
                                                 saveAs=None)
            elif feature == 'turn':
                # discrete data
                #plot_timeseries_turn()
                pass
            else:
                ax = plot_timeseries(df=timeseries_strain,
                                     x='timestamp',
                                     y=feature,
                                     window=100,
                                     n_frames_video=9000,
                                     title=feature,
                                     ax=ax,
                                     saveAs=None)   
                
            # save plots separately if plotting each strain separately
            if not MULTI_STRAIN:
                saveAs = save_dir / 'timeseries_plots' / strain / '{}.pdf'.format(feature)
                saveAs.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(saveAs, dpi=300)
                
        # save single plot if plotting multiple strains together  
        if MULTI_STRAIN:
            saveAs = save_dir / 'timeseries_plots' / '{}.pdf'.format(feature)
            saveAs.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(saveAs, dpi=300)            
    
#%% Main
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot worm motion mode timeseries for videos in \
                                     filenames summaries file")
    parser.add_argument('-f', '--filenames_summaries_path', help="Tierpsy filenames summaries path", 
                        default=FILENAMES_SUMMARIES_PATH, type=str)
    parser.add_argument('--metadata_path', help="Path to metadata file", 
                        default=METADATA_PATH, type=str)
    parser.add_argument('--strain_list_path', help="Path to text file with list of strains to plot",
                        default=STRAIN_LIST_PATH, type=str)
    parser.add_argument('--fset_path', help="Path to text file with list of features to plot", 
                        default=FEATURE_SET_PATH, type=str)
    parser.add_argument('--save_dir', help="Path to save timeseries plots", 
                        default=SAVE_DIR, type=str)
    args = parser.parse_args()
    
    if args.save_dir is None:
        args.save_dir = SAVE_DIR #Path(args.filenames_summaries_path).parent
    
    strain_list = None if args.strain_list_path is None else read_list_from_file(args.strain_list_path)
    fset = None if args.fset_path is None else read_list_from_file(args.fset_path)
    
    # plot_timeseries_from_filenames_summaries(filenames_summaries_path=args.filenames_summaries_path, 
    #                                          fset=fset, 
    #                                          save_dir=args.save_dir)

    plot_timeseries_from_metadata(metadata_path=args.metadata_path, 
                                  group_by='gene_name',
                                  strain_list=strain_list, 
                                  fset=fset,
                                  motion_mode='stationary', # 'forwards', 'backwards', 'stationary', 
                                  save_dir=Path(args.save_dir) / 'timeseries')
    
