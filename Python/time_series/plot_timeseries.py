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
METADATA_PATH = '/Volumes/hermes$/KeioScreen2_96WP/AuxiliaryFiles/20210928/20210928_day_metadata.csv'
FEATURE_SET_PATH = '/Volumes/hermes$/KeioScreen2_96WP/AuxiliaryFiles/selected_features_timeseries.txt'
STRAIN_LIST_PATH = '/Volumes/hermes$/KeioScreen2_96WP/AuxiliaryFiles/selected_strains_timeseries.txt'
SAVE_DIR = '/Users/sm5911/Documents/Keio_Conf_Screen'

BLUELIGHT_FRAMES = [(1500,1751),(4000,4251),(6500,6751)]

#%% Function

def add_bluelight_to_plot(fig, ax, bluelight_frames=BLUELIGHT_FRAMES, alpha=0.75):
    """ Add lines to plot to indicate video frames where bluelight stimulus was delivered 
    
        Inputs
        ------
        fig, ax : figure and axes from plt.subplots()
        bluelight_frames : list of tuples (start, end) 
    """
    assert type(bluelight_frames) == list or type(bluelight_frames) == tuple
    if not type(bluelight_frames) == list:
        bluelight_frames = [bluelight_frames]
    
    for bt in bluelight_frames:
        (start, stop) = bt       
        ax.axvspan(start, stop, facecolor='blue', alpha=alpha)
     
    return fig, ax

def plot_timeseries_hydra(filename, x, y, save_dir=None, window=100, n_frames_video=9000, **kwargs):
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
        # TODO: hue='worm_index'

        fig, ax = add_bluelight_to_plot(fig, ax)
        
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
            plot_timeseries_hydra(filename=filename, 
                                  x='timestamp', 
                                  y=feature, 
                                  saveDir=args.save_dir / feature / stem, 
                                  window=None)      
        break
    
def plot_timeseries(df, x='timestamp', y='motion_mode', window=None, title=None, figsize=(12,6), 
                    ax=None):
    """ 
    Inputs
    ------
    df : pd.DataFrame
        pandas dataframe of compiled timeseries results for a given sample
    x : str
        column name of x variable for timeseries plot (default='timestamp')
    y : str
        column name of y variable, ie. feature. Motion mode is plotted by default
    """
    
    # average feature data for strain at each timestamp
    grouped_timestamp = df.groupby(x)
    
    df_mean = grouped_timestamp.mean() # mean of sample across all recordings: plates/wells
    df_std = grouped_timestamp.std() # standard deviation to compute error

    # crop timeseries data to standard length (optional)
    if n_frames_video:
        df = df[df[x] <= n_frames_video]
    
    if y == 'motion_mode':
        # discrete data   
        motion_dict = {-1:'backwards', 0:'paused', 1:'forwards'}
    elif y == 'turn':
        # discrete data
        turn_dict = {0:'straight', 1:'turn'}
        df['turn_type'] = ['NA' if pd.isna(t) else turn_dict[int(t)] for t in df['turn']]
        y = 'turn_type' 
        
    # plot timeseries
    def _plot_timeseries(df, x, y, ax, figsize):
        """ """
        if ax is None:              
            fig, ax = plt.subplots(figsize=figsize)


        return fig, ax
     
    if average_wells:
        _plot_timeseries(df, x, y, ax=ax)
    else:      
        wells_list = list(df['well_name'].unique())
        
        # get data for each well in turn
        grouped_well = df.groupby('well_name')
        for well in wells_list:
            well_data = grouped_well.get_group(well)
            
            _plot_timeseries(df, x, y, ax=ax)

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
        if ax is None:              
            fig, ax = plt.subplots(figsize=figsize)

        #sns.scatterplot(x=well_mean.index, y=well_mean.values, ax=ax) # hue=well_mean.index
        ax.scatter(x=well_mean.index, y=well_mean.values, c=colours, ls='-', marker='.', **kwargs)
        ax.set_xlim(0, xmax)
        ax.axhline(0, 0, xmax, ls='--', marker='o') 
        ax.fill_between(well_mean.index, well_mean-well_std, well_mean+well_std, 
                        where=(well_mean > 0), facecolor='lightblue', alpha=0.5) # egdecolor=None
        ax.fill_between(well_mean.index, well_mean-well_std, well_mean+well_std, 
                        where=(well_mean < 0), facecolor='red', alpha=0.5) # egdecolor=None        
        # TODO: hue='worm_index'

        fig, ax = add_bluelight_to_plot(fig, ax)
        
        # sns.scatterplot(x=x, y=y, data=timeseries_data, **kwargs)
        if save_dir is not None:
            Path(save_dir).mkdir(exist_ok=True, parents=True)
            plt.savefig(Path(save_dir) / 'roaming_state_{}.png'.format(well))
            plt.close()
        else:
            plt.show()       
  
def plot_timeseries_turn(df, window=None, title=None, figsize=(12,6), ax=None):
    """ """
    turn_dict = {0:'straight', 1:'turn'}
    df['turn_type'] = ['NA' if pd.isna(t) else turn_dict[int(t)] for t in df['turn']]

def plot_timeseries_motion_mode(df, window=None, title=None, figsize=(12,6), ax=None):
    """ """
    
    
  
def plot_timeseries_from_metadata(metadata_path, 
                                  group_by='gene_name', 
                                  strain_list=None, 
                                  fset=['motion_mode'],
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
    
    aux_dir = Path(metadata_path).parent
    res_dir = Path(str(aux_dir).replace('/AuxiliaryFiles/','/Results/'))
    save_dir = res_dir if save_dir is None else Path(save_dir)
    
    metadata = pd.read_csv(args.metadata_path, dtype={"comments":str, "source_plate_id":str})
        
    if strain_list is None:
        strain_list = metadata[group_by].unique()
    else:
        assert type(strain_list)==list and all(s in metadata[group_by].unique() for s in strain_list)
        
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
            featuresN_list = [str(res_dir / i / 'metadata_featuresN.hdf5') 
                              for i in strain_df['imgstore']] 
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

    ### Plotting timeseries for each strain
    print("Plotting timeseries")
    for strain in tqdm(strain_list):
        
        # read timeseries data for strain
        ts_save_dir = save_dir / 'timeseries_data' / 'timeseries_{}.csv'.format(strain)
        timeseries_strain = pd.read_csv(ts_save_dir)
        
        for feature in fset:
            if feature == 'motion_mode':
                plot_timeseries_motion_mode()
            elif feature == 'turn':
                # discrete data
                plot_timeseries_turn()
                        
            
            plot_timeseries(df=timeseries_strain,
                            x='timestamp',
                            y=feature,
                            window=100,
                            n_frames_video=9000,
                            save_dir=save_dir)    
    
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
                                  save_dir=Path(args.save_dir) / 'timeseries')
    
