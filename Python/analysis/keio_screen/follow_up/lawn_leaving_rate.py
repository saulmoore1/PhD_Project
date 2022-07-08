#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keio screen lawn-leaving assay

@author: sm5911
@date: 25/06/22
"""

#%% Imports

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from time import time
from pathlib import Path
from matplotlib import path as mpath
from matplotlib import pyplot as plt

from filter_data.filter_trajectories import filter_worm_trajectories
from time_series.plot_timeseries import add_bluelight_to_plot

from tierpsytools.analysis.statistical_tests import bootstrapped_ci

#%% Globals

PROJECT_DIR = "/Users/sm5911/Documents/Keio_Supplements" # local

# Filter parameters
THRESHOLD_DURATION = 25 # threshold trajectory length (n frames) / 25 fps => 1 second
THRESHOLD_MOVEMENT = 10 # threshold movement (n pixels) * 12.4 microns per pixel => 124 microns
THRESHOLD_LEAVING_DURATION = 50 # n frames a worm has to leave food for to be called a true leaving event

FPS = 25
BLUELIGHT_TIMEPOINTS_SECONDS = [(30*60,30*60+10),(31*60,31*60+10),(32*60,32*60+10)]
SMOOTHING = 250

#%% Functions

def on_food(traj_df, poly_dict):
    
    for key, values in poly_dict.items():
        polygon = mpath.Path(values, closed=True)
        traj_df[key] = polygon.contains_points(traj_df[['x','y']])
    
    return traj_df

def leaving_events(df, 
                   on_food_col='Poly_0', 
                   threshold_n_frames=50): # 50 frames = 2 seconds (25fps)
    """ Calculate leaving events as when a worm midbody centroid leaves the annotated coordinates 
        of the lawn 
    """
    
    leaving_event_list = []
    
    wormIDs = df['worm_id'].unique()
    grouped_worm = df.groupby('worm_id')
    for worm in wormIDs:
        worm_df = grouped_worm.get_group(worm)
        
        _leave_mask = np.where(worm_df[on_food_col].astype(int).diff() == -1)[0]
        _enter_mask = np.where(worm_df[on_food_col].astype(int).diff() == 1)[0]
        leaving = worm_df.iloc[_leave_mask].index
        entering = worm_df.iloc[_enter_mask].index
        
        # if there is a leaving event
        if len(leaving) > 0:
            # if the worm does not return
            if len(entering) == 0:
                # compare to end of trajectory
                entering = np.array([worm_df.index[-1]])
            # if there is also an entering event
            elif len(entering) > 0:
                # if worm leaves/enters an equal number of times
                if len(leaving) == len(entering):
                    # if worm enters first, then leaves
                    if entering[0] < leaving[0]:
                        # ignore first entering event + compare leaving duration to end of trajectory 
                        # by adding end of trajectory index as last 'entering' event
                        entering = entering[1:]
                        entering = np.insert(entering, len(entering), worm_df.index[-1])
                    # if the worm returned to the food
                    elif leaving[0] < entering[0]:
                        pass
                # if worm leaves/enters an unequal number of times
                elif len(leaving) != len(entering):
                    # if worm leaves first
                    if leaving[0] < entering[0]:
                        # compare to end (add end of trajectory index as last 'entering event')
                        entering = np.insert(entering, len(entering), worm_df.index[-1])
                    # if worm enters first
                    elif entering[0] < leaving[0]:
                        # ignore first entering event
                        entering = entering[1:]
               
            # calculate leaving duration
            leaving_duration = entering - leaving
            
            # -1 to index the frame just before the leaving event to know which food it left from
            leaving_df = worm_df.loc[leaving - 1] 

            # append columns for leaving duration to leaving frame data for worm on food
            leaving_df['duration_n_frames'] = pd.Series(leaving_duration, 
                                                               index=leaving_df.index)
            # append as rows to out-dataframe
            leaving_event_list.append(leaving_df)
            
    # compile leaving events for all worms in video
    if len(leaving_event_list) == 0:
        leaving_events_df = None
    else:
        leaving_events_df = pd.concat(leaving_event_list, axis=0)
    
    # Filter for worms that left food for longer than threshold_n_frames (n frames after leaving)
    if threshold_n_frames is not None and leaving_events_df is not None:
        short_leaving_df = leaving_events_df[leaving_events_df['duration_n_frames'] < threshold_n_frames]
        print("Removing %d (%.1f%%) leaving events < %d frames" % (short_leaving_df.shape[0], 
              (short_leaving_df.shape[0]/leaving_events_df.shape[0])*100, threshold_n_frames))
        leaving_events_df = leaving_events_df[leaving_events_df['duration_n_frames'] >= threshold_n_frames]

    # TODO: could also filter by distance from food edge (spatial thresholding)

    return leaving_events_df

def fraction_on_food(metadata, 
                     food_coords_dir, 
                     threshold_duration=None, 
                     threshold_movement=None,
                     threshold_leaving_duration=50):
    """ Calculate the mean fraction of worms on food in each timestamp of each video (6-well only) """
    
    video_frac_path = Path(food_coords_dir) / 'video_fraction_on_food.csv'
    leaving_events_path = Path(food_coords_dir) / 'leaving_events.csv'
    
    if video_frac_path.exists() and leaving_events_path.exists():
        print("Found compiled information for the fraction of worms on food")
        video_frac_df = pd.read_csv(video_frac_path, header=0, index_col=0)
        leaving_events_df = pd.read_csv(leaving_events_path, header=0, index_col=0)
        
    else:
        coordsfilelist = [str(food_coords_dir) + s.split('RawVideos')[-1] + '/lawn_coordinates.csv' 
                          for s in metadata['filename']]
        
        # subset for files that have lawn coordinates annotated
        mask = [Path(file).exists() for file in coordsfilelist]
        n_coords = sum(mask)
        if n_coords < metadata.shape[0]:
            print("%d files found with no annotations" % (metadata.shape[0] - n_coords))
        
        metadata = metadata.loc[metadata.index[mask],:]
        maskedfilelist = [s.replace('RawVideos','MaskedVideos') + '/metadata.hdf5' for s in 
                          metadata['filename']]
        coordsfilelist = np.array(coordsfilelist)[mask].tolist()
    
        assert all(str(Path(i).parent).split('MaskedVideos')[-1] == 
                   str(Path(j).parent).split(str(food_coords_dir))[-1] 
                   for i, j in zip(maskedfilelist, coordsfilelist))
        
        video_frac_list = []
        leaving_events_list = []
        print("Calculating fraction on/off food")
        for i, (maskedfile, featurefile, coordsfile) in enumerate(tqdm(zip(
                maskedfilelist, metadata['featuresN_filename'], coordsfilelist), total=metadata.shape[0])):
            
            # if i == 49:
            #     break
            assert (str(Path(maskedfile).parent).split('MaskedVideos')[-1] == 
                    str(Path(coordsfile).parent).split(str(food_coords_dir))[-1])
            
            # load coordinates of food lawns (user labelled)
            f = open(coordsfile, 'r').read()
            poly_dict = eval(f) # use 'evaluate' to read as dictionary not string
            
            # load coordinates of worm trajectories
            with h5py.File(featurefile, 'r') as f:
                traj_df = pd.DataFrame({'x': f['trajectories_data']['coord_x'],
                                        'y': f['trajectories_data']['coord_y'],
                                        'frame_number': f['trajectories_data']['frame_number'],
                                        'worm_id': f['trajectories_data']['worm_index_joined']})
            # from tierpsytools.read_data.get_timeseries import read_timeseries
            # traj_df = read_timeseries(featurefile, names=None, only_wells=None) ### SLOW!
                    
            # filter based on thresholds of trajectory movement/duration
            traj_df, _stats = filter_worm_trajectories(traj_df,
                                                       threshold_move=threshold_movement, 
                                                       threshold_time=threshold_duration,
                                                       microns_per_pixel=12.4)
            # TODO: store stats / investigate number of bad worm trajectories?
            
            # compute whether each wormID in each timestamp is on or off food + append results
            traj_df = on_food(traj_df, poly_dict)
            
            # calculate fraction of worms on food in each timestamp
            grouped_frame = traj_df.groupby('frame_number')
            on_food_frac = grouped_frame['Poly_0'].sum() / grouped_frame['Poly_0'].count()
            on_food_frac.name = featurefile
            
            video_frac_list.append(on_food_frac)
            
            # compute leaving events for each wormID in video
            leaving_df = leaving_events(traj_df, threshold_n_frames=threshold_leaving_duration)

            if leaving_df is None:
                print("WARNING: Could not find leaving events for %s" % featurefile)
            else:
                # append file info
                leaving_df['masked_video_path'] = maskedfile
                leaving_events_list.append(leaving_df[
                    ['frame_number','worm_id','duration_n_frames','masked_video_path']])
                
        # save fraction of worms in each timestamp of each video to file
        video_frac_df = pd.concat(video_frac_list, axis=1)
        video_frac_df.to_csv(video_frac_path, index=True, header=True)
        
        # save leaving event information to file
        leaving_events_df = pd.concat(leaving_events_list, axis=0)
        leaving_events_df.to_csv(leaving_events_path, index=False, header=True)
            
    return video_frac_df, leaving_events_df

def leaving_histogram(leaving_events_df, threshold_leaving_duration=None, save_dir=None):
        
    # Get histogram bin positions prior to plotting
    bins = np.histogram(leaving_events_df['duration_n_frames'], bins=300)[1]

    if threshold_leaving_duration is not None:
        # filter dataframe and store filtered leaving events separately
        filtered_leaving_df = leaving_events_df[leaving_events_df['duration_n_frames'] < 
                                                threshold_leaving_duration]
        leaving_events_df = leaving_events_df[leaving_events_df['duration_n_frames'] >= 
                                              threshold_leaving_duration]
    
    # Plot histogram of leaving event durations + threshold for leaving event identification
    print("Plotting histogram of leaving durations..")
    plt.close("all")
    fig, ax = plt.subplots(figsize=(12,8), dpi=300)
    ax.hist(leaving_events_df['duration_n_frames'].values.astype(int), 
             bins=bins, color='skyblue')
    ax.hist(filtered_leaving_df['duration_n_frames'].values.astype(int), 
             bins=bins, color='gray', hatch='/')
    plt.rcParams['hatch.color'] = 'lightgray'
    ax.set_xlabel("Duration after leaving food (n frames)", fontsize=8, labelpad=10)
    ax.set_ylabel("Number of leaving events", fontsize=8, labelpad=10)
    
    # plot threshold for leaving event selection
    plt.xlim(0, leaving_events_df['duration_n_frames'].max()) # zoom-in on the very short leaving durations
    # plt.xticks(np.arange(0, threshold_leaving_duration*5+1, threshold_leaving_duration))
    plt.tick_params(labelsize=6)
    ax.axvline(threshold_leaving_duration, ls='--', lw=1, color='k')
    # from matplotlib import transforms    
    # trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) # transform y axis only
    ax.set_title("Threshold Leaving Duration = {0} frames".format(threshold_leaving_duration),
                 fontsize=5) # ha='left', va='top', rotation=-90, transform=trans
    plt.subplots_adjust(left=0.15, bottom=0.18, right=None, top=None)
    
    # save histogram
    if save_dir is not None:
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        save_path = Path(save_dir) / "leaving_duration_histogram.pdf"
        plt.savefig(save_path, format='pdf')
        plt.close()
    else:
        plt.show()

    return

def timeseries_on_food(metadata, 
                       group_by,
                       video_frac_df,
                       control='BW-none-nan',
                       save_dir=None, 
                       smoothing=None,
                       bluelight_frames=None,
                       palette='tab10',
                       error=True):
    
    assert video_frac_df.shape[0] == video_frac_df.index.nunique()
    
    timeseries_data_path = Path(save_dir) / 'timeseries_fraction_on_food.csv'
    
    if Path(timeseries_data_path).exists():
        timeseries_frac_df = pd.read_csv(timeseries_data_path, header=0, index_col=None)

    else:
        # group metadata by treatment + compute fraction on/off food
        grouped = metadata.groupby(group_by)

        group_frac_list = []
        
        for group in grouped.groups.keys():
            group_meta = grouped.get_group(group)
                            
            # video_frac_df = video_frac_df.iloc[:,~video_frac_df.columns.duplicated()]
            assert not any(video_frac_df.columns.duplicated())
            assert group_meta['featuresN_filename'].nunique() == group_meta['featuresN_filename'].shape[0]
    
            _mask = video_frac_df.columns.isin(group_meta['featuresN_filename'].unique())        
            cols = video_frac_df.columns[_mask].tolist()
            
            # mean + std fraction on food in each frame across videos for treatment group
            group_frac_df = video_frac_df[cols]
            mean = group_frac_df.mean(axis=1)         
            
            frac_mean = pd.DataFrame.from_dict({group_by:group, 'mean':mean}).reset_index()

            if error:
                try:
                    err = group_frac_df.apply(lambda x: bootstrapped_ci(x,func=np.mean,n_boot=100), axis=1)
                    lower, upper = [i[0] for i in err.values], [i[1] for i in err.values]
                    frac_mean = pd.DataFrame.from_dict({group_by:group, 'mean':mean, 
                                                        'lower':lower, 'upper':upper}).reset_index()
                except Exception as e:
                    print("WARNING:! Could not compute error for %s:\n%s" % (group, e))
                    
            group_frac_list.append(frac_mean)
            
        timeseries_frac_df = pd.concat(group_frac_list, axis=0)
        
        # save timeseries fraction to file
        timeseries_frac_df.to_csv(timeseries_data_path, header=True, index=False)
                    
    if smoothing:
        timeseries_frac_df = timeseries_frac_df.set_index(['frame_number', group_by]).rolling(
            window=smoothing, center=True).mean().reset_index()

    # plot timeseries for each group vs control
    grouped_timeseries = timeseries_frac_df.groupby(group_by)
    control_ts = grouped_timeseries.get_group(control)
    
    max_n_frames = timeseries_frac_df['frame_number'].max()
    groups_list = [i for i in grouped_timeseries.groups.keys() if i != control]
    
    print("Plotting timeseries fraction of worms on food...")
    for group in tqdm(groups_list):
        group_ts = grouped_timeseries.get_group(group)
    
        colour_dict = dict(zip([control, group], sns.color_palette(palette=palette, n_colors=2)))

        plt.close('all')
        fig, ax = plt.subplots(figsize=(15,6))
        
        sns.lineplot(x='frame_number', y='mean', data=control_ts, 
                     color=colour_dict[control], ax=ax, label=control)
        sns.lineplot(x='frame_number', y='mean', data=group_ts, 
                     color=colour_dict[group], ax=ax, label=group)

        if error:
            ax.fill_between(control_ts['frame_number'], 
                            control_ts['lower'], 
                            control_ts['upper'], 
                            color=colour_dict[control], edgecolor=None, alpha=0.25)
            ax.fill_between(group_ts['frame_number'], 
                            group_ts['lower'], 
                            group_ts['upper'], 
                            color=colour_dict[group], edgecolor=None, alpha=0.25)

        # add decorations
        if bluelight_frames is not None:
            ax = add_bluelight_to_plot(ax, bluelight_frames=bluelight_frames, alpha=0.25)
        
        ax.set_xlim(0, max_n_frames)
        xticks = [0,7500,15000,22500,30000,37500,45000,52500,60000]
        xticklabels = [0,5,10,15,20,25,30,35]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('Time (seconds)', labelpad=10)
        ax.set_ylabel('Fraction of worms feeding', labelpad=10)
        
        if save_dir:
            plt.savefig(Path(save_dir) / 'timeseries_fraction_on_{}.pdf'.format(group), dpi=300)
        
    return

def timeseries_leaving(metadata,
                       group_by,
                       leaving_events_df,
                       control='BW-none-nan',
                       save_dir=None,
                       bluelight_frames=None,
                       smoothing=None, # default 10-second binning window for smoothing
                       show_error=True):
    
    # # TODO: leaving rate timeseries
    # moving_count = df[food].rolling(smooth_window,center=True).sum()
    # ax.plot(moving_count, color=colour_dict[labels[i]], ls='-') # x=np.arange(xlim)

    return

def lawn_leaving_rate(metadata, 
                      food_coords_dir, 
                      group_by='treatment',
                      control='BW-none-nan',
                      threshold_duration=None, 
                      threshold_movement=None,
                      threshold_leaving_duration=50):
    
    print("Estimating time spent on vs off the lawn in each video..")
    video_frac_df, leaving_events_df = fraction_on_food(metadata, 
                                                        food_coords_dir, 
                                                        threshold_duration, 
                                                        threshold_movement,
                                                        threshold_leaving_duration=None)
    
    # filter leaving events (after having saved full leaving event data to file)
    if threshold_leaving_duration is not None:
        short_leaving_df = leaving_events_df[leaving_events_df['duration_n_frames'] < threshold_leaving_duration]
        print("Removing %d (%.1f%%) leaving events < %d frames" % (short_leaving_df.shape[0], 
              (short_leaving_df.shape[0]/leaving_events_df.shape[0])*100, threshold_leaving_duration))
        leaving_events_df = leaving_events_df[leaving_events_df['duration_n_frames'] >= threshold_leaving_duration]

    # plot histogram of leaving duration
    leaving_histogram(leaving_events_df, 
                      threshold_leaving_duration=threshold_leaving_duration, 
                      save_dir=food_coords_dir)
    
    timeseries_on_food(metadata,
                       group_by=group_by,
                       video_frac_df=video_frac_df,
                       control=control,
                       save_dir=food_coords_dir / 'timeseries',
                       bluelight_frames=[(i*FPS, j*FPS) for (i, j) in BLUELIGHT_TIMEPOINTS_SECONDS],
                       smoothing=SMOOTHING, # default 10-second binning window for smoothing
                       show_error=True)
    
    timeseries_leaving(metadata,
                       group_by=group_by,
                       leaving_events_df=leaving_events_df,
                       control=control,
                       save_dir=food_coords_dir / 'timeseries_leaving',
                       bluelight_frames=[(i*FPS, j*FPS) for (i, j) in BLUELIGHT_TIMEPOINTS_SECONDS],
                       smoothing=SMOOTHING, # default 10-second binning window for smoothing
                       show_error=True)
                
    return video_frac_df, leaving_events_df

#%% Main

if __name__ == '__main__':
    toc = time()

    metadata_path = Path(PROJECT_DIR) / "metadata.csv"
    food_coords_dir = Path(PROJECT_DIR) / "lawn_leaving"

    # load project metadata
    metadata = pd.read_csv(metadata_path, header=0, index_col=None)
    
    ##### SUPPLEMENT ANALYSIS #####
    if 'Supplements' in PROJECT_DIR:
        # subset for metadata for a single window (so there are no duplicate filenames)
        metadata = metadata[metadata['window']==0]

        # subset for paraquat results only
        metadata = metadata[metadata['drug_type'].isin(['paraquat','none'])]
        
        # treatment names for experiment conditions
        metadata['treatment'] = metadata[['food_type','drug_type','imaging_plate_drug_conc']
                                         ].astype(str).agg('-'.join, axis=1)
        
    video_frac_df, leaving_events_df = lawn_leaving_rate(metadata,
                                                         food_coords_dir=food_coords_dir,
                                                         group_by='treatment',
                                                         control='BW-none-nan',
                                                         threshold_movement=THRESHOLD_MOVEMENT,
                                                         threshold_duration=THRESHOLD_DURATION,
                                                         threshold_leaving_duration=THRESHOLD_LEAVING_DURATION)
    
    print("Done! (Time taken: %.1f seconds" % (time() - toc))
    
