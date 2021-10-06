#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check trajectory filtering for worms tracked on 'hit' strains selected from initial Keio screen, 
prior to follow-up analyses:
- How many trajectories were too short?
- How many trajectories did not move much?

@author: sm5911
@date: 30/09/2021

"""

#%% Imports

import pandas as pd
from pathlib import Path
from time import time
from tqdm import tqdm

from read_data.paths import get_save_dir
from read_data.read import read_list_from_file, load_json, load_topfeats, get_skeleton_data
from filter_data.filter_trajectories import filter_worm_trajectories

#%% Globals

JSON_PARAMETERS_PATH = "analysis/20210406_parameters_keio_screen.json"

STRAIN_LIST_PATH = "/Volumes/hermes$/KeioScreen_96WP/Analysis/52_selected_strains_from_initial_keio_top100_lowest_pval_tierpsy16.txt"

STIM_TYPE = "bluelight"

THRESH_TRAJ_DURATION = 25 # frames
THRESH_TRAJ_DISTANCE = 10 # pixels, 1 pixel = 12.4 microns

#%% Functions


#%% Main

if __name__ == "__main__":
    tic = time()

    args = load_json(JSON_PARAMETERS_PATH)
        
    # load features + metadata
    features = pd.read_csv(Path(args.save_dir) / 'features.csv')
    metadata = pd.read_csv(Path(args.save_dir) / 'metadata.csv', dtype={"comments":str})
     
    # record strain list
    assert not metadata['gene_name'].isna().any()
    all_strains = list(metadata['gene_name'].unique())
    
    # load hit strains
    hit_strains = read_list_from_file(STRAIN_LIST_PATH)
    assert all(strain in all_strains for strain in hit_strains)

    # load tierpsy top feature set + subset (columns) for top feats only
    if args.n_top_feats is not None:
        top_feats_path = Path(args.tierpsy_top_feats_dir) / "tierpsy_{}.csv".format(str(args.n_top_feats))        
        topfeats = load_topfeats(top_feats_path, add_bluelight=True, 
                                 remove_path_curvature=True, header=None)

        # Drop features that are not in results
        top_feats_list = [feat for feat in list(topfeats) if feat in features.columns]
        features = features[top_feats_list]

    save_dir = get_save_dir(args)

    grouped = metadata.join(features).groupby('gene_name')
    
    strain_skel_data = []
    col_names = ['short','stationary','total']

    for strain in tqdm(hit_strains):
        strain_data = grouped.get_group(strain)
        
        imgstore_name_stim = strain_data['imgstore_name_{}'.format(STIM_TYPE)].to_list()
        
        # maskedvideos / p / metadata.hdf5
        skeleton_file_list = [Path(args.project_dir) / 'Results' / p / 'metadata_skeletons.hdf5'
                              for p in imgstore_name_stim if type(p)==str]
        
        skeleton_data = pd.DataFrame(index=range(len(skeleton_file_list)), 
                                     columns=col_names)
        
        for i, skelfile in enumerate(skeleton_file_list):
            skeldata = get_skeleton_data(skelfile, rig='Hydra', dataset='trajectories_data')
            
            _, stats = filter_worm_trajectories(skeldata,
                                                threshold_move=THRESH_TRAJ_DISTANCE,
                                                threshold_time=THRESH_TRAJ_DURATION,
                                                fps=25,
                                                microns_per_pixel=12.4,
                                                worm_id_col='worm_index_joined',
                                                x_coord_col='coord_x', 
                                                y_coord_col='coord_y',
                                                verbose=False)
            # record trajectories that are too short or immobile
            skeleton_data.loc[i, 'short'] = stats['short_worm_trajectories']
            skeleton_data.loc[i, 'stationary'] = stats['stationary_worm_trajectories']
            skeleton_data.loc[i, 'total'] = stats['total_worm_trajectories']
            
        skeleton_data['gene_name'] = strain
        strain_skel_data.append(skeleton_data)
        
    strain_skel_data = pd.concat(strain_skel_data, axis=0)
    for col in col_names:
        strain_skel_data[col] = strain_skel_data[col].astype(float)
    
    mean_strain_skel = strain_skel_data.groupby('gene_name').mean()
    
    toc = time()
    print('Time taken: %.1f seconds' % (toc-tic))

