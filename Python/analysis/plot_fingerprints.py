#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Tierpsy Fingerprints

@author: sm5911
@date: 3/6/21

"""

#%% IMPORTS

import argparse
import numpy as np
import pandas as pd
from time import time 
from pathlib import Path

from read_data.read import load_json #load_topfeats
from statistical_testing.perform_keio_stats import average_control_keio
from tierpsytools.analysis.fingerprints import tierpsy_fingerprints

#%% GLOBALS

JSON_PARAMETERS_PATH = "analysis/20210406_parameters_keio_screen.json"
CLUSTERS_PATH = ("/Users/sm5911/Tierpsy_Versions/tierpsy-tools-python" + 
                 "/examples/fingerprints/cluster_features/feature_clusters.csv")
FEATURES_PATH = "/Users/sm5911/Documents/Keio_Screen/features.csv"
METADATA_PATH = "/Users/sm5911/Documents/Keio_Screen/metadata.csv"
FEATURES_NO_ALIGN_PATH = "/Users/sm5911/Documents/Keio_Screen/features_align_blue=False.csv"
METADATA_NO_ALIGN_PATH = "/Users/sm5911/Documents/Keio_Screen/metadata_align_blue=False.csv"

#%% FUNCTIONS

def cluster_analysis(features, n_clusters=100, saveDir=None):
    """ Cluster analysis of features for Tierpsy fingerprints """

    from scipy.stats import zscore
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.metrics import pairwise_distances_argmin_min
    
    # Z-normalise the features data
    featZ = features.apply(zscore, axis=0)

    # Drop features with NaN values after normalising
    n_cols = len(featZ.columns)
    featZ.dropna(axis=1, inplace=True)
    n_dropped = n_cols - len(featZ.columns)
    if n_dropped > 0:
        print("Dropped %d features after normalisation (NaN)" % n_dropped)

    ### Cluster analysis
    
    column_linkage = linkage(featZ.T, method='complete', metric='correlation')
    n_clusters = min(n_clusters, len(featZ.columns))
    clusters = fcluster(column_linkage, n_clusters, criterion='maxclust')
    un,n = np.unique(clusters, return_counts=True)
    cluster_centres = (featZ.T).groupby(by=clusters).mean() # get cluster centres
    
    # get index of closest feature to cluster centre
    central, _ = pairwise_distances_argmin_min(cluster_centres, featZ.T, metric='cosine')
    assert(np.unique(central).shape[0] == n_clusters)
    central = featZ.columns.to_numpy()[central]
    
    # make cluster dataframe
    df = pd.DataFrame(index=featZ.columns, columns=['group_label', 'stat_label', 'motion_label'])
    df['group_label'] = clusters
    stats = np.array(['10th', '50th', '90th', 'IQR'])
    df['stat_label'] = [np.unique([x for x in stats if x in ft]) for ft in df.index]
    df['stat_label'] = df['stat_label'].apply(lambda x: x[0] if len(x)==1 else np.nan)
    motions = np.array(['forward', 'backward', 'paused'])
    df['motion_label'] = [[x for x in motions if x in ft] for ft in df.index]
    df['motion_label'] = df['motion_label'].apply(lambda x: x[0] if len(x)==1 else np.nan)
    df['representative_feature'] = False
    df.loc[central, 'representative_feature'] = True
    df = df.fillna('none')
    
    # save feature clusters results
    if saveDir is not None:
        Path(saveDir).mkdir(exist_ok=True, parents=True)
        df.to_csv(saveDir / 'feature_clusters.csv', index=True) # save cluster df to file        
    
    return df

def plot_fingerprints(features, metadata, group_by, control, saveDir, strain_list=None, 
                      clusters_path=None, n_fingers=None):
    """ Do not align features by bluelight stimulus """
    
    if clusters_path is not None and Path(clusters_path).exists():
        clusters = pd.read_csv(clusters_path, index_col=0)

    if strain_list is not None:
        assert all(strain in metadata[group_by] for strain in strain_list)
    else:
        strain_list = list(metadata[group_by].unique())
    
    ### Fingerprints
    
    fingerprint_dir = Path(saveDir) / 'fingerprints'  
    n_fingers = len(strain_list) if n_fingers is None else n_fingers
            
    fingers = {}
    for group in strain_list[:n_fingers]:
        if group == control:
            continue
        if not metadata[metadata[group_by]==group].shape[0] > 1:
            print("skipping %s (n=1)" % group)
            continue

        print('Getting fingerprint of %s: %s' % (group_by, group))
        (fingerprint_dir / group).mkdir(exist_ok=True, parents=True)
        mask = metadata[group_by].isin([control, group])
        finger = tierpsy_fingerprints(bluelight=True, 
                                      test='Mann-Whitney', 
                                      multitest_method='fdr_by',
                                      significance_threshold=0.05, 
                                      groups=(clusters if clusters_path is not None else None), 
                                      test_results=None)
    
        # Fit the fingerprint (run univariate tests and create the profile)
        finger.fit(features[mask], metadata.loc[mask, group_by], control=control)
        
        # Plot and save the fingerprint for this strain
        print("Plotting fingerprint")
        finger.plot_fingerprints(merge_bluelight=False, #feature_names_as_xticks=True,
                                  saveto=(fingerprint_dir / group /'fingerprint.png'))
    
        # Plot and save boxplots for all the representative features
        print("Plotting boxplots")
        finger.plot_boxplots(features[mask],
                             metadata.loc[mask, group_by], 
                             (fingerprint_dir / group), 
                             control=control)
        
        fingers[group] = finger # Store the fingerprint object
    
    return fingers

#%% MAIN

if __name__ == "__main__":
    tic = time()
    parser = argparse.ArgumentParser(description="Read clean features and etadata and find 'hit' \
                                                  Keio knockout strains that alter worm behaviour")
    parser.add_argument('-j', '--json', help="Path to JSON parameters file", 
                        default=JSON_PARAMETERS_PATH, type=str)
    parser.add_argument('--features_path', help="Path to feature summaries file", 
                        default=FEATURES_PATH, type=str)
    parser.add_argument('--metadata_path', help="Path to metadata file", 
                        default=METADATA_PATH, type=str)
    parser.add_argument('--clusters_path', help="Path to cluster analysis results (for clustering only)", 
                        default=CLUSTERS_PATH, type=str)
    parser.add_argument('--features_no_align_path', help=("Path to feature summaries file " + 
                        "(no align bluelight, for clustering only)"), 
                        default=FEATURES_NO_ALIGN_PATH, type=str)
    parser.add_argument('--metadata_no_align_path', help=("Path to metadata file " + 
                        "(no align bluelight, for clustering only)"), 
                        default=METADATA_NO_ALIGN_PATH, type=str)

    args = parser.parse_args()
    
    args = load_json(args.json)
        
    # Read clean feature summaries + metadata
    print("Loading metadata and feature summary results...")
    features = pd.read_csv(FEATURES_PATH)
    metadata = pd.read_csv(METADATA_PATH, dtype={'comments':str, 'source_plate_id':str})
    # TODO: do not align features for bluelight conditions for cluster_analysis
    
    # # Load Tierpsy Top feature set + subset (columns) for top feats only
    # if args.n_top_feats is not None:
    #     top_feats_path = Path(args.tierpsy_top_feats_dir) / "tierpsy_{}.csv".format(str(args.n_top_feats))        
    #     topfeats = load_topfeats(top_feats_path, add_bluelight=True, 
    #                              remove_path_curvature=True, header=None)

    #     # Drop features that are not in results
    #     top_feats_list = [feat for feat in list(topfeats) if feat in features.columns]
    #     features = features[top_feats_list]

    if args.collapse_control:
        print("Collapsing control data (mean of each day)")
        features, metadata = average_control_keio(features, metadata)
                            
    # Record mean sample size per group
    mean_sample_size = int(np.round(metadata.join(features).groupby([args.grouping_variable], 
                                                                    as_index=False).size().mean()))
    print("Mean sample size: %d" % mean_sample_size)

    # plot tierpsy fingerprints (no clusters)
    fingers = plot_fingerprints(features, metadata, 
                                group_by='gene_name', 
                                control='wild_type', 
                                saveDir=args.save_dir, 
                                clusters_path=None,
                                n_fingers=10)

    ## Compile and clean feature summaries (no align bluelight)
    #clusters_path = Path(args.save_dir) / 'feature_clusters.csv'

    # # Read clean feature summaries + metadata (no align bluelight)
    # print("Loading metadata and feature summary results...")
    # features_no_align = pd.read_csv(FEATURES_NO_ALIGN_PATH)
    # metadata_no_align = pd.read_csv(METADATA_NO_ALIGN_PATH, dtype={'comments':str, 'source_plate_id':str})

    # cluster_df = cluster_analysis(features_no_align, n_clusters=100, saveDir=args.save_dir)    
    # # Perform cluster analysis, default using pre-made clusters
    # plot_fingerprints(features, metadata, 
    #                   group_by='gene_name', 
    #                   control='wild_type', 
    #                   saveDir=args.save_dir, 
    #                   clusters_path=CLUSTERS_PATH,
    #                   n_fingers=None)
    
    toc = time()
    print("Done in %.1f seconds" % (toc - tic))
    