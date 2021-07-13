#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster Keio Hit Strains

@author: sm5911
@date: 12/07/2021

"""

#%% Imports

import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from time import time
from pathlib import Path
from matplotlib import pyplot as plt
#%matplotlib inline
from scipy.stats import zscore
from scipy.cluster.hierarchy import dendrogram, linkage #fcluster, cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
#from sklearn import metrics as skmet

from tierpsytools.analysis.clustering_tools import hierarchical_purity

from clustering.hierarchical_clustering import plot_clustermap
from filter_data.clean_feature_summaries import subset_results
from read_data.read import load_json, load_topfeats, read_list_from_file

#%% Globals

JSON_PARAMETERS_PATH = "analysis/20210406_parameters_keio_screen.json"
FEATURES_PATH = "/Users/sm5911/Documents/Keio_Screen/features.csv"
METADATA_PATH = "/Users/sm5911/Documents/Keio_Screen/metadata.csv"
HIT_STRAINS_PATH = "/Users/sm5911/Documents/Keio_Screen/Top16/hit_strains.txt"

#%% Functions

def average_strain_data(features, metadata, groups_column='gene_name'):
    """ Average results for strains with multiple observations """

    meta_cols = metadata.columns.to_list()    
    data = pd.concat([metadata[groups_column], features], axis=1)
    mean_data = data.groupby(groups_column).mean()
    df = metadata.merge(mean_data, how='right', on=groups_column)
    df = df.groupby(groups_column).first().reset_index()
    metadata = df[meta_cols]
    features = df[[c for c in df.columns if c not in meta_cols]]
    
    return features, metadata
    
#%% Main

if __name__ == "__main__":
    tic = time()
    parser = argparse.ArgumentParser(description="Cluster Keio hit strains into similar phenogroups")
    parser.add_argument('-j', '--json', help="Path to JSON parameters file", 
                        default=JSON_PARAMETERS_PATH, type=str)
    parser.add_argument('--features_path', help="Path to feature summaries file", 
                        default=FEATURES_PATH, type=str)
    parser.add_argument('--metadata_path', help="Path to metadata file", 
                        default=METADATA_PATH, type=str)
    parser.add_argument('--strain_list_path', help="Path to list of strains to cluster", 
                        default=HIT_STRAINS_PATH, type=str)
    args = parser.parse_args()

    # Read clean feature summaries + metadata
    print("Loading metadata and feature summary results...")
    features = pd.read_csv(args.features_path)
    metadata = pd.read_csv(args.metadata_path, dtype={'comments':str, 'source_plate_id':str})
    
    strain_list = read_list_from_file(args.strain_list_path)
    print("%d strains found in: %s" % (len(strain_list), args.strain_list_path))
    
    args = load_json(args.json)
    save_path = Path(args.save_dir) / "clustering"
    
    # control = args.control_dict[args.grouping_variable]
    # strain_list = [control] + hit_strains

    # Load Tierpsy Top feature set + subset (columns) for top feats only
    if args.n_top_feats is not None:
        top_feats_path = Path(args.tierpsy_top_feats_dir) / "tierpsy_{}.csv".format(str(args.n_top_feats))        
        topfeats = load_topfeats(top_feats_path, add_bluelight=True, 
                                  remove_path_curvature=True, header=None)

        # Drop features that are not in results
        top_feats_list = [feat for feat in list(topfeats) if feat in features.columns]
        features = features[top_feats_list]

    # Subset for hit strains in strain list provided
    features, metadata = subset_results(features, 
                                        metadata, 
                                        column=args.grouping_variable, 
                                        groups=strain_list)
    
    # Average strain data
    #features, metadata = average_strain_data(features, metadata, groups_column=args.grouping_variable)
    
    # Normalise data
    featZ = features.apply(zscore, axis=0)

    # Drop features with NaN values after normalising
    n_cols = len(featZ.columns)
    featZ.dropna(axis=1, inplace=True)
    n_dropped = n_cols - len(featZ.columns)
    if n_dropped > 0:
        print("Dropped %d features after normalisation (NaN)" % n_dropped)

    # Hierarchical purity
    (distances, 
     clusters, 
     purity, 
     purity_rand) = hierarchical_purity(data=featZ, 
                                        labels=metadata[args.grouping_variable], 
                                        linkage_matrix=None, 
                                        linkage_method='average',
                                        criterion='distance', 
                                        n_random=100)
    
    # Hierarchical clustering
    
    ## set float precision to 4 d.p.
    #np.set_printoptions(precision=4, suppress=True)
    
    linkage_method = 'average' # see docs for options: ?scipy.cluster.hierarchy.linkage
    distance_metric = 'euclidean' # see docs for options: ?scipy.spatial.distance.pdist
    
    ##### METHOD 1 - Use seaborn clustermap, then extract dendrogram linkage and distances
    cg = plot_clustermap(featZ, 
                         metadata, 
                         group_by=args.grouping_variable,
                         col_linkage=None,
                         method=linkage_method,
                         metric=distance_metric,
                         saveto=save_path / "HCA_{}.pdf".format(linkage_method+'_'+distance_metric),
                         figsize=[15,10],
                         sns_colour_palette="Pastel1",
                         sub_adj={'top':0.98,'bottom':0.02,'left':0.02,'right':0.9})
    plt.close()

    # extract distances from clustermap dendrogram
    calculated_linkage = cg.dendrogram_row.calculated_linkage
    dg = cg.dendrogram_row.dendrogram
    
    # plt.figure(figsize=(10,7))
    # print(dg)
    # plt.show()

    ##### METHOD 2 - Compute distances between points with pdist, then compute linkage for dendrogram
    pdistances = pdist(X=featZ, metric=distance_metric)
    Z = linkage(y=pdistances, method=linkage_method, metric=distance_metric)
    
    plt.subplots(figsize=(15,6))
    dendrogram(Z, 
               truncate_mode=None, # 'lastp', 'level'
               #p=10,
               #show_contracted=True,
               leaf_rotation=90,
               leaf_font_size=5)
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.05, right=0.95)
    plt.savefig(save_path / 'dendrogram.pdf', dpi=300)
    plt.close()

    n_clusters = 10 # from looking at the dendrogram and estimating the number of visible clusters

    # Perform clustering + fit dataset to assign each datapoint to a cluster
    Hclustering = AgglomerativeClustering(#distance_threshold=10, 
                                          n_clusters=n_clusters,
                                          affinity=distance_metric, 
                                          linkage=linkage_method)
    y_hc = Hclustering.fit_predict(featZ)
 
    # plot scatterplot of clusters and cluster centres
    cols = sns.color_palette(palette="tab10", n_colors=n_clusters)
    plt.figure(figsize=(10,7))
    for i in range(n_clusters):
        plt.scatter(x=features.loc[y_hc == i, 'length_90th_bluelight'],
                    y=features.loc[y_hc == i, 'curvature_head_abs_90th_bluelight'], 
                    c=np.array(cols[i]).reshape(1,-1))
        plt.xlabel('length_90th_bluelight', labelpad=12, fontsize=12)
        plt.ylabel('curvature_head_abs_90th_bluelight', labelpad=12, fontsize=12)
    plt.savefig(save_path / 'clusters_scatter.png', dpi=600)
    plt.close()
    
    # skmet.accuracy_score(y, # true labels for groups (unknown)
    #                      Hclustering.labels_ # predicted group labels
    #                      )
    
# =============================================================================
#     # K means clustering
# 
#     from sklearn.cluster import KMeans
#     km = KMeans(n_clusters=n_clusters_HCA, init='k-means++', max_iter=100, n_init=1,
#                 verbose=True)
# =============================================================================
