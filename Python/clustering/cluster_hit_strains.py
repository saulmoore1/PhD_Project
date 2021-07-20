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
from matplotlib.colors import rgb2hex, colorConverter
#%matplotlib inline
from scipy.stats import zscore
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from sklearn.cluster import AgglomerativeClustering
#from sklearn import metrics as skmet

from tierpsytools.analysis.clustering_tools import hierarchical_purity
from filter_data.clean_feature_summaries import subset_results
from read_data.read import load_json, load_topfeats, read_list_from_file

#%% Globals

JSON_PARAMETERS_PATH = "analysis/20210406_parameters_keio_screen.json"
FEATURES_PATH = "/Users/sm5911/Documents/Keio_Screen/features.csv"
METADATA_PATH = "/Users/sm5911/Documents/Keio_Screen/metadata.csv"
HIT_STRAINS_PATH = "/Users/sm5911/Documents/Keio_Screen/Top16/hit_strains.txt"

LINKAGE_METHOD = 'average' # 'ward' - see docs for options: ?scipy.cluster.hierarchy.linkage
DISTANCE_METRIC = 'euclidean' # 'cosine' - see docs for options: ?scipy.spatial.distance.pdist

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

def plot_dendrogram(Z, labels=None, figsize=(15,6), saveAs=None, color_threshold=None):
    """ Plot dendrogram from cluster linkage array (contains the hierarchical clustering information) 
        Z -  cluster linkage array
        colour_threshold -  dendrogram 'clusters' are coloured based on this distance cut-off 
    """
    
    
    plt.ioff() if saveAs is not None else plt.ion()
    plt.subplots(figsize=figsize)
    den = dendrogram(Z,
                     truncate_mode=None, # 'lastp', 'level'
                     #p=10,
                     #show_contracted=True,
                     labels=labels,
                     leaf_rotation=90,
                     leaf_font_size=5,
                     color_threshold=color_threshold)

    if color_threshold is not None:
        # plot a horizontal cut-off line
        plt.axhline(y=color_threshold, c='gray', ls='--')

    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.05, right=0.95)

    if saveAs is not None:
        plt.savefig(saveAs, dpi=300)
        plt.close()
    else:
        plt.show()
        
    return den

def cluster_linkage(featZ, saveDir=None, method=LINKAGE_METHOD, metric=DISTANCE_METRIC): 
    """ METHOD 2 
        - Use seaborn clustermap, then extract dendrogram linkage and distances
        
        Returns
    """
    from clustering.hierarchical_clustering import plot_clustermap
    
    cg = plot_clustermap(featZ, 
                         metadata, 
                         group_by=args.grouping_variable,
                         col_linkage=None,
                         method=method,
                         metric=metric,
                         saveto=saveDir / "HCA_{}.pdf".format(method + '_' + metric),
                         figsize=[15,10],
                         sns_colour_palette="Pastel1",
                         sub_adj={'top':0.98,'bottom':0.02,'left':0.02,'right':0.9})
    plt.close()

    # extract distances from clustermap dendrogram
    Z = cg.dendrogram_row.calculated_linkage
    
    cg_featZ = cg.data    
    # dg = cg.dendrogram_row.dendrogram

    return Z, cg_featZ

def cluster_linkage2(features, metadata, groupby='gene_name', 
                     method=LINKAGE_METHOD, metric=DISTANCE_METRIC):
    """ METHOD 1 
        - Compute distances between points with pdist, then compute linkage for dendrogram
    """
    
    # Average strain data
    mean_feat, mean_meta = average_strain_data(features, metadata, groups_column=groupby)
    mean_featZ = mean_feat.apply(zscore, axis=0)

    pdistances = pdist(X=mean_featZ, metric=metric)
    Z = linkage(y=pdistances, method=method, metric=metric)

    return Z, mean_featZ
            
def plot_clusters(X, clusters, saveAs=None):
    """ """

    plt.close('all')    
    plt.figure(figsize=(10,7))

    # plot clusters
    plt.scatter(X.loc[:,'length_90th_bluelight'],
                X.loc[:,'motion_mode_backward_frequency_bluelight'],
                c=clusters, cmap='Set1', s=60, edgecolor='k')
    
    if saveAs is not None:
        plt.savefig(saveAs)
    else:
        plt.show()

def plot_clusters2(Z, y_hc, saveAs=None):
    """ Scatterplot of clusters using computed distances between points in euclidean space
        Z - cluster linkage array
        y_hc - hierarchical clustering fit prediction on Z using AgglomerativeClustering
    """
    
    n_clusters = len(np.unique(y_hc))
    cols = sns.color_palette(palette="Set1", n_colors=n_clusters)

    # plot scatterplot of clusters and cluster centres    
    plt.close('all')    
    plt.figure(figsize=(10,7))
    for i in range(n_clusters):
        plt.scatter(x=Z[y_hc == i, 0],
                    y=Z[y_hc == i, 1], 
                    c=np.array(cols[i]).reshape(1,-1),
                    s=60,
                    edgecolor='k')
    plt.xlabel('axis 1', labelpad=12, fontsize=12)
    plt.ylabel('axis 2', labelpad=12, fontsize=12)
    
    if saveAs is not None:
        plt.savefig(saveAs)
        plt.close()
    else:
        plt.show()

def plot_elbow(Z, saveAs, n_chosen=None):
    """ Estimate the number of clusters by finding the clustering step where the acceleration 
        of distance growth is the largest (ie. the "strongest elbow")
        
        Input
        -----
        Z - cluster linkage matrix
        
        Output
        ------
        k - estimated number of clusters
    """
        
    dist = Z[:, 2]
    dist_rev = dist[::-1]
    idxs = np.arange(1, len(dist) + 1)
    
    acceleration = np.diff(dist, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]

    k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
    print("clusters:", k)

    plt.close('all')    
    plt.figure(figsize=(10,8))
    plt.plot(idxs, dist_rev, c='blue')
    plt.plot(idxs[:-2] + 1, acceleration_rev, c='orange')    
    plt.title("Elbow plot )", fontsize=25, pad=15)
    plt.xlabel("N clusters", fontsize=18, labelpad=10)
    plt.ylabel("Acceleration of distances", fontsize=18, labelpad=10)
    plt.axvline(x=k, ls='-', c='gray')
    
    if n_chosen is not None:
        plt.axvline(x=n_chosen, ls='--', c='gray')
    
    if saveAs is not None:
        plt.savefig(saveAs)
    else:
        plt.show()
        
    return k

class Clusters(dict):
    def _repr_html_(self):
        html = '<table style="border: 0;">'
        for c in self:
            hx = rgb2hex(colorConverter.to_rgb(c))
            html += '<tr style="border: 0;">' \
            '<td style="background-color: {0}; ' \
                       'border: 0;">' \
            '<code style="background-color: {0};">'.format(hx)
            html += c + '</code></td>'
            html += '<td style="border: 0"><code>' 
            html += repr(self[c]) + '</code>'
            html += '</td></tr>'

        html += '</table>'

        return html
    
def get_cluster_classes(den, label='ivl'):
    """ Get list of samples in each cluster """
    
    from collections import defaultdict    
    cluster_idxs = defaultdict(list)
    
    for c, pi in zip(den['color_list'], den['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))
    
    cluster_classes = Clusters()
    for c, l in cluster_idxs.items():
        i_l = [den[label][i] for i in l]
        cluster_classes[c] = i_l
    
    return cluster_classes

def get_clust_graph(df, numclust, transpose=False, saveAs=None):
    if transpose==True:
        aml=df.transpose()
        xl="x-axis"
    else:
        aml=df
        xl="y-axis"
        
    data_dist = pdist(aml.transpose())
    data_link = linkage(data_dist,  metric=DISTANCE_METRIC, method=LINKAGE_METHOD)
    
    plt.close('all')
    plt.figure(figsize=(10,7))
    B=dendrogram(data_link, labels=list(aml.columns), p=numclust, truncate_mode="lastp", 
                 get_leaves=True, count_sort='ascending', show_contracted=True)
    #myInd = [i for i, c in zip(B['ivl'], B['color_list']) if c=='g']
    get_cluster_classes(B)
    ax=plt.gca()
    ax.tick_params(axis='x', which='major', labelsize=8)
    ax.tick_params(axis='y', which='major', labelsize=15)
    plt.xlabel(xl)
    #plt.set_size_inches(18.5, 10.5)
    plt.ylabel('Distance')
    if saveAs is not None:
        plt.savefig(saveAs)
    else:
        plt.show()
        
    return get_cluster_classes(B)

def give_cluster_assigns(df, numclust, transpose=True):
    if transpose==True:
        data_dist = pdist(df.transpose())
        data_link = linkage(data_dist,  metric=DISTANCE_METRIC, method=LINKAGE_METHOD)
        cluster_assigns=pd.Series(fcluster(data_link, numclust, criterion='maxclust', 
                                           monocrit=None), index=df.columns)
    else:
        data_dist = pdist(df)
        data_link = linkage(data_dist,  metric=DISTANCE_METRIC, method=LINKAGE_METHOD)
        cluster_assigns=pd.Series(fcluster(data_link, numclust, criterion='maxclust', 
                                           monocrit=None), index=df.index)
    for i in range(1, numclust + 1):
        print("Cluster ", str(i), ": ( N =", len(cluster_assigns[cluster_assigns==i].index), 
              ")", ", ".join(list(cluster_assigns[cluster_assigns == i].index)))
        
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
        
    # Normalise data
    featZ = features.apply(zscore, axis=0)

    # Drop features with NaN values after normalising
    n_cols = len(featZ.columns)
    featZ.dropna(axis=1, inplace=True)
    n_dropped = n_cols - len(featZ.columns)
    if n_dropped > 0:
        print("Dropped %d features after normalisation (NaN)" % n_dropped)
    
    ##### Hierarchical clustering #####

    # Cluster linkage array
    #Z, X = cluster_linkage(featZ, saveDir=save_path)
    Z, X = cluster_linkage2(features, metadata, groupby='gene_name')

    # Compare (correlates) actual pairwise distances of all samples to hierarchical clustering 
    # distances. The closer the value to 1 the better the clustering preserves original distances.
    c, coph_dists = cophenet(Z, pdist(X)); print("Cophenet:", c)
                
    # save dendrogram
    den = plot_dendrogram(Z, saveAs=save_path / 'dendrogram.pdf', color_threshold=4.2) #default color_threshold = 0.7*max(Z[:,2])

    ##### Cluster Analysis #####
    # The number of clusters can be inferred in several ways:
    #   1. By choosing a max_distance parameter to cut the tree into clustered groups
    #   2. By estimating the greatest decline in the rate of change of an 'elbow' plot -- the 'elbow' method

    n_clusters = 6 # from looking at the dendrogram/elbow plot and estimating the number of clusters
        
    print("Suggested number of clusters ('elbow' method):")
    k = plot_elbow(Z, saveAs=save_path / 'elbow_plot.png', n_chosen=n_clusters)
    # from the elbow plot the mathematical choice is 2 clusters, but there looks to be visibly like 
    # anywhere from 5-8 clusters as the elbow is quite gradual

    ## METHOD 1
    clusters = fcluster(Z, t=n_clusters, criterion='maxclust') # t=max_d, criterion='distance'
    plot_clusters(X, clusters, saveAs=save_path / 'clusters_scatter.png')

    cluster_classes = get_cluster_classes(den, label='ivl')
    
    get_clust_graph(X, numclust=n_clusters, transpose=True, saveAs=save_path / 'dendrogram_clusters.png')
    
    ## METHOD 2 
    # Perform clustering + fit dataset to assign each datapoint to a cluster
    Hclustering = AgglomerativeClustering(n_clusters=n_clusters,
                                          affinity=DISTANCE_METRIC, 
                                          linkage=LINKAGE_METHOD)
    y_hc = Hclustering.fit_predict(Z)
    plot_clusters2(Z, y_hc, saveAs=save_path / 'clusters_scatterZ.png')

    # skmet.accuracy_score(y, # true labels for groups (unknown)
    #                      Hclustering.labels_ # predicted group labels
    #                      )
        
    # Test hierarchical purity
    (_distances, 
     _clusters, 
     _purity,
     _purity_rand) = hierarchical_purity(data=X, 
                                         labels=clusters, 
                                         linkage_matrix=None, 
                                         linkage_method='average',
                                         criterion='distance', 
                                         n_random=100)
    
    # Compare with distances computed by Eleni's function
    distances = Z[:,[2]].flatten()
    assert all(np.round(distances, 6) == np.round(_distances, 6))
    
# =============================================================================
#     # K-means clustering
# 
#     from sklearn.cluster import KMeans
#     km = KMeans(n_clusters=n_clusters_HCA, init='k-means++', max_iter=100, n_init=1,
#                 verbose=True)
# =============================================================================
