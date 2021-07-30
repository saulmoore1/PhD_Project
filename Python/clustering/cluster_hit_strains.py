#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hierarchical clustering of Keio (hit) strains for GO enrichment analysis

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
from collections import defaultdict    
from scipy.stats import zscore
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from sklearn.cluster import AgglomerativeClustering #KMeans
from sklearn.decomposition import PCA
#from sklearn import metrics as skmet

from tierpsytools.analysis.clustering_tools import hierarchical_purity

from read_data.read import load_json, load_topfeats, read_list_from_file
from clustering.hierarchical_clustering import plot_clustermap

#%% Globals

JSON_PARAMETERS_PATH = "analysis/20210406_parameters_keio_screen.json"
FEATURES_PATH = "/Users/sm5911/Documents/Keio_Screen/features.csv"
METADATA_PATH = "/Users/sm5911/Documents/Keio_Screen/metadata.csv"
HIT_STRAINS_PATH = "/Users/sm5911/Documents/Keio_Screen/Top256/hit_strains.txt"

# Clustering parameters
LINKAGE_METHOD = 'average' # 'ward' - see docs for options: ?scipy.cluster.hierarchy.linkage
DISTANCE_METRIC = 'euclidean' # 'cosine' - see docs for options: ?scipy.spatial.distance.pdist
N_NEIGHBOURS = 5 # number of closest strains to each 

# Estimating distance and/or number of clusters from looking at the dendrogram/heatmap:
MAX_DISTANCE = 55 # chosen from visual inspection of dendrogram 
N_CLUSTERS = 10 # chosen from visual inspection of heatmap

#%% Functions

def average_strain_data(features, metadata, groups_column='gene_name'):
    """ Average results for strains with multiple observations """

    meta_cols = metadata.columns.to_list()    
    data = pd.concat([metadata[groups_column], features], axis=1)
    mean_data = data.groupby(groups_column).mean()
    df = metadata.merge(mean_data, how='right', on=groups_column)
    df = df.groupby(groups_column).first().reset_index()
    
    feat = df[[c for c in df.columns if c not in meta_cols]]
    meta = df[meta_cols]
    
    return feat, meta

def dropNaN(featZ):
    """ Drop features with NaN values after normalising """
    
    n_cols = len(featZ.columns)
    featZ.dropna(axis=1, inplace=True)
    n_dropped = n_cols - len(featZ.columns)
    
    if n_dropped > 0:
        print("Dropped %d features after normalisation (NaN)" % n_dropped)
        
    return featZ

def plot_dendrogram(Z, labels=None, figsize=(15,7), saveAs=None, color_threshold=None):
    """ Plot dendrogram from cluster linkage array (contains the hierarchical clustering information) 
        Z -  cluster linkage array
        colour_threshold -  dendrogram 'clusters' are coloured based on this distance cut-off 
    """
    
    plt.close('all')
    plt.subplots(figsize=figsize)
    sns.set_style("whitegrid")
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

def cluster_linkage_seaborn(features, metadata, groupby='gene_name', saveDir=None, 
                    method=LINKAGE_METHOD, metric=DISTANCE_METRIC): 
    """ METHOD 2 
        - Use seaborn clustermap, then extract dendrogram linkage and distances
        
        Returns
        -------
        Z - cluster linkage array
        cg_featZ - data used for cluster mapping
        
    """
    
    # Normalise data
    featZ = features.apply(zscore, axis=0)
    featZ = dropNaN(featZ) # drop NaN values after normalising

    plt.close('all')
    cg = plot_clustermap(featZ, 
                         metadata,
                         group_by=groupby,
                         col_linkage=None,
                         method=method,
                         metric=metric,
                         saveto=(saveDir / "HCA_{}.pdf".format(method + '_' + metric) if 
                                 saveDir is not None else None),
                         figsize=[15,10],
                         sns_colour_palette="Pastel1",
                         sub_adj={'top':0.98,'bottom':0.02,'left':0.02,'right':0.9})
    plt.close()

    # extract distances from clustermap dendrogram
    Z = cg.dendrogram_row.linkage
       
    # extract mean df (one sample per row)
    mean_featZ = cg.data

    # extract row labels from clustermap heatmap
    labels = sorted(metadata[groupby].unique())
    mean_featZ.index = labels # strain names as index    
    # TODO: Is this the correct way to obtain the row labels from seaborn cluster grid / dendrogram?
    # dg = cg.dendrogram_row.dendrogram
    
    return Z, mean_featZ

def cluster_linkage_pdist(features, metadata, groupby='gene_name', saveDir=None,
                     method=LINKAGE_METHOD, metric=DISTANCE_METRIC):
    """ METHOD 1 
        - Compute distances between points with pdist, then compute linkage for dendrogram
        
        Returns
        -------
        Z - cluster linkage array
        mean_featZ - data used for cluster mapping

    """
        
    # Normalise data
    featZ = features.apply(zscore, axis=0)
    featZ = dropNaN(featZ) # drop NaN values after normalising

    # Average strain data
    mean_featZ, mean_meta = average_strain_data(featZ, metadata, groups_column=groupby)

    # strain names as index
    mean_featZ.index = mean_meta[groupby]
    
    pdistances = pdist(X=mean_featZ, metric=metric)
    Z = linkage(y=pdistances, method=method, metric=metric)

    return Z, mean_featZ
 
def plot_squareform(X, metric=DISTANCE_METRIC, saveAs=None):
    """ Plot squareform distance matrix of pairwise distances between samples """
    
    # Squareform matrix of pairwise distances of all samples from each other
    pdistances = pdist(X=X, metric=DISTANCE_METRIC)
    sq_dist = squareform(pdistances)

    # Plot squareform distance matrix of pairwise distances between strains        
    plt.close('all')
    plt.figure(figsize=(8,8))
    plt.imshow(sq_dist)
   
    if saveAs is not None:
        plt.savefig(saveAs, dpi=600)
    else:
        plt.show()

    return sq_dist

def nearest_neighbours(X, 
                       strain_list=None, 
                       distance_metric=DISTANCE_METRIC, 
                       n_neighbours=N_NEIGHBOURS, 
                       saveDir=None):
    """ Save ranked distances to N closest neighbours from each strain in strain_list and 
        return dataframe of distances of all strains from strains in strain_list 
    """
    
    if strain_list is None:
        strain_list = X.index.to_list()
 
    sq_dist = plot_squareform(X=X, metric=distance_metric, 
                              saveAs=(save_path / 'squareform_pdist.png' if save_path is not 
                                      None else None))    
    # sq_dist_sorted = np.sort(sq_dist, axis=1) # add [:,::-1] to sort in descending order
   
    # convert sqdist to dataframe and subset rows for hit strains only
    distances_df = pd.DataFrame(sq_dist, index=X.index, columns=X.index)
    hit_distances_df = distances_df.loc[distances_df.index.isin(strain_list)]
    
    # For each hit strain, rank all other strains by distance from it + save to file
    #hit_distances_dict = {}
    for hit in distances_df.index:
        hit_distances = distances_df.loc[hit].sort_values(ascending=True).reset_index()[1:n_neighbours+1]
        hit_distances.columns = ['gene_name', 'distance']
        #hit_distances_dict['hit'] = hit_distances
        
        if saveDir is not None:
            saveDir.mkdir(exist_ok=True, parents=True)
            hit_distances.to_csv(saveDir / '{}_neighbours.csv'.format(hit), index=False, header=True)        
    
    return hit_distances_df
         
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
    print("N clusters suggested (elbow method):", k)

    plt.close('all')    
    plt.figure(figsize=(10,8))
    plt.plot(idxs, dist_rev, c='blue')
    plt.plot(idxs[:-2] + 1, acceleration_rev, c='orange')    
    plt.title("Elbow plot (predicted k = {})".format(k), fontsize=25, pad=15)
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

def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:d}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

def plot_cluster_histogram(clusters, saveDir=None):
    """ Plot histogram from fcluster array of cluster labels """
    
    plt.close('all')    
    fig, ax = plt.subplots(1,1)
    sns.set_style('white')
    sns.histplot(clusters, bins=k, ax=ax)    
    show_values_on_bars(ax)
    plt.title('n={} bins'.format(k))
    plt.xlabel('Clusters', labelpad=10)
    plt.ylabel('Number of strains', labelpad=10)
    
    if saveDir is not None:
        plt.savefig(Path(saveDir) / 'clusters_histogram.png', dpi=300)
    else:
        plt.show()
        
def plot_clusters(X, clusters, kde=False, saveAs=None, figsize=(9,8)):
    """ Scatterplot of clusters in principal component space 
    
        Inputs
        ------
        X - features dataframe
        clusters - fcluster array of cluster labels 
    """

    # Normalise data
    featZ = X.apply(zscore, axis=0)
    featZ = dropNaN(featZ) # drop NaN values after normalising

    # Fit PCA model to normalised data + project onto PCs
    pca = PCA(n_components=2)
    projected = pca.fit_transform(featZ)

    # Compute explained variance ratio of component axes
    ex_var=np.var(projected, axis=0)
    ex_var_ratio = ex_var/np.sum(ex_var)
    
    # Store the results for first few PCs in dataframe
    projected_df = pd.DataFrame(data=projected, columns=['PC1','PC2'], index=featZ.index)
    
    # Create colour palette for cluster labels
    cluster_labels = list(np.unique(clusters))
    colours = sns.color_palette('Set1', len(cluster_labels))
    palette = dict(zip(cluster_labels, colours))           

    # group data by clusters for plotting
    data = pd.DataFrame(clusters, columns=['cluster'], index=projected_df.index).join(projected_df)
    grouped = data.groupby('cluster')
            
    # Plot PCA + colour clusters
    plt.close('all')
    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=figsize)
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter',x='PC1', y='PC2', label=key, color=palette[key])
    if kde:
        sns.kdeplot(x='PC1', y='PC2', data=data, hue='cluster', palette=palette, fill=True,
                    alpha=0.25, thresh=0.05, levels=2, bw_method="scott", bw_adjust=1)
        
    ax.set_xlabel('Principal Component 1 (%.1f%%)' % (ex_var_ratio[0]*100), fontsize=20, labelpad=12)
    ax.set_ylabel('Principal Component 2 (%.1f%%)' % (ex_var_ratio[1]*100), fontsize=20, labelpad=12)
    ax.set_title("2-component PCA (n={} clusters)".format(k), fontsize=20)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    ax.legend(cluster_labels, frameon=False, loc=(1.01, 0.62),#'upper right', 
              fontsize=15, markerscale=1.5)
    ax.grid(False)

    if saveAs is not None:
        plt.savefig(saveAs, dpi=300)
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
        plt.scatter(x=Z[y_hc==i, 0], 
                    y=Z[y_hc==i, 1], 
                    c=np.array(cols[i]).reshape(1,-1), s=60, edgecolor='k')
    plt.title('Cluster distances (euclidean, ')
    plt.xlabel('axis 1', labelpad=12, fontsize=12)
    plt.ylabel('axis 2', labelpad=12, fontsize=12)
    
    if saveAs is not None:
        plt.savefig(saveAs)
        plt.close()
    else:
        plt.show()

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

    # # Subset for hit strains in strain list provided
    # from filter_data.clean_feature_summaries import subset_results
    # hit_features, hit_metadata = subset_results(features, 
    #                                             metadata, 
    #                                             column=args.grouping_variable, 
    #                                             groups=strain_list)
    
    # Load Tierpsy Top feature set + subset (columns) for top feats only
    if args.n_top_feats is not None:
        top_feats_path = Path(args.tierpsy_top_feats_dir) / "tierpsy_{}.csv".format(str(args.n_top_feats))        
        topfeats = load_topfeats(top_feats_path, add_bluelight=True, 
                                  remove_path_curvature=True, header=None)

        # Drop features that are not in results
        top_feats_list = [feat for feat in list(topfeats) if feat in features.columns]
        features = features[top_feats_list]
          
    ##### Hierarchical clustering #####

    # Cluster linkage array
    Z, X = cluster_linkage_seaborn(features, metadata, groupby='gene_name', saveDir=save_path)
    _Z, _X = cluster_linkage_pdist(features, metadata, groupby='gene_name', saveDir=save_path)
    
    # Assert that the two methods are identical (within limits of machine precision)
    assert (np.round(Z, 6) == np.round(_Z, 6)).all()
    assert (X == _X).all().all()

    # Find nearest neighbours to hit strains by ranking the computed sqaureform distance matrix to each strain
    distances_df = nearest_neighbours(X=X, strain_list=strain_list, distance_metric=DISTANCE_METRIC, 
                                      n_neighbours=N_NEIGHBOURS, saveDir=save_path / 'nearest_neighbours')
  
    # Compare pairwise distances between all samples to hierarchical clustering distances 
    # The closer the value to 1 the better the clustering preserves original distances
    c, coph_dists = cophenet(Z, pdist(X)); print("Cophenet:", c)
    
    # save dendrogram
    den = plot_dendrogram(Z, saveAs=save_path / 'dendrogram.pdf', color_threshold=MAX_DISTANCE) 
    #default color_threshold = 0.7*max(Z[:,2])

    ##### Cluster Analysis #####
    # The number of clusters can be inferred in several ways:
    #   1. By choosing a max_distance parameter to cut the tree into clustered groups
    #   2. By estimating the greatest decline in the rate of change of an 'elbow' plot -- the 'elbow' method

    # METHOD 1 - Maximum distance cut-off (inferred from dendrogram)
    clusters = fcluster(Z, t=MAX_DISTANCE, criterion='distance')
    k = len(np.unique(clusters))
    print("N clusters chosen from dendrogram: %d (distance: %.1f)" % (k, MAX_DISTANCE))
    
    # METHOD 2 - N clusters (inferred from heatmap/elbow plot)
    # For all 3874 strains, there looks to be anywhere from 5-10 visible clusters from the heatmap
    # From the elbow plot the mathematical choice is 2 clusters, but the elbow looks quite gradual 
    print("N clusters chosen from heatmap: %d" % N_CLUSTERS)
    _k = plot_elbow(Z, saveAs=save_path / 'elbow_plot.png', n_chosen=N_CLUSTERS)
    _clusters = fcluster(Z, t=N_CLUSTERS, criterion='maxclust') # t=_k
    
    # Plot histogram of n strains in each cluster
    plot_cluster_histogram(clusters, saveDir=save_path)

    # Plot clusters as scatter plot in PCA space
    plot_clusters(X, clusters, kde=False, saveAs=save_path / 'PCA_clusters={}.pdf'.format(k))

    # Get list of groups in each cluster
    cluster_classes = get_cluster_classes(den, label='ivl')  
    get_clust_graph(X, numclust=N_CLUSTERS, transpose=True, saveAs=save_path / 'dendrogram_clusters.png')    
    
    ## ALTERNATIVE METHOD WITH SCIKIT-LEARN 
    # Perform clustering + fit dataset to assign each datapoint to a cluster
    Hclustering = AgglomerativeClustering(distance_threshold=MAX_DISTANCE,
                                          #n_clusters=N_CLUSTERS,
                                          affinity=DISTANCE_METRIC, 
                                          linkage=LINKAGE_METHOD)
    y_hc = Hclustering.fit_predict(Z)
    plot_clusters2(Z, y_hc, saveAs=save_path / 'clusters_scatterZ.png')

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
#    skmet.accuracy_score(y, # true labels for groups (unknown)
#                         Hclustering.labels_ # predicted group labels
#                         )
# =============================================================================
    
# =============================================================================
#     # K-means clustering
# 
#     km = KMeans(n_clusters=n_clusters_HCA, init='k-means++', max_iter=100, n_init=1,
#                 verbose=True)
# =============================================================================

