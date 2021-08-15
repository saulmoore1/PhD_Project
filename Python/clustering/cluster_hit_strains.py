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
from scipy.stats import zscore
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from sklearn.decomposition import PCA

from tierpsytools.analysis.clustering_tools import hierarchical_purity

from read_data.read import load_json, load_topfeats, read_list_from_file
from write_data.write import write_list_to_file
from filter_data.clean_feature_summaries import subset_results
from clustering.hierarchical_clustering import plot_clustermap

#%% Globals

JSON_PARAMETERS_PATH = "analysis/20210406_parameters_keio_screen.json"
FEATURES_PATH = "/Users/sm5911/Documents/Keio_Screen/features.csv"
METADATA_PATH = "/Users/sm5911/Documents/Keio_Screen/metadata.csv"

# Hit strains to compare distances from each to all other strains OR just hit strains only
HIT_STRAINS_PATH = "/Users/sm5911/Documents/Keio_Screen/Top256/hit_strains.txt"
HIT_STRAINS_ONLY = True

# Clustering parameters
LINKAGE_METHOD = 'average' # 'ward' - see docs for options: ?scipy.cluster.hierarchy.linkage
DISTANCE_METRIC = 'euclidean' # 'cosine' - see docs for options: ?scipy.spatial.distance.pdist

# Estimate EITHER distance OR number of clusters from looking at the dendrogram/heatmap:
MAX_DISTANCE = 15 # METHOD 1 - distance cut-off chosen from visual inspection of dendrogram
N_CLUSTERS = None # METHOD 2 - number of clusters chosen from visual inspection of heatmap/elbow plot

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
                         saveto=(saveDir / "heatmap_{}.pdf".format(method + '_' + metric) if 
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

    return sq_dist

def nearest_neighbours(X, 
                       strain_list=None, 
                       distance_metric=DISTANCE_METRIC, 
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
   
    # Convert squareform distance matrix to dataframe and subset rows for hit strains only
    sq_dist_df = pd.DataFrame(sq_dist, index=X.index, columns=X.index)
    hit_distances_df = sq_dist_df.loc[sq_dist_df.index.isin(strain_list)]
    
    # For each hit strain, rank all other strains by distance from it 
    # and store nearest neighbour gene names and distances separately
    names_dict = {}
    distances_dict = {}
    for hit in hit_distances_df.index:
        hit_distances_sorted = hit_distances_df.loc[hit].sort_values(ascending=True)
        names_dict[hit] = hit_distances_sorted.index
        distances_dict[hit] = hit_distances_sorted.values
        
    names_df = pd.DataFrame.from_dict(names_dict).T
    distances_df = pd.DataFrame.from_dict(distances_dict).T
        
    # save ranked nearest neighbours gene names and corresponding distances to file
    if saveDir is not None:
        saveDir.mkdir(exist_ok=True, parents=True)
        names_df.to_csv(saveDir / 'nearest_neighbours_names.csv', index=True, header=True)
        distances_df.to_csv(saveDir / 'nearest_neighbours_distances.csv', index=True, header=True)
        
    return names_df, distances_df
         
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

def get_cluster_classes(X, clusters, saveDir=None):
    """ Return dictionary of samples (values) in each cluster (keys) """
    
    assert len(X.index) == len(clusters)
    
    data = X.copy()
    data['clusters'] = clusters        
    grouped = data.groupby('clusters')
    
    cluster_classes_dict = {}
    for c in np.unique(clusters):
         cluster_strains = grouped.get_group(c).index.to_list()        
         cluster_classes_dict[c] = cluster_strains
         
         if saveDir is not None:
             Path(saveDir).mkdir(exist_ok=True, parents=True)
             write_list_to_file(cluster_strains, save_path=saveDir / 'cluster_{}'.format(c))
             
    return cluster_classes_dict

def show_values_on_bars(axs):
    """ Helper function to plot values on bars in barplot """
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

def plot_cluster_histogram(clusters, saveAs=None):
    """ Plot histogram from fcluster array of cluster labels 
    
        Inputs 
        ------
        clusters - fcluster array of cluster labels
        saveAs - path to save, str 
    """
    
    k = len(np.unique(clusters))
    
    plt.close('all')    
    fig, ax = plt.subplots(1,1)
    sns.set_style('white')
    sns.histplot(clusters, bins=k, ax=ax)    
    show_values_on_bars(ax)
    plt.title('n={} bins'.format(k))
    plt.xlabel('Clusters', labelpad=10)
    plt.ylabel('Number of strains', labelpad=10)
    
    if saveAs is not None:
        plt.savefig(saveAs, dpi=300)
    else:
        plt.show()
        
def plot_clusters_pca(X, clusters, kde=False, saveAs=None, figsize=(9,8)):
    """ Scatterplot of clusters in principal component space 
    
        Inputs
        ------
        X - features dataframe
        clusters - fcluster array of cluster labels 
        kde - show kernel density on plot, bool
        saveAs - path to save, str
        figsize - figure size (x,y), tuple
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
    
    k = len(np.unique(clusters))
    ax.set_title("2-component PCA (n={} clusters)".format(k), fontsize=20)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    ax.legend(cluster_labels, frameon=False, loc=(1.01, 0.62),#'upper right', 
              fontsize=15, markerscale=1.5)
    ax.grid(False)

    if saveAs is not None:
        plt.savefig(saveAs, dpi=300)
    else:
        plt.show()

def plot_clusters_distance(Z, y_hc, saveAs=None):
    """ Scatterplot of clusters using computed distances between points in euclidean space using
        sklearn.cluster.AgglomerativeClustering
    
        Inputs
        ------
        Z - cluster linkage array
        y_hc - hierarchical clustering fit prediction on Z using AgglomerativeClustering
        saveAs - path to save, str
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

    assert (MAX_DISTANCE is None) or (N_CLUSTERS is None) # make sure to choose only one method
    
    # Read clean feature summaries + metadata
    print("Loading metadata and feature summary results...")
    features = pd.read_csv(args.features_path)
    metadata = pd.read_csv(args.metadata_path, dtype={'comments':str, 'source_plate_id':str})
    
    # Read list of hit strains from file (optional)
    if args.strain_list_path is not None:
        strain_list = read_list_from_file(args.strain_list_path)
        print("%d strains found in: %s" % (len(strain_list), args.strain_list_path))
    else:
        strain_list = None
    
    args = load_json(args.json)

    # Subset for hit strains in strain list provided
    if HIT_STRAINS_ONLY and strain_list is not None:
        print("Subsetting results for hit strains only")
        features, metadata = subset_results(features, 
                                            metadata, 
                                            column=args.grouping_variable, 
                                            groups=strain_list)
    
    # Load Tierpsy Top feature set + subset (columns) for top feats only
    if args.n_top_feats is not None:
        top_feats_path = Path(args.tierpsy_top_feats_dir) / "tierpsy_{}.csv".format(str(args.n_top_feats))        
        topfeats = load_topfeats(top_feats_path, add_bluelight=True, 
                                  remove_path_curvature=True, header=None)

        # Drop features that are not in results
        top_feats_list = [feat for feat in list(topfeats) if feat in features.columns]
        features = features[top_feats_list]

    n_strains, n_feats = len(metadata[args.grouping_variable].unique()), len(features.columns)
    save_path = Path(args.save_dir) / "clustering" / ("%d_strains_%d_features" % (n_strains, n_feats))
         
    ##### Hierarchical clustering #####

    # Cluster linkage array
    Z, X = cluster_linkage_seaborn(features, metadata, groupby=args.grouping_variable, saveDir=save_path)
    _Z, _X = cluster_linkage_pdist(features, metadata, groupby=args.grouping_variable, saveDir=save_path)
    
    # Assert that the two methods are identical (within limits of machine precision)
    assert (np.round(Z, 6) == np.round(_Z, 6)).all()
    assert (X == _X).all().all()

    # save dendrogram
    den = plot_dendrogram(Z, saveAs=save_path / 'dendrogram.pdf', 
                          color_threshold=(MAX_DISTANCE if MAX_DISTANCE is not None else None)) 
                          #default color_threshold = 0.7*max(Z[:,2])

    # Compare pairwise distances between all samples to hierarchical clustering distances 
    # The closer the value to 1 the better the clustering preserves original distances
    c, coph_dists = cophenet(Z, pdist(X)); print("Cophenet:", c)

    # Find nearest neighbours to hit strains by ranking the computed sqaureform distance matrix to all other strains
    names_df, distances_df = nearest_neighbours(X=X,
                                                strain_list=strain_list, 
                                                distance_metric=DISTANCE_METRIC, 
                                                saveDir=save_path / 'nearest_neighbours')
  
    ##### Cluster Analysis #####
    # The number of clusters can be inferred in several ways:
    #   1. By choosing a max_distance parameter to cut the tree into clustered groups
    #   2. By estimating the greatest decline in the rate of change of an 'elbow' plot -- the 'elbow' method

    # METHOD 1 - Maximum distance cut-off (inferred from dendrogram)
    if MAX_DISTANCE is not None:
        clusters = fcluster(Z, t=MAX_DISTANCE, criterion='distance')
        N_CLUSTERS = len(np.unique(clusters))
        print("N clusters chosen from dendrogram: %d (distance: %.1f)" % (N_CLUSTERS, MAX_DISTANCE))
    
    # METHOD 2 - N clusters (inferred from heatmap/elbow plot)
    # For all 3874 strains, there looks to be anywhere from 5-10 visible clusters from the heatmap
    # From the elbow plot the mathematical choice is 2 clusters, but the elbow looks quite gradual
    elif N_CLUSTERS is not None:
        print("N clusters chosen from heatmap: %d" % N_CLUSTERS)
        clusters = fcluster(Z, t=N_CLUSTERS, criterion='maxclust')
    
    # Elbow plot (return suggested number of clusters)
    k = plot_elbow(Z, saveAs=save_path / 'elbow_plot.png', n_chosen=N_CLUSTERS)
    
    # Plot histogram of n strains in each cluster
    plot_cluster_histogram(clusters, saveAs=save_path / 'clusters_histogram.png')

    # Plot clusters as scatter plot in PCA space
    plot_clusters_pca(X, clusters, kde=False, saveAs=save_path / 'PCA_clusters={}.pdf'.format(N_CLUSTERS))

# =============================================================================
#     ## ALTERNATIVE METHOD WITH SCIKIT-LEARN 
#     from sklearn.cluster import AgglomerativeClustering
#     from sklearn import metrics as skmet
#
#     # Perform clustering + fit dataset to assign each datapoint to a cluster
#     Hclustering = AgglomerativeClustering(distance_threshold=(MAX_DISTANCE is MAX_DISTANCE 
#                                                               is not None else None),
#                                           n_clusters=(N_CLUSTERS if N_CLUSTERS 
#                                                       is not None else None),
#                                           affinity=DISTANCE_METRIC, 
#                                           linkage=LINKAGE_METHOD)
#     y_hc = Hclustering.fit_predict(Z)
#     plot_clusters_distance(Z, y_hc, saveAs=save_path / 'clusters_scatterZ.png')

#     skmet.accuracy_score(y,                   #true labels for groups (unknown)
#                          Hclustering.labels_) #predicted group labels
# =============================================================================

    # Get list of groups in each cluster
    cluster_classes = get_cluster_classes(X, clusters, saveDir=save_path / 'cluster_classes')
    
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
    
    # Compare with distances computed by Eleni's function - they should be the same
    distances = Z[:,[2]].flatten()
    assert all(np.round(distances, 6) == np.round(_distances, 6))
    
# =============================================================================
#     # K-means clustering
#     from sklearn.cluster import KMeans
#     km = KMeans(n_clusters=N_CLUSTERS, init='k-means++', max_iter=100, n_init=1, verbose=True)
# =============================================================================
