#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nearest neighbour analysis of hit strain selected from Top100 strains ranked by lowest p-value of 
any feature by t-test (Tierpsy 16, fdr_bh)

Loads the 59 hit strains curated from lowest ranked 100 strains by p-value for any feature 
(t-test, p<0.05, fdr_bh, Tierpsy 16) from the initial screen, computes the Euclidean distance 
between strains in phenotype space, and finds the 3 nearest neighbours to each hit strain to expand 
the candidate strain list and increase the chance of finding interesting behaviour-modifying strains

@author: sm5911
@date: 12/07/2021

"""

#%% Imports

import numpy as np
import pandas as pd
from time import time
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.stats import zscore
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, cophenet

from read_data.read import read_list_from_file
from write_data.write import write_list_to_file
from clustering.hierarchical_clustering import plot_clustermap
from tierpsytools.preprocessing.filter_data import select_feat_set

#%% Globals

PROJECT_DIR = Path("/Users/sm5911/Documents/Keio_Screen_Initial")
FEATURES_PATH = PROJECT_DIR / "features.csv"
METADATA_PATH = PROJECT_DIR / "metadata.csv"

# Load 59 hit strains curated from lowest ranked 100 strains by pvalue from initial screen
# for nearest neighbour analysis too expand gene set
CONF_STRAIN_LIST_PATH = PROJECT_DIR /\
    "59_selected_strains_from_initial_keio_top100_lowest_pval_tierpsy16.txt"
SAVE_DIR = PROJECT_DIR / "nearest_neighbours"

N_NEIGHBOURS = 3 # Number of neighbours to record

LINKAGE_METHOD = 'average' # 'ward' - see docs for options: ?scipy.cluster.hierarchy.linkage
DISTANCE_METRIC = 'euclidean' # 'cosine' - see docs for options: ?scipy.spatial.distance.pdist

N_TIERPSY_FEATURES = 256

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

def cluster_linkage_seaborn(features, 
                            metadata, 
                            groupby='gene_name', 
                            saveDir=None, 
                            method='average', 
                            metric='euclidean'): 
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

def cluster_linkage_pdist(features, 
                          metadata, 
                          groupby='gene_name', 
                          saveDir=None,
                          method='average', 
                          metric='euclidean'):
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
 
def plot_squareform(X, metric='euclidean', saveAs=None):
    """ Plot squareform distance matrix of pairwise distances between samples """
    
    # Squareform matrix of pairwise distances of all samples from each other
    pdistances = pdist(X=X, metric=metric)
    sq_dist = squareform(pdistances)

    # Plot squareform distance matrix of pairwise distances between strains        
    plt.close('all')
    plt.figure(figsize=(8,8))
    plt.imshow(sq_dist)
   
    if saveAs is not None:
        plt.savefig(saveAs, dpi=600)

    return sq_dist

def nearest_neighbours(X, 
                       distance_metric='euclidean', 
                       strain_list=None, 
                       saveDir=None):
    """ Rank distances from each strain to all others to find closest neighbours using the distance 
        metric provided, and return matching dataframes of names and distances
        
        Inputs
        ------
        X
        strain_list
        
        Returns
        -------
        names_df, distances_df
    """
    
    if strain_list is None:
        strain_list = X.index.to_list()

    if saveDir is not None:
        saveDir.mkdir(exist_ok=True, parents=True)
        
    # compute squareform euclidean distances between each strain
    sq_dist = plot_squareform(X=X, metric=distance_metric, 
                              saveAs=(saveDir / 'squareform_pdist.png' if saveDir is not None 
                                      else None))
    # sq_dist_sorted = np.sort(sq_dist, axis=1) # add [:,::-1] to sort in descending order
   
    # Convert squareform distance matrix to dataframe and subset rows for hit strains only
    sq_dist_df = pd.DataFrame(sq_dist, index=X.index, columns=X.index)
    hit_distances_df = sq_dist_df.loc[sq_dist_df.index.isin(strain_list),:]
    
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
        
    # save ranked nearest neighbour names along with corresponding distances to file
    if saveDir is not None:
        names_df.to_csv(saveDir / 'nearest_neighbours_names.csv', index=True, header=True)
        distances_df.to_csv(saveDir / 'nearest_neighbours_distances.csv', index=True, header=True)
        
    return names_df, distances_df
  
#%% Main

if __name__ == "__main__":
    tic = time()
    
    # Read clean feature summaries + metadata
    print("Loading metadata and feature summary results...")
    features = pd.read_csv(FEATURES_PATH)
    metadata = pd.read_csv(METADATA_PATH, dtype={'comments':str, 'source_plate_id':str})
        
    # Load Tierpsy Top feature set + subset (columns) for top feats only
    if N_TIERPSY_FEATURES is not None:
        assert N_TIERPSY_FEATURES in [8,16,256,'2k']
        features = select_feat_set(features, 
                                   tierpsy_set_name='tierpsy_{}'.format(N_TIERPSY_FEATURES), 
                                   append_bluelight=True)
        
    n_strains, n_feats = metadata['gene_name'].nunique(), len(features.columns)
    save_path = Path(SAVE_DIR) / ("%d_strains_%d_features" % (n_strains, n_feats))
         
    ##### Hierarchical clustering #####

    # Cluster linkage array
    print("Computing cluster linkage array...")
    Z, X = cluster_linkage_seaborn(features, 
                                   metadata, 
                                   groupby='gene_name', 
                                   saveDir=save_path, 
                                   method=LINKAGE_METHOD, 
                                   metric=DISTANCE_METRIC)
    _Z, _X = cluster_linkage_pdist(features, 
                                   metadata, 
                                   groupby='gene_name',
                                   saveDir=save_path, 
                                   method=LINKAGE_METHOD, 
                                   metric=DISTANCE_METRIC)
    
    # Assert that the two methods are identical
    assert np.allclose(Z, _Z)

    # Compare pairwise distances between all samples to hierarchical clustering distances 
    # The closer the value to 1 the better the clustering preserves original distances
    c, coph_dists = cophenet(Z, pdist(X)); print("Cophenet: %.3f" % c)

    # Find nearest neighbours by ranking the computed sqaureform distance matrix between all strains
    names_df, distances_df = nearest_neighbours(X=X,
                                                strain_list=None,
                                                distance_metric=DISTANCE_METRIC, 
                                                saveDir=save_path)
    
    # Load cherry-picked hit strains list (n=59) from top100 lowest p-value (any Tierpsy 16 feature, 
    # fdr_bh) selected for confirmation screening
    selected_strain_list = read_list_from_file(CONF_STRAIN_LIST_PATH)
    
    # Record confirmation screen hit strains in/not in initial hit strain list
    conf_strain_initial_list = [s for s in selected_strain_list if s in names_df.index]
    conf_strain_not_initial_list = [s for s in selected_strain_list if s not in names_df.index]
    neighbour_names_df = names_df.loc[conf_strain_initial_list, 1:N_NEIGHBOURS]
    neighbour_names_df.to_excel(Path(save_path) / "nearest_{}_neighbours.xlsx".format(N_NEIGHBOURS))
    
    # Save N nearest neighbours list
    neighbour_list = np.unique(np.asmatrix(neighbour_names_df).flatten().tolist())
    neighbour_list = sorted(set(neighbour_list) - set(selected_strain_list))
    write_list_to_file(neighbour_list, Path(save_path) / "neighbour_strains.txt")

    new_strain_list = sorted(set(neighbour_list).union(set(selected_strain_list)))
    atp_genes = [s for s in names_df.index if s.startswith('atp') and s not in new_strain_list]
    nuo_genes = [s for s in names_df.index if s.startswith('nuo') and s not in new_strain_list]
    extra_strains = ['fiu','fhuE','fhuA','tonB','exbD','exbB','entA','entB','entC','entE','entF',
                     'fes','cirA']
    extra_strain_list = atp_genes + nuo_genes + extra_strains
    assert not any(s in new_strain_list for s in extra_strain_list)
    write_list_to_file(extra_strain_list, Path(save_path) / "extra_strains_added.txt")

    new_strain_list.extend(extra_strain_list)

    # Save expanded list of hit strains for confirmational screen, including the N closest strains 
    # to each hit strain selected from the initial screen
    print("Saving new hit strain list of %d genes to file" % len(new_strain_list))
    write_list_to_file(sorted(new_strain_list), Path(save_path) /\
                       "{}_selected_strains_for_confirmation_screen.txt".format(len(new_strain_list)))
       
    print("Done in %.1f seconds" % (time()-tic))
    