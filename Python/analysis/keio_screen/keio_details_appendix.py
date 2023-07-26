#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appendix for thesis detailing strains selected for confirmation screen

@author: sm5911
@date: 24/07/2023

"""

#%% Imports

import pandas as pd
from scipy.stats import zscore
from scipy.spatial.distance import pdist, squareform
from clustering.nearest_neighbours import dropNaN, average_strain_data
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes

#%% Globals

METADATA_PATH = "/Users/sm5911/Documents/Keio_Screen_Confirmation/metadata.csv"
FEATURES_PATH = "/Users/sm5911/Documents/Keio_Screen_Confirmation/features.csv"

STRAIN_ORDER_PATH = "/Users/sm5911/Documents/Keio_Screen_Confirmation/Top256/gene_name/Plots/fdr_by/heatmaps/gene_name_clustermap_label_strain_order.csv"

SAVE_PATH = "/Users/sm5911/Documents/Keio_Screen_Confirmation/thesis_appendix_confirmation_screen_details.csv"

P_VALUE_THRESHOLD = 0.05
FDR_METHOD = 'fdr_bh'
DISTANCE_METRIC = 'euclidean'

#%% Functions

def ttest(metadata,
          features,
          group_by='gene_name',
          control='wild_type',
          feat='speed_50th_bluelight',
          pvalue_threshold=0.05,
          fdr_method='fdr_bh'):
    
    """ Perform ANOVA and t-tests to compare worm speed on each treatment vs control """
        
    assert all(metadata.index == features.index)
    features = features[[feat]]
             
    # Perform t-tests
    stats_t, pvals_t, reject_t = univariate_tests(X=features,
                                                  y=metadata[group_by],
                                                  control=control,
                                                  test='t-test',
                                                  comparison_type='binary_each_group',
                                                  multitest_correction=fdr_method,
                                                  alpha=pvalue_threshold)
    
    effect_sizes_t = get_effect_sizes(X=features,
                                      y=metadata[group_by],
                                      control=control,
                                      linked_test='t-test')
    
    stats_t.columns = ['stats_' + str(c) for c in stats_t.columns]
    pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
    reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
    effect_sizes_t.columns = ['effect_size_' + str(c) for c in effect_sizes_t.columns]
    ttest_results = pd.concat([stats_t, pvals_t, reject_t, effect_sizes_t], axis=1)
        
    nsig = sum(reject_t.sum(axis=1) > 0)
    print("%d significant features between any %s vs %s (t-test, P<%.2f, %s)" %\
          (nsig, group_by, control, pvalue_threshold, fdr_method))

    return ttest_results


def main():
    
    metadata = pd.read_csv(METADATA_PATH, header=0, index_col=False, dtype={'comments':str})
    features = pd.read_csv(FEATURES_PATH, header=0, index_col=False)
    
    strain_order = pd.read_csv(STRAIN_ORDER_PATH, header=0, index_col=0)
    assert set(metadata['gene_name'].unique()) == set(strain_order['strain_name'])
    
    # append mean and std speed for each strain
    speed_mean = []
    speed_std = []
    grouped = metadata.groupby('gene_name')
    for strain in strain_order['strain_name']:
        strain_meta = grouped.get_group(strain)
        strain_feat = features.reindex(strain_meta.index)

        speed_mean.append(strain_feat['speed_50th_bluelight'].mean())
        speed_std.append(strain_feat['speed_50th_bluelight'].std())
        
    strain_order['speed_mean'] = speed_mean
    strain_order['speed_std'] = speed_std

    # append p-value for speed_50th_bluelight vs BW wild-type   
    ttest_results = ttest(metadata,
                          features,
                          group_by='gene_name',
                          control='wild_type',
                          feat='speed_50th_bluelight',
                          pvalue_threshold=P_VALUE_THRESHOLD,
                          fdr_method=FDR_METHOD)
    
    pvals = ttest_results[[c for c in ttest_results if 'pvals_' in c]]
    effect_sizes = ttest_results[[c for c in ttest_results if 'effect_size_' in c]]

    speed_pval = []
    speed_effect_size = []    
    for strain in strain_order['strain_name']:
        if strain == 'wild_type':
            speed_pval.append('N/A')
            speed_effect_size.append('N/A')
        else:
            speed_pval.append(pvals.loc['speed_50th_bluelight', 'pvals_' + strain])
            speed_effect_size.append(effect_sizes.loc['speed_50th_bluelight', 'effect_size_' + strain])
    
    strain_order['p-value'] = speed_pval
    strain_order['effect_size'] = speed_effect_size
    
    # append euclidean distance from wild_type
    featZ = features.apply(zscore, axis=0)
    featZ = dropNaN(featZ) # dropped 1 feature after normalising (NaN)

    # average strain data
    mean_featZ, mean_meta = average_strain_data(featZ, metadata, groups_column='gene_name') # X == mean_featZ

    # Squareform matrix of pairwise distances of all samples from each other
    pdistances = pdist(X=mean_featZ, metric=DISTANCE_METRIC)
    sq_dist = pd.DataFrame(squareform(pdistances), 
                           index=mean_meta['gene_name'].values, 
                           columns=mean_meta['gene_name'].values)

    euclidean_distances = sq_dist.loc['wild_type',:].reindex(strain_order['strain_name'].values)
    strain_order['Euclidean_distance'] = euclidean_distances.values
    
    # save appendix to file
    print("Saving appendix to file..")
    strain_order.to_csv(SAVE_PATH, header=True, index=False)
    print('Done!')

    return


#%% Main

if __name__ == "__main__":
    main()
    
    
    
    
    
    
