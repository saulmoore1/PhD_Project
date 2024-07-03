#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appendix for thesis detailing strains selected for confirmation screen

@author: sm5911
@date: 24/07/2023

"""

#%% Imports

import numpy as np
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

def round_pvalue(pval):
    """ P-values given to 2 significant figures unless p<0.0001. For p-values between 0.001 and 0.20,
        report p-value to the nearest thousandth 
    """
    
    if type(pval) == str:
        return pval
    elif pval < 0.0001:
        pval = "<0.0001"
    elif pval >= 0.001 and pval < 0.2:
        pval = np.round(pval, decimals=3)
    else:
        pval = float('%.2g' % pval)   
    
    return pval

def round_effect_size(effect_size, decimals=2):
    """ Function to round effect size and handle N/A string """
    if type(effect_size) == str:
        return effect_size
    else:
        effect_size = np.round(effect_size, decimals=decimals)
    
    return effect_size

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
    
    appendix_df = pd.read_csv(STRAIN_ORDER_PATH, header=0, index_col=0)
    assert set(metadata['gene_name'].unique()) == set(appendix_df['strain_name'])

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

    euclidean_distances = sq_dist.loc['wild_type',:].reindex(appendix_df['strain_name'].values)
    appendix_df['euclidean_distance'] = euclidean_distances.values
    
    # append mean and std speed for each strain
    speed_mean = []
    speed_std = []
    grouped = metadata.groupby('gene_name')
    for strain in appendix_df['strain_name']:
        strain_meta = grouped.get_group(strain)
        strain_feat = features.reindex(strain_meta.index)
        
        speed_mean.append(strain_feat['speed_50th_bluelight'].mean())
        speed_std.append(strain_feat['speed_50th_bluelight'].std())
        
    appendix_df['speed_mean'] = speed_mean
    appendix_df['speed_sd'] = speed_std

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
    for strain in appendix_df['strain_name']:
        if strain == 'wild_type':
            speed_pval.append(np.nan)
            speed_effect_size.append(np.nan)
        else:
            speed_pval.append(pvals.loc['speed_50th_bluelight', 'pvals_' + strain])
            speed_effect_size.append(effect_sizes.loc['speed_50th_bluelight', 'effect_size_' + strain])
    
    appendix_df['effect_size'] = speed_effect_size
    appendix_df['pval'] = speed_pval

    # # rank by euclidean distance
    # appendix_df = appendix_df.sort_values(by="euclidean_distance", ascending=False)
    
    # rank by p-value
    appendix_df = appendix_df.sort_values(by='pval', ascending=True)
        
    # round values
    appendix_df['euclidean_distance'] = np.round(appendix_df['euclidean_distance'].values, decimals=2)
    appendix_df['speed_mean'] = np.round(appendix_df['speed_mean'].values, decimals=2)
    appendix_df['speed_sd'] = np.round(appendix_df['speed_sd'].values, decimals=2)
    appendix_df['effect_size'] = np.round(appendix_df['effect_size'].values, decimals=2)
    appendix_df['pval'] = [round_pvalue(p) for p in appendix_df['pval'].values]
    
    # rename colums
    appendix_df.columns = ["Strain Name", 
                           "Euclidean Distance",
                           "Mean Speed (um s^-1)", 
                           "SD Speed (um s^-1)",
                           "Effect size (Cohen's d)",
                           "P-value"]
        
    # save appendix to file
    print("Saving appendix to file..")
    appendix_df.to_csv(SAVE_PATH, header=True, index=False)
    print('Done!')

    return


#%% Main

if __name__ == "__main__":
    main()
    
    
    
    
    
    
