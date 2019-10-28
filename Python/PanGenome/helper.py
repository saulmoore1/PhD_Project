#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HELPER FUNCTIONS FOR PANGENOME ASSAY - DATA WRANGLING & VISUALISATION

@author: sm5911
@date: 28/10/2019

Acknowledgements: Luigi Feriani (Github - lferiani) for the function: pcainfo()

"""

#%% IMPORTS & DEPENDENCIES

import os
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

#%% FUNCTIONS

def ranksumtest(test_data, control_data):
    """ 
    Wilcoxon rank sum test (column-wise between 2 dataframes of equal dimensions)
    
    Returns
    -------
    2 lists: a list of test statistics, and a list of associated p-values
    """
    
    colnames = list(test_data.columns)
    J = len(colnames)
    statistics = np.zeros(J)
    pvalues = np.zeros(J)
    
    for j in range(J):
        test_feat_data = test_data[colnames[j]]
        control_feat_data = control_data[colnames[j]]
        statistic, pval = stats.ranksums(test_feat_data, control_feat_data)
        pvalues[j] = pval
        statistics[j] = statistic
        
    return statistics, pvalues

def savefig(Path, tight_layout=True, tellme=True, saveFormat='eps', **kwargs):
    """ Helper function for easy plot saving. A simple wrapper for plt.savefig """
    if tellme:
        print("Saving figure:", os.path.basename(Path))
    if tight_layout:
        plt.tight_layout()
    if saveFormat == 'eps':
        plt.savefig(Path, format=saveFormat, dpi=600, **kwargs)
    else:
        plt.savefig(Path, format=saveFormat, dpi=600, **kwargs)
    if tellme:
        print("Done.")
        
def pcainfo(pca, zscores, PC=1, n_feats2print=10):
    """ A function to plot PCA explained variance, and print the most 
        important features in the given principal component (P.C.) """
    
    # Input error handling
    PC = int(PC)
    if PC == 0:
        PC += 1
    elif PC < 1:
        PC = abs(PC)
        
    cum_expl_var_frac = np.cumsum(pca.explained_variance_ratio_)

    # Plot explained variance
    fig, ax = plt.subplots()
    plt.plot(range(1,len(cum_expl_var_frac)+1),
             cum_expl_var_frac,
             marker='o')
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('explained $\sigma^2$')
    ax.set_ylim((0,1.05))
    fig.tight_layout()
    
    # Print important features
    important_feats = zscores.columns[np.argsort(pca.components_[PC-1]**2)[-n_feats2print:][::-1]]
    
    print("\nTop %d features in Principal Component %d:\n" % (n_feats2print, PC))
    for feat in important_feats:
        print(feat)

    return important_feats, fig