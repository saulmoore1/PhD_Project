#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HELPER FUNCTIONS FOR PANGENOME ASSAY - DATA WRANGLING & VISUALISATION

@author: sm5911
@date: 28/10/2019

Acknowledgements: Luigi Feriani (Github - lferiani) for the function: pcainfo()

"""

#%% IMPORTS

import os, re, itertools
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro
from sklearn.covariance import MinCovDet
from matplotlib import pyplot as plt
from matplotlib.axes._axes import _log as mpl_axes_logger # Work-around for Axes3D plot colour warnings
from mpl_toolkits.mplot3d import Axes3D

#%% FUNCTIONS
# TODO: Replace with existing functions.... :(

def lookforfiles(root_dir, regex, depth=None, exact=False):
    """ A function to find files in a given directory that match a given
        regular expression pattern. Directory search depth may be specified
        using the 'depth' and 'exact' optional arguments.
    """
    filelist = []
    # Iterate over all files within sub-directories contained within the starting root directory
    for root, subdir, files in os.walk(root_dir, topdown=True):
        if depth:
            start_depth = root_dir.count(os.sep)
            if root.count(os.sep) - start_depth < depth:
                for file in files:
                    if re.search(pattern=regex, string=file):
                        if exact:
                            if os.path.join(root, file).count(os.sep) - start_depth == depth:
                                filelist.append(os.path.join(root, file))
                        else: # if exact depth is not specified, return all matches to specified depth
                            filelist.append(os.path.join(root, file))
        else: # if depth argument is not provided, return all matches
            for file in files:
                if re.search(pattern=regex, string=file):
                    filelist.append(os.path.join(root, file))
    return(filelist)
    
#%%
def ranksumtest(test_data, control_data):
    """ Wilcoxon rank sum test (column-wise between 2 dataframes of equal dimensions)
        Returns 2 lists: a list of test statistics, and a list of associated p-values
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

#%%
def savefig(Path, tight_layout=True, tellme=True, saveFormat='eps', **kwargs):
    """ Helper function for easy plot saving. Simple wrapper for plt.savefig """
    if tellme:
        print("Saving figure:", os.path.basename(Path))
    if tight_layout:
        plt.tight_layout()
    if saveFormat == 'eps':
        plt.savefig(Path, format=saveFormat, dpi=300, **kwargs)
    else:
        plt.savefig(Path, format=saveFormat, dpi=600, **kwargs)
    if tellme:
        print("Done.")
        
#%%        
def pcainfo(pca, zscores, PC=1, n_feats2print=10):
    """ A function to plot PCA explained variance, and print the most 
        important features in the given principal component (P.C.)
    """
    
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
    ax.set_xlabel('Number of Principal Components', fontsize=15)
    ax.set_ylabel('explained $\sigma^2$', fontsize=15)
    ax.set_ylim((0,1.05))
    fig.tight_layout()
    
    # Print important features
    important_feats = zscores.columns[np.argsort(pca.components_[PC-1]**2)[-n_feats2print:][::-1]]
    
    print("\nTop %d features in Principal Component %d:\n" % (n_feats2print, PC))
    for feat in important_feats:
        print(feat)

    return important_feats, fig

#%%
def plotPCA(projected_df, grouping_variable, var_subset=None, savepath=None,\
            title=None, n_component_axes=2, rotate=False):
    """ A function to plot PCA projections and colour by a given categorical 
        variable (eg. grouping_variable = 'food type'). 
        Optionally, a subset of the data can be plotted for the grouping variable 
        (eg. var_subset=[list of foods]). 
    """

    # TODO: Plot features that have greatest influence on PCA (eg. PC1)
        
    if var_subset == None or len(var_subset) == 0:
        var_subset = list(projected_df[grouping_variable].unique())

    plt.close('all')

    # OPTION 1: Plot PCA - 2 principal components
    if n_component_axes == 2:
        plt.rc('xtick',labelsize=15)
        plt.rc('ytick',labelsize=15)
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=[10,10])
        
        # Create colour palette for plot loop
        palette = itertools.cycle(sns.color_palette("gist_rainbow", len(var_subset)))
        
        for g_var in var_subset:
            g_var_projected_df = projected_df[projected_df[grouping_variable]==g_var]
            sns.scatterplot(g_var_projected_df['PC1'], g_var_projected_df['PC2'], color=next(palette), s=50)
        ax.set_xlabel('Principal Component 1', fontsize=20, labelpad=12)
        ax.set_ylabel('Principal Component 2', fontsize=20, labelpad=12)
        if title:
            ax.set_title(title, fontsize=20) # title = 'Top256 features 2-Component PCA'
        if len(var_subset) <= 15:
            plt.tight_layout(rect=[0.04, 0, 0.84, 0.96])
            ax.legend(var_subset, frameon=False, loc=(1, 0.85), fontsize=15)
        ax.grid()

        # Save PCA scatterplot of first 2 PCs
        if savepath:
            savefig(savepath, tight_layout=False, tellme=True, saveFormat='eps') # rasterized=True

        plt.show(); plt.pause(2)
                            
    # OPTION 2: Plot PCA - 3 principal components
    elif n_component_axes == 3:
        # Work-around for 3D plot colour warnings
        mpl_axes_logger.setLevel('ERROR')
    
        plt.rc('xtick',labelsize=12)
        plt.rc('ytick',labelsize=12)
        fig = plt.figure(figsize=[10,10])
        ax = Axes3D(fig) # ax = fig.add_subplot(111, projection='3d')
        
        # Create colour palette for plot loop
        palette = itertools.cycle(sns.color_palette("gist_rainbow", len(var_subset)))
        
        for g_var in var_subset:
            g_var_projected_df = projected_df[projected_df[grouping_variable]==g_var]
            ax.scatter(xs=g_var_projected_df['PC1'], ys=g_var_projected_df['PC2'], zs=g_var_projected_df['PC3'],\
                       zdir='z', s=30, c=next(palette), depthshade=False)
        ax.set_xlabel('Principal Component 1', fontsize=15, labelpad=12)
        ax.set_ylabel('Principal Component 2', fontsize=15, labelpad=12)
        ax.set_zlabel('Principal Component 3', fontsize=15, labelpad=12)
        if title:
            ax.set_title(title, fontsize=20)
        if len(var_subset) <= 15:
            ax.legend(var_subset, frameon=False, fontsize=12)
            #ax.set_rasterized(True)
        ax.grid()
        
        # Save PCA scatterplot of first 3 PCs
        if savepath:
            savefig(savepath, tight_layout=False, tellme=True, saveFormat='png') # rasterized=True

        # Rotate the axes and update plot        
        if rotate:
            for angle in range(0, 360):
                ax.view_init(270, angle)
                plt.draw(); plt.pause(0.001)
        else:
            plt.show(); plt.pause(2)
    else:
        print("Please select from n_component_axes = 2 or 3.")
                
#%%
def MahalanobisOutliers(featMatProjected, extremeness=2., showplot=False):
    """ A function to determine to return a list of outlier indices using the
        Mahalanobis distance. 
        Outlier threshold = std(Mahalanobis distance) * extremeness degree 
        [extreme_values=2, very_extreme_values=3 --> according to 68-95-99.7 rule]
    """
    # NB: Euclidean distance puts more weight than it should on correlated variables
    # Chicken and egg situation, we can’t know they are outliers until we calculate 
    # the stats of the distribution, but the stats of the distribution are skewed outliers!
    # Mahalanobis gets around this by weighting by robust estimation of covariance matrix
    
    # Fit a Minimum Covariance Determinant (MCD) robust estimator to data 
    robust_cov = MinCovDet().fit(featMatProjected[:,:10]) # Use the first 10 principal components
    
    # Get the Mahalanobis distance
    MahalanobisDist = robust_cov.mahalanobis(featMatProjected[:,:10])
    
    # Colour PCA by Mahalanobis distance
    if showplot:
        projectedTable = pd.DataFrame(featMatProjected[:,:10],\
                              columns=['PC' + str(n+1) for n in range(10)])
        plt.close('all')
        plt.rc('xtick',labelsize=15)
        plt.rc('ytick',labelsize=15)
        fig, ax = plt.subplots(figsize=[10,10])
        ax.set_facecolor('#F7FFFF')
        plt.scatter(np.array(projectedTable['PC1']), np.array(projectedTable['PC2']), c=MahalanobisDist)
        plt.title('Mahalanobis Distance for Outlier Detection', fontsize=20)
        plt.colorbar()

    k = np.std(MahalanobisDist) * extremeness
    upper_t = np.mean(MahalanobisDist) + k
    outliers = []
    for i in range(len(MahalanobisDist)):
        if (MahalanobisDist[i] >= upper_t):
            outliers.append(i)
    print("Outliers found: %d" % len(outliers))
            
    return np.array(outliers)

#%%
def check_normality(fullresults, meta_colnames, BACTERIAL_STRAINS,\
                    p_value_threshold=0.05, is_normal_threshold=0.95):
    print("Checking for normality in feature summaries for each food..")
    prop_features_normal = pd.Series(data=None, index=BACTERIAL_STRAINS, name='prop_normal')
    for food in BACTERIAL_STRAINS:
        data = fullresults[fullresults['food_type_upper']==food]
        if not data.shape[0] > 2:
            print("Not enough data for normality test for %s" % food)
        else:
            feats2test = [feature for feature in data.columns if feature not in meta_colnames]
            normality_results = pd.DataFrame(data=None, index=['stat','pval'], columns=feats2test)
            data = data.dropna(axis=1, how='all')
            for f, feature in enumerate(feats2test):
                try:
                    stat, pval = shapiro(data[feature])
                    # NB: UserWarning: Input data for shapiro has range zero # Some features for that strain contain all zeros - shapiro(np.zeros(5))
                    normality_results.loc['stat',feature] = stat
                    normality_results.loc['pval',feature] = pval
                    
                    ## Q-Q plots to visualise whether data fit Gaussian distribution
                    #from statsmodels.graphics.gofplots import qqplot
                    #qqplot(data[feature], line='s')
                    
                except Exception as EE:
                    print("WARNING: %s" % EE)
                    
            prop_normal = (normality_results.loc['pval'] < p_value_threshold).sum()/len(feats2test)
            prop_features_normal.loc[food] = prop_normal
            print("%.1f%% (n=%d) of features are normal for %s" % (prop_normal*100, data.shape[0], food))

    # Determine whether to perform parametric or non-parametric statistics
    # NB: Null hypothesis - feature summary results for individual strains are normally distributed (Gaussian)
    total_prop_normal = np.mean(prop_features_normal)
    if total_prop_normal > is_normal_threshold:
        print("""More than %d%% of features (%.1f%%) were found to obey a normal (Gaussian) distribution, 
        so parametric analyses will be preferred.""" % (is_normal_threshold*100, total_prop_normal*100))
        is_normal = True
    else:
        print("""Less than %d%% of features (%.1f%%) were found to obey a normal (Gaussian) distribution, 
        so non-parametric analyses will be preferred.""" % (is_normal_threshold*100, total_prop_normal*100))
        is_normal = False
            
    return prop_features_normal, is_normal
