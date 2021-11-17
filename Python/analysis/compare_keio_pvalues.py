#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script to compare correlation of p-values from initial Keio screen vs confirmation screen,
- Hope to see a visible trend in p-values between the screens, with the confirmational screen 
  showing lower p-values in general, when -log10 p-values are plotted on a scatter plot
  
1. for each gene, plot significant feature p-values only
2. for each feature, plot p-values of all genes OR significant genes only
3. for each gene, correlation of initial vs confirm mean values for all features, eg. corr(feature_vector_initial, feature_vector_confirm)

@author: sm5911
@date: 15/08/2021

"""

#%% Imports
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt

from sklearn import linear_model
from sklearn.metrics import r2_score # mean_squared_error
# from scipy.stats import linregress

#%% Globals

N_TOP_FEATS = 256
PVAL_THRESHOLD = 0.05

FEATURE_SET_PATH = '/Users/sm5911/Tierpsy_Versions/tierpsy-tools-python/tierpsytools/extras/feat_sets/tierpsy_{}.csv'.format(N_TOP_FEATS)

KEIO_STATS_PATH = '/Users/sm5911/Documents/Keio_Screen/Top{}/gene_name/Stats/fdr_by/t-test_results_uncorrected.csv'.format(N_TOP_FEATS)
KEIO_CONF_STATS_PATH = '/Users/sm5911/Documents/Keio_Conf_Screen/Top{}/gene_name/Stats/fdr_by/t-test_results_uncorrected.csv'.format(N_TOP_FEATS)

KEIO_FEATURES_PATH = '/Users/sm5911/Documents/Keio_Screen/features.csv'
KEIO_CONF_FEATURES_PATH = '/Users/sm5911/Documents/Keio_Conf_Screen/features.csv'

SAVE_DIR = '/Users/sm5911/Documents/Keio_Conf_Screen/Top{}'.format(N_TOP_FEATS)

#%% Functions

def strain_pval_pairplot(pvals, pvals2, strain_list=None, saveAs=None):
    """ Pairplot of correlation between initial vs confirmation screen p-values for each strain """
    
    # subset for hit strains / selected features only
    if strain_list is not None:
        assert all(s in pvals.columns for s in strain_list)
    else:
        strain_list = pvals.columns.to_list()
    
    # reshape for pairplot
    _pvals = pvals.reset_index(drop=None).melt(id_vars=['index'], 
                                               var_name='gene_name', 
                                               value_name='p1')
    _pvals2 = pvals2.reset_index(drop=None).melt(id_vars=['index'], 
                                                 var_name='gene_name', 
                                                 value_name='p2')
    paired = pd.merge(_pvals, _pvals2, how='inner', on=['index','gene_name']).set_index('index')
    
    # -log10 transformation
    paired['p1'] = - np.log10(paired['p1'])
    paired['p2'] = - np.log10(paired['p2'])
    
    grouped_strain = paired.groupby('gene_name')
    
    n = int(np.ceil(np.sqrt(len(pvals.columns))))
    plt.close('all')
    fig, axs = plt.subplots(nrows=n, ncols=n, figsize=(10,10))
    strain_counter = 0
    for i in range(n):
        for ii in range(n):
            try: 
                strain = strain_list[strain_counter]
                strain_pvals = grouped_strain.get_group(strain)
    
                sns.scatterplot(x='p1', y='p2', data=strain_pvals, ax=axs[i,ii], 
                                marker='+', s=10, color='k')
                
                # perform linear regression fit
                X = strain_pvals['p1'].values.reshape((-1, 1))
                Y = strain_pvals['p2']
                model = linear_model.LinearRegression()
                model = model.fit(X, Y)
    
                axs[i,ii].plot(X, model.predict(X), "r-", lw=1)
 
                # Perform linear regression fit
                # m, b = np.polyfit(pvals, pvals2, deg=1) # m = slope, b = intercept
                # y_pred = np.poly1d([m, b])(pvals) # y_pred = m * pvals + b
                
                # Estimate correlation coefficient of determination
                # You can calculate coefficient of determination (r2) by:
                # 1. sklearn.metrics.r2_score(y,y_pred)
                # 2. numpy.corrcoef(x,y)[0,1]**2 
                # 3. scipy.stats.linregress(x,y)[2]**2
                
                # #1
                # r2 = r2_score(pvals2, y_pred)
                r2 = r2_score(Y, model.predict(X))

                # #2 alternative method to find r2 using correlation matrix:
                # correlation_matrix = np.corrcoef(initial.values, confirm.values)
                # correlation_xy = correlation_matrix[0,1]
                # r2 = correlation_xy ** 2
                
                # #3
                # # scipy.stats.linregress is a ready function for the linear regression fit
                # slope, intercept, r_value, p_value, std_err = linregress(pvals, pvals2)

                if r2 > 0.5:
                    axs[i,ii].text(0.5, 0.5, strain, transform=axs[i,ii].transAxes, 
                                   fontsize=10, c='k', horizontalalignment='center')
                # text = f"$y={m:0.3f}\;x{b:+0.3f}$\n$R^2 = {r2:0.3f}$"
                # plt.gca().text(0, 1.1, text,transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')

            except Exception as E:
                print(E)
                
            # x and y labels only  for middle columns and rows, respectively
            if i == int(n / 2) and ii == 0:
                axs[i,ii].set_ylabel('Confirmation screen (-log10 p-value)', 
                                     fontsize=15, labelpad=10)
            else:
                axs[i,ii].set_ylabel('')
            if i == 0 and ii == int(n / 2):
                axs[i,ii].set_xlabel('Initial screen (-log10 p-value)', 
                                     fontsize=15, labelpad=10)
            else:
                axs[i,ii].set_xlabel('')
            # x and y ticks sete to false
            axs[i,ii].axes.xaxis.set_visible(False)
            axs[i,ii].axes.yaxis.set_visible(False)
    
            strain_counter += 1
            
    if saveAs is not None:
        Path(saveAs).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(saveAs, dpi=600)
    
    return fig, axs

def strain_pval_plot(pvals, pvals2, strain_list=None, sig_feats_only=True, 
                     pval_threshold=PVAL_THRESHOLD, saveDir=None, figsize=(8,8)):
    """ Plot of correlation of p-values from initial Keio screen vs confirmation screen
        - for each strain separately 
        - significant feature p-values only 
    """
    
    if strain_list is not None:
        assert type(strain_list) == list
        assert all(s in pvals.columns for s in strain_list)
    else:
        strain_list = pvals.columns.to_list()    
    
    # for each strain...
    errlog = []
    for strain in tqdm(strain_list):
        initial = pvals[strain]
        confirm = pvals2[strain]
        
        # subset for significant features only
        if sig_feats_only:
            initial = initial[initial < pval_threshold]
        
        if initial.shape[0] < 3:
            errlog.append(strain)
            continue
        
        # rank by lowest p-value from initial screen
        initial = initial.sort_values(ascending=True)
        confirm = confirm[initial.index]
        
        # -log10 transformation
        initial = - np.log10(initial)
        confirm = - np.log10(confirm)
        
        if saveDir is not None:
            saveDir.mkdir(parents=True, exist_ok=True)
            save_path = saveDir / '{}_pval_corr.pdf'.format(strain)
        
        plt.close('all')
        fig, ax = plt.subplots(figsize=figsize)
        sns.scatterplot(x=initial.values, y=confirm.values, marker='o', s=50, color='k')
                
        # perform linear regression fit
        X = initial.values.reshape((-1, 1))
        Y = confirm.values
        model = linear_model.LinearRegression()
        model = model.fit(X, Y)
        ax.plot(X, model.predict(X), "r-", lw=3)
 
        # m = slope, b = intercept
        m, b = np.polyfit(initial.values, confirm.values, deg=1) 
             
        # coefficient of determination, r2
        r2 = r2_score(Y, model.predict(X)) #r2_score(confirm, confirm_pred)
                        
        ax.text(0.5, 1.05, strain, transform=ax.transAxes, fontsize=20, c='k', 
                horizontalalignment='center')
        ax.text(0.02, 0.95, "y = {0:.2f} * x + {1:.2f}".format(m,b), transform=ax.transAxes, 
                fontsize=15, c='k', horizontalalignment='left')
        ax.text(0.02, 0.9, "$r^2$={:.2f}".format(r2), transform=ax.transAxes, fontsize=15, c='k', 
                horizontalalignment='left')   
        ax.set_xlabel('Initial screen p-value (-log10)', fontsize=18, labelpad=12)
        ax.set_ylabel('Confirmation screen p-value (-log10)', fontsize=18, labelpad=12)
        
        if saveDir is not None:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show(); plt.pause(2)
        
    return errlog

def feature_pval_plot(pvals, pval2, feature_list=None, sig_strains_only=True, 
                      pval_threshold=PVAL_THRESHOLD, saveDir=None, figsize=(8,8),
                      plt_strain_names=True):
    """ Plot of correlation of p-values of all genes OR significant genes only, for each feature """

    if feature_list is not None:
        assert type(feature_list) == list
        assert all(s in pvals.index for s in feature_list)
    else:
        feature_list = pvals.index.to_list()    
    
    # for each feature...
    errlog = []
    for feature in tqdm(feature_list):
        initial = pvals.loc[feature]
        confirm = pvals2.loc[feature]
        
        # subset for significant strains only
        if sig_strains_only:
            initial = initial[initial < pval_threshold]
        
        if initial.shape[0] < 3:
            errlog.append(feature)
            continue
        
        # rank by lowest p-value from initial screen
        initial = initial.sort_values(ascending=True)
        confirm = confirm[initial.index]
        
        # -log10 transformation
        initial = - np.log10(initial)
        confirm = - np.log10(confirm)
        
        if saveDir is not None:
            saveDir.mkdir(parents=True, exist_ok=True)
            save_path = saveDir / '{}_pval_corr.pdf'.format(feature)
        
        plt.close('all')
        fig, ax = plt.subplots(figsize=figsize)
        if plt_strain_names:
            sns.scatterplot(x=initial.values, y=confirm.values, marker='.', s=40, color='k')
        else:
            sns.scatterplot(x=initial.values, y=confirm.values, marker='o', s=50, color='k')
                
        # perform linear regression fit
        X = initial.values.reshape((-1, 1))
        Y = confirm.values
        model = linear_model.LinearRegression()
        model = model.fit(X, Y)
        ax.plot(X, model.predict(X), "r-", lw=3)
 
        # m = slope, b = intercept
        m, b = np.polyfit(initial.values, confirm.values, deg=1) 
             
        # coefficient of determination, r2
        r2 = r2_score(Y, model.predict(X)) #r2_score(confirm, confirm_pred)
                  
        strain_list = initial.index.to_list()
        if plt_strain_names:
            for strain in strain_list:
                x, y = initial[strain], confirm[strain]
                ax.text(x, y, '{}\n'.format(strain), fontsize=8, c='k', verticalalignment='bottom', 
                        horizontalalignment='center', linespacing=0)
            
        ax.text(0.5, 1.05, feature, transform=ax.transAxes, fontsize=15, c='k', 
                horizontalalignment='center')
        ax.text(0.02, 0.95, "y = {0:.2f} * x + {1:.2f}".format(m,b), transform=ax.transAxes, 
                fontsize=15, c='k', horizontalalignment='left')
        ax.text(0.02, 0.9, "$r^2$={:.2f}".format(r2), transform=ax.transAxes, fontsize=15, c='k', 
                horizontalalignment='left')   
        ax.set_xlabel('Initial screen p-value (-log10)', fontsize=18, labelpad=12)
        ax.set_ylabel('Confirmation screen p-value (-log10)', fontsize=18, labelpad=12)
        
        if saveDir is not None:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show(); plt.pause(2)
        
    return errlog

def strain_mean_plot(initial_feat, initial_meta, confirm_feat, confirm_meta,
                     strain_list=None, feature_list=None, saveDir=None, figsize=(8,8)):
    """ Plot of correlation of mean values of Keio initial vs confirm screen for all features for 
        each gene, eg. corr(feature_vector_initial, feature_vector_confirm) 
    """
    
    assert set(initial_feat.index) == set(initial_meta.index)
    assert set(confirm_feat.index) == set(confirm_meta.index)
    
    initial_strains = list(initial_meta['gene_name'].unique())
    confirm_strains = list(confirm_meta['gene_name'].unique())
    
    if strain_list is not None:
        assert type(strain_list) == list
        assert all(s in initial_strains and s in confirm_strains for s in strain_list)
    else:
        # shared strains only (present in both screens, ie. hit strains)
        strain_list = list(set(initial_strains).intersection(set(confirm_strains))) 

    if feature_list is not None:
        assert type(feature_list) == list
        assert all(f in initial_feat.columns and f in confirm_feat.columns for f in feature_list)
    else:
        # shared features only
        feature_list = list(set(initial_feat.columns).intersection(set(confirm_feat.columns))) 
        
    for strain in tqdm(strain_list):
        # subset for strain and feature(s) of interest
        initial_strain_meta = initial_meta[initial_meta['gene_name'] == strain]
        initial_strain_feat = initial_feat[feature_list].reindex(initial_strain_meta.index)
        
        confirm_strain_meta = confirm_meta[confirm_meta['gene_name'] == strain]
        confirm_strain_feat = confirm_feat[feature_list].reindex(confirm_strain_meta.index)
        
        # rank features by lowest log10 abs mean value from initial screen
        initial_log10absmean = np.log10(initial_strain_feat.mean(axis=0).abs()).sort_values(ascending=True)
        confirm_log10absmean = np.log10(confirm_strain_feat.mean(axis=0).abs())[initial_log10absmean.index]

        if saveDir is not None:
            saveDir.mkdir(parents=True, exist_ok=True)
            save_path = saveDir / '{}_mean_corr.pdf'.format(strain)
        
        plt.close('all')
        fig, ax = plt.subplots(figsize=figsize)
        sns.scatterplot(x=initial_log10absmean.values, y=confirm_log10absmean.values, 
                        marker='o', s=50, color='k')
                
        # perform linear regression fit
        X = initial_log10absmean.values.reshape((-1, 1))
        Y = confirm_log10absmean.values
        model = linear_model.LinearRegression()
        model = model.fit(X, Y)
        ax.plot(X, model.predict(X), "r-", lw=3)
 
        # m = slope, b = intercept
        m, b = np.polyfit(initial_log10absmean.values, confirm_log10absmean.values, deg=1) 
             
        # coefficient of determination, r2
        r2 = r2_score(Y, model.predict(X)) #r2_score(confirm, confirm_pred)
        
        ax.text(0.5, 1.05, strain, transform=ax.transAxes, fontsize=15, c='k', 
                horizontalalignment='center')
        ax.text(0.02, 0.95, "y = {0:.2f} * x + {1:.2f}".format(m,b), transform=ax.transAxes, 
                fontsize=15, c='k', horizontalalignment='left')
        ax.text(0.02, 0.9, "$r^2$={:.2f}".format(r2), transform=ax.transAxes, fontsize=15, c='k', 
                horizontalalignment='left')   
        ax.set_xlabel('Log10 mean feature value (Initial screen)', fontsize=18, labelpad=12)
        ax.set_ylabel('Log10 mean feature value (Confirmation screen)', fontsize=18, labelpad=12)
        
        if saveDir is not None:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show(); plt.pause(2)
        
    return

#%% Main

if __name__ == "__main__":
    
    # Load p-values from initial Keio screen
    pvals = pd.read_csv(KEIO_STATS_PATH, index_col=0)
    pvals = pvals[[c for c in pvals.columns if 'pval' in c]]
    pvals.columns = [c.split('pvals_')[-1] for c in pvals.columns]
    
    # Load p-values from confirmational Keio screen
    pvals2 = pd.read_csv(KEIO_CONF_STATS_PATH, index_col=0)
    pvals2 = pvals2[[c for c in pvals2.columns if 'pval' in c]]
    pvals2.columns = [c.split('pvals_')[-1] for c in pvals2.columns]
    
    # assert index match
    assert set(pvals.index) == set(pvals2.index) 

    # subset for shared columns only (strains present in both screens, ie. hit strains)
    shared = list(set(pvals.columns).intersection(set(pvals2.columns)))
    pvals, pvals2 = pvals[shared], pvals2[shared]

    # fig, axs = strain_pval_pairplot(pvals, pvals2, strain_list=strain_list, saveAs=Path(SAVE_DIR) / 'pairplot.pdf')
    # plt.tight_layout(pad=0.2)
    # plt.show()
    strain_list = pvals.columns.to_list()
    
    # load Tierpsy top features (and expand for bluelight)
    feature_list = pd.read_csv(FEATURE_SET_PATH, header=None)[0].to_list()
    feature_list = [f + s for s in ['_prestim','_bluelight','_poststim'] for f in feature_list]
    assert all(f in pvals.index for f in feature_list)
    
    # initial vs confirm screen - plot p-value corr of significant features for each strain
    errlog = strain_pval_plot(pvals, pvals2, strain_list=strain_list, 
                              saveDir=Path(SAVE_DIR) / 'p-value_strain_corr', figsize=(8,8))
    print("\nCould not plot p-value correlation for %d strains (not enough significant features)" % len(errlog))
    
    # initial vs confirm screen - plot p-value corr of significant strains for each feature
    errlog = feature_pval_plot(pvals, pvals2, feature_list=feature_list, 
                              saveDir=Path(SAVE_DIR) / 'p-value_feature_corr', figsize=(8,8))
    print("\nCould not plot p-value correlation for %d feature (not enough significant strains)" % len(errlog))
       
    # load feature summaries results + metadata for intital and confirmation screens
    initial_feat = pd.read_csv(KEIO_FEATURES_PATH)
    initial_meta = pd.read_csv(KEIO_FEATURES_PATH.replace('features.csv','metadata.csv'),
                               dtype={'comments':str})
    
    confirm_feat = pd.read_csv(KEIO_CONF_FEATURES_PATH)
    confirm_meta = pd.read_csv(KEIO_CONF_FEATURES_PATH.replace('features.csv','metadata.csv'), 
                               dtype={'comments':str})
    
    # subset for selected strains
    initial_meta = initial_meta[initial_meta['gene_name'].isin(strain_list)]
    initial_feat = initial_feat.reindex(initial_meta.index)
    
    confirm_meta = confirm_meta[confirm_meta['gene_name'].isin(strain_list)]
    confirm_feat = confirm_feat.reindex(confirm_meta.index)
    
    # initial vs confirm screen - plot of mean value corr of all features for each strain
    strain_mean_plot(initial_feat, initial_meta, confirm_feat, confirm_meta,
                     strain_list=strain_list, feature_list=feature_list, 
                     saveDir=Path(SAVE_DIR) / 'mean_value_strain_corr', figsize=(8,8))
       
    