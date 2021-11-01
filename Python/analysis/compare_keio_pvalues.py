#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script to compare p-values of initial Keio screen with those of confirmational Keio screen, to
check for correlation
- Hope to see a visible trend in p-values between the screens, with the confirmational screen 
  showing lower p-values in generalwhen -log10 p-values are plotted on a scatter plot
  
@author: sm5911
@date: 15/08/2021

"""

#%% Imports
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt

from sklearn import linear_model
from sklearn.metrics import r2_score # mean_squared_error
# from scipy.stats import linregress

#%% Globals

strain_list = None #['wild_type','fepB','fepD','fes','atpB','nuoC','sdhD','entA']

N_TOP_FEATS = 16

KEIO_STATS_PATH = '/Users/sm5911/Documents/Keio_Screen/Top{}/gene_name/Stats/fdr_by/t-test_results_uncorrected.csv'.format(N_TOP_FEATS)
KEIO_CONF_STATS_PATH = '/Users/sm5911/Documents/Keio_Conf_Screen/Top{}/gene_name/Stats/fdr_by/t-test_results_uncorrected.csv'.format(N_TOP_FEATS)
SAVE_DIR = '/Users/sm5911/Documents/Keio_Conf_Screen/Top{}/p-value_corr'.format(N_TOP_FEATS)

#%% Functions

def strain_pval_pairplot(pvals, pvals2, strain_list=None, saveAs=None):
    
    assert set(pvals.index) == set(pvals2.index) # assert features index matches

    # Subset for shared columns only (strains present in both screens, ie. hit strains)
    shared = list(set(pvals.columns).intersection(set(pvals2.columns)))
    pvals, pvals2 = pvals[shared], pvals2[shared]
    
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

                # #2
                # correlation_matrix = np.corrcoef(pvals, pvals2)
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
           
#%% Main

if __name__ == "__main__":
    
    # Load p-values from initial Keio screen (Top256)
    pvals = pd.read_csv(KEIO_STATS_PATH, index_col=0)
    pvals = pvals[[c for c in pvals.columns if 'pval' in c]]
    pvals.columns = [c.split('pvals_')[-1] for c in pvals.columns]
    
    # Load p-values from confirmational Keio screen (Top256)
    pvals2 = pd.read_csv(KEIO_CONF_STATS_PATH, index_col=0)
    pvals2 = pvals2[[c for c in pvals2.columns if 'pval' in c]]
    pvals2.columns = [c.split('pvals_')[-1] for c in pvals2.columns]
    
    fig, axs = strain_pval_pairplot(pvals, pvals2, strain_list=strain_list, saveAs=Path(SAVE_DIR) / 'pairplot.pdf')
    plt.tight_layout(pad=0.2)
    plt.show()
    
        