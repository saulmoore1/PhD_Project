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

import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import linregress

TIERPSY_FEATSET = 256 # 16
FDR_METHOD = 'fdr_bh'

# Load p-values from initial Keio screen (Top256)
keio_stats_path = "/Users/sm5911/Documents/Keio_Screen/Top{}".format(TIERPSY_FEATSET) \
                  + "/gene_name/Stats_{}/t-test_results.csv".format(FDR_METHOD)
pvals = pd.read_csv(keio_stats_path, index_col=0)
pvals = pvals[[c for c in pvals.columns if 'pval' in c]]
pvals.columns = [c.split('pvals_')[-1] for c in pvals.columns]

# Load p-values from confirmational Keio screen (Top256)
keio2_stats_path = "/Users/sm5911/Documents/Keio_Screen2/Top{}".format(TIERPSY_FEATSET) \
                   + "/gene_name/Stats_{}/t-test_results.csv".format(FDR_METHOD)
pvals2 = pd.read_csv(keio2_stats_path, index_col=0)
pvals2 = pvals2[[c for c in pvals2.columns if 'pval' in c]]
pvals2.columns = [c.split('pvals_')[-1] for c in pvals2.columns]

# Subset for shared columns only (strains present in both screens, ie. hit strains)
shared = list(set(pvals.columns).intersection(set(pvals2.columns)))
pvals, pvals2 = pvals[shared], pvals2[shared]

# -log10 transformation
pvals = - np.log10(pvals.values.flatten())
pvals2 = - np.log10(pvals2.values.flatten())

# Perform linear regression fit
m, b = np.polyfit(pvals, pvals2, deg=1) # m = slope, b = intercept
y_pred = np.poly1d([m, b])(pvals) # y_pred = m * pvals + b

# Estimate correlation coefficient of determination
# You can calculate coefficient of determination (r2) by:
# 1. sklearn.metrics.r2_score(y,y_pred)
# 2. numpy.corrcoef(x,y)[0,1]**2 
# 3. scipy.stats.linregress(x,y)[2]**2

#1
r2 = r2_score(pvals2, y_pred)

#2
correlation_matrix = np.corrcoef(pvals, pvals2)
correlation_xy = correlation_matrix[0,1]
r2 = correlation_xy ** 2

#3
# scipy.stats.linregress is a ready function for the linear regression fit
slope, intercept, r_value, p_value, std_err = linregress(pvals, pvals2)

# Assert that the linregress method is identical (within limits of machine precision)
assert ((np.round(slope,6) == np.round(m,6)) and 
        (np.round(intercept,6) == np.round(b,6)) and 
        (np.round(r_value**2,6) == np.round(r2,6)))

# Plot scatterplot
plt.close('all')
plt.figure(figsize=(8,7))
plt.plot(pvals, pvals2, '+', ms=10, mec='k')
plt.plot(pvals, y_pred, "r--", lw=1)

text = f"$y={m:0.3f}\;x{b:+0.3f}$\n$R^2 = {r2:0.3f}$"
plt.gca().text(0, 1.1, text,transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
plt.xlabel('Initial screen (-log10 p-value)', fontsize=15)
plt.ylabel('Confirmation screen (-log10 p-value)', fontsize=15)
plt.title(FDR_METHOD, fontsize=15)

# Save figure
SAVE_PATH = "/Users/sm5911/Documents/Keio_Screen2/Top{}".format(TIERPSY_FEATSET) \
            + "/p-value_corr/{}_corr.pdf".format(FDR_METHOD)
Path(SAVE_PATH).parent.mkdir(exist_ok=True, parents=True)
plt.savefig(SAVE_PATH, dpi=300)
plt.show()
