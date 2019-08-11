#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 18:36:49 2019

@author: sm5911
"""

#%% IMPORTS

# General imports
import os, itertools#, umap (NB: Need to install umap library in anaconda first!)
import pandas as pd
import numpy as np
import seaborn as sns; sns.set(color_codes=True)
from matplotlib import pyplot as plt
from scipy.stats import f_oneway, zscore
from statsmodels.stats import multitest as smm # AnovaRM
from statsmodels.stats.multicomp import MultiComparison, pairwise_tukeyhsd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Custom imports
#sys.path.insert(0, '/Users/sm5911/Documents/GitHub/PhD_Project/Python') # OPTIONAL: Path to GitHub functions
from SM_plot import pcainfo
from SM_save import savefig
from SM_clean import cleanSummaryResults


#%% PRE-AMBLE

# Global variables
PROJECT_NAME = 'MicrobiomeAssay'
PROJECT_ROOT_DIR = '/Volumes/behavgenom$/Saul/' + PROJECT_NAME
DATA_DIR = '/Volumes/behavgenom$/Priota/Data/' + PROJECT_NAME

verbose = True
nan_threshold = 0.75
p_value_threshold = 0.05

# Select imaging date(s) for analysis
IMAGING_DATES = ['20190704','20190705','20190711','20190712','20190718'] 


#%% READ + FILTER + CLEAN SUMMARY RESULTS
# - Subset to look at results for first video snippets only
# - Subset to look at results for L4-preconditioned worms only
# - Remove columns with all zeros
# - Remove columns with too many NaNs (>75%)
# - Impute remaining NaN values (using mean feature value for each food)

# Read feature summary results
results_inpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'fullresults.csv')
full_results_df = pd.read_csv(results_inpath, dtype={"comments" : str})

L4_1st_snippets_df, droppedFeatsList_NaN, droppedFeatsList_allZero = cleanSummaryResults(full_results_df)

droppedlist_out = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Dropped_Features_NaN_List.txt')
fid = open(droppedlist_out, 'w')
print(*droppedFeatsList_NaN, file=fid)
fid.close()

droppedlist_out = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Dropped_Features_AllZeros_List.txt')
fid = open(droppedlist_out, 'w')
print(*droppedFeatsList_allZero, file=fid)
fid.close()
    
# Record the bacterial strain names for use in analyses
bacterial_strains = list(np.unique(L4_1st_snippets_df['food_type'].str.lower()))


#%% OP50 CONTROL DATA ACROSS DAYS: STATS (ANOVAs) + BOX PLOTS
# - Does N2 worm behaviour on OP50 control vary across experiment days?
# - Perform ANOVA to see if features vary across imaging days for OP50 control
# - Perform Tukey HSD post-hoc analyses for pairwise differences between imaging days
# - Highlight outlier imaging days and investigate reasons why
# - Save list of top significant features for outlier days - are they size-related features?
#   (worms are larger? pre-fed earlier? camera focus/FOV adjusted? skewed by non-worm tracked objects?
#   Did not record time when worms were refed! Could be this. If so, worms will be bigger across all foods on that day) 

# Plot OP50 control top10 size-skewed features for each food - do they all differ for outlier date? If so, worms are likely just bigger.
# PCA: For just OP50 control - colour by imaging date - do they cluster visibly? If so, we have time-dependence = NOT GREAT 
# => Consider excluding that date on the basis of un-standardised development times since refeeding?

test = f_oneway
dates2exclude = []#['20190711','20190718'] 
n_top_features = 10

# Subset data for OP50 only (across imaging days)
OP50_dates_df = L4_1st_snippets_df[L4_1st_snippets_df['food_type'].str.lower()=='op50']            

# Exclude certain imaging dates from analyses
OP50_dates_df = OP50_dates_df[~OP50_dates_df['date_yyyymmdd'].isin(dates2exclude)]

# Drop columns that contain only zeros
n_cols = len(OP50_dates_df.columns)
OP50_dates_df = OP50_dates_df.drop(columns=OP50_dates_df.columns[(OP50_dates_df == 0).all()])
zero_cols = n_cols - len(OP50_dates_df.columns)
if verbose:
    print("Dropped %d feature summaries for OP50 control (all zeros)" % zero_cols)

# Select non-data columns to drop for statistics
non_data_columns = OP50_dates_df.columns[0:25]

# Record a list of feature column names
feature_colnames = OP50_dates_df.columns[25:]

# One-way ANOVA with Bonferroni correction for repeated measures
print("Performing One-Way ANOVAs (for each feature) to investigate whether control OP50 results vary across imaging dates...")
OP50_over_time_results_df = pd.DataFrame(index=['stat','pval'], columns=feature_colnames)
for feature in OP50_dates_df.columns[25:]:
    test_stat, test_pvalue = test(*[OP50_dates_df[OP50_dates_df['date_yyyymmdd']==date][feature]\
                                        for date in OP50_dates_df['date_yyyymmdd'].unique()])
    OP50_over_time_results_df.loc['stat',feature] = test_stat
    OP50_over_time_results_df.loc['pval',feature] = test_pvalue

# Bonferroni correction for multiple comparisons
_corrArray = smm.multipletests(OP50_over_time_results_df.loc['pval'], alpha=p_value_threshold,\
                               method='fdr_bh', is_sorted=False, returnsorted=False)

# Get pvalues for features that passed the Benjamini/Hochberg (non-negative) correlation test
pvalues_corrected = _corrArray[1][_corrArray[0]]

# Add pvalues to 1-way ANOVA results dataframe
OP50_over_time_results_df = OP50_over_time_results_df.append(pd.Series(name='pval_corrected'))
OP50_over_time_results_df.loc['pval_corrected', _corrArray[0]] = pvalues_corrected

n_sigfeats = sum(OP50_over_time_results_df.loc['pval_corrected'] < p_value_threshold)

print("%d / %d (%.1f%%) of features show significant variation across imaging dates for OP50 control (ANOVA)" % \
      (n_sigfeats, len(OP50_over_time_results_df.columns), n_sigfeats/len(OP50_over_time_results_df.columns)*100))

# Record name of statistical test
test_name = str(test).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]

# Save test statistics to file
stats_outpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Stats', 'L4_snippet_1',\
                             test_name, 'OP50_control_across_days.csv')
directory = os.path.dirname(stats_outpath)
if not os.path.exists(directory):
    os.makedirs(directory)
    
# Save test results to CSV
OP50_over_time_results_df.to_csv(stats_outpath)

# Tally total number of significantly different pairwise comparison
n_sigdiff_pairwise_beforeBF = 0
n_sigdiff_pairwise_afterBF = 0

# Tukey HSD post-hoc pairwise differences between dates for each feature
for feature in feature_colnames:
    # Tukey HSD post-hoc analysis (no Bonferroni correction!)
    tukeyHSD = pairwise_tukeyhsd(OP50_dates_df[feature], OP50_dates_df['date_yyyymmdd'])
    n_sigdiff_pairwise_beforeBF += sum(tukeyHSD.reject)
    
    # Tukey HSD post-hoc analysis (Bonferroni correction)
    tukeyHSD_BF = MultiComparison(OP50_dates_df[feature], OP50_dates_df['date_yyyymmdd'])
    n_sigdiff_pairwise_afterBF += sum(tukeyHSD_BF.tukeyhsd().reject)   
    
total_comparisons = len(feature_colnames) * 6
reject_H0_percentage = n_sigdiff_pairwise_afterBF / total_comparisons * 100
print("%d / %d (%.1f%%) of pairwise-comparisons of imaging dates (%d features) show significant variation for OP50 control (TukeyHSD)" %\
      (n_sigdiff_pairwise_afterBF, total_comparisons, reject_H0_percentage, len(feature_colnames)))

# TODO: Reverse-engineer p-values using mean/std 
#from statsmodels.stats.libqsturng import psturng
##studentized range statistic
#rs = res2[1][2] / res2[1][3]
#pvalues = psturng(np.abs(rs), 3, 27)

# Mantel test instead..???

# Boxplots for most important features across days!
pvals = OP50_over_time_results_df.loc['pval_corrected']
n_sigfeats = sum(pvals < p_value_threshold)

if pvals.isna().all():
    print("No signficant features found across days for OP50 control!")
elif n_sigfeats > 0:
    # Rank p-values in ascending order
    ranked_pvals = pvals.sort_values(ascending=True)
            
    # Select the top few p-values
    topfeats = ranked_pvals[:n_top_features]
            
    if n_sigfeats < n_top_features:
        print("Only %d features found to vary significantly across days" % n_sigfeats)
        # Drop NaNs
        topfeats = topfeats.dropna(axis=0)
        # Drop non-sig feats
        topfeats = topfeats[topfeats < p_value_threshold]
        
    if verbose:
        print("\nTop %d features for OP50 that differ significantly across days:\n" % len(topfeats))
        print(*[feat + '\n' for feat in list(topfeats.index)])

    # for f, feature in enumerate(feature_colnames[0:25]):
    for f, feature in enumerate(topfeats.index):
        print("P-value for '%s': %s" % (feature, str(topfeats[feature])))
        OP50_topfeat_df = OP50_dates_df[['date_yyyymmdd', feature]]
        
        # Plot boxplots of OP50 control across days for most significant features
        plt.close('all')
        fig = plt.figure(figsize=[10,6])
        ax = fig.add_subplot(1,1,1)
        sns.boxplot(x='date_yyyymmdd', y=feature, data=OP50_dates_df)
        ax.set_xlabel('Imaging Date (YYYYMMDD)', fontsize=15, labelpad=12)
        ax.set_title(feature, fontsize=20, pad=20)
        
        # TODO: Add reverse-engineered pvalues to plot
        
        # Save plot
        plots_outpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'L4_snippet_1',\
                                     'OP50', feature + '_across_days.eps')
        directory = os.path.dirname(plots_outpath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        savefig(plots_outpath, tellme=True, saveFormat='eps')
        

#%% OP50 CONTROL DATA ACROSS DAYS: PCA with the MOST SIGNIFICANT FEATURES (ANOVAs)

PCs_to_keep = 10   
dates2exclude = []#[20190711]

# Read feature summary results
results_inpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'fullresults.csv')
full_results_df = pd.read_csv(results_inpath)

# Prepare data for PCA
L4_1st_snippets_df, droppedFeatsList_NaN, droppedFeatsList_allZero = cleanSummaryResults(full_results_df)

OP50_dates_df = L4_1st_snippets_df[L4_1st_snippets_df['food_type'].str.upper()=='OP50']

# Exclude certain imaging dates from analyses
OP50_dates_df = OP50_dates_df[~OP50_dates_df['date_yyyymmdd'].isin(dates2exclude)]
        
# Select non-data columns to drop for PCA
non_data_columns = OP50_dates_df.columns[0:25]

data = OP50_dates_df.drop(columns=non_data_columns)

# Normalise the data before PCA
zscores = data.apply(zscore, axis=0)

# Drop features with NaN values after normalising
zscores.dropna(axis=1, inplace=True)

# Perform PCA on extracted features
print("\nPerforming Principal Components Analysis (PCA)...")

# Fit the PCA model with the normalised data
pca = PCA()
pca.fit(zscores)

# Plot summary data from PCA: explained variance (most important features)
important_feats, fig = pcainfo(pca, zscores, PC=1, n_feats2print=10)

# Save plot of PCA explained variance
if len(dates2exclude) > 0:
    plotpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'L4_snippet_1', 'OP50',\
                            'PCA', 'L4_snippet_1' + '_no_{0}'.format(str(dates2exclude)) + '_PCA_explained.eps')
else:
    plotpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'L4_snippet_1', 'OP50',\
                            'PCA', 'L4_snippet_1' + '_PCA_explained.eps')
directory = os.path.dirname(plotpath)
if not os.path.exists(directory):
    os.makedirs(directory)
savefig(plotpath, tight_layout=True, tellme=True, saveFormat='eps')

# Project data (zscores) onto PCs
projected = pca.transform(zscores) # A matrix is produced
# NB: Could also have used pca.fit_transform()

# Store the results for first few PCs in dataframe
projected_df = pd.DataFrame(projected[:,:PCs_to_keep],\
                            columns=['PC' + str(n+1) for n in range(PCs_to_keep)])

# Add concatenate projected PC results to metadata
projected_df.set_index(OP50_dates_df.index, inplace=True) # Do not lose video snippet index position
OP50_dates_projected_df = pd.concat([OP50_dates_df[non_data_columns], projected_df], axis=1)

# 2D Plot - first 2 PCs - OP50 Control (coloured by imaging date)

# Plot first 2 principal components
plt.close('all'); plt.ion()
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize=15, labelpad=12)
ax.set_ylabel('Principal Component 2', fontsize=15, labelpad=12)
ax.set_title('2 Component PCA', fontsize=20)

# Create colour palette for plot loop
imaging_dates = list(OP50_dates_projected_df['date_yyyymmdd'].unique())
palette = itertools.cycle(sns.color_palette("bright", len(imaging_dates)))

for date in imaging_dates:
    date_projected_df = OP50_dates_projected_df[OP50_dates_projected_df['date_yyyymmdd']==int(date)]
    sns.scatterplot(date_projected_df['PC1'], date_projected_df['PC2'], color=next(palette), s=100)
ax.legend(imaging_dates)
ax.grid()

# Save scatterplot of first 2 PCs
if len(dates2exclude) > 0:
    plotpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'L4_snippet_1', 'OP50',\
                            'PCA', 'L4_snippet_1' + '_no_{0}'.format(str(dates2exclude)) + '_1st_2PCs.eps')
else:
    plotpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'L4_snippet_1', 'OP50',\
                            'PCA', 'L4_snippet_1' + '_1st_2PCs.eps')
savefig(plotpath, tight_layout=True, tellme=True, saveFormat='eps')

plt.show(); plt.pause(2)

# Plot 3 PCs - OP50 across imaging dates
rotate = True
depthshade = False

# Plot first 3 principal components
plt.close('all')
fig = plt.figure(figsize=[8,8])
ax = Axes3D(fig) # ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Principal Component 1', fontsize=15, labelpad=12)
ax.set_ylabel('Principal Component 2', fontsize=15, labelpad=12)
ax.set_zlabel('Principal Component 3', fontsize=15, labelpad=12)
ax.set_title('3 Component PCA', fontsize=20)

# Re-initialise colour palette for plot loop
palette = itertools.cycle(sns.color_palette("bright", len(imaging_dates)))

for date in imaging_dates:
    date_projected_df = OP50_dates_projected_df[OP50_dates_projected_df['date_yyyymmdd']==int(date)]
    ax.scatter(xs=date_projected_df['PC1'], ys=date_projected_df['PC2'], zs=date_projected_df['PC3'],\
               zdir='z', color=next(palette), s=50, depthshade=depthshade)
ax.legend(imaging_dates)
ax.grid()

# Save scatterplot of first 3 PCs
if len(dates2exclude) > 0:
    plotpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'L4_snippet_1',\
                            'Dimensionality_Reduction', 'Principal_Components_Analysis',\
                            'L4_snippet_1', + '_no_{0}'.format(str(dates2exclude)) + '_1st_3PCs.eps')
else:
    plotpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'L4_snippet_1', 'OP50',\
                            'PCA', 'L4_snippet_1' + '_1st_3PCs.eps')
savefig(plotpath, tight_layout=False, tellme=True, saveFormat='eps')

# Rotate the axes and update
if rotate:
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw(); plt.pause(0.001)
else:
    plt.show(); plt.pause(1)


