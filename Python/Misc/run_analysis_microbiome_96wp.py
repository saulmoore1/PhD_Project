#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSYCHOBIOTICS: ANALYSIS OF MICROBIOME CORE SET (96WP ASSAY)

A script written to visualise and interpret results for the quantitative 
behavioural analysis of freely moving N2 C. elegans raised monoxenically on 
various bacterial strains cultered from the C. elegans gut microbiome, and 
compared to N2 performance on standard laboratory strains of E. coli, with OP50 
as the control. 

@author: sm5911
@date: 13/10/2019

"""

#%% IMPORTS & DEPENDENCIES

import os, sys, time, copy, decimal, itertools, umap
import numpy as np
import pandas as pd
import subprocess as sp
import seaborn as sns
from scipy.stats import kruskal, zscore#, ttest_ind, f_oneway
from statsmodels.stats import multitest as smm # AnovaRM
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib import transforms
from matplotlib.axes._axes import _log as mpl_axes_logger # Work-around for Axes3D plot colour warnings
from mpl_toolkits.mplot3d import Axes3D

# Paths to Github + local helper functions
sys.path.insert(0, '/Users/sm5911/Documents/GitHub/PhD_Project/Python')
sys.path.insert(1, '/Users/sm5911/OneDrive - Imperial College London/Psychobiotics/Code/')

# Custom imports
from SM_calculate import ranksumtest
from SM_save import savefig
from SM_plot import pcainfo
from run_control_analysis_96wp import control_variation

# Record script start time
bigtic = time.time()

#%% GLOBAL PARAMETERS (USER-DEFINED)

PROJECT_ROOT_DIR = '/Volumes/behavgenom$/Saul/MicrobiomeAssay96WP'             # Project root directory

IMAGING_DATES = os.listdir(os.path.join(PROJECT_ROOT_DIR, "MaskedVideos"))     # Get list of experiment imaging dates                      

PROCESS_METADATA = False                                                       # Process the metadata prior to analysis?

PROCESS_FEATURE_SUMMARY_RESULTS = False                                        # Process feature summary files?

STATISTICS = True                                                              # Compute Wilcoxon rank-sum + Kruskal-Wallis tests?

ANALYSE_OP50_CONTROL_ACROSS_DAYS = True

CONTROL_STRAIN = 'OP50'                                                        # Control strain, E. coli OP50

nan_threshold = 0.5                                                            # Threshold proportion of NaN values to drop a feature column from analyses
p_value_threshold = 0.05                                                       # Threshold p-value for statistical significance
filter_size_related_feats = True                                               # Drop size-related features from analysis?
n_top_features = 10                                                            # Number of top-ranked features to plot (boxplot comparison between foods for each feature)
show_plots = True                                                              # Show figures?

# Dimensionality reduction parameters
PCs_to_keep = 10                                                               # Number of principle components to use for PCA
rotate = True                                                                  # PCA - Rotate 3-D plots?
depthshade = False                                                             # PCA - Shade colours on 3-D plots to show depth?
perplexity = [10,15,20,30,40]                                                  # tSNE - Parameter range for t-SNE mapping eg. np.arange(5,51,1)
n_neighbours = [10,15,20,30,40]                                                # UMAP - Number of neighbours parameter for UMAP projections eg. np.arange(3,31,1)
min_dist = 0.3                                                                 # UMAP - Minimum distance parameter for UMAP projections

#%% PROCESS / LOAD METADATA

# Use subprocess to call 'process_metadata_96WP.py'
# Optional: pass imaging dates as arguments to the script
metafilepath = os.path.join(PROJECT_ROOT_DIR, "AuxiliaryFiles", "metadata.csv")

if PROCESS_METADATA:
    print("\nProcessing metadata file...")
    SCRIPT_PATH = "/Users/sm5911/OneDrive - Imperial College London/Psychobiotics/Code/process_metadata_96wp.py"
    process_metadata = sp.Popen([sys.executable, SCRIPT_PATH, metafilepath, *IMAGING_DATES])
    process_metadata.communicate()
    print("Complete.")

# Read metadata
metadata = pd.read_csv(metafilepath)
print("\nMetadata file loaded.")

# Subset metadata to remove remaining entries with missing filepaths
is_filename = [isinstance(path, str) for path in metadata['filename']]
if any(list(~np.array(is_filename))):
    print("WARNING: Could not find filepaths for %d entries in metadata.\n\t Omitting these files from analysis..." \
          % sum(list(~np.array(is_filename))))
    metadata = metadata[list(np.array(is_filename))]
    # Reset index
    metadata.reset_index(drop=True, inplace=True)

#%% PROCESS / LOAD FEATURE SUMMARY RESULTS

if PROCESS_FEATURE_SUMMARY_RESULTS:
    tic = time.time()
    print("\nProcessing feature summary results...")
    SCRIPT_PATH = "/Users/sm5911/OneDrive - Imperial College London/Psychobiotics/Code/process_feature_summary_96wp.py"
    process_feature_summary = sp.Popen([sys.executable, SCRIPT_PATH, metafilepath, *IMAGING_DATES])
    process_feature_summary.communicate()
    toc = time.time()
    print("Complete (time taken: %d seconds)" % int(toc-tic))

# TODO: Compare outcome with 'No worm' comments in metadata
#       Are there no feature summaries because there were no worms dispensed into those wells?

# Read full results (metadata + feature summaries)
# NB: pre-allocate 'comments' column as 'str' dtype for faster read (contains many empty lines)
resultspath = os.path.join(PROJECT_ROOT_DIR, "Results", "fullresults.csv")
fullresults = pd.read_csv(resultspath, dtype={"comments":str})
print("Feature summary results loaded.")

# Record strain names for which we have results
BACTERIAL_STRAINS = list(fullresults['food_type'].unique())
TEST_STRAINS = [strain for strain in BACTERIAL_STRAINS if strain != CONTROL_STRAIN]

# Record metadata + feature summary column names in fullresults
meta_colnames = list(metadata.columns)
feat_colnames = [featcol for featcol in fullresults.columns if featcol not in meta_colnames]

#%% FILTER / CLEAN RESULTS

# Drop rows that have no results (empty wells?)
n_rows = len(fullresults)
fullresults = fullresults[fullresults[feat_colnames].sum(axis=1) != 0]
print("Dropped %d row entries with no feature summary results (eg. no worms in well)" % (n_rows - len(fullresults)))

# Split results into metadata + feature results
results_meta = fullresults[meta_colnames]
results_feats = fullresults[feat_colnames]

# Drop feature columns with too many NaN values
# NB: All dropped features here have to do with the 'food_edge' (which is undefined, so NaNs are expected)
results_feats = results_feats.dropna(axis=1, thresh=nan_threshold)
feat_colnames_nonan = results_feats.columns
nan_cols = len(feat_colnames) - len(feat_colnames_nonan)
droppedfeats_nan = [col for col in feat_colnames if col not in feat_colnames_nonan]
print("Dropped %d feature columns with too many NaNs" % nan_cols)

# Impute remaining NaN values (using global mean feature value for each food)
n_nans = results_feats.isna().sum(axis=0).sum()
if n_nans > 0:
    print("Imputing %d missing values using global mean value for each feature" % n_nans)  
    results_feats = results_feats.fillna(results_feats.mean(axis=0))
else:
    print("No need to impute! No remaining NaN values found in feature summary results.")

# Drop feature columns that contain only zeros
results_feats = results_feats.drop(columns=results_feats.columns[(results_feats == 0).all()])
feat_colnames_nonanzero = results_feats.columns
zero_cols = len(feat_colnames_nonan) - len(feat_colnames_nonanzero)
droppedfeats_allzero = [col for col in feat_colnames_nonan if col not in feat_colnames_nonanzero]
print("Dropped %d feature columns with all zeros" % zero_cols)

# Re-combine into full results dataframe
fullresults = pd.concat([results_meta, results_feats], axis=1, sort=False)

# Extract OP50 control data from clean feature summaries
OP50_control = fullresults[fullresults['food_type'] == "OP50"]

# Save OP50 control data
PATH_OP50 = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Control', 'OP50_control_results.csv')
directory = os.path.dirname(PATH_OP50) # make folder if it does not exist
if not os.path.exists(directory):
    os.makedirs(directory)
OP50_control.to_csv(PATH_OP50)

# Re-record feature column names after dropping some features
feat_colnames = [featcol for featcol in fullresults.columns if featcol not in meta_colnames]

# OPTIONAL: Filter out size-related features
if filter_size_related_feats:
    size_feat_keys = ['blob','box','width','length','area']
    size_features = []
    for feature in feat_colnames:
        for key in size_feat_keys:
            if key in feature:
                size_features.append(feature)  
    feats2keep = [feat for feat in feat_colnames if feat not in size_features]
    cols2keep = meta_colnames + feats2keep
    print("Dropped %d features that are size-related" % (len(fullresults.columns)-len(cols2keep)))
    fullresults = fullresults[cols2keep]

#%% # Analyse results for OP50 control across days 
# NB: High behavioural variation across days may affect any conclusions about differences on food

if ANALYSE_OP50_CONTROL_ACROSS_DAYS:
    control_variation(path_to_control_data=PATH_OP50, feature_colnames=feat_colnames)

## Use subprocess to call 'food_behaviour_control.py', passing imaging dates as arguments to the script
#    SCRIPT_PATH = '/Users/sm5911/OneDrive - Imperial College London/Psychobiotics/Code/run_control_analysis_96wp.py'
#    print("\nRunning script: '%s' to analyse OP50 control variation across days:" % SCRIPT_PATH)
#    food_behaviour_control = sp.Popen([sys.executable, SCRIPT_PATH, PATH_OP50, *feat_colnames])
#    food_behaviour_control.communicate()

#%% PERFORM STATISTICAL TESTS - To look for behavioural features that differ significantly between worms on different foods

# TODO: First look to see if response data are homoscedastic / normally distributed
#featurefullresults.insert(0, 'is_normal', '')
#fullresults[feat_colnames]
#
#if not is_normal:
    
if STATISTICS:
    # Record number of decimal places of threshold p-value, for print statements to std_out
    p_decims = abs(int(decimal.Decimal(str(p_value_threshold)).as_tuple().exponent))
    
    # Perform rank-sum test (aka. Wilcoxon / Mann-Whitney U)
    
    # Pre-allocate dataframes for storing test statistics and p-values
    test_stats_df = pd.DataFrame(index=list(TEST_STRAINS), columns=feat_colnames)
    test_pvalues_df = pd.DataFrame(index=list(TEST_STRAINS), columns=feat_colnames)
    sigdifffeats_df = pd.DataFrame(index=test_pvalues_df.index, columns=['N_sigdiff_beforeBF','N_sigdiff_afterBF'])
    
    # Compare each strain to OP50: compute t-test statistics for each feature
    for t, food in enumerate(TEST_STRAINS):
        print("Computing rank-sum tests for OP50 vs %s..." % food)
            
        # Grab feature summary results for that food
        test_food_df = fullresults[fullresults['food_type'] == food]
        
        # Drop non-data columns
        test_data = test_food_df.drop(columns=meta_colnames)
        control_data = OP50_control.drop(columns=meta_colnames)
                    
        # Drop columns that contain only zeros
        n_cols = len(test_data.columns)
        test_data.drop(columns=test_data.columns[(test_data == 0).all()], inplace=True)
        control_data.drop(columns=control_data.columns[(control_data == 0).all()], inplace=True)
        
        # Use only shared feature summaries between control data and test data
        shared_colnames = control_data.columns.intersection(test_data.columns)
        test_data = test_data[shared_colnames]
        control_data = control_data[shared_colnames]
    
        zero_cols = n_cols - len(test_data.columns)
        if zero_cols > 0:
            print("Dropped %d feature summaries for %s (all zeros)" % (zero_cols, food))
    
        # Perform rank-sum tests comparing between foods for each feature (max features = 4539)
        test_stats, test_pvalues = ranksumtest(test_data, control_data)
    
        # Add test results to out-dataframe
        test_stats_df.loc[food][shared_colnames] = test_stats
        test_pvalues_df.loc[food][shared_colnames] = test_pvalues
        
        # Record the names and number of significant features 
        sigdiff_feats = test_pvalues_df.columns[np.where(test_pvalues < p_value_threshold)]
        sigdifffeats_df.loc[food,'N_sigdiff_beforeBF'] = len(sigdiff_feats)
                
    # Bonferroni corrections for multiple comparisons
    sigdifffeatslist = []
    test_pvalues_corrected_df = pd.DataFrame(index=test_pvalues_df.index, columns=test_pvalues_df.columns)
    for food in test_pvalues_df.index:
        # Locate pvalue results (row) for food
        food_pvals = test_pvalues_df.loc[food] # pd.Series object
        
        # Perform Bonferroni correction for multiple comparisons on t-test pvalues
        _corrArray = smm.multipletests(food_pvals.values, alpha=p_value_threshold, method='fdr_bh',\
                                       is_sorted=False, returnsorted=False)
        
        # Get pvalues for features that passed the Benjamini/Hochberg (non-negative) correlation test
        pvalues_corrected = _corrArray[1][_corrArray[0]]
        
        # Add pvalues to dataframe of corrected ttest pvalues
        test_pvalues_corrected_df.loc[food, _corrArray[0]] = pvalues_corrected
        
        # Record the names and number of significant features (after Bonferroni correction)
        sigdiff_feats = pd.Series(test_pvalues_corrected_df.columns[np.where(_corrArray[1] < p_value_threshold)])
        sigdiff_feats.name = food
        sigdifffeatslist.append(sigdiff_feats)
        sigdifffeats_df.loc[food,'N_sigdiff_afterBF'] = len(sigdiff_feats)
    
    # Concatenate into dataframe of features for each food that differ significantly from behaviour on OP50
    sigdifffeats_food_df = pd.concat(sigdifffeatslist, axis=1, ignore_index=True, sort=False)
    sigdifffeats_food_df.columns = test_pvalues_df.index
    
    # Save test statistics to file
    stats_outpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Stats', 'ranksum_test_results.csv')
    sigfeats_outpath = stats_outpath.replace('_results.csv', '_significant_features.csv')
    directory = os.path.dirname(stats_outpath) # make folder if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    test_pvalues_df.to_csv(stats_outpath) # Save test results as CSV
    sigdifffeats_food_df.to_csv(sigfeats_outpath, index=False) # Save feature list as text file
        
    #%% Kruskal-Wallis (ANOVA) + Tukey HSD post-hoc tests for pairwise differences 
    #   between foods for each feature
        
    print("\nComputing Kruskal-Wallis tests between foods for each feature...")
    
    # Keep only necessary columns for 1-way ANOVAs
    stats_cols = copy.deepcopy(feat_colnames)
    stats_cols.insert(0, 'food_type')
    test_data = fullresults.loc[:,stats_cols]
    
    # Perform 1-way ANOVAs for each feature between test strains
    test_pvalues_df = pd.DataFrame(index=['stat','pval'], columns=feat_colnames)
    for f, feature in enumerate(feat_colnames):
        if f % 100 == 0:
            print("Analysing feature: %d/%d" % (f, len(feat_colnames)))        
        # Perform test and capture outputs: test statistic + p value
        test_stat, test_pvalue = kruskal(*[test_data[test_data['food_type']==food][feature]\
                                          for food in test_data['food_type'].unique()])
        test_pvalues_df.loc['stat',feature] = test_stat
        test_pvalues_df.loc['pval',feature] = test_pvalue
    
    # Perform Bonferroni correction for multiple comparisons on 1-way ANOVA pvalues
    _corrArray = smm.multipletests(test_pvalues_df.loc['pval'], alpha=p_value_threshold, method='fdr_bh',\
                                   is_sorted=False, returnsorted=False)
    
    # Get pvalues for features that passed the Benjamin/Hochberg (non-negative) correlation test
    pvalues_corrected = _corrArray[1][_corrArray[0]]
    
    # Add pvalues to 1-way ANOVA results dataframe
    test_pvalues_df = test_pvalues_df.append(pd.Series(name='pval_corrected'))
    test_pvalues_df.loc['pval_corrected', _corrArray[0]] = pvalues_corrected
    
    # Store names of features that show significant differences across the test bacteria
    sigdiff_feats = test_pvalues_df.columns[np.where(_corrArray[1] < p_value_threshold)]
    print("Complete!\n%d features exhibit significant differences between foods (Kruskal-Wallis test, Bonferroni)" % len(sigdiff_feats))
    
    # Compile list to store names of significant features
    sigfeats_out = pd.Series(test_pvalues_df.columns[np.where(test_pvalues_df.loc['pval_corrected'] < p_value_threshold)])
    sigfeats_out.name = 'significant_features_kruskal'
    sigfeats_out = pd.DataFrame(sigfeats_out)
    
    # Save test statistics to file
    stats_outpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Stats', 'kruskal_test_results.csv')
    sigfeats_outpath = stats_outpath.replace('_results.csv', '_significant_features.csv')
    directory = os.path.dirname(stats_outpath) # make folder if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    test_pvalues_df.to_csv(stats_outpath) # Save test results as CSV
    sigfeats_out.to_csv(sigfeats_outpath, index=False) # Save feature list as text file
    
    # TODO: Perform post-hoc analyses (eg.Tukey HSD) for pairwise comparisons between foods for each feature
       
#%% GLOBAL PLOTTING VARIABLES
    
# Make dictionary of colours for plotting
colour_dictionary = dict(zip(BACTERIAL_STRAINS, sns.color_palette(palette="gist_rainbow",\
                                                                  n_colors=len(BACTERIAL_STRAINS))))

# Divide results into (1) data (feature summaries) and (2) non-data (metadata)
results_feats = fullresults.loc[:,feat_colnames]
results_meta = fullresults.loc[:,meta_colnames]   

# Read list of important features (highlighted by previous research - see Javer, 2018 paper)
featslistpath = os.path.join(PROJECT_ROOT_DIR, 'AuxiliaryFiles', 'top256_tierpsy_no_blob_no_eigen_only_abs_no_norm.csv')
top256features = pd.read_csv(featslistpath)

# Take first set of 256 features (it does not matter which set is chosen)
top256features = top256features[top256features.columns[0]]

n_feats = len(top256features)
top256features = [feat for feat in top256features if feat in results_feats.columns]
print("Dropping %d size-related features from Top256" % (n_feats - len(top256features)))

#%% BOX PLOTS - INDIVIDUAL PLOTS OF TOP RANKED FEATURES FOR EACH FOOD
# - Rank features by pvalue significance (lowest first) and select the Top 10 features for each food
# - Plot boxplots of the most important features for each food compared to OP50
# - Plot features separately with feature as title and in separate folders for each food

# Load test results (pvalues) for plotting
# NB: Non-parametric ranksum test preferred over t-test as many features may not be normally distributed
stats_inpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Stats', 'ranksum_test_results.csv')
test_pvalues_df = pd.read_csv(stats_inpath, index_col=0)

print("\nPlotting box plots of top %d highest-ranked features by rank-sum test:\n" % len(top256features))
for i, food in enumerate(test_pvalues_df.index):
    pvals = test_pvalues_df.loc[food]
    n_sigfeats = sum(pvals < p_value_threshold)
    n_nonnanfeats = np.logical_not(pvals.isna()).sum()
    if pvals.isna().all():
        print("No signficant features found for %s!" % food)
    elif n_sigfeats > 0:       
        # Rank p-values in ascending order
        ranked_pvals = pvals.sort_values(ascending=True)
        # Drop NaNs
        ranked_pvals = ranked_pvals.dropna(axis=0)
        topfeats = ranked_pvals[:n_top_features] # Select the top ranked p-values
        topfeats = topfeats[topfeats < p_value_threshold] # Drop non-sig feats           
        if n_sigfeats < n_top_features:
            print("Only %d significant features found for %s" % (n_sigfeats, food))
        print("\nTop %d features for %s:\n" % (len(topfeats), food))
        print(*[feat + '\n' for feat in list(topfeats.index)])

        # Subset feature summary results for test-food + OP50-control only
        plot_df = fullresults[np.logical_or(fullresults['food_type'].str.upper()=="OP50",\
                                            fullresults['food_type']==food)] 
        # Colour/legend parameters
        labels = list(plot_df['food_type'].unique())
        colour_dict = {key:value for key, value in colour_dictionary.items() if key in labels}
          
        # OPTIONAL: Plot box plots for cherry-picked (relatable) features
        # topfeats = pd.Series(index=['curvature_midbody_abs_90th'])
                                    
        # Boxplots of OP50 vs test-food for each top-ranked significant feature
        
        for f, feature in enumerate(topfeats.index):
            plt.close('all')
            sns.set_style('darkgrid')
            fig = plt.figure(figsize=[10,8])
            ax = fig.add_subplot(1,1,1)
            sns.boxplot(x="food_type", y=feature, data=plot_df, palette=colour_dict,\
                        showfliers=False, showmeans=True,\
                        meanprops={"marker":"x", "markersize":5, "markeredgecolor":"k"},\
                        flierprops={"marker":"x", "markersize":15, "markeredgecolor":"r"})
            sns.swarmplot(x="food_type", y=feature, data=plot_df, s=10, marker=".", color='k')
            ax.set_xlabel('Bacterial Strain (Food)', fontsize=15, labelpad=12)
            ax.set_ylabel(feature, fontsize=15, labelpad=12)
            ax.set_title(feature, fontsize=20, pad=40)

            # Add plot legend
            patches = []
            for l, key in enumerate(colour_dict.keys()):
                patch = mpatches.Patch(color=colour_dict[key], label=key)
                patches.append(patch)
                if key == 'OP50':
                    continue
                else:
                    ylim = plot_df[plot_df['food_type'].str.upper()==key][feature].max()
                    pval = test_pvalues_df.loc[key, feature]
                    if isinstance(pval, float) and pval < p_value_threshold:
                        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                        ax.text(l - 0.1, 1, 'p={:g}'.format(float('{:.2g}'.format(pval))),\
                        fontsize=13, color='k', verticalalignment='bottom', transform=trans)
            plt.tight_layout(rect=[0.04, 0, 0.84, 0.96])
            plt.legend(handles=patches, labels=colour_dict.keys(), loc=(1.02, 0.8),\
                      borderaxespad=0.4, frameon=False, fontsize=15)

            # Save figure
            plots_outpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots',\
                                         food, feature + '_{0}.eps'.format(f + 1))
            directory = os.path.dirname(plots_outpath)
            if not os.path.exists(directory):
                os.makedirs(directory)
            savefig(plots_outpath, tellme=True, saveFormat='eps')
            if show_plots:
                plt.show(); plt.pause(2)

#%% PLOT FEATURE SUMMARIES -- ALL FOODS (for Avelino's Top 256)
# - Investigate Avelino's top 256 features to look for any differences between foods (see paper: Javer et al, 2018)
# - Plot features that show significant differences from behaviour on OP50
    
# Only plot features in Top256 list
features2plot = [feature for feature in top256features if feature in test_pvalues_df.columns]
   
# Only plot features displaying significant differences between any food and OP50 control
features2plot = [feature for feature in features2plot if (test_pvalues_df[feature] < p_value_threshold).any()]
print("Dropped %d insignificant feature from Top256." % (len(top256features) - len(features2plot)))

# OPTIONAL: Plot cherry-picked features
#features2plot = ['speed_50th','curvature_neck_abs_50th','major_axis_50th','angular_velocity_neck_abs_50th']

# Seaborn boxplots with swarmplot overlay for each feature - saved to file
tic = time.time()
plt.ioff()
sns.set(color_codes=True); sns.set_style('darkgrid')
for f, feature in enumerate(features2plot):
    plotpath_out = os.path.join(PROJECT_ROOT_DIR, "Results", "Plots", "All",\
                                "Top256_Javer_2018", feature + '.eps')
    print("Plotting feature: '%s'" % feature)
    # Seaborn boxplots for each feature
    fig = plt.figure(figsize=[12,7])
    ax = fig.add_subplot(1,1,1)
    sns.boxplot(x="food_type", y=feature, data=fullresults, showfliers=False, showmeans=True,\
                meanprops={"marker":"x", "markersize":5, "markeredgecolor":"k"},\
                flierprops={"marker":"x", "markersize":15, "markeredgecolor":"r"})
    sns.swarmplot(x="food_type", y=feature, data=fullresults, s=10, marker=".", color='k')
    ax.set_xlabel('Bacterial Strain (Test Food)', fontsize=15, labelpad=12)
    ax.set_ylabel(feature, fontsize=15, labelpad=12)
    ax.set_title(feature, fontsize=20, pad=20)                             # Set title
    labels = [lab.get_text().upper() for lab in ax.get_xticklabels()]      # Get x-labels
    ax.set_xticklabels(labels, rotation=45)                                # Rotate x-labels
    labels.remove('OP50')
    for l, food in enumerate(labels):
        ylim = fullresults[fullresults['food_type']==food][feature].max()
        pval = test_pvalues_df.loc[food, feature]
        if isinstance(pval, float) and pval < p_value_threshold:
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(l + 0.75, 0, 'p={:g}'.format(float('{:.2g}'.format(pval))),\
                    fontsize=10, color='k', verticalalignment='bottom', transform=trans)
    plt.subplots_adjust(top=0.9,bottom=0.2,left=0.1,right=0.95)
    
    # Save boxplots
    directory = os.path.dirname(plotpath_out)
    if not os.path.exists(directory):
        os.makedirs(directory)
    savefig(plotpath_out, tellme=False, saveFormat='eps')
    if show_plots:
        plt.show(); plt.pause(2)
    plt.close(fig) # Close plot
            
toc = time.time()
print("Time taken: %.1f seconds" % (toc - tic))

#%% NORMALISE THE DATA

# Normalise the data (z-scores)
zscores = results_feats.apply(zscore, axis=0)

# Drop features with NaN values after normalising
zscores.dropna(axis=1, inplace=True)
print("Dropped %d features after normalisation (NaN)" % (len(results_feats.columns)-len(zscores.columns)))

top256featcols = zscores.columns
top256featcols = [feat for feat in top256featcols if feat in top256features]

useTop256 = False
if useTop256:
    print("Using top 256 features for dimensionality reduction...")
    zscores = zscores[top256featcols]

# NB: In general, for the curvature and angular velocity features we should only 
# use the 'abs' versions, because the sign is assigned based on whether the worm 
# is on its left or right side and this is not known for the multiworm tracking data

#%% HIERARCHICAL CLUSTERING (HEATMAP) - Top256 Features
# Clustermap of features by foods, to see if data cluster into
# groups for each food - does OP50 control form a nice group?
# NB: This has to be restricted to Top256 else plot will be too large
                    
plotroot = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'All', 'HCA')

# Heatmap (clustergram) of Top10 features per food (n=45)
plt.close('all')
row_colours = results_meta['food_type'].map(colour_dictionary)
sns.set(font_scale=0.6)
g = sns.clustermap(zscores[top256featcols], row_colors=row_colours, figsize=[18,15], xticklabels=3)
plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
patches = []
for l, key in enumerate(colour_dictionary.keys()):
    patch = mpatches.Patch(color=colour_dictionary[key], label=key)
    patches.append(patch)
plt.legend(handles=patches, labels=colour_dictionary.keys(),\
           borderaxespad=0.4, frameon=False, loc=(-3.5, -9), fontsize=12)
plt.subplots_adjust(top=0.985,bottom=0.385,left=0.09,right=0.945,hspace=0.2,wspace=0.2)

# Save clustermap and features of interest
cluster_outpath = os.path.join(plotroot, 'Hierarchical_Clustering_Top256.eps')
directory = os.path.dirname(cluster_outpath)
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig(cluster_outpath, tight_layout=True, dpi=300, saveFormat='eps')
plt.show(); plt.pause(1)

#%% PRINCIPAL COMPONENTS ANALYSIS (PCA) - ALL FOODS Top256

# TODO: Plot features that have greatest influence on PCA (eg. PC1)

tic = time.time()
plotroot = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'All', 'PCA')

# Perform PCA on extracted features
print("\nPerforming Principal Components Analysis (PCA)...")

# Fit the PCA model with the normalised data
pca = PCA()
pca.fit(zscores)

# Plot summary data from PCA: explained variance (most important features)
important_feats, fig = pcainfo(pca, zscores, PC=1, n_feats2print=10)

# Save plot of PCA explained variance
plotpath = os.path.join(plotroot, 'PCA_explained.eps')
directory = os.path.dirname(plotpath)
if not os.path.exists(directory):
    os.makedirs(directory)
savefig(plotpath, tight_layout=True, tellme=True, saveFormat='eps')

# Project data (zscores) onto PCs
projected = pca.transform(zscores) # A matrix is produced
# NB: Could also have used pca.fit_transform() OR decomposition.TruncatedSVD().fit_transform()

# Store the results for first few PCs in dataframe
projected_df = pd.DataFrame(projected[:,:PCs_to_keep],\
                            columns=['PC' + str(n+1) for n in range(PCs_to_keep)])

# TODO: Store PCA important features list + plot for each feature

# Add concatenate projected PC results to metadata
projected_df.set_index(fullresults.index, inplace=True) # Do not lose index position
projected_df = pd.concat([fullresults[meta_colnames], projected_df], axis=1)

#%% 2D Plot - first 2 PCs - ALL FOODS

# Plot first 2 principal components
plt.close('all'); plt.ion()
plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=[10,10])

# Create colour palette for plot loop
palette = itertools.cycle(sns.color_palette("gist_rainbow", len(BACTERIAL_STRAINS)))
for food in BACTERIAL_STRAINS:
    food_projected_df = projected_df[projected_df['food_type']==food]
    sns.scatterplot(food_projected_df['PC1'], food_projected_df['PC2'], color=next(palette), s=100)
ax.set_xlabel('Principal Component 1', fontsize=20, labelpad=12)
ax.set_ylabel('Principal Component 2', fontsize=20, labelpad=12)
if useTop256:
    ax.set_title('Top256 features 2-Component PCA', fontsize=20)
else: 
    ax.set_title('All features 2-Component PCA', fontsize=20)
plt.tight_layout(rect=[0.04, 0, 0.84, 0.96])
ax.legend(BACTERIAL_STRAINS, frameon=False, loc=(1, 0.1), fontsize=15)
ax.grid()

# Save scatterplot of first 2 PCs
plotpath = os.path.join(plotroot, 'PCA_2PCs.eps')
savefig(plotpath, tight_layout=True, tellme=True, saveFormat='eps')

plt.show(); plt.pause(2)

#%% 3D Plot - first 3 PCs - ALL FOODS

# Work-around for 3D plot colour warnings
mpl_axes_logger.setLevel('ERROR')

# Plot first 3 principal components
plt.close('all')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
fig = plt.figure(figsize=[10,10])
ax = Axes3D(fig) # ax = fig.add_subplot(111, projection='3d')

# Create colour palette for plot loop
palette = itertools.cycle(sns.color_palette("gist_rainbow", len(BACTERIAL_STRAINS)))

for food in BACTERIAL_STRAINS:
    food_projected_df = projected_df[projected_df['food_type']==food]
    ax.scatter(xs=food_projected_df['PC1'], ys=food_projected_df['PC2'], zs=food_projected_df['PC3'],\
               zdir='z', s=50, c=next(palette), depthshade=depthshade)
ax.set_xlabel('Principal Component 1', fontsize=15, labelpad=12)
ax.set_ylabel('Principal Component 2', fontsize=15, labelpad=12)
ax.set_zlabel('Principal Component 3', fontsize=15, labelpad=12)
if useTop256:
    ax.set_title('Top256 features 3-Component PCA', fontsize=20)
else: 
    ax.set_title('All features 3-Component PCA', fontsize=20)
ax.legend(BACTERIAL_STRAINS, frameon=False, fontsize=12)
ax.grid()
#ax.set_rasterized(True)

# Save scatterplot of first 2 PCs
plotpath = os.path.join(plotroot, 'PCA_3PCs.eps')
savefig(plotpath, tight_layout=False, tellme=True, saveFormat='eps') # rasterized=True

# Rotate the axes and update
if rotate:
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw(); plt.pause(0.001)
else:
    plt.show(); plt.pause(1)
    
#%% t-distributed Stochastic Neighbour Embedding (t-SNE)

plotroot = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'All', 'tSNE')

# Perform tSNE on extracted features
print("\nPerforming t-distributed stochastic neighbour embedding (t-SNE)...")

perplexity=[20,25,30,35,40]
for perplex in perplexity:
    # 2-COMPONENT t-SNE
    tSNE_embedded = TSNE(n_components=2, init='random', random_state=42,\
                         perplexity=perplex, n_iter=3000).fit_transform(zscores)
    tSNE_results_df = pd.DataFrame(tSNE_embedded, columns=['tSNE_1', 'tSNE_2'])
    tSNE_results_df.shape
    
    # Add tSNE results to metadata
    tSNE_results_df.set_index(fullresults.index, inplace=True) # Do not lose video snippet index position
    tSNE_results_df = pd.concat([fullresults[meta_colnames], tSNE_results_df], axis=1)
    
    # Plot 2-D tSNE
    plt.close('all')
    plt.rc('xtick',labelsize=12)
    plt.rc('ytick',labelsize=12)
    fig = plt.figure(figsize=[10,10])
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('tSNE Component 1', fontsize=15, labelpad=12)
    ax.set_ylabel('tSNE Component 2', fontsize=15, labelpad=12)
    if useTop256:
        ax.set_title('Top256 features 2-component tSNE (perplexity={0})'.format(perplex), fontsize=20)
    else:
        ax.set_title('All features 2-component tSNE (perplexity={0})'.format(perplex), fontsize=20)        
            
    # Create colour palette for plot loop
    palette = itertools.cycle(sns.color_palette("gist_rainbow", len(BACTERIAL_STRAINS))) # 'gist_rainbow'
    
    for food in BACTERIAL_STRAINS:
        food_tSNE_df = tSNE_results_df[tSNE_results_df['food_type']==food]
        sns.scatterplot(food_tSNE_df['tSNE_1'], food_tSNE_df['tSNE_2'], color=next(palette), s=100)
    plt.tight_layout(rect=[0.04, 0, 0.84, 0.96])
    ax.legend(BACTERIAL_STRAINS, frameon=False, loc=(1, 0.1), fontsize=15)
    ax.grid()
    
    plotpath = os.path.join(plotroot, 'tSNE_perplex={0}.eps'.format(perplex))
    directory = os.path.dirname(plotpath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    savefig(plotpath, tight_layout=True, tellme=True, saveFormat='eps')
    
    plt.show(); plt.pause(2)
       
#%% Uniform Manifold Projection (UMAP)

plotroot = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'All', 'UMAP')

# Perform UMAP on extracted features
print("\nPerforming uniform manifold projection (UMAP)...")

# Perform UMAP
for n in n_neighbours:
    UMAP_projection = umap.UMAP(n_neighbors=n,\
                                min_dist=min_dist,\
                                metric='correlation').fit_transform(zscores)
    
    UMAP_projection_df = pd.DataFrame(UMAP_projection, columns=['UMAP_1', 'UMAP_2'])
    UMAP_projection_df.shape
    
    # Add tSNE results to metadata
    UMAP_projection_df.set_index(fullresults.index, inplace=True) # Do not lose video snippet index position
    UMAP_projection_df = pd.concat([fullresults[meta_colnames], UMAP_projection_df], axis=1)
    
    # Plot 2-D UMAP
    plt.close('all')
    sns.set_style('whitegrid')
    plt.rc('xtick',labelsize=12)
    plt.rc('ytick',labelsize=12)
    fig = plt.figure(figsize=[11,10])
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('UMAP Component 1', fontsize=15, labelpad=12)
    ax.set_ylabel('UMAP Component 2', fontsize=15, labelpad=12)
    if useTop256:
        ax.set_title('Top256 features 2-component UMAP (n_neighbours={0})'.format(n), fontsize=20)
    else:
        ax.set_title('All features 2-component UMAP (n_neighbours={0})'.format(n), fontsize=20)
            
    # Create colour palette for plot loop
    palette = itertools.cycle(sns.color_palette("gist_rainbow", len(BACTERIAL_STRAINS)))
    
    for food in BACTERIAL_STRAINS:
        food_UMAP_df = UMAP_projection_df[UMAP_projection_df['food_type']==food]
        sns.scatterplot(food_UMAP_df['UMAP_1'], food_UMAP_df['UMAP_2'], color=next(palette), s=100)
    plt.tight_layout(rect=[0.04, 0, 0.84, 0.96])
    ax.legend(BACTERIAL_STRAINS, frameon=False, loc=(1, 0.1), fontsize=15)
    ax.grid()
    
    plotpath = os.path.join(plotroot, 'UMAP_n_neighbours={0}.eps'.format(n))
    directory = os.path.dirname(plotpath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    savefig(plotpath, tight_layout=True, tellme=True, saveFormat='eps')
    
    plt.show(); plt.pause(2)
 
#%%
    
bigtoc = time.time()
print("Total time taken (seconds): %.1f" % (bigtoc - bigtic))