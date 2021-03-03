#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sm5911
@date: 17/08/2019

Bacterial Effects on Caenorhabditis elegans Behaviour

A script written to analyse worm behaviour on various bacterial foods, and
compare Tierpsy-generated quantitative features of posture and locomotion to
highlight behavioural differences between foods.
The script does the following:

- Reads project metadata and fills in missing filepaths 
- Checks for Tierpsy results files (features/skeletons/intensities)
- Compiles feature summary results for selected results/treatments
- OPTIONAL: perform analyses with all / microbiome / miscellaneous bacterial strains
- OPTIONAL: perform analyses on L4-preconditioned / Day1-naive worms
- Performs statistical analyses to test for:
  1. Significant differences in N2 worm behaviour on different foods
     - T-test/ranksum tests (pairwise for all features) between each food and OP50 control
     - ANOVA/Kruskal tests across all foods
  2. Significant differences in controls over time
     - ANOVA/Kriskal tests for OP50 control across imaging days  
     - Impacts of confounders/random variables are considered
- Performs dimensionality reduction:
  1. Principal Components Analysis (PCA) 
  2. t-distributed Stochastic Neighbour Embedding (t-SNE)
  3. Uniform Manifold Projection (UMAP)
- Extracts siginificant features for each food for visualisation

"""

# TODO: Write script to compare differences between L4-prefed vs naive Day1 worms on different foods 


#%% PRE-AMBLE

# General imports
import os, sys, time, itertools, decimal, umap
import subprocess as sp
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib import transforms as transforms
from matplotlib.axes._axes import _log as mpl_axes_logger # Work-around for Axes3D plot colour warnings
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind, f_oneway, kruskal, zscore
from statsmodels.stats import multitest as smm # AnovaRM

# Custom imports
#sys.path.insert(0, '/Users/sm5911/Documents/GitHub/PhD_Project/Python') # OPTIONAL: Path to GitHub functions
from SM_find import changepath, lookforfiles, findworms
from SM_read import gettrajdata
from SM_calculate import ranksumtest
from SM_plot import manuallabelling, pcainfo
from SM_save import savefig
from SM_clean import filterSummaryResults


#%% GLOBAL PARAMETERS

PROJECT_NAME = 'MicrobiomeAssay'
PROJECT_ROOT_DIR = '/Volumes/behavgenom$/Saul/' + PROJECT_NAME
DATA_DIR = '/Volumes/behavgenom$/Priota/Data/' + PROJECT_NAME

verbose = True # Print statements to track script progress?

# Preprocessing
PROCESS_METADATA = True # Process metadata file?
PROCESS_FEATURE_SUMMARY_RESULTS = True # Process feature summary files?
CHECK_FEATURESN_RESULTS = True

# Which set of bacterial strains to analyse?
# OPTIONAL: Investigate microbiome/miscellaneous/all(both) strains
MICROBIOME = True
MISCELLANEOUS = False
ANALYSE_OP50_CONTROL_ACROSS_DAYS = False

# OPTIONAL: Select a 12-min video snippet to analyse
snippet = 0 # Which video segment to analyse? # TODO: Add option - 'ALL' - loop??

# OPTIONAL: L4-prefed worms (long exposure to food) or Naive adults
preconditioned_from_L4 = 'yes' # 'yes' for L4-prefed worms, or 'no' for naive worms

# Remove size-related features from analyses? (found to exhibit high variation across iamging dates)
filter_size_related_feats = True

# Statistics parameters
nan_threshold = 0.05 # Threshold proportion NaNs to drop features from analysis
p_value_threshold = 0.05 # P-vlaue threshold for statistical analyses

# Plot parameters
overwrite_top256 = True # Overwrite/replace previous existing plots?
show_plots = False

# Dimensionality reduction parameters
useTop256 = True                # Restrict dimensionality reduction inputs to Avelino's top 256 feature list?
test2use = 'ranksumtest'        # Preferred test results to use to select top features for dimensionality reduction
n_top_features = 10             # HCA - Number of top features to include in HCA (for union across foods HCA)
PCs_to_keep = 10                # PCA - Number of principal components to record
rotate = True                   # PCA - Rotate 3-D plots?
depthshade = False              # PCA - Shade colours on 3-D plots to show depth?
perplexity = [10,15,20,25,30]   # tSNE - Parameter range for t-SNE mapping eg. np.arange(5,51,1)
n_neighbours = [10,15,20,25,30] # UMAP - Number of neighbours parameter for UMAP projections eg. np.arange(3,31,1)
min_dist = 0.3                  # UMAP - Minimum distance parameter for UMAP projections

# Trajectory filtering parameters
thresh_movement = 10 # Threshold trajectory length for filtering tracked objects (for true worms)
thresh_duration = 50 # Threshold duration an object must be tracked for to be identified as a worm

MANUAL_LABELLING = False
FILTER_TRAJECTORY_DATA = False

# Select imaging date(s) for analysis
IMAGING_DATES = ['20190704','20190705','20190711','20190712','20190718','20190719',\
                 '20190725','20190726','20190801']

# Bacterial Strains
TEST_STRAINS = [# MICROBIOME STRAINS - Schulenburg et al microbiome (core set)
                'BIGB0170','BIGB0172','BIGB393','CENZENT1','JUB19','JUB44',\
                'JUB66','JUB134','MYB10','MYB11','MYB71','PM',\
                # MISCELLANEOUS STRAINS
                'MYB9','MYB27','MYB45','MYB53','MYB131','MYB181','MG1655','2783',\
                'MARBURG','DA1880','DA1885']

BACTERIAL_STRAINS = []
if MICROBIOME:
    BACTERIAL_STRAINS.extend(TEST_STRAINS[:12])
    BACTERIAL_STRAINS.insert(0, 'OP50')
if MISCELLANEOUS:
    BACTERIAL_STRAINS.extend(TEST_STRAINS[12:])
    if 'OP50' not in BACTERIAL_STRAINS:
        BACTERIAL_STRAINS.insert(0, 'OP50')

if MICROBIOME and MISCELLANEOUS:
    PATH = 'All_Strains'
elif MICROBIOME:
    PATH = 'Microbiome_Strains'
elif MISCELLANEOUS:
    PATH = 'Miscellaneous_Strains'
      
if preconditioned_from_L4:
    FOOD_EXPOSURE = 'L4_preconditioned'
else:
    FOOD_EXPOSURE = 'Day1_naive'

#%% PROCESS METADATA

# Use subprocess to call 'process_metadata.py', passing imaging dates as arguments to the script
if PROCESS_METADATA:
    print("\nProcessing metadata file...")
    metafilepath = os.path.join(DATA_DIR, "AuxiliaryFiles", "metadata.csv")
    process_metadata = sp.Popen([sys.executable, "process_metadata.py",\
                                 metafilepath, *IMAGING_DATES])
    process_metadata.communicate()

# Read metadata (CSV file)
metafilepath = os.path.join(PROJECT_ROOT_DIR, "metadata.csv")
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


#%% PROCESS FEATURE SUMMARY RESULTS

# Use subprocess to call 'process_metadata.py', passing imaging dates as arguments to the script
if PROCESS_FEATURE_SUMMARY_RESULTS:
    print("\nProcessing feature summary results...")
    process_feature_summary = sp.Popen([sys.executable, "process_feature_summary.py",\
                                        metafilepath, *IMAGING_DATES])
    process_feature_summary.communicate()
    
    
#%% CHECK FEATURES N RESULTS - TRY TO READ TRAJECTORY DATA

if CHECK_FEATURESN_RESULTS:
    ERROR_LIST = []
    print("Checking for missing or corrupt results files...")
    for i, maskedvideodir in enumerate(metadata.filename):
        featuresfilepath = changepath(maskedvideodir + ('/%.6d.hdf5' % snippet), returnpath='features')
        if i % 10 == 0:
            print("%d/%d" % (i, len(metadata.filename)))
        try:
            data = gettrajdata(featuresfilepath)
        except Exception as EE:
            print("ERROR: Failed to read file: %s\n%s" % (maskedvideodir, EE))
            ERROR_LIST.append(maskedvideodir)
    
    # Save error log to file
    if ERROR_LIST:
        print("WARNING: %d maskedvideos found with incomplete or missing results files!" % len(ERROR_LIST))
        error_outpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Unprocessed_MaskedVideos.txt')
        directory = os.path.dirname(error_outpath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        fid = open(error_outpath, 'w')
        print(ERROR_LIST, file=fid)
        fid.close()
    else:
        print("Check complete! All results files present.")


#%% FILTER SUMMARY RESULTS
# - Subset (rows) for desired bacterial strains only
# - Subset (rows) to look at results for given video snippet only
# - Subset (rows) to look at results for L4-preconditioned/naive adults worms
# - Remove (columns) features with all zeros 
# - Remove (columns) features with too many NaNs (>75%)
# - Remove (columns) size-related features that exhibit high variation across days
# - Impute remaining NaN values (with global mean OR with mean for that food)

# Read feature summary results
results_inpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'fullresults.csv')
full_results_df = pd.read_csv(results_inpath, dtype={"comments" : str})

# Record feature columns names for filtering/cleaning
feature_column_names = list(full_results_df.columns[25:])

results_df, droppedFeats_NaN, droppedFeats_allZero = filterSummaryResults(full_results_df,\
                                                     impute_NaNs_by_group=False,\
                                                     preconditioned_from_L4=preconditioned_from_L4,
                                                     featurecolnames=feature_column_names,\
                                                     snippet=snippet,\
                                                     nan_threshold=nan_threshold)

# =============================================================================
# droppedlist_out = os.path.join(PROJECT_ROOT_DIR, 'Results', PATH, 'Dropped_Features_NaN.txt')
# directory = os.path.dirname(droppedlist_out)
# if not os.path.exists(directory):
#     os.makedirs(directory)
# fid = open(droppedlist_out, 'w')
# print(*droppedFeats_NaN, file=fid)
# fid.close()
# 
# droppedlist_out = droppedlist_out.replace('NaN', 'AllZero')
# fid = open(droppedlist_out, 'w')
# print(*droppedFeats_allZero, file=fid)
# fid.close()
# =============================================================================

# Filter for selected bacterial strains
results_df = results_df[results_df['food_type'].isin(BACTERIAL_STRAINS)]

# Filter for selected imaging dates
results_df = results_df[results_df['date_yyyymmdd'].isin(IMAGING_DATES)]

# Filter out size-related features
if filter_size_related_feats:
    size_feat_keys = ['blob','box','width','length','area']
    size_features = []
    for feature in results_df.columns:
        for key in size_feat_keys:
            if key in feature:
                size_features.append(feature)         
    feats2keep = [feat for feat in results_df.columns if feat not in size_features]
    print("Dropped %d features that are size-related" % (len(results_df.columns)-len(feats2keep)))
    results_df = results_df[feats2keep]

# Record the bacterial strain names for use in analyses
bacterial_strains = list(np.unique(results_df['food_type'].str.upper()))
test_bacteria = [strain for strain in bacterial_strains if strain != "OP50"]

# Record feature column names + non-data columns to drop for statistics
colnames_all = results_df.columns
colnames_nondata = results_df.columns[:25]
colnames_data = results_df.columns[25:]


#%% ANALYSE OP50 CONTROL DATA - VARIATION ACROSS DAYS
# TODO: Fix script to investigate control OP50 variation (across days/temp/humidity/etc)

# Extract OP50 control data from feature summaries
OP50_control_df = results_df[results_df['food_type'].str.upper() == "OP50"]

# Save OP50 control data
PATH_OP50 = os.path.join(PROJECT_ROOT_DIR, 'Results', 'OP50_control',\
                            FOOD_EXPOSURE, 'snippet_{0}'.format(snippet),\
                            'OP50_control_results.csv')

directory = os.path.dirname(PATH_OP50) # make folder if it does not exist
if not os.path.exists(directory):
    os.makedirs(directory)
OP50_control_df.to_csv(PATH_OP50)

# Use subprocess to call 'food_behaviour_control.py', passing imaging dates as arguments to the script
SCRIPT_NAME_OP50 = 'run_control_analysis.py'

if ANALYSE_OP50_CONTROL_ACROSS_DAYS:
    print("\nRunning script: '%s' to analyse OP50 control data..." % SCRIPT_NAME_OP50)
    food_behaviour_control = sp.Popen([sys.executable, SCRIPT_NAME_OP50, PATH_OP50],\
                                      stdout=open(os.devnull, "w"), stderr=sp.STDOUT)
    food_behaviour_control.communicate()


#%% PERFORM STATISTICAL TESTS 
# - T-TESTS + RANK-SUM TESTS for pairwise differences in features between each food and OP50 control
# - ANOVAs + KRUSKAL-WALLIS TESTS for features that vary significantly across all foods
# - Bonferroni correction for multiple comparisons (many features are also highly correlated)
# - Records, for each food, the top features that are significantly different

#   TODO: Look to see if response data are homoscedastic / not normally distributed
tests = [ttest_ind, ranksumtest, f_oneway, kruskal]#, AnovaRM]

# Perform statistical tests
for test in tests:
    # Record number of decimal places of requested pvalue - for print statements to std_out
    p_decims = abs(int(decimal.Decimal(str(p_value_threshold)).as_tuple().exponent))
    
    # Record name of statistical test
    test_name = str(test).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]
    
    # T-tests and Ranksum Tests
    if test == ttest_ind or test == ranksumtest:
        # Pre-allocate dataframes for storing test statistics and p-values
        test_stats_df = pd.DataFrame(index=list(test_bacteria), columns=colnames_data)
        test_pvalues_df = pd.DataFrame(index=list(test_bacteria), columns=colnames_data)
        
        sigdifffeats_df = pd.DataFrame(index=test_pvalues_df.index, columns=['N_sigdiff_beforeBF','N_sigdiff_afterBF'])

        # Compare each strain to OP50: compute t-test statistics for each feature
        for t, food in enumerate(test_bacteria):
            print("Computing '%s' tests for %s vs OP50..." % (test_name, food))
                
            # Filter feature summary results for that food
            test_food_df = results_df[results_df['food_type'].str.upper() == food]
            
            # Drop non-data columns
            test_data = test_food_df.drop(columns=colnames_nondata)
            control_data = OP50_control_df.drop(columns=colnames_nondata)
                        
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

            # Perform t-tests/ranksums comparing between foods for each feature (max features = 4539)
            test_stats, test_pvalues = test(test_data, control_data)
            
            # Add t-test/ranksum results to out-dataframe
            test_stats_df.loc[food][shared_colnames] = test_stats
            test_pvalues_df.loc[food][shared_colnames] = test_pvalues
            
            # Record the names and number of significant features 
            sigdiff_feats = colnames_data[np.where(test_pvalues < p_value_threshold)]
            sigdifffeats_df.loc[food,'N_sigdiff_beforeBF'] = len(sigdiff_feats)
                    
        # Bonferroni corrections for multiple comparisons
        test_pvalues_corrected_df = pd.DataFrame(index=test_pvalues_df.index, columns=test_pvalues_df.columns)
        
        sigdifffeatslist = []
        for food in test_pvalues_df.index:
            # Locate pvalue results (row) for food
            food_pvals_df = test_pvalues_df.loc[food] # pd.Series object
            
            # Perform Bonferroni correction for multiple comparisons on t-test pvalues
            _corrArray = smm.multipletests(food_pvals_df.values, alpha=p_value_threshold, method='fdr_bh',\
                                           is_sorted=False, returnsorted=False)
            
            # Get pvalues for features that passed the Benjamini/Hochberg (non-negative) correlation test
            pvalues_corrected = _corrArray[1][_corrArray[0]]
            
            # Add pvalues to dataframe of corrected ttest pvalues
            test_pvalues_corrected_df.loc[food, _corrArray[0]] = pvalues_corrected
            
            # Record the names and number of significant features (after Bonferroni correction)
            sigdiff_feats = test_pvalues_corrected_df.columns[np.where(_corrArray[1] < p_value_threshold)]
            sigdifffeatslist.append(sigdiff_feats)
            sigdifffeats_df.loc[food,'N_sigdiff_afterBF'] = len(sigdiff_feats)
            
            if verbose:
                print("%d features found for worms on %s that differ from worms on OP50 (p<%.{0}f, Bonferroni)".format(p_decims)\
                      % (len(sigdiff_feats), food, p_value_threshold))
        if verbose:
            print('\n', sigdifffeats_df)
            
        # Compile dictionary to store full list of significant features for each food (t-tests/ranksum tests)
        sigfeats_out = {food:list(feats) for food, feats in zip(test_bacteria, sigdifffeatslist)}
        sigfeats_out = pd.DataFrame.from_dict(sigfeats_out, orient='index').T
        
        test_pvalues_df = test_pvalues_corrected_df
        
    # ANOVA and Kruskal-Wallis Tests    
    elif test == f_oneway or test == kruskal:
        print("Computing '%s' tests between foods for each feature..." % test_name)
        
        # Keep only necessary columns for 1-way ANOVAs
        test_cols = list(colnames_data)
        test_cols.insert(0, 'food_type')
        test_data = results_df[test_cols]
        
        # Drop columns that contain only zeros
        n_cols = len(test_data.columns)
        test_data = test_data.drop(columns=test_data.columns[(test_data == 0).all()])
        
        zero_cols = n_cols - len(test_data.columns)
        if zero_cols > 0:
            print("Dropped %d feature summaries for %s (all zeros)" % (zero_cols, test_name))

        # Perform 1-way ANOVAs for each feature between the test bacteria 
        # (NB: Post-hoc analysis (eg.Tukey HSD) allows for pairwise comparisons between foods for each feature)
        test_pvalues_df = pd.DataFrame(index=['stat','pval'], columns=colnames_data)
        for feature in test_data.columns[1:]:
            # Perform test and capture outputs: test statistic + p value
            test_stat, test_pvalue = test(*[test_data[test_data['food_type']==food][feature]\
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
        if verbose:
            print("Complete!\n%d features exhibit significant differences between foods ('%s' test, Bonferroni)"\
                  % (len(sigdiff_feats), test_name))
            print('\n', test_pvalues_df)
        
        # Compile list to store names of significant features
        sigfeats_out = pd.Series(test_pvalues_df.columns[np.where(test_pvalues_df.loc['pval_corrected'] < p_value_threshold)])
        sigfeats_out.name = 'significant_features_' + test_name
        sigfeats_out = pd.DataFrame(sigfeats_out)
        
    # Save test statistics to file
    stats_outpath = os.path.join(PROJECT_ROOT_DIR, 'Results', PATH, 'Stats',\
                                 FOOD_EXPOSURE, 'snippet_{0}'.format(snippet),\
                                 test_name, test_name + '_results.csv')
    sigfeats_outpath = stats_outpath.replace('_results.csv', '_significant_features.csv')
    directory = os.path.dirname(stats_outpath) # make folder if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    test_pvalues_df.to_csv(stats_outpath) # Save test results as CSV
    sigfeats_out.to_csv(sigfeats_outpath, index=False) # Save feature list as text file

# TODO: MANOVA (because response data are not independent - a dimensionality reduction technique involving eigenvalues)?
    
    
#%% VARIABLES - Plotting and feature analysis
    
# Make dictionary of colours for plotting
colour_dictionary = dict(zip(bacterial_strains, sns.color_palette(palette="gist_rainbow",\
                                                                  n_colors=len(bacterial_strains))))

# Divide dataframe into 2 dataframes: data (feature summaries) and non-data (metadata)
data_df = results_df[colnames_data]
nondata_df = results_df[colnames_nondata]   

# Read list of important features (highlighted by previous research - see Javer, 2018 paper)
featslistpath = os.path.join(PROJECT_ROOT_DIR, 'Data', 'top256_tierpsy_no_blob_no_eigen_only_abs_no_norm.csv')
top256features = pd.read_csv(featslistpath)

# Take first set of 256 features (it does not matter which set is chosen)
top256features = top256features[top256features.columns[0]]

n_feats = len(top256features)
top256features = [feat for feat in top256features if feat in data_df.columns]
print("Dropping %d size-related features from Top256" % (n_feats - len(top256features)))
 

#%% BOX PLOTS - INDIVIDUAL PLOTS OF TOP RANKED FEATURES FOR EACH FOOD
# - Rank features by pvalue significance (lowest first) and select the Top 10 features for each food
# - Plot boxplots of the most important features for each food compared to OP50
# - Plot features separately with feature as title and in separate folders for each food

# Load test results (pvalues) for plotting
test_inpath = os.path.join(PROJECT_ROOT_DIR, 'Results', PATH, 'Stats', FOOD_EXPOSURE, 'snippet_{0}'.format(snippet),\
                           test2use, test2use + '_results.csv')
test_pvalues_df = pd.read_csv(test_inpath, index_col=0)

# NB: non-parametric ranksum test preferred over t-test
print("\nPlotting box plots of top %d highest ranked features by '%s':\n" % (n_top_features, test2use))
for i, food in enumerate(test_pvalues_df.index):
    pvals = test_pvalues_df.loc[food]
    n_sigfeats = sum(pvals < p_value_threshold)
    n_nonnanfeats = np.logical_not(pvals.isna()).sum()
    if pvals.isna().all():
        print("No signficant features found for %s!" % food)
    elif n_sigfeats > 0:       
        # Rank p-values in ascending order
        ranked_pvals = pvals.sort_values(ascending=True)
        topfeats = ranked_pvals[:n_top_features] # Select the top ranked p-values
        topfeats = topfeats.dropna(axis=0) # Drop NaNs
        topfeats = topfeats[topfeats < p_value_threshold] # Drop non-sig feats           
        if verbose:
            if n_sigfeats < n_top_features:
                print("Only %d significant features found for %s" % (n_sigfeats, food))
            print("\nTop %d features for %s:\n" % (len(topfeats), food))
            print(*[feat + '\n' for feat in list(topfeats.index)])

        # Subset feature summary results for test-food + OP50-control only
        plot_df = results_df[np.logical_or(results_df['food_type'].str.upper()=="OP50",\
                                           results_df['food_type'].str.upper()==food)] 
        # Colour/legend parameters
        labels = list(plot_df['food_type'].str.upper().unique())
        colour_dict = {key:value for key, value in colour_dictionary.items() if key in labels}
          
        # OPTIONAL: Plot box plots for cherry-picked (relatable) features
#            topfeats = pd.Series(index=['curvature_midbody_abs_90th'])
                                    
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
            plots_outpath = os.path.join(PROJECT_ROOT_DIR, 'Results', PATH, 'Plots',\
                                         FOOD_EXPOSURE, 'snippet_{0}'.format(snippet), food,\
                                         feature + '_' + test2use + '_{0}.eps'.format(f + 1))
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

# OPTIONAL: Plot cherry-picked features
features2plot = ['speed_90th','speed_50th','speed_hips_50th','speed_midbody_50th',\
                 'curvature_neck_abs_50th','curvature_midbody_abs_50th','curvature_hips_abs_50th','curvature_tail_abs_50th',\
                 'major_axis_50th',\
                 'angular_velocity_head_base_abs_50th','angular_velocity_tail_base_abs_50th','angular_velocity_neck_abs_50th','angular_velocity_midbody_abs_50th']

# Seaborn boxplots with swarmplot overlay for each feature - saved to file
tic = time.time()
plt.ioff()
sns.set(color_codes=True); sns.set_style('darkgrid')
for f, feature in enumerate(features2plot):
    plotpath_out = os.path.join(PROJECT_ROOT_DIR, "Results", PATH, "Plots", \
                                FOOD_EXPOSURE, 'snippet_{0}'.format(snippet), "All",\
                                "Top256_Javer_2018", feature + '.eps')
    if not os.path.exists(plotpath_out) or overwrite_top256:
        print("Plotting feature: '%s'" % feature)
        # Seaborn boxplots for each feature
        fig = plt.figure(figsize=[12,7])
        ax = fig.add_subplot(1,1,1)
        sns.boxplot(x="food_type", y=feature, data=results_df, showfliers=False, showmeans=True,\
                         meanprops={"marker":"x", "markersize":5, "markeredgecolor":"k"},\
                         flierprops={"marker":"x", "markersize":15, "markeredgecolor":"r"})
        sns.swarmplot(x="food_type", y=feature, data=results_df, s=10, marker=".", color='k')
        ax.set_xlabel('Bacterial Strain (Test Food)', fontsize=15, labelpad=12)
        ax.set_ylabel(feature, fontsize=15, labelpad=12)
        ax.set_title(feature, fontsize=20, pad=20)
        labels = [lab.get_text().upper() for lab in ax.get_xticklabels()]
        labels.remove('OP50')
        for l, food in enumerate(labels):
            ylim = results_df[results_df['food_type'].str.upper()==food][feature].max()
            pval = test_pvalues_df.loc[food, feature]
            if isinstance(pval, float) and pval < p_value_threshold:
                trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                ax.text(l + 0.75, 0, '(p={:g})'.format(float('{:.2g}'.format(pval))),\
                        fontsize=10, color='k', verticalalignment='bottom', transform=trans)
        # Save boxplots
        directory = os.path.dirname(plotpath_out)
        if not os.path.exists(directory):
            os.makedirs(directory)
        savefig(plotpath_out, tellme=False, saveFormat='eps')
        plt.close(fig) # Close plot
            
toc = time.time()
print("Time taken: %.1f seconds" % (toc - tic))


#%% NORMALISE THE DATA

# Normalise the data (z-scores)
zscores = data_df.apply(zscore, axis=0)

# Drop features with NaN values after normalising
zscores.dropna(axis=1, inplace=True)

if useTop256:
    print("Using top 256 features for dimensionality reduction...")
    zscores = zscores[top256features]

# NB: In general, for the curvature and angular velocity features we should only 
# use the 'abs' versions, because the sign is assigned based on whether the worm 
# is on its left or right side and this is not known for the multiworm tracking data


#%% HIERARCHICAL CLUSTERING (HEATMAP) - ALL FOODS - AVELINO TOP 256
# - Scikit-learn clustermap of features by foods, to see if they cluster into
#   groups for each food - does OP50 control form a nice group?
                    
plotroot = os.path.join(PROJECT_ROOT_DIR, 'Results', PATH, 'Plots', FOOD_EXPOSURE,\
                        'snippet_{0}'.format(snippet), 'All', 'HCA')

# Heatmap (clustergram) of Top10 features per food (n=45)
plt.close('all')
row_colours = nondata_df['food_type'].map(colour_dictionary)
sns.set(font_scale=0.6)
g = sns.clustermap(zscores, row_colors=row_colours, figsize=[18,15], xticklabels=3)
plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
patches = []
for l, key in enumerate(colour_dictionary.keys()):
    patch = mpatches.Patch(color=colour_dictionary[key], label=key)
    patches.append(patch)
plt.legend(handles=patches, labels=colour_dictionary.keys(),\
           borderaxespad=0.4, frameon=False, loc=(-3.5, -9), fontsize=12)
plt.subplots_adjust(top=0.985,bottom=0.385,left=0.09,right=0.945,hspace=0.2,wspace=0.2)

# Save clustermap and features of interest
cluster_outpath = os.path.join(plotroot, 'HCA_Top256_features.eps')
directory = os.path.dirname(cluster_outpath)
if not os.path.exists(directory):
    os.makedirs(directory)
savefig(cluster_outpath, tight_layout=False, tellme=True, saveFormat='eps')
plt.show(); plt.pause(1)

                    
#%% PRINCIPAL COMPONENTS ANALYSIS (PCA) - ALL FOODS Top256

# TODO: Plot features that have greatest influence on PCA (eg. PC1)

tic = time.time()
plotroot = os.path.join(PROJECT_ROOT_DIR, 'Results', PATH, 'Plots', FOOD_EXPOSURE,\
                        'snippet_{0}'.format(snippet), 'All', 'PCA')

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
projected_df.set_index(results_df.index, inplace=True) # Do not lose video snippet index position
projected_df = pd.concat([results_df[colnames_nondata], projected_df], axis=1)


#%% 2D Plot - first 2 PCs - ALL FOODS

# Plot first 2 principal components
plt.close('all'); plt.ion()
plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=[10,10])

# Create colour palette for plot loop
palette = itertools.cycle(sns.color_palette("gist_rainbow", len(bacterial_strains)))
for food in bacterial_strains:
    food_projected_df = projected_df[projected_df['food_type'].str.upper()==food]
    sns.scatterplot(food_projected_df['PC1'], food_projected_df['PC2'], color=next(palette), s=100)
ax.set_xlabel('Principal Component 1', fontsize=20, labelpad=12)
ax.set_ylabel('Principal Component 2', fontsize=20, labelpad=12)
if useTop256:
    ax.set_title('Top256 features 2-Component PCA', fontsize=20)
else: 
    ax.set_title('All features 2-Component PCA', fontsize=20)
plt.tight_layout(rect=[0.04, 0, 0.84, 0.96])
ax.legend(bacterial_strains, frameon=False, loc=(1, 0.1), fontsize=15)
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
palette = itertools.cycle(sns.color_palette("gist_rainbow", len(bacterial_strains)))

for food in bacterial_strains:
    food_projected_df = projected_df[projected_df['food_type'].str.upper()==food]
    ax.scatter(xs=food_projected_df['PC1'], ys=food_projected_df['PC2'], zs=food_projected_df['PC3'],\
               zdir='z', s=50, c=next(palette), depthshade=depthshade)
ax.set_xlabel('Principal Component 1', fontsize=15, labelpad=12)
ax.set_ylabel('Principal Component 2', fontsize=15, labelpad=12)
ax.set_zlabel('Principal Component 3', fontsize=15, labelpad=12)
if useTop256:
    ax.set_title('Top256 features 3-Component PCA', fontsize=20)
else: 
    ax.set_title('All features 3-Component PCA', fontsize=20)
ax.legend(bacterial_strains, frameon=False, fontsize=12)
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

plotroot = os.path.join(PROJECT_ROOT_DIR, 'Results', PATH, 'Plots', FOOD_EXPOSURE,\
                        'snippet_{0}'.format(snippet), 'All', 'tSNE')

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
    tSNE_results_df.set_index(results_df.index, inplace=True) # Do not lose video snippet index position
    tSNE_results_df = pd.concat([results_df[colnames_nondata], tSNE_results_df], axis=1)
    
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
    palette = itertools.cycle(sns.color_palette("gist_rainbow", len(bacterial_strains))) # 'gist_rainbow'
    
    for food in bacterial_strains:
        food_tSNE_df = tSNE_results_df[tSNE_results_df['food_type'].str.upper()==food]
        sns.scatterplot(food_tSNE_df['tSNE_1'], food_tSNE_df['tSNE_2'], color=next(palette), s=100)
    plt.tight_layout(rect=[0.04, 0, 0.84, 0.96])
    ax.legend(bacterial_strains, frameon=False, loc=(1, 0.1), fontsize=15)
    ax.grid()
    
    plotpath = os.path.join(plotroot, 'tSNE_perplex={0}.eps'.format(perplex))
    directory = os.path.dirname(plotpath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    savefig(plotpath, tight_layout=True, tellme=True, saveFormat='eps')
    
    plt.show(); plt.pause(2)
    
    
#%% Uniform Manifold Projection (UMAP)

plotroot = os.path.join(PROJECT_ROOT_DIR, 'Results', PATH, 'Plots', FOOD_EXPOSURE,\
                        'snippet_{0}'.format(snippet), 'All', 'UMAP')

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
    UMAP_projection_df.set_index(results_df.index, inplace=True) # Do not lose video snippet index position
    UMAP_projection_df = pd.concat([results_df[colnames_nondata], UMAP_projection_df], axis=1)
    
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
    palette = itertools.cycle(sns.color_palette("gist_rainbow", len(bacterial_strains)))
    
    for food in bacterial_strains:
        food_UMAP_df = UMAP_projection_df[UMAP_projection_df['food_type'].str.upper()==food]
        sns.scatterplot(food_UMAP_df['UMAP_1'], food_UMAP_df['UMAP_2'], color=next(palette), s=100)
    plt.tight_layout(rect=[0.04, 0, 0.84, 0.96])
    ax.legend(bacterial_strains, frameon=False, loc=(1, 0.1), fontsize=15)
    ax.grid()
    
    plotpath = os.path.join(plotroot, 'UMAP_n_neighbours={0}.eps'.format(n))
    directory = os.path.dirname(plotpath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    savefig(plotpath, tight_layout=True, tellme=True, saveFormat='eps')
    
    plt.show(); plt.pause(2)

 
#%% Linear Discriminant Analysis (LDA)
# - Projection of data along top 2 most influential eigenvectors (not any one feature)

# TODO: Linear Discriminant Analysis??
    
    
#%% MANUAL LABELLING OF FOOD REGIONS (To check time spent feeding throughout video)

# Return list of pathnames for masked videos in the data directory under given imaging dates
maskedfilelist = []
date_total = []
for i, expDate in enumerate(IMAGING_DATES):
    tmplist = lookforfiles(os.path.join(DATA_DIR, "MaskedVideos", expDate), ".*.hdf5$")
    date_total.append(len(tmplist))
    maskedfilelist.extend(tmplist)
print("%d masked videos found for imaging dates provided:\n%s" % (len(maskedfilelist), [*zip(IMAGING_DATES, date_total)]))

first_snippets = [snip for snip in maskedfilelist if ('/%.6d.hdf5' % snippet) in snip]
print("\nManual labelling:\n%d masked video snippets found for %d assay recordings (duration: 2hrs)"\
      % (len(maskedfilelist), len(first_snippets)))

# Manual labelling of food regions in each assay using 1st video snippet, 1st frame
if MANUAL_LABELLING:
    plt.ion() # Interactive plotting (for user input when labelling plots)
    tic = time.time()
    for i, maskedfilepath in enumerate(first_snippets):    
        # Manually outline + assign labels to food regions + save coordinates + trajectory overlay to file           
        manuallabelling(maskedfilepath, n_poly=1, out_dir='MicrobiomeAssay', save=True, skip=True)       
    print("Manual labelling complete!\n(Time taken: %d seconds.)\n" % (time.time() - tic))

# TODO: Automate food labelling using (U-Net) ML algorithm -- pre-trained already by Luigi


#%% PLOT + FILTER WORM TRAJECTORIES

# TODO: Filter worm trajectories
if FILTER_TRAJECTORY_DATA:
    for i, maskedvideodir in enumerate(metadata.filename):
        featuresfilepath = changepath(maskedvideodir + ('/%.6d.hdf5' % snippet), returnpath='features')
        try:
            trajectory_df = gettrajdata(featuresfilepath)
            print(trajectory_df.head())
            findworms(trajectory_df, threshold_move=thresh_movement,\
                      threshold_time=thresh_duration,\
                      tellme=True)
        except Exception as EE:
            print(EE)


#%%
# =============================================================================
#         if test == AnovaRM:
#             # Create long-format df with condition treatment columns for food_type: FOOD vs OP50
#             dfs2concat = {food: test_data, 'op50': control_data}
#             test_data_aovRM = pd.concat(dfs2concat).reset_index(drop=False)
#             test_data_aovRM = pd.melt(frame = test_data_aovRM,\
#                                       id_vars = 'level_0',\
#                                       value_vars = test_data_aovRM.columns[2:],\
#                                       var_name = 'Feature',\
#                                       value_name = 'Summary')
#             aovrm = AnovaRM(data_df=test_data_aovRM, depvar=, subject='level_0', within=['Feature'])
##            data_df = pd.read_csv('../Sandbox/example_AnovaRM.csv', sep=',')
##            res = AnovaRM(data_df, depvar='RT', subject='SubID', within=['TrialType'], aggregate_func='mean')
##            print(res.fit())
#         else:  
# =============================================================================

# =============================================================================
# #%% HIERARCHICAL CLUSTERING (HEATMAP) - ALL FOODS - STATS TEST TOP FEATURES
# 
# # Read significant features list for each food
# sigfeats_in = os.path.join(PROJECT_ROOT_DIR, 'Results', PATH, 'Stats', FOOD_EXPOSURE, 'snippet_{0}'.format(snippet),\
#                            test2use, test2use + '_significant_features.csv')
# sigfeats_df = pd.read_csv(sigfeats_in)
#
# # Normalise the data (z-scores)
# zscores = data_df.apply(zscore, axis=0)
# 
# # Drop features with NaN values after normalising
# zscores.dropna(axis=1, inplace=True)
# 
# # Store top features for each food in dictionary
# food_sigfeats_dict = {food:None for food in test_bacteria}
# for food in food_sigfeats_dict.keys():
#     sigfeats_food = list(sigfeats_df[food])
#     sigfeats_food = [feature for i, feature in enumerate(sigfeats_food[:n_top_features])\
#                      if type(sigfeats_food[i])==str]
#     if len(sigfeats_food) > 0:
#         # Get pvalues for features
#         pvals = test_pvalues_df.loc[food, sigfeats_food]
#         # Rank features by significance (test p-value)
#         ranked_pvals = pvals.sort_values(ascending=True)   
#         # Warn if less than requested number of sig feats were in fact significant
#         if len(ranked_pvals) < n_top_features:
#             n_feats2use = len(ranked_pvals)
#             print("WARNING: Only %d/%d significant features found for '%s'"\
#                   % (n_feats2use, n_top_features, food))
#         else:
#             n_feats2use = n_top_features
#             
#         # Record the top most important features
#         sigfeats_food = ranked_pvals[:n_feats2use]
#         food_sigfeats_dict[food] = list(sigfeats_food.index)      
#     
# feats2plt = []
# for food, values in food_sigfeats_dict.items():
#     if values:
#         feats2plt = list(set(values) | set(feats2plt))
# print("Performing cluster analysis with %d significant features (Top%d union)" % (len(feats2plt), n_top_features))
# 
# # Heatmap (clustergram) of Top10 features per food (n=45)
# plt.close('all')
# row_colours = nondata_df['food_type'].map(colour_dictionary)
# sns.set(font_scale=0.8)
# g = sns.clustermap(zscores[feats2plt], row_colors=row_colours, figsize=[18,15])
# plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
# patches = []
# for l, key in enumerate(colour_dictionary.keys()):
#     patch = mpatches.Patch(color=colour_dictionary[key], label=key)
#     patches.append(patch)
# plt.legend(handles=patches, labels=colour_dictionary.keys(),\
#            borderaxespad=0.4, frameon=False, loc=(-3.5, -9), fontsize=12)
# plt.subplots_adjust(top=0.985,bottom=0.4,left=0.09,right=0.94,hspace=0.2,wspace=0.2)
# 
# # Save clustermap and features of interest
# cluster_outpath = os.path.join(plotroot, 'HCA_n={0}_features_'.format(len(feats2plt)) + test_name + '.eps')
# directory = os.path.dirname(cluster_outpath)
# if not os.path.exists(directory):
#     os.makedirs(directory)
# savefig(cluster_outpath, tight_layout=False, tellme=True, saveFormat='eps')
# 
# clusterfeats_outpath = os.path.join(plotroot, 'HCA_features_list_' + test_name + '.txt')
# fid = open(clusterfeats_outpath, 'w')
# print(feats2plt, file=fid)
# fid.close()
# 
# plt.show(); plt.pause(1)
# =============================================================================

