#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bacterial effects on Caenorhabditis elegans behaviour 
- Microbiome strains
- Bacillus and E. coli strains

This script reads Tierpsy results for experimental data collected as oart of the 
preliminary screening of Schulenberg Lab bacterial strains isolated from the C. 
elegans gut microbiome, as well as several other strains previously reported
to affect C. elegans behaviour.

It does the following: 
    - Reads the project metadata file, and completes missing filepath info
    - Checks for results files (features/skeletons/intensities)
    - Extracts feature summaries of interest for visualisation
    - Statistical analysis of feature summaries to look for significant differences
      in N2 worm behaviour on different foods
    - PCA, t-SNE and UMAP to decipher the most important features influencing 
      these differences
    - Visualisation of most significant features for each food

@author: sm5911
@date: 07/07/2019

"""


#%% IMPORTS

# General imports
import os, sys, time, itertools, decimal#, umap (NB: Need to install umap library in anaconda first!)
import subprocess as sp
import pandas as pd
import numpy as np
import seaborn as sns; sns.set(color_codes=True)
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from scipy.stats import ttest_ind, f_oneway, kruskal, zscore
from statsmodels.stats import multitest as smm # AnovaRM
from statsmodels.stats.multicomp import MultiComparison, pairwise_tukeyhsd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Custom imports
#sys.path.insert(0, '/Users/sm5911/Documents/GitHub/PhD_Project/Python') # OPTIONAL: Path to GitHub functions
from SM_find import changepath, lookforfiles, findworms
from SM_read import gettrajdata
from SM_calculate import ranksumtest
from SM_plot import manuallabelling, pcainfo
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


#%% PROCESS METADATA

# NB: Imaging date '20190711' was found to be an outlier experiment day and will be dropped from analysis
# - OP50 control differed significantly on this date
# - Coincidentally, the 2 Bacillus strains (Str+) were tested on this date

# Use subprocess to call 'process_metadata.py', passing imaging dates as arguments to the script
print("\nProcessing metadata file...")
process_metadata = sp.Popen([sys.executable, "process_metadata.py", *IMAGING_DATES])
process_metadata.communicate()


#%% PROCESS FEATURE SUMMARY RESULTS

# Use subprocess to call 'process_metadata.py', passing imaging dates as arguments to the script
print("\nProcessing feature summary results...")
metadata_filepath = os.path.join(PROJECT_ROOT_DIR, "metadata.csv")
process_feature_summary = sp.Popen([sys.executable, "process_feature_summary.py", metadata_filepath, *IMAGING_DATES])
process_feature_summary.communicate()


#%% READ METADATA

# Read metadata (CSV file)
metafilepath = os.path.join(PROJECT_ROOT_DIR, "metadata.csv")
metadata = pd.read_csv(metafilepath)
print("\nMetadata file loaded.")


#%% REMOVE MISSING FILEPATHS FROM METADATA

# Subset metadata to remove remaining entries with missing filepaths
is_filename = [isinstance(path, str) for path in metadata['filename']]
if any(list(~np.array(is_filename))):
    print("WARNING: Could not find filepaths for %d entries in metadata.\n\t These files will be omitted from further analyses!" \
          % sum(list(~np.array(is_filename))))
    metadata = metadata[list(np.array(is_filename))]
    # Reset index
    metadata.reset_index(drop=True, inplace=True)
    

#%% CHECK FEATURES N RESULTS - TRY TO READ TRAJECTORY DATA

if verbose:
    ERROR_LIST = []
    for i, maskedvideodir in enumerate(metadata.filename):
        if i % 10 == 0:
            print("Checking for missing/corrupt results files: %d/%d" % (i, len(metadata.filename)))
        try:
            featuresfilepath = changepath(maskedvideodir + '/000000.hdf5', returnpath='features')
            data = gettrajdata(featuresfilepath)
        except Exception as EE:
            print("ERROR:", EE)
            ERROR_LIST.append(maskedvideodir)
    
    # Save error log to file
    if ERROR_LIST:
        print("WARNING: %d maskedvideos found with incomplete or missing results files!" % len(ERROR_LIST))
        error_outpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Error_logs', 'Unprocessed_MaskedVideos.txt')
        directory = os.path.dirname(error_outpath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        fid = open(error_outpath, 'w')
        print(ERROR_LIST, file=fid)
        fid.close()
    else:
        print("Check complete! All results files present.")


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


#%% PERFORM STATISTICAL TESTS 
# - Looks at first segment only  
# - Starts with L4-prefed worms (long exposure to food)
#   TODO: Looks to see if response data are homoscedastic or not normally distributed
# - T-TESTS + RANK-SUM TESTS for pairwise differences in features between each food and OP50 control
# - ANOVAs + KRUSKAL-WALLIS TESTS for features that vary significantly across all foods (incl. OP50/DA1880/DA1885)
# - Bonferroni correction for multiple comparisons (also, many features are highly correlated)
# - For each food, produces a list of top features that are significantly different

tests = [ttest_ind, ranksumtest, f_oneway, kruskal]#, AnovaRM]

test_bacteria = [strain for strain in bacterial_strains if strain != 'op50']

# Filter feature summary results for OP50 control
OP50_control_df = L4_1st_snippets_df[L4_1st_snippets_df['food_type'].str.lower() == "op50"]

# Select non-data columns to drop for statistics
non_data_columns = L4_1st_snippets_df.columns[0:25]

# Record a list of feature column names
feature_colnames = L4_1st_snippets_df.columns[25:]

# Perform statistical tests
for test in tests:
    # Record number of decimal places of requested pvalue - for print statements to std_out
    p_decims = abs(int(decimal.Decimal(str(p_value_threshold)).as_tuple().exponent))
    
    # Record name of statistical test
    test_name = str(test).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]
    
    # T-tests and Ranksum Tests
    if test == ttest_ind or test == ranksumtest:
        # Pre-allocate dataframes for storing test statistics and p-values
        test_stats_df = pd.DataFrame(index=list(test_bacteria), columns=feature_colnames)
        test_pvalues_df = pd.DataFrame(index=list(test_bacteria), columns=feature_colnames)
        
        sigdifffeats_df = pd.DataFrame(index=test_pvalues_df.index, columns=['N_sigdiff_beforeBF','N_sigdiff_afterBF'])

        # Compare each strain to OP50: compute t-test statistics for each feature
        for t, food in enumerate(test_bacteria):
            print("Computing '%s' tests for %s vs OP50..." % (test_name, food.upper()))
                
            # Filter feature summary results for that food
            test_food_df = L4_1st_snippets_df[L4_1st_snippets_df['food_type'].str.lower() == food]
            
            # Drop non-data columns
            test_data = test_food_df.drop(columns=non_data_columns)
            control_data = OP50_control_df.drop(columns=non_data_columns)
                        
            # Drop columns that contain only zeros
            n_cols = len(test_data.columns)
            test_data.drop(columns=test_data.columns[(test_data == 0).all()], inplace=True)
            control_data.drop(columns=control_data.columns[(control_data == 0).all()], inplace=True)
            
            # Use only shared feature summaries between control data and test data
            shared_colnames = control_data.columns.intersection(test_data.columns)
            test_data = test_data[shared_colnames]
            control_data = control_data[shared_colnames]

            zero_cols = n_cols - len(test_data.columns)
            if verbose:
                print("Dropped %d feature summaries for %s (all zeros)" % (zero_cols, food.upper()))

            # Perform t-tests/ranksums comparing between foods for each feature (max features = 4539)
            test_stats, test_pvalues = test(test_data, control_data)
            
            # Add t-test/ranksum results to out-dataframe
            test_stats_df.loc[food][shared_colnames] = test_stats
            test_pvalues_df.loc[food][shared_colnames] = test_pvalues
            
            # Record the names and number of significant features 
            sigdiff_feats = feature_colnames[np.where(test_pvalues < p_value_threshold)]
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
                print("%d features found for worms on %s that differ from worms on OP50 (p<%.{0}f, after Bonferroni correction for multiple comparisons)".format(p_decims) %\
                      (len(sigdiff_feats), food.upper(), p_value_threshold))
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
        test_cols = list(feature_colnames)
        test_cols.insert(0, 'food_type')
        test_data = L4_1st_snippets_df[test_cols]

        # Perform 1-way ANOVAs for each feature between the test bacteria 
        # (NB: Post-hoc analysis (eg.Tukey HSD) allows for pairwise comparisons between foods for each feature)
        test_pvalues_df = pd.DataFrame(index=['stat','pval'], columns=feature_colnames)
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
            print("Complete!\n%d features found to exhibit significant differences between test bacteria by %s test (after Bonferroni correction for mutliple comparisons)"\
                  % (len(sigdiff_feats), test_name))
            print('\n', test_pvalues_df)
        
        # Compile list to store names of significant features
        sigfeats_out = pd.Series(test_pvalues_df.columns[np.where(test_pvalues_df.loc['pval_corrected'] < p_value_threshold)])
        sigfeats_out.name = 'significant_features_' + test_name
        sigfeats_out = pd.DataFrame(sigfeats_out)
        
    # Save test statistics to file
    stats_outpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Stats', 'L4_snippet_1',\
                                 test_name, test_name + '_results.csv')
    sigfeats_outpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Stats', 'L4_snippet_1',\
                                    test_name, test_name + '_significant_features.csv')
    
    # Make parent directory if it does not exist
    directory = os.path.dirname(stats_outpath)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    # Save test results to CSV
    test_pvalues_df.to_csv(stats_outpath)
    
    # Save feature list to text file
    sigfeats_out.to_csv(sigfeats_outpath, index=False)

# TODO: MANOVA (because response data are not independent, dimensionality reduction techniques involving eigenvalues)?
# TODO: Make into a script/functions and use to look for differences between L4-prefed vs. Naive adults 
# - time dependence? does it take time for food-behavioural effects to emerge?
# naive_1st_snippets_df = first_snippets_df[first_snippets_df['preconditioned_from_L4'].str.lower() == "no"]
# test_stats_L4vsNaive, test_pvalues_L4vsNaive = test(naive_1st_snippets_df.drop(columns=non_data_columns),\
#                                                     OP50_control_df.drop(columns=non_data_columns), axis=0)
   
#%% PLOT SUMMARY STATISTICS (for Avelino's Top 256)
# - Investigate Avelino's top 256 features to look for any differences between foods (see paper: Javer et al, 2018)
# - Plot features that show significant differences from behaviour on OP50

tic = time.time()
test_names = ['ranksumtest','ttest_ind']

for test_name in test_names:
    # Load test results (pvalues) for plotting
    test_inpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Stats', 'L4_snippet_1', test_name, test_name + '_results.csv')
    test_pvalues_df = pd.read_csv(test_inpath, index_col=0)
    
    # Read list of important features (highlighted by previous research - see Javer, 2018 paper)
    featslistpath = os.path.join(PROJECT_ROOT_DIR,'Data','top256_tierpsy_no_blob_no_eigen_only_abs_no_norm.csv')
    top256features = pd.read_csv(featslistpath)
    
    # Take first set of 256 features (it does not matter which set is chosen)
    top256features = top256features[top256features.columns[0]]
    
    # Only investigate features for which we have results
    features2plot = [feature for feature in top256features if feature in test_pvalues_df.columns]
    
    # Filter for features displaying significant differences between any food and OP50 control
    features2plot = [feature for feature in features2plot if (test_pvalues_df[feature] < p_value_threshold).any()]
    
    # Seaborn boxplots with swarmplot overlay for each feature - saved to file
    plt.ioff()
    for f, feature in enumerate(features2plot):
        # NB: Currently n=6 L4 first snippet replicates for each bacterial food (1 imaging date per food) to date
        plotpath_out = os.path.join(PROJECT_ROOT_DIR, "Results", "Plots", "L4_snippet_1", "Top256_Javer_2018",\
                            test_name, feature + '_L4_snippet_1' + '.eps')
        if not os.path.exists(plotpath_out):
            print("Plotting feature: '%s'" % feature)
            # Seaborn boxplots for each feature
            fig = plt.figure(figsize=[12,7])
            ax = fig.add_subplot(1,1,1)
            sns.boxplot(x="food_type", y=feature, data=L4_1st_snippets_df, showfliers=False, showmeans=True,\
                             meanprops={"marker":"x", "markersize":5, "markeredgecolor":"k"},\
                             flierprops={"marker":"x", "markersize":15, "markeredgecolor":"r"})
            sns.swarmplot(x="food_type", y=feature, data=L4_1st_snippets_df, s=10, marker=".", color='k')
            ax.set_xlabel('Bacterial Strain (Test Food)', fontsize=15, labelpad=12)
            ax.set_ylabel(feature, fontsize=15, labelpad=12)
            ax.set_title(feature, fontsize=20, pad=20)
            food_labels = [lab.get_text().lower() for lab in ax.get_xticklabels()]
            food_labels.remove('op50')
            for l, food in enumerate(food_labels):
                ylim = L4_1st_snippets_df[L4_1st_snippets_df['food_type'].str.lower()==food][feature].max()
                pval = test_pvalues_df.loc[food, feature]
                if isinstance(pval, float) and pval < p_value_threshold:
                    ax.text(l + 0.9, ylim, '*', fontsize=25, color='k', verticalalignment='bottom')
                    ax.text(l + 0.7, ylim + ylim/10, 'p={:g}'.format(float('{:.2g}'.format(pval))),\
                            fontsize=10, color='k', verticalalignment='bottom')
            # Save boxplots
            directory = os.path.dirname(plotpath_out)
            if not os.path.exists(directory):
                os.makedirs(directory)
            savefig(plotpath_out, tellme=False, saveFormat='eps')
            plt.close(fig) # Close plot
toc = time.time()
print("Time taken: %.1f seconds" % (toc - tic))


#%% BOX PLOTS - INDIVIDUAL PLOTS FOR TOP RANKED FEATURES FOR EACH FOOD
# - Rank features by pvalue significance (lowest first) and select the Top 10 features for each food
# - Plot boxplots of the most important features for each food compared to OP50
# - Plot features separately with feature as title and in separate folders for each food

# Parameters
test_names = ['ttest_ind','ranksumtest']
n_top_features = 10
show_plot = False

if not show_plot:
    plt.ioff()
for test_name in test_names:
    print("\nPlotting box plots of top %d highest ranked features by '%s':\n" % (n_top_features, test_name))
    # Read test results (pvalues)
    test_results_inpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Stats', 'L4_snippet_1', test_name, test_name + '_results.csv')
    test_pvalues_df = pd.read_csv(test_results_inpath, index_col=0)

    for i, food in enumerate(test_pvalues_df.index.str.upper()):
        pvals = test_pvalues_df.loc[food.lower()]
        n_sigfeats = sum(pvals < p_value_threshold)
        n_nonnanfeats = np.logical_not(pvals.isna()).sum()
        if pvals.isna().all():
            print("No signficant features found for %s!" % food)
        elif n_sigfeats > 0:       
            # Rank p-values in ascending order
            ranked_pvals = pvals.sort_values(ascending=True)
            
            # Select the top few p-values
            topfeats = ranked_pvals[:n_top_features]
            
            if n_sigfeats < n_top_features:
                print("Only %d significant features found for %s" % (n_sigfeats, food))
                # Drop NaNs
                topfeats = topfeats.dropna(axis=0)
                # Drop non-sig feats
                topfeats = topfeats[topfeats < p_value_threshold]
                
            if verbose:
                print("\nTop %d features for %s:\n" % (len(topfeats), food))
                print(*[feat + '\n' for feat in list(topfeats.index)])

            # Subset L4 1st snippet feature summary results for test-food + OP50-control only
            data = L4_1st_snippets_df[np.logical_or(L4_1st_snippets_df['food_type'].str.upper()=="OP50",\
                                                    L4_1st_snippets_df['food_type'].str.upper()==food)] 
            
            # Colour/legend parameters
            food_labels = list(data['food_type'].unique())
            colour_dict = dict(zip(food_labels, sns.color_palette(palette="bright", n_colors=len(food_labels))))
                        
            # Boxplots of OP50 vs test-food for each top-ranked significant feature
            for f, feature in enumerate(topfeats.index):
                plt.close('all')
                fig = plt.figure(figsize=[10,8])
                ax = fig.add_subplot(1,1,1)
                sns.boxplot(x="food_type", y=feature, data=data, palette=colour_dict,\
                            showfliers=False, showmeans=True,\
                            meanprops={"marker":"x", "markersize":5, "markeredgecolor":"k"},\
                            flierprops={"marker":"x", "markersize":15, "markeredgecolor":"r"})
                sns.swarmplot(x="food_type", y=feature, data=data, s=10, marker=".", color='k')
                ax.set_xlabel('Bacterial Strain (Food)', fontsize=15, labelpad=12)
                ax.set_ylabel(feature, fontsize=15, labelpad=12)
                ax.set_title(feature, fontsize=20, pad=20)

                # Add plot legend
                patches = []
                for l, key in enumerate(colour_dict.keys()):
                    patch = mpatches.Patch(color=colour_dict[key], label=key)
                    patches.append(patch)
                    if key == 'OP50':
                        continue
                    else:
                        ylim = data[data['food_type'].str.upper()==key][feature].max()
                        pval = test_pvalues_df.loc[key.lower(), feature]
                        if isinstance(pval, float) and pval < p_value_threshold:
                            ax.text(l - 0.03, ylim, '*', fontsize=25, color='k', verticalalignment='bottom')
                            ax.text(l + 0.02, ylim + ylim/50, 'p={:g}'.format(float('{:.2g}'.format(pval))),\
                                    fontsize=10, color='k', verticalalignment='bottom')
                plt.tight_layout(rect=[0, 0, 0.88, 0.98])
                plt.legend(handles=patches, labels=colour_dict.keys(), loc=(1.02, 0.8),\
                          borderaxespad=0.4, frameon=False, fontsize=15)
                plt.show(); plt.pause(1)
                
                # Save figure
                plots_outpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'L4_snippet_1', food, test_name,\
                                             feature + '_Rank_{0}.eps'.format(f + 1))
                directory = os.path.dirname(plots_outpath)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                savefig(plots_outpath, tellme=True, saveFormat='eps')
                
                if show_plot:
                    plt.show(); plt.pause(2)


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


#%% HIERARCHICAL CLUSTERING (HEATMAP) - ALL FOODS
# - Scikit-learn clustermap of features by foods, to see if they cluster into
#   groups for each food - does OP50 control form a nice group?
  
test_name = 'ranksumtest'    
n_top_feats_per_food = 10

# Divide dataframe into 2 dataframes: data (feature summaries) and non-data (metadata)
colnames_all = list(L4_1st_snippets_df.columns)
colnames_nondata = colnames_all[:25]
colnames_data = colnames_all[25:]
L4_1st_snippets_data = L4_1st_snippets_df[colnames_data]
L4_1st_snippets_nondata = L4_1st_snippets_df[colnames_nondata]

# Read significant features list for each food
sigfeats_in = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Stats', 'L4_snippet_1',\
                           test_name, test_name + '_significant_features.csv')
sigfeats_df = pd.read_csv(sigfeats_in)

# Read test results (pvalues)
results_inPath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Stats', 'L4_snippet_1',\
                              test_name, test_name + '_results.csv')
test_pvalues_df = pd.read_csv(results_inPath, index_col=0)

#n_sigfeats = sigfeats_df.shape[0]-sigfeats_df.isna().sum()

# Store top features for each food in dictionary
TEST_BACTERIA = [food.upper() for food in L4_1st_snippets_df['food_type'].unique() if food != 'OP50']
food_sigfeats_dict = {food.upper():None for food in TEST_BACTERIA}
for food in food_sigfeats_dict.keys():
    sigfeats_food = list(sigfeats_df[food.lower()])
    sigfeats_food = [feature for i, feature in enumerate(sigfeats_food[:n_top_feats_per_food]) if type(sigfeats_food[i])==str]
    if len(sigfeats_food) > 0:
        # Get pvalues for features
        pvals = test_pvalues_df.loc[food.lower(), sigfeats_food]
        # Rank features by significance (test p-value)
        ranked_pvals = pvals.sort_values(ascending=True)   
        # Warn if less than requested number of sig feats were in fact significant
        if len(ranked_pvals) < n_top_feats_per_food:
            n_feats2use = len(ranked_pvals)
            print("WARNING: Only %d/%d significant features found for '%s'"\
                  % (n_feats2use, n_top_feats_per_food, food))
        else:
            n_feats2use = n_top_feats_per_food
            
        # Record the top most important features
        sigfeats_food = ranked_pvals[:n_feats2use]
        food_sigfeats_dict[food.upper()] = list(sigfeats_food.index)      
    
feats2plt = []
for food, values in food_sigfeats_dict.items():
    if values:
        feats2plt = list(set(values) | set(feats2plt))
print("%d significant features found to differ from OP50 for one or more foods (union)" % len(feats2plt))

# Normalise the data (z-scores)
zscores = L4_1st_snippets_data.apply(zscore, axis=0)

# Heatmap (clustergram) of Top10 features per food (n=45)
plt.close('all')
bacterial_strains = list(L4_1st_snippets_nondata['food_type'].unique())
colour_dict = dict(zip(bacterial_strains, sns.color_palette("bright", len(bacterial_strains))))
row_colours = L4_1st_snippets_nondata['food_type'].map(colour_dict)
sns.clustermap(zscores[feats2plt], row_colors=row_colours, figsize=[12,9])
patches = []
for l, key in enumerate(colour_dict.keys()):
    patch = mpatches.Patch(color=colour_dict[key], label=key)
    patches.append(patch)
plt.legend(handles=patches, labels=colour_dict.keys(),\
           borderaxespad=0.4, frameon=False, loc=(-3.5, -6), fontsize=12)

# Save clustermap and features of interest
cluster_outpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'L4_snippet_1',\
                               'Dimensionality_Reduction', 'Hierarchical_Clustering_Analysis',\
                               'Hierarchical_clustering_n={0}_features.eps'.format(len(feats2plt)))
directory = os.path.dirname(cluster_outpath)
if not os.path.exists(directory):
    os.makedirs(directory)
savefig(cluster_outpath, tight_layout=False, tellme=True, saveFormat='eps')

clusterfeats_outpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'L4_snippet_1',\
                                    'Dimensionality_Reduction', 'Hierarchical_Clustering',\
                                    'Hierarchical_clustering_features_list.txt')
fid = open(clusterfeats_outpath, 'w')
print(feats2plt, file=fid)
fid.close()

plt.show(); plt.pause(1)

    
#%% PRINCIPAL COMPONENTS ANALYSIS (PCA) - ALL FOODS
# - Investigate first video snippets for L4-preconditioned worms to begin with

PCs_to_keep = 10
dates2exclude = []#[20190711]

tic = time.time()
 
# Read feature summary results
results_inpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'fullresults.csv')
full_results_df = pd.read_csv(results_inpath)

# Prepare data for PCA
L4_1st_snippets_df, droppedFeatsList_NaN, droppedFeatsList_allZero = cleanSummaryResults(full_results_df)

# OPTIONAL: Exclude certain imaging dates from PCA
L4_1st_snippets_df = L4_1st_snippets_df[~L4_1st_snippets_df['date_yyyymmdd'].isin(dates2exclude)]

# Select non-data columns to drop for PCA
non_data_columns = L4_1st_snippets_df.columns[0:25]

data = L4_1st_snippets_df.drop(columns=non_data_columns)

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
plotpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'L4_snippet_1', 'PCA',\
                        'L4_snippet_1' + '_PCA_explained.eps')
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

# TODO: Store PCA important features list + plot for each feature

# Add concatenate projected PC results to metadata
projected_df.set_index(L4_1st_snippets_df.index, inplace=True) # Do not lose video snippet index position
L4_1st_snippet_projected_df = pd.concat([L4_1st_snippets_df[non_data_columns], projected_df], axis=1)


#%% 2D Plot - first 2 PCs - ALL FOODS

# Plot first 2 principal components
plt.close('all'); plt.ion()
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize=15, labelpad=12)
ax.set_ylabel('Principal Component 2', fontsize=15, labelpad=12)
ax.set_title('2 Component PCA', fontsize=20)

# 
bacterial_strains = L4_1st_snippet_projected_df['food_type'].unique()

# Create colour palette for plot loop
palette = itertools.cycle(sns.color_palette("bright", len(bacterial_strains)))

for food in bacterial_strains:
    food_projected_df = L4_1st_snippet_projected_df[L4_1st_snippet_projected_df['food_type'].str.upper()==food]
    sns.scatterplot(food_projected_df['PC1'], food_projected_df['PC2'], color=next(palette), s=100)
ax.legend(bacterial_strains)
ax.grid()

# Save scatterplot of first 2 PCs
plotpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'L4_snippet_1', 'PCA', 'L4_snippet_1' + '_PCA_1st_2PCs.eps')
savefig(plotpath, tight_layout=True, tellme=True, saveFormat='eps')

plt.show(); plt.pause(2)

#%% 3D Plot - first 3 PCs - ALL FOODS

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

# Create colour palette for plot loop
palette = itertools.cycle(sns.color_palette("bright", len(bacterial_strains)))

for food in bacterial_strains:
    food_projected_df = L4_1st_snippet_projected_df[L4_1st_snippet_projected_df['food_type'].str.upper()==food]
    ax.scatter(xs=food_projected_df['PC1'], ys=food_projected_df['PC2'], zs=food_projected_df['PC3'],\
               zdir='z', s=50, c=next(palette), depthshade=depthshade)
ax.legend(bacterial_strains)
ax.grid()

# Save scatterplot of first 2 PCs
plotpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'L4_snippet_1', 'PCA', 'L4_snippet_1' + '_PCA_1st_3PCs.eps')
savefig(plotpath, tight_layout=False, tellme=True, saveFormat='eps')

# Rotate the axes and update
if rotate:
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw(); plt.pause(0.001)
else:
    plt.show(); plt.pause(1)
    
#%% t-distributed Stochastic Neighbour Embedding (t-SNE)

# TODO: t-SNE analysis
    
    
#%% Uniform Manifold Projection (UMAP)

# TODO: U-MAP analysis

 
#%% Linear Discriminant Analysis (LDA)
# - Projection of data along top 2 most influential eigenvectors (not any one feature)

# TODO: Linear Discriminant Analysis 

    
#%% MANUAL LABELLING OF FOOD REGIONS (To check time spent feeding throughout video)

MANUAL = False

# Return list of pathnames for masked videos in the data directory under given imaging dates
maskedfilelist = []
date_total = []
for i, expDate in enumerate(IMAGING_DATES):
    tmplist = lookforfiles(os.path.join(DATA_DIR, "MaskedVideos", expDate), ".*.hdf5$")
    date_total.append(len(tmplist))
    maskedfilelist.extend(tmplist)
print("%d masked videos found for imaging dates provided:\n%s" % (len(maskedfilelist), [*zip(IMAGING_DATES, date_total)]))

first_snippets = [snip for snip in maskedfilelist if "000000.hdf5" in snip]
print("\nManual labelling:\n%d masked video snippets found for %d assay recordings (duration: 2hrs)" % (len(maskedfilelist), len(first_snippets)))

# Manual labelling of food regions in each assay using 1st video snippet, 1st frame
plt.ion() # Interactive plotting (for user input when labelling plots)
tic = time.time()
if MANUAL:
    for i, maskedfilepath in enumerate(first_snippets):    
        # Manually outline + assign labels to food regions + save coordinates + trajectory overlay to file           
        manuallabelling(maskedfilepath, n_poly=1, out_dir='MicrobiomeAssay', save=True, skip=True)       
    print("Manual labelling complete!\n(Time taken: %d seconds.)\n" % (time.time() - tic))

# TODO: Automate food labelling using (U-Net) ML algorithm -- pre-trained already by Luigi

#%% PLOT WORM TRAJECTORIES
    
thresh_movement = 10
thresh_duration = 50

for dirname in metadata['filename']:
    maskedfilepath = os.path.join(dirname, '000000.hdf5')
    featuresfilepath = changepath(maskedfilepath, returnpath='features')
    trajectory_df = gettrajdata(featuresfilepath)
    findworms(trajectory_df, threshold_move=thresh_movement, threshold_time=thresh_duration, tellme=True)    
        
          
#%% FILTER WORM TRAJECTORIES









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
#             aovrm = AnovaRM(data=test_data_aovRM, depvar=, subject='level_0', within=['Feature'])
##            data = pd.read_csv('../Sandbox/example_AnovaRM.csv', sep=',')
##            res = AnovaRM(data, depvar='RT', subject='SubID', within=['TrialType'], aggregate_func='mean')
##            print(res.fit())
#         else:  
# =============================================================================
