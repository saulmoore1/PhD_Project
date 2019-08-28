#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sm5911
@date: 11/08/2019

Bacterial effects on Caenorhabditis elegans behaviour 
- FOOD BEHAVIOUR CONTROL

OP50 Control across imaging days

"""

#%% IMPORTS

# General imports
import os, itertools, time#, umap (NB: Need to install umap library in anaconda first!)
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


#%% MAIN

if __name__ == '__main__':
    # PRE-AMBLE
    tic = time.time()
    
    # Global parameters
    PROJECT_NAME = 'MicrobiomeAssay'
    PROJECT_ROOT_DIR = '/Volumes/behavgenom$/Saul/' + PROJECT_NAME
    
    verbose = True # Print statements to track script progress?
    preprocessing = False # Process metadata file and feature summaries files?
    
    # Which set of bacterial strains to analyse?
    # - OPTION: Investigate microbiome/miscellaneous/all(both) strains
    MICROBIOME = True
    MISCELLANEOUS = True
    
    # - OPTION: Look at one 12-min video snippet
    snippet = 0 # Which video segment to analyse? # TODO: Add option - 'ALL'
    
    # - OPTION: L4-prefed worms (long exposure to food) or Naive adults
    preconditioned_from_L4 = True # TODO: Make so can analyse prefed vs naive adults
    
    # Remove size-related features from analyses? (found to exhibit high variation across iamging dates)
    filter_size_related_feats = False
    
    # Statistics parameters
    test = f_oneway
    nan_threshold = 0.75 # Threshold proportion NaNs to drop features from analysis
    p_value_threshold = 0.05 # P-vlaue threshold for statistical analyses
    
    # Dimensionality reduction parameters
    n_top_feats_per_food = 10       # HCA - Number of top features to include in HCA (for union across foods HCA)
    useTop256 = True                # PCA/tSNE/UMAP - Restrict to Avelino's top 256 feature list?
    PCs_to_keep = 10                # PCA - Number of principal components to record
    rotate = True                   # PCA - Rotate 3-D plots?
    depthshade = False              # PCA - Shade colours on 3-D plots to show depth?
    perplexity = [10,15,20,25,30]   # tSNE - Parameter range for t-SNE mapping eg. np.arange(5,51,1)
    n_neighbours = [10,15,20,25,30] # UMAP - Number of neighbours parameter for UMAP projections eg. np.arange(3,31,1)
    min_dist = 0.3                  # UMAP - Minimum distance parameter for UMAP projections
    
    # Select imaging date(s) for analysis
    IMAGING_DATES = ['20190704','20190705','20190711','20190712','20190718','20190719',\
                     '20190725','20190726']
    
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
    
    results_df, droppedFeats_NaN, droppedFeats_allZero = cleanSummaryResults(full_results_df,\
                                                         impute_NaNs_by_group=False,\
                                                         preconditioned_from_L4=preconditioned_from_L4,\
                                                         snippet=snippet, nan_threshold=nan_threshold)
    
    droppedlist_out = os.path.join(PROJECT_ROOT_DIR, 'Results', 'OP50_Control', 'Dropped_Features_NaN.txt')
    directory = os.path.dirname(droppedlist_out)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fid = open(droppedlist_out, 'w')
    print(*droppedFeats_NaN, file=fid)
    fid.close()
    
    droppedlist_out = droppedlist_out.replace('NaN', 'AllZero')
    fid = open(droppedlist_out, 'w')
    print(*droppedFeats_allZero, file=fid)
    fid.close()
    
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
    
    # Extract OP50 control data from feature summaries
    OP50_control_df = results_df[results_df['food_type'].str.lower() == "op50"]
    
    # Record the bacterial strain names for use in analyses
    bacterial_strains = list(np.unique(results_df['food_type'].str.lower()))
    test_bacteria = [strain for strain in bacterial_strains if strain != "op50"]
    
    # Record feature column names + non-data columns to drop for statistics
    colnames_all = results_df.columns
    colnames_nondata = results_df.columns[:25]
    colnames_data = results_df.columns[25:]
    
  
    #%% READ + FILTER + CLEAN SUMMARY RESULTS
                
    # Drop columns that contain only zeros
    n_cols = len(OP50_control_df.columns)
    OP50_control_df = OP50_control_df.drop(columns=OP50_control_df.columns[(OP50_control_df == 0).all()])
    zero_cols = n_cols - len(OP50_control_df.columns)
    if verbose:
        print("Dropped %d feature summaries for OP50 control (all zeros)" % zero_cols)
    
    if filter_size_related_feats:
        # Filter out size-related features
        size_feat_keys = ['blob','box','width','length','area']
        size_features = []
        for feature in OP50_control_df.columns:
            for key in size_feat_keys:
                if key in feature:
                    size_features.append(feature)         
        feats2keep = [feat for feat in OP50_control_df.columns if feat not in size_features]
        OP50_control_df = OP50_control_df[feats2keep]
    
    # Record non-data columns to drop for statistics
    non_data_columns = OP50_control_df.columns[0:25]
    
    # Record a list of feature column names
    feature_colnames = OP50_control_df.columns[25:]
    
    
    #%% OP50 CONTROL DATA ACROSS DAYS: STATS (ANOVAs)
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
    
    # One-way ANOVA with Bonferroni correction for repeated measures
    print("Performing One-Way ANOVAs (for each feature) to investigate whether control OP50 results vary across imaging dates...")
    OP50_over_time_results_df = pd.DataFrame(index=['stat','pval'], columns=feature_colnames)
    for feature in OP50_control_df.columns[25:]:
        test_stat, test_pvalue = test(*[OP50_control_df[OP50_control_df['date_yyyymmdd']==date][feature]\
                                            for date in OP50_control_df['date_yyyymmdd'].unique()])
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
    stats_outpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'OP50_Control', 'Stats', 'L4_snippet_1',\
                                 test_name, 'OP50_control_across_days_' + test_name + '.csv')
    directory = os.path.dirname(stats_outpath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    OP50_over_time_results_df.to_csv(stats_outpath)
    
    # Compile list to store names of significant features
    sigfeats_out = OP50_over_time_results_df.loc['pval_corrected'].sort_values(ascending=True)
    sigfeats_out = sigfeats_out[sigfeats_out < p_value_threshold]
    sigfeats_out.name = 'p_value_' + test_name
    
    # Save significant features list to CSV
    sigfeats_outpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'OP50_Control', 'Stats', 'L4_snippet_1',\
                                    test_name, 'OP50_control_across_days_significant_features_' + test_name + '.csv')
    # Save feature list to text file
    sigfeats_out.to_csv(sigfeats_outpath)
    
    # Tally total number of significantly different pairwise comparisons
    n_sigdiff_pairwise_beforeBF = 0
    n_sigdiff_pairwise_afterBF = 0
    
    # Tukey HSD post-hoc pairwise differences between dates for each feature
    for feature in feature_colnames:
        # Tukey HSD post-hoc analysis (no Bonferroni correction!)
        tukeyHSD = pairwise_tukeyhsd(OP50_control_df[feature], OP50_control_df['date_yyyymmdd'])
        n_sigdiff_pairwise_beforeBF += sum(tukeyHSD.reject)
        
        # Tukey HSD post-hoc analysis (Bonferroni correction)
        tukeyHSD_BF = MultiComparison(OP50_control_df[feature], OP50_control_df['date_yyyymmdd'])
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
    
    # Mantel test?
    
    
    #%% Boxplots for most important features across days
    
    plotroot = os.path.join(PROJECT_ROOT_DIR, 'Results', 'OP50_Control', 'Plots', 'L4_snippet_1', 'OP50')
                 
    pvals = OP50_over_time_results_df.loc['pval_corrected']
    n_sigfeats = sum(pvals < p_value_threshold)
    
    if pvals.isna().all():
        print("No signficant features found across days for OP50 control!")
    elif n_sigfeats > 0:
        # Rank p-values in ascending order
        ranked_pvals = pvals.sort_values(ascending=True)
                
        # Select the top few p-values
        topfeats = ranked_pvals[:n_top_feats_per_food]
                
        if n_sigfeats < n_top_feats_per_food:
            print("Only %d features found to vary significantly across days" % n_sigfeats)
            # Drop NaNs
            topfeats = topfeats.dropna(axis=0)
            # Drop non-sig feats
            topfeats = topfeats[topfeats < p_value_threshold]
            
        if verbose:
            print("\nTop %d features for OP50 that differ significantly across days (ANOVA):\n" % len(topfeats))
            print(*[feat + '\n' for feat in list(topfeats.index)])
    
        # for f, feature in enumerate(feature_colnames[0:25]):
        for f, feature in enumerate(topfeats.index):
            print("P-value for '%s': %s" % (feature, str(topfeats[feature])))
            OP50_topfeat_df = OP50_control_df[['date_yyyymmdd', feature]]
            
            # Plot boxplots of OP50 control across days for most significant features
            plt.close('all')
            fig = plt.figure(figsize=[10,6])
            ax = fig.add_subplot(1,1,1)
            sns.boxplot(x='date_yyyymmdd', y=feature, data=OP50_control_df)
            ax.set_xlabel('Imaging Date (YYYYMMDD)', fontsize=15, labelpad=12)
            ax.set_title(feature, fontsize=20, pad=20)
            
            # TODO: Add reverse-engineered pvalues to plot
            
            # Save plot
            plots_outpath = os.path.join(plotroot, feature + '_across_days.eps')
            directory = os.path.dirname(plots_outpath)
            if not os.path.exists(directory):
                os.makedirs(directory)
            savefig(plots_outpath, tellme=True, saveFormat='eps')
            
    
    #%% PCA of OP50 CONTROL DATA ACROSS DAYS
    
    plotroot = os.path.join(PROJECT_ROOT_DIR, 'Results', 'OP50_Control', 'Plots', 'L4_snippet_1', 'OP50')
    
    # Read list of important features (highlighted by previous research - see Javer, 2018 paper)
    featslistpath = os.path.join(PROJECT_ROOT_DIR,'Data','top256_tierpsy_no_blob_no_eigen_only_abs_no_norm.csv')
    top256features = pd.read_csv(featslistpath)
    
    # Take first set of 256 features (it does not matter which set is chosen)
    top256features = top256features[top256features.columns[0]]   
    n_feats = len(top256features)
    top256features = [feat for feat in top256features if feat in OP50_control_df.columns]
#    print("Dropping %d size-related features from Top256" % (n_feats - len(top256features)))

    # Drop all but top256 columns for PCA
    data = OP50_control_df[top256features]
    
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
    plotpath = os.path.join(plotroot, 'PCA', 'L4_snippet_{0}'.format(snippet) + '_PCA_explained.eps')
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
    projected_df.set_index(OP50_control_df.index, inplace=True) # Do not lose video snippet index position
    OP50_dates_projected_df = pd.concat([OP50_control_df[non_data_columns], projected_df], axis=1)
    
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
    plotpath = os.path.join(plotroot, 'PCA', 'L4_snippet_1'+'_1st_2PCs.eps')
    savefig(plotpath, tight_layout=True, tellme=True, saveFormat='eps')
    
    plt.show(); plt.pause(2)
    
    # Plot 3 PCs - OP50 across imaging dates
    
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
    plotpath = os.path.join(plotroot, 'PCA', 'L4_snippet_1' + '_1st_3PCs.eps')
    savefig(plotpath, tight_layout=False, tellme=True, saveFormat='eps')
    
    # Rotate the axes and update
    if rotate:
        for angle in range(0, 360):
            ax.view_init(30, angle)
            plt.draw(); plt.pause(0.001)
    else:
        plt.show(); plt.pause(1)
    
    toc = time.time()
    print("OP50 control analysis complete.\n(Time taken: %d seconds)" % (toc-tic))
