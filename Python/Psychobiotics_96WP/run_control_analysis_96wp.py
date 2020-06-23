#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONTROL VARIATION ACROSS DAYS

Analyse control data: look for variation across experiment days (Kruskal-Wallis) 
and plot: (a) boxplots of the most important features that vary across days, (b)

@author: sm5911
@date: 26/10/2019

"""

#%% IMPORTS

# General imports
import os, sys, time#, umap
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns; sns.set(color_codes=True)
from matplotlib import pyplot as plt
from scipy.stats import shapiro, kruskal, f_oneway, zscore
from sklearn.decomposition import PCA
from statsmodels.stats import multitest as smm#, AnovaRM
#from statsmodels.stats.multicomp import MultiComparison, pairwise_tukeyhsd
#from statsmodels.multivariate.manova import MANOVA

# Custom imports
# Path to Github helper functions (USER-DEFINED path to local copy of Github repo)
PATH_LIST = ['/Users/sm5911/Tierpsy_Versions/tierpsy-tools-python/',
             '/Users/sm5911/Documents/GitHub/PhD_Project/Python/Psychobiotics_96WP']
for sysPath in PATH_LIST:
    if sysPath not in sys.path:
        sys.path.insert(0, sysPath)
    
# Custom imports
from my_helper import savefig, pcainfo, plotPCA, MahalanobisOutliers

#%% FUNCTIONS

def removeOutliersMahalanobis(df, features_to_analyse=None):
    """ Remove outliers from dataset based on Mahalanobis distance metric 
        between points in PCA space. """

    if features_to_analyse:
        data = df[features_to_analyse]
    else:
        data = df
            
    # Normalise the data before PCA
    zscores = data.apply(zscore, axis=0)
    
    # Drop features with NaN values after normalising
    colnames_before = list(zscores.columns)
    zscores.dropna(axis=1, inplace=True)
    colnames_after = list(zscores.columns)
    nan_cols = [col for col in colnames_before if col not in colnames_after]
    if len(nan_cols) > 0:
        print("Dropped %d features with NaN values after normalization:\n%s" %\
              (len(nan_cols), nan_cols))

    print("\nPerforming PCA for outlier removal...")
    
    # Fit the PCA model with the normalised data
    pca = PCA()
    pca.fit(zscores)
    
    # Project data (zscores) onto PCs
    projected = pca.transform(zscores) # A matrix is produced
    # NB: Could also have used pca.fit_transform()

    # Plot summary data from PCA: explained variance (most important features)
    important_feats, fig = pcainfo(pca, zscores, PC=1, n_feats2print=10)        
    
    # Remove outliers: Use Mahalanobis distance to exclude outliers from PCA
    indsOutliers = MahalanobisOutliers(projected, showplot=True)
    
    # Get outlier indices in original dataframe
    indsOutliers = np.array(data.index[indsOutliers])
    plt.pause(5); plt.close()
    
    # Drop outlier(s)
    print("Dropping %d outliers from analysis" % len(indsOutliers))
    df = df.drop(index=indsOutliers)
        
    return df, indsOutliers


def doPCA(df, grouping_variable, features_to_analyse, plot_save_dir=None, PCs_to_keep=10):
    """ Perform PCA to investigate a 'grouping_variable' of interest, for a 
        given set of 'features_to_analyse' """
        
    data = df[features_to_analyse]
    
    # Normalise the data before PCA
    zscores = data.apply(zscore, axis=0)
    
    # Drop features with NaN values after normalising
    colnames_before = list(zscores.columns)
    zscores.dropna(axis=1, inplace=True)
    colnames_after = list(zscores.columns)
    nan_cols = [col for col in colnames_before if col not in colnames_after]
    if len(nan_cols) > 0:
        print("Dropped %d features with NaN values after normalization:\n%s" %\
              (len(nan_cols), nan_cols))

    print("\nPerforming Principal Components Analysis (PCA)...")
    
    # Fit the PCA model with the normalised data
    pca = PCA()
    pca.fit(zscores)
    
    # Project data (zscores) onto PCs
    projected = pca.transform(zscores) # A matrix is produced
    # NB: Could also have used pca.fit_transform()

    # Plot summary data from PCA: explained variance (most important features)
    important_feats, fig = pcainfo(pca, zscores, PC=1, n_feats2print=10)        
    
    if plot_save_dir:
        # Save plot of PCA explained variance
        PCAplotroot = Path(plot_save_dir) / 'PCA'
        PCAplotroot.mkdir(exist_ok=True, parents=True)
        PCAplotpath = PCAplotroot / ('control_variation_in_' + 
                                     grouping_variable + 
                                     '_PCA_explained.eps')
        savefig(PCAplotpath, tight_layout=True, tellme=True, saveFormat='eps')
        plt.pause(2); plt.close()
    else:
        PCAplotpath=None
        plt.show(); plt.pause(2); plt.close()
        
    # Store the results for first few PCs in dataframe
    projected_df = pd.DataFrame(projected[:,:PCs_to_keep],
                                columns=['PC' + str(n+1) for n in range(PCs_to_keep)])                
    
    # Add concatenate projected PC results to metadata
    projected_df.set_index(df.index, inplace=True) # Do not lose video snippet index position
    
    df = pd.concat([df, projected_df], axis=1)

    # Plot PCA - Variation in control data with respect to a given variable (eg. date_recording_yyyymmdd)
            
    # 2-PC
    if plot_save_dir:
        PCAplotpath = Path(str(PCAplotpath).replace('_PCA_explained', 
                                                    '_PCA_2_components'))
    title = "2-Component PCA: Control variation in\n\
     '{0}'".format(grouping_variable) + " (Top256 features)"
    plotPCA(df, grouping_variable, var_subset=None, savepath=PCAplotpath, 
            title=title, n_component_axes=2)
    plt.pause(2); plt.close()
    
    # 3-PC
    if plot_save_dir:
        PCAplotpath = Path(str(PCAplotpath).replace('_PCA_2_components', 
                                                    '_PCA_3_components'))
    title = "3-Component PCA: Control variation in\n\
     '{0}'".format(grouping_variable) + " (Top256 features)"
    plotPCA(df, grouping_variable, var_subset=None, savepath=PCAplotpath, 
            title=title, n_component_axes=3, rotate=False)
    plt.pause(2)
    
    return df


def topfeats_boxplots_by_group(df, test_results_df, grouping_variable, 
                               plot_save_dir=None, p_value_threshold=0.05, 
                               n_topfeats=5):
    """ Boxplots of variation in most significant features (ANOVA, p < p_value_threshold) 
        with respect to given grouping variable eg. 'imaging_run_number' """
    
    if plot_save_dir:
        # Ensure directory exists to save plots
        plot_save_dir.mkdir(exist_ok=True, parents=True)
   
    pvals_corrected = test_results_df.loc['pval_corrected']
    n_sigfeats = sum(pvals_corrected < p_value_threshold)
    
    if pvals_corrected.isna().all():
        print("No signficant features found in control with respect to '%s'" % grouping_variable)
    elif n_sigfeats > 0:
        # Rank p-values in ascending order
        ranked_pvals = pvals_corrected.sort_values(ascending=True)
         
        # Drop non-sig feats
        ranked_pvals = ranked_pvals[ranked_pvals < p_value_threshold]
        
        # Select the first n pvalues for plotting
        topfeats = ranked_pvals[:n_topfeats]
                
        if n_sigfeats < n_topfeats:
            print("WARNING: Only %d features found to vary significantly with respect to '%s'"\
                  % (n_sigfeats, grouping_variable))
            
        print("\nTop %d features found to differ significantly with respect to '%s':\n"\
              % (len(topfeats), grouping_variable))
        print(*[feat + '\n' for feat in list(topfeats.index)])
    
        # for f, feature in enumerate(features_to_analyse[0:25]):
        for feature in topfeats.index:
            print("P-value for '%s': %s" % (feature, str(topfeats[feature])))
            feat_df = df[[grouping_variable, feature]]
            
            # Plot boxplots of control across days for most significant features
            plt.close('all')
            fig = plt.figure(figsize=[10,6])
            ax = fig.add_subplot(1,1,1)
            sns.boxplot(x=grouping_variable, y=feature, data=feat_df)
            ax.set_xlabel(grouping_variable, fontsize=15, labelpad=12)
            ax.set_title(feature, fontsize=20, pad=20)
            
            # TODO: Add pvalues to plot?
            
            if plot_save_dir:
                # Save plot
                plots_outpath = plot_save_dir / (feature + '_wrt_' + grouping_variable + '.eps')
                savefig(plots_outpath, tellme=True, saveFormat='eps')            
                plt.close()
            else:
                plt.show(); plt.pause(5)


def topfeats_ANOVA_by_group(df, grouping_variable, features_to_analyse,\
                            TEST=f_oneway, p_value_threshold=0.05):
    print("\nTESTING: %s\n" % grouping_variable)    
    if not len(df[grouping_variable].unique()) > 1:
        print("Need at least two groups for %s to perform statistics!" % grouping_variable)
    else:
        # Record name of statistical test used (kruskal/f_oneway)
        test_name = str(TEST).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]
    
        # One-way ANOVA (with Bonferroni correction for repeated measures)
        print("""Performing %s tests for each feature to investigate
        whether control results vary with respect to '%s':""" % (test_name, grouping_variable))
        
        test_results_df = pd.DataFrame(index=['stat','pval'], columns=features_to_analyse)
        for feature in features_to_analyse:
            test_stat, test_pvalue = TEST(*[df[df[grouping_variable]==g_var][feature]\
                                                for g_var in df[grouping_variable].unique()])
            test_results_df.loc['stat',feature] = test_stat
            test_results_df.loc['pval',feature] = test_pvalue
        
        # Bonferroni correction for multiple comparisons
        _corrArray = smm.multipletests(test_results_df.loc['pval'], alpha=p_value_threshold,\
                                       method='fdr_bh', is_sorted=False, returnsorted=False)
        
        # Get pvalues for features that passed the Benjamini/Hochberg (non-negative) correlation test
        pvalues_corrected = _corrArray[1][_corrArray[0]]
        
        # Add corrected pvalues to results dataframe
        test_results_df = test_results_df.append(pd.Series(name='pval_corrected', dtype='float64'))
        test_results_df.loc['pval_corrected', _corrArray[0]] = pvalues_corrected
        
        n_sigfeats = sum(test_results_df.loc['pval_corrected'] < p_value_threshold)
        print("""%d/%d (%.1f%%) of features show significant variation (%s)
        with respect to '%s'""" % (n_sigfeats, len(test_results_df.columns), 
        n_sigfeats/len(test_results_df.columns)*100, test_name, grouping_variable))
                    
        # Compile list to store names of significant features
        sigfeats_out = test_results_df.loc['pval_corrected'].sort_values(ascending=True) # Rank pvalues by significance
        sigfeats_out = sigfeats_out[sigfeats_out < p_value_threshold]
        sigfeats_out.name = 'p_value_' + test_name
        
#        # TODO: TukeyHSD ?
#        # Tally total number of significantly different pairwise comparisons
#        n_sigdiff_pairwise_beforeBF = 0
#        n_sigdiff_pairwise_afterBF = 0
#        
#        # Tukey HSD post-hoc pairwise differences between dates for each feature
#        for feature in features_to_analyse:
#            # Tukey HSD post-hoc analysis (no Bonferroni correction!)
#            tukeyHSD = pairwise_tukeyhsd(df[feature], df[grouping_variable])
#            n_sigdiff_pairwise_beforeBF += sum(tukeyHSD.reject)
#            
#            # Tukey HSD post-hoc analysis (Bonferroni correction)
#            tukeyHSD_BF = MultiComparison(df[feature], df[grouping_variable])
#            n_sigdiff_pairwise_afterBF += sum(tukeyHSD_BF.tukeyhsd().reject)   
#            
#        total_comparisons = len(features_to_analyse) * 6
#        reject_H0_percentage = n_sigdiff_pairwise_afterBF / total_comparisons * 100
#        
#        print("""%d / %d (%.1f%%) of pairwise-comparisons (%d features) 
#        show significant variation in control (TukeyHSD) with respect to '%s'""" %\
#        (n_sigdiff_pairwise_afterBF, total_comparisons, reject_H0_percentage,\
#         len(features_to_analyse), grouping_variable))
            
#        # TODO: Reverse-engineer p-values using mean/std?
#        from statsmodels.stats.libqsturng import psturng
#        # Studentized range statistic
#        rs = res2[1][2] / res2[1][3]
#        pvalues = psturng(np.abs(rs), 3, 27)
            
#         # TODO: MANOVA (date, temp, humid, etc)
#        
#         maov = MANOVA.from_formula('' + '' + '' ~ , data=df)
#         print(maov.mv_test())
        
        return test_results_df, sigfeats_out
    

def check_normality(df, features_to_analyse, p_value_threshold=0.05):
    """ Check control data for normality and decide whether to use 
        parametric (one-way ANOVA) or non-parametric (Kruskal-Wallis) 
        statistics """
    
    is_normal_threshold = 1 - p_value_threshold

    normality_results = pd.DataFrame(data=None, index=['stat','pval'], columns=features_to_analyse)
    for f, feature in enumerate(features_to_analyse):
        try:
            stat, pval = shapiro(df[feature])
            # NB: UserWarning: Input data for shapiro has range zero 
            # Some features contain all zeros - shapiro(np.zeros(5))
            normality_results.loc['stat',feature] = stat
            normality_results.loc['pval',feature] = pval
        except Exception as EE:
            print("WARNING: %s" % EE)
            
    prop_normal = (normality_results.loc['pval'] < p_value_threshold).sum()/len(features_to_analyse)    
    if prop_normal > is_normal_threshold:
        print("""More than %d%% of control features (%.1f%%) were found to obey a 
        normal (Gaussian) distribution, so parametric analyses will be 
        preferred.""" % (is_normal_threshold*100, prop_normal*100))
        TEST = f_oneway
    else:
        print("""Less than %d%% of control features (%.1f%%) were found to obey a 
        normal (Gaussian) distribution, so non-parametric analyses will be 
        preferred.""" % (is_normal_threshold*100, prop_normal*100))
        TEST = kruskal
    return TEST


def control_variation(df, outDir, features_to_analyse, 
                      variables_to_analyse=["date_yyyymmdd"], 
                      remove_outliers=True, 
                      p_value_threshold=0.05, 
                      PCs_to_keep=10):
    """ A function written to analyse control variation over time across with respect 
        to a defined grouping variable (factor), eg. day of experiment, run number, 
        duration of L1 diapause, camera/rig ID, etc. """
           
    # Record non-data columns before dropping feature columns   
    other_colnames = [col for col in df.columns if col not in features_to_analyse]
        
    # Drop columns that contain only zeros
    colnames_before = list(df.columns)
    AllZeroFeats = df[features_to_analyse].columns[(df[features_to_analyse] == 0).all()]
    df = df.drop(columns=AllZeroFeats)
    colnames_after = list(df.columns)
    zero_cols = [col for col in colnames_before if col not in colnames_after]
    if len(zero_cols) > 0:
        print("Dropped %d features with all-zero summaries:\n%s" % (len(zero_cols), zero_cols))
    
    # Record feature column names after dropping zero data
    features_to_analyse = [feat for feat in df.columns if feat not in other_colnames]
    
    # Remove outliers from the dataset 
    if remove_outliers:
        df, indsOutliers = removeOutliersMahalanobis(df, features_to_analyse)
        remove_outliers = False 
        # NB: Ensure Mahalanobis operation is performed only once!

    # Check for normality in features to analyse in order decide which 
    # statistical test to use: one-way ANOVA (parametric) or Kruskal-Wallis 
    # (non-parametric) test
    TEST = check_normality(df, features_to_analyse, p_value_threshold)

    # Record name of statistical test used (kruskal/f_oneway)
    test_name = str(TEST).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]

    # CONTROL VARIATION: STATS (ANOVAs)
    # - Does N2 worm behaviour on control vary across experiment days? 
    #       (worms are larger? Shorter L1 diapuase? Camera focus/FOV adjusted? Skewed by non-worm tracked objects?
    #       Did not record time when worms were refed! Could be this. If so, worms will be bigger across all foods on that day) 
    # - Perform ANOVA to see if features vary across imaging days for control
    # - Perform Tukey HSD post-hoc analyses for pairwise differences between imaging days
    # - Highlight outlier imaging days and investigate reasons why
    # - Save list of top significant features for outlier days - are they size-related features?
    for grouping_variable in variables_to_analyse:
        print("\nTESTING: %s\n" % grouping_variable)
        
        if not len(df[grouping_variable].unique()) > 1:
            print("Need at least two groups for stats to investigate %s" % grouping_variable)
        else:
            print("Performing %s tests for '%s'" % (test_name, grouping_variable))            
    
            test_results_df, sigfeats_out = \
                topfeats_ANOVA_by_group(df, 
                                        grouping_variable, 
                                        features_to_analyse,
                                        TEST,
                                        p_value_threshold)
            
            # Ensure directory exists to save results
            Path(outDir).mkdir(exist_ok=True, parents=True)
            
            # Define outpaths
            froot = 'control_variation_in_' + grouping_variable + '_' + test_name
            stats_outpath = outDir / (froot + "_results.csv")
            sigfeats_outpath = outDir / (froot + "_significant_features.csv")
                                   
            # Save test statistics + significant features list to file
            test_results_df.to_csv(stats_outpath)
            sigfeats_out.to_csv(sigfeats_outpath, header=False)

            # Box plots
            plotDir = outDir / "Plots"
            topfeats_boxplots_by_group(df, 
                                       test_results_df, 
                                       grouping_variable,
                                       plot_save_dir=plotDir, #save to plotDir
                                       p_value_threshold=p_value_threshold)
                        
            # PCA (coloured by grouping variable, eg. experiment date)
            df = doPCA(df, 
                       grouping_variable, 
                       features_to_analyse,
                       plot_save_dir = plotDir,
                       PCs_to_keep = PCs_to_keep)
            
#%% MAIN
        
if __name__ == '__main__':
    tic = time.time()
    if not len(sys.argv) >= 2:
        print("Please provide path to control data, followed by an unpacked list\
        of feature column names as inputs.")
    else:
        print("\nRunning script", os.path.basename(sys.argv[0]), "...")
        path_to_control_data = sys.argv[1]
        outDir = Path(path_to_control_data).parent
        features_to_analyse = list(sys.argv[2:])
        
        # Variables to analyse
        variables_to_analyse = ['date_yyyymmdd','imaging_run_number',\
                                'imaging_plate_id','master_stock_plate_ID',\
                                'instrument_name','well_name']
        
        # Load control summary results
        df = pd.read_csv(path_to_control_data, comment='#')
        print("Control data loaded.")
        
        # Analyse control variation over time
        control_variation(df, features_to_analyse, variables_to_analyse, 
                          outDir, remove_outliers=True, p_value_threshold=0.05, 
                          PCs_to_keep=10)
        
    toc = time.time()
    print("Control analysis complete.\n(Time taken: %d seconds)" % (toc-tic))
