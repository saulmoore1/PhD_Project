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
import pandas as pd
import seaborn as sns; sns.set(color_codes=True)
from matplotlib import pyplot as plt
from scipy.stats import shapiro, kruskal, f_oneway, zscore
from sklearn.decomposition import PCA
from statsmodels.stats import multitest as smm#, AnovaRM
#from statsmodels.stats.multicomp import MultiComparison, pairwise_tukeyhsd
#from statsmodels.multivariate.manova import MANOVA

# Path to Github/local helper functions (USER-DEFINED: Path to local copy of my Github repo)
PATH = '/Users/sm5911/Documents/GitHub/PhD_Project/Python/Psychobiotics_96WP'
if PATH not in sys.path:
    sys.path.insert(0, PATH)
    
# Custom imports
from helper import savefig, pcainfo, plotPCA, MahalanobisOutliers

#%% FUNCTIONS

def doPCA(CONTROL_DF, grouping_variable, feature_column_names, meta_colnames,\
          plot_save_dir, remove_outliers = False, PCs_to_keep=10):
    data = CONTROL_DF[feature_column_names]
    
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
    
    if remove_outliers:
        # Remove outliers: Use Mahalanobis distance to exclude outliers from PCA
        indsOutliers = MahalanobisOutliers(projected, showplot=True)
        plt.pause(5); plt.close()
        
        # Drop outlier observation(s)
        print("Dropping %d outliers from analysis" % len(indsOutliers))
        #indsOutliers = data.index[indsOutliers]
        data = data.drop(index=indsOutliers)
        CONTROL_DF = CONTROL_DF.drop(index=indsOutliers)
        
        # Re-normalise data
        zscores = data.apply(zscore, axis=0)
        
        # Drop features with NaN values after normalising
        zscores.dropna(axis=1, inplace=True)
        print("Dropped %d features after normalisation (NaN)" % (len(data.columns)-len(zscores.columns)))
                
        # Project data on PCA axes again
        pca = PCA()
        pca.fit(zscores)
        projected = pca.transform(zscores) # project data (zscores) onto PCs
        important_feats, fig = pcainfo(pca=pca, zscores=zscores, PC=1, n_feats2print=10)
        
        remove_outliers = False # Only perform outlier removal once on the dataset 

    # Save plot of PCA explained variance
    PCAplotroot = os.path.join(plot_save_dir, 'PCA')
    if not os.path.exists(PCAplotroot):
        os.makedirs(PCAplotroot)
    PCAplotpath = os.path.join(PCAplotroot, 'control_variation_in_'\
                               + grouping_variable + '_PCA_explained.eps')
    savefig(PCAplotpath, tight_layout=True, tellme=True, saveFormat='eps')
    plt.pause(2); plt.close()
    
    # Store the results for first few PCs in dataframe
    projected_df = pd.DataFrame(projected[:,:PCs_to_keep],\
                                columns=['PC' + str(n+1) for n in range(PCs_to_keep)])                
    
    # Add concatenate projected PC results to metadata
    projected_df.set_index(CONTROL_DF.index, inplace=True) # Do not lose video snippet index position
    CONTROL_PROJECTED_DF = pd.concat([CONTROL_DF[meta_colnames], projected_df], axis=1)

    # Plot PCA - Variation in control data with respect to a given variable (eg. date_recording_yyyymmdd)
            
    # 2-PC
    plt.close()
    PCAplotpath = PCAplotpath.replace('_PCA_explained', '_PCA_2_components')
    title = "2-Component PCA: Control variation in\n\
     '{0}'".format(grouping_variable) + " (Top256 features)"
    plotPCA(projected_df=CONTROL_PROJECTED_DF, grouping_variable=grouping_variable,\
            var_subset=None, savepath=PCAplotpath, title=title, n_component_axes=2)
    plt.pause(2)
    
    # 3-PC
    plt.close()
    PCAplotpath = PCAplotpath.replace('_PCA_2_components', '_PCA_3_components')
    title = "3-Component PCA: Control variation in\n\
     '{0}'".format(grouping_variable) + " (Top256 features)"
    plotPCA(projected_df=CONTROL_PROJECTED_DF, grouping_variable=grouping_variable,\
            var_subset=None, savepath=PCAplotpath, title=title, n_component_axes=3, rotate=False)
    plt.pause(2)
    
    return CONTROL_DF, CONTROL_PROJECTED_DF, remove_outliers

#%%
def boxplots_topfeats(CONTROL_DF, TEST_RESULTS_DF, grouping_variable, plot_save_dir,\
             p_value_threshold=0.05, n_topfeats=5):    
    """ Plot and save box plots for a given variable """

    if not os.path.exists(plot_save_dir):
        os.makedirs(plot_save_dir)
   
    pvals_corrected = TEST_RESULTS_DF.loc['pval_corrected']
    n_sigfeats = sum(pvals_corrected < p_value_threshold)
    
    if pvals_corrected.isna().all():
        print("No signficant features found in control with respect to '%s'!" % grouping_variable)
    elif n_sigfeats > 0:
        # Rank p-values in ascending order
        ranked_pvals = pvals_corrected.sort_values(ascending=True)
         
        # Drop NaNs
        ranked_pvals = ranked_pvals.dropna(axis=0)
        
        # Drop non-sig feats
        ranked_pvals = ranked_pvals[ranked_pvals < p_value_threshold]

        # Select the first n pvalues for plotting
        topfeats = ranked_pvals[:n_topfeats]
                
        if n_sigfeats < n_topfeats:
            print("WARNING: Only %d features found to vary significantly with respect to '%s'"\
                  % (n_sigfeats, grouping_variable))
            
        print("\nTop %d features for control that differ significantly with respect to '%s':\n"\
              % (len(topfeats), grouping_variable))
        print(*[feat + '\n' for feat in list(topfeats.index)])
    
        # for f, feature in enumerate(feature_column_names[0:25]):
        for feature in topfeats.index:
            print("P-value for '%s': %s" % (feature, str(topfeats[feature])))
            control_feat_df = CONTROL_DF[[grouping_variable, feature]]
            
            # Plot boxplots of control across days for most significant features
            plt.close('all')
            fig = plt.figure(figsize=[10,6])
            ax = fig.add_subplot(1,1,1)
            sns.boxplot(x=grouping_variable, y=feature, data=control_feat_df)
            ax.set_xlabel(grouping_variable, fontsize=15, labelpad=12)
            ax.set_title(feature, fontsize=20, pad=20)
            
            # TODO: Add reverse-engineered pvalues to plot?
            
            # Save plot
            plots_outpath = os.path.join(plot_save_dir, feature + '__by_'\
                                         + grouping_variable + '.eps')
            savefig(plots_outpath, tellme=True, saveFormat='eps')            
            plt.pause(5); plt.close()


#%%
def control_stats(CONTROL_DF, grouping_variable, feature_column_names,\
                  TEST=kruskal, p_value_threshold=0.05):
    print("\nTESTING: %s\n" % grouping_variable)    
    if not len(CONTROL_DF[grouping_variable].unique()) > 1:
        print("Need at least two groups for stats to investigate %s" % grouping_variable)
    else:
        # Record name of statistical test used (kruskal/f_oneway)
        test_name = str(TEST).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]
    
        # Kruskal-Wallis tests (ie. non-parametric one-way ANOVA) with Bonferroni correction for repeated measures
        print("""Performing '%s' tests for each feature to investigate 
        whether control results vary with respect to '%s':""" % (test_name, grouping_variable))
        TEST_RESULTS_DF = pd.DataFrame(index=['stat','pval'], columns=feature_column_names)
        for feature in feature_column_names:
            test_stat, test_pvalue = TEST(*[CONTROL_DF[CONTROL_DF[grouping_variable]==g_var][feature]\
                                                for g_var in CONTROL_DF[grouping_variable].unique()])
            TEST_RESULTS_DF.loc['stat',feature] = test_stat
            TEST_RESULTS_DF.loc['pval',feature] = test_pvalue
        
        # Bonferroni correction for multiple comparisons
        _corrArray = smm.multipletests(TEST_RESULTS_DF.loc['pval'], alpha=p_value_threshold,\
                                       method='fdr_bh', is_sorted=False, returnsorted=False)
        
        # Get pvalues for features that passed the Benjamini/Hochberg (non-negative) correlation test
        pvalues_corrected = _corrArray[1][_corrArray[0]]
        
        # Add pvalues to 1-way ANOVA results dataframe
        TEST_RESULTS_DF = TEST_RESULTS_DF.append(pd.Series(name='pval_corrected'))
        TEST_RESULTS_DF.loc['pval_corrected', _corrArray[0]] = pvalues_corrected
        
        n_sigfeats = sum(TEST_RESULTS_DF.loc['pval_corrected'] < p_value_threshold)
        
        print("%d / %d (%.1f%%) of features for control show significant variation (%s) with respect to '%s'" % \
              (n_sigfeats, len(TEST_RESULTS_DF.columns), n_sigfeats/len(TEST_RESULTS_DF.columns)*100,\
               test_name, grouping_variable))
                    
        # Compile list to store names of significant features
        sigfeats_out = TEST_RESULTS_DF.loc['pval_corrected'].sort_values(ascending=True) # Rank pvalues by significance
        sigfeats_out = sigfeats_out[sigfeats_out < p_value_threshold]
        sigfeats_out.name = 'p_value_' + test_name
        
#        # TODO: TukeyHSD:
#        # Tally total number of significantly different pairwise comparisons
#        n_sigdiff_pairwise_beforeBF = 0
#        n_sigdiff_pairwise_afterBF = 0
#        
#        # Tukey HSD post-hoc pairwise differences between dates for each feature
#        for feature in feature_column_names:
#            # Tukey HSD post-hoc analysis (no Bonferroni correction!)
#            tukeyHSD = pairwise_tukeyhsd(CONTROL_DF[feature], CONTROL_DF[grouping_variable])
#            n_sigdiff_pairwise_beforeBF += sum(tukeyHSD.reject)
#            
#            # Tukey HSD post-hoc analysis (Bonferroni correction)
#            tukeyHSD_BF = MultiComparison(CONTROL_DF[feature], CONTROL_DF[grouping_variable])
#            n_sigdiff_pairwise_afterBF += sum(tukeyHSD_BF.tukeyhsd().reject)   
#            
#        total_comparisons = len(feature_column_names) * 6
#        reject_H0_percentage = n_sigdiff_pairwise_afterBF / total_comparisons * 100
#        
#        print("""%d / %d (%.1f%%) of pairwise-comparisons (%d features) 
#        show significant variation in control (TukeyHSD) with respect to '%s'""" %\
#        (n_sigdiff_pairwise_afterBF, total_comparisons, reject_H0_percentage,\
#         len(feature_column_names), grouping_variable))
            
#        # TODO: Reverse-engineer p-values using mean/std?
#        from statsmodels.stats.libqsturng import psturng
#        # Studentized range statistic
#        rs = res2[1][2] / res2[1][3]
#        pvalues = psturng(np.abs(rs), 3, 27)
            
#         # TODO: MANOVA (date, temp, humid, etc)
#        
#         maov = MANOVA.from_formula('' + '' + '' ~ , data=CONTROL_DF)
#         print(maov.mv_test())
        
        return TEST_RESULTS_DF, sigfeats_out
    
    
#%%
def check_normality(CONTROL_DF, feature_column_names, p_value_threshold=0.05, is_normal_threshold=0.95):
    # Check control data for normality - to decide whether to use parametric/non-parametric statistics
    normality_results = pd.DataFrame(data=None, index=['stat','pval'], columns=feature_column_names)
    for f, feature in enumerate(feature_column_names):
        try:
            stat, pval = shapiro(CONTROL_DF[feature])
            # NB: UserWarning: Input data for shapiro has range zero # Some features contain all zeros - shapiro(np.zeros(5))
            normality_results.loc['stat',feature] = stat
            normality_results.loc['pval',feature] = pval
        except Exception as EE:
            print("WARNING: %s" % EE)
            
    prop_normal = (normality_results.loc['pval'] < p_value_threshold).sum()/len(feature_column_names)    
    if prop_normal > is_normal_threshold:
        print("More than %d%% of control features (%.1f%%) were found to obey a normal (Gaussian) distribution\
         so parametric analyses will be preferred." % (is_normal_threshold*100, prop_normal*100))
        TEST = f_oneway
    else:
        print("Less than %d%% of control features (%.1f%%) were found to obey a normal (Gaussian) distribution\
         so non-parametric analyses will be preferred." % (is_normal_threshold*100, prop_normal*100))
        TEST = kruskal
    return TEST

#%%
def control_variation(path_to_control_data, feature_column_names,\
                      variables_to_analyse = ["date_recording_yyyymmdd"],\
                      remove_outliers = True):
    """ A function written to analyse control variation over time across with respect 
        to a defined grouping variable (factor), eg. experiment day, run number, 
        duration of L1 diapause, camera/rig ID, etc. """

    # LOCAL PARAMETERS
    is_normal_threshold = 0.95 # Threshold for parametric/non-parametric stats
    p_value_threshold = 0.05 # P-value threshold for statistical analyses
    n_topfeats = 5 # HCA - Number of top features to include in HCA
    PCs_to_keep = 10 # PCA - Number of principal components to record

    # Load control summary results
    CONTROL_DF = pd.read_csv(path_to_control_data)
    print("Control data loaded.")

    # Record non-data columns before dropping feature columns   
    meta_colnames = [col for col in CONTROL_DF.columns if col not in feature_column_names]
        
    # Drop columns that contain only zeros
    colnames_before = list(CONTROL_DF.columns)
    AllZeroFeats = CONTROL_DF[feature_column_names].columns[(CONTROL_DF[feature_column_names] == 0).all()]
    CONTROL_DF = CONTROL_DF.drop(columns=AllZeroFeats)
    colnames_after = list(CONTROL_DF.columns)
    zero_cols = [col for col in colnames_before if col not in colnames_after]
    if len(zero_cols) > 0:
        print("Dropped %d features with all-zero summaries:\n%s" % (len(zero_cols), zero_cols))
    
    # Record a list of feature column names
    feature_column_names = [feat for feat in CONTROL_DF.columns if feat not in meta_colnames]
        
    TEST = check_normality(CONTROL_DF, feature_column_names, p_value_threshold, is_normal_threshold)

    # Record name of statistical test used (kruskal/f_oneway)
    test_name = str(TEST).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]

    # Do for all variables in list of variables to investigate

    # CONTROL VARIATION: STATS (ANOVAs)
    # - Does N2 worm behaviour on control vary across experiment days? 
    #       (worms are larger? Shorter L1 diapuase? Camera focus/FOV adjusted? Skewed by non-worm tracked objects?
    #       Did not record time when worms were refed! Could be this. If so, worms will be bigger across all foods on that day) 
    # - Perform ANOVA to see if features vary across imaging days for control
    # - Perform Tukey HSD post-hoc analyses for pairwise differences between imaging days
    # - Highlight outlier imaging days and investigate reasons why
    # - Save list of top significant features for outlier days - are they size-related features?
    
    # Plot control top10 size-skewed features for each food - do they all differ for outlier date? If so, worms are likely just bigger.
    # PCA: For just control - colour by imaging date - do they cluster visibly? If so, we have time-dependence = NOT GREAT 
    for grouping_variable in variables_to_analyse:
        print("\nTESTING: %s\n" % grouping_variable)
        
        if not len(CONTROL_DF[grouping_variable].unique()) > 1:
            print("Need at least two groups for stats to investigate %s" % grouping_variable)
        else:
            print("Performing '%s' tests for '%s'" % (test_name, grouping_variable))            
    
            TEST_RESULTS_DF, sigfeats_out = control_stats(CONTROL_DF,\
                                                          grouping_variable,\
                                                          feature_column_names,\
                                                          TEST,\
                                                          p_value_threshold)
            # Ensure directory exists to save results in
            DIRPATH = os.path.dirname(path_to_control_data)    
            stats_outpath = os.path.join(DIRPATH, grouping_variable, 'Stats',\
                                         'control_variation_in_' +\
                                         grouping_variable + '_' +\
                                         test_name + '_results.csv')
            directory = os.path.dirname(stats_outpath) 
            if not os.path.exists(directory):
                os.makedirs(directory)
                    
            # Save test statistics to file
            TEST_RESULTS_DF.to_csv(stats_outpath)

            # Save significant features list to CSV
            sigfeats_outpath = stats_outpath.replace("_results.csv",\
                                                     "_significant_features.csv")
            sigfeats_out.to_csv(sigfeats_outpath, header=False)

            # Box plots
            plot_save_dir = os.path.join(DIRPATH, grouping_variable, "Plots")

            boxplots_topfeats(CONTROL_DF, TEST_RESULTS_DF, grouping_variable,\
                              plot_save_dir, p_value_threshold, n_topfeats)
                        
            # PCA of CONTROL DATA ACROSS DAYS
            CONTROL_DF, CONTROL_PROJECTED_DF, remove_outliers = doPCA(CONTROL_DF,\
                                                                      grouping_variable,\
                                                                      feature_column_names,\
                                                                      meta_colnames,\
                                                                      plot_save_dir,\
                                                                      remove_outliers,\
                                                                      PCs_to_keep)
        
#%% MAIN
        
if __name__ == '__main__':
    tic = time.time()
    if not len(sys.argv) >= 2:
        print("Please provide path to control data, followed by an unpacked list\
        of feature column names as inputs.")
    else:
        print("\nRunning script", os.path.basename(sys.argv[0]), "...")
        path_to_control_data = sys.argv[1]
        feature_column_names = list(sys.argv[2:])
        
        # Variables to analyse
        variables_to_analyse = ['date_recording_yyyymmdd','run_number',\
                                'plate_number','master_stock_plate_ID',\
                                'instrument_name','well_number']
        
        # Analyse control variation over time
        control_variation(path_to_control_data, feature_column_names,\
                          variables_to_analyse, remove_outliers=True)   
        
    toc = time.time()
    print("Control analysis complete.\n(Time taken: %d seconds)" % (toc-tic))
