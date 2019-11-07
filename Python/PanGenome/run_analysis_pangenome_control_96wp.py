#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OP50 CONTROL VARIATION ACROSS DAYS

Analyse control data: look for variation across experiment days (Kruskal-Wallis) 
and plot: (a) boxplots of the most important features that vary across days, (b)

@author: sm5911
@date: 26/10/2019

"""

#%% IMPORTS

# General imports
import os, sys, itertools, time#, umap
import pandas as pd
import seaborn as sns; sns.set(color_codes=True)
from matplotlib import pyplot as plt
from scipy.stats import zscore, kruskal#, f_oneway
from statsmodels.stats import multitest as smm#, AnovaRM
from statsmodels.stats.multicomp import MultiComparison, pairwise_tukeyhsd
#from statsmodels.multivariate.manova import MANOVA
from sklearn.decomposition import PCA
from matplotlib.axes._axes import _log as mpl_axes_logger # Work-around for Axes3D plot colour warnings
from mpl_toolkits.mplot3d import Axes3D

# Path to Github / local helper functions
sys.path.insert(0, '/Users/sm5911/Documents/GitHub/PhD_Project/Python/PanGenome')

# Custom imports
from helper import savefig, pcainfo

# TODO: Colour PCA plots: refeeding time to track (morn vs afternoon), and bleaching variation, temp + humidity

#%% MAIN
def control_variation(path_to_control_data, feature_column_names, grouping_variable="date_recording_yyyymmdd"):
    """ A function written to analyse control data variation over time across
        different experiment days. """

#%%    
    DIRPATH = os.path.dirname(path_to_control_data)
                        
    # Statistics parameters
    test = kruskal # f_oneway
    TukeyHSD = False
    p_value_threshold = 0.05 # P-vlaue threshold for statistical analyses
    
    # Dimensionality reduction parameters
    useTop256 = True                # Restrict dimensionality reduction inputs to Avelino's top 256 feature list?
    n_top_feats_per_food = 10       # HCA - Number of top features to include in HCA (for union across foods HCA)
    PCs_to_keep = 10                # PCA - Number of principal components to record
    depthshade = False              # PCA - Shade colours on 3-D plots to show depth?
#    perplexity = [10,15,20,25,30]   # tSNE - Parameter range for t-SNE mapping
#    n_neighbours = [10,15,20,25,30] # UMAP - Number of neighbours parameter for UMAP projections
#    min_dist = 0.3                  # UMAP - Minimum distance parameter for UMAP projections
  
    #%% READ + FILTER + CLEAN SUMMARY RESULTS
    
    CONTROL_DF = pd.read_csv(path_to_control_data, index_col=0)
    print("Control data loaded.")

    # Record non-data columns before dropping features for statistics    
    meta_colnames = [col for col in CONTROL_DF.columns if col not in feature_column_names]
        
    # Drop columns that contain only zeros
    colnames_before = list(CONTROL_DF.columns)
    CONTROL_DF = CONTROL_DF.drop(columns=CONTROL_DF.columns[(CONTROL_DF == 0).all()])
    colnames_after = list(CONTROL_DF.columns)
    zero_cols = [col for col in colnames_before if col not in colnames_after]
    if len(zero_cols) > 0:
        print("Dropped %d features with all-zero summaries:\n%s" % (len(zero_cols), zero_cols))
    
    # Record a list of feature column names
    feat_names = [col for col in CONTROL_DF.columns if col not in meta_colnames]
    
    #%% OP50 CONTROL DATA ACROSS DAYS: STATS (ANOVAs)
    # - Does N2 worm behaviour on OP50 control vary across experiment days?
    # - Perform ANOVA to see if features vary across imaging days for OP50 control
    # - Perform Tukey HSD post-hoc analyses for pairwise differences between imaging days
    # - Highlight outlier imaging days and investigate reasons why
    # - Save list of top significant features for outlier days - are they size-related features?
    #   (worms are larger? Shorter L1 diapuase? Camera focus/FOV adjusted? Skewed by non-worm tracked objects?
    #   Did not record time when worms were refed! Could be this. If so, worms will be bigger across all foods on that day) 
    
    # Plot OP50 control top10 size-skewed features for each food - do they all differ for outlier date? If so, worms are likely just bigger.
    # PCA: For just OP50 control - colour by imaging date - do they cluster visibly? If so, we have time-dependence = NOT GREAT 
    # => Consider excluding that date on the basis of un-standardised development times since refeeding?
    
    # Kruskal-Wallis tests (ie. non-parametric one-way ANOVA) with Bonferroni correction for repeated measures
    print("""Performing Kruskal-Wallis tests for each feature to investigate 
             whether control OP50 results vary across imaging dates:""")
    TEST_RESULTS_DF = pd.DataFrame(index=['stat','pval'], columns=feat_names)
    for feature in feat_names:
        test_stat, test_pvalue = test(*[CONTROL_DF[CONTROL_DF[grouping_variable]==g_var][feature]\
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
    
    print("%d / %d (%.1f%%) of features show significant variation across imaging dates for OP50 control (ANOVA/Kruskal)" % \
          (n_sigfeats, len(TEST_RESULTS_DF.columns), n_sigfeats/len(TEST_RESULTS_DF.columns)*100))
    
    # Record name of statistical test used (kruskal/f_oneway)
    test_name = str(test).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]
    
    # Save test statistics to file
    stats_outpath = os.path.join(DIRPATH, 'OP50_control_across_days_' + test_name + '_stats.csv')
    TEST_RESULTS_DF.to_csv(stats_outpath)
    
    # Compile list to store names of significant features
    sigfeats_out = TEST_RESULTS_DF.loc['pval_corrected'].sort_values(ascending=True) # Rank pvalues by significance
    sigfeats_out = sigfeats_out[sigfeats_out < p_value_threshold]
    sigfeats_out.name = 'p_value_' + test_name
    
    # Save significant features list to CSV
    sigfeats_outpath = os.path.join(DIRPATH, 'OP50_control_across_days_' + test_name + '_sigfeats.csv')
    sigfeats_out.to_csv(sigfeats_outpath, header=False)
    
    if TukeyHSD:
        # Tally total number of significantly different pairwise comparisons
        n_sigdiff_pairwise_beforeBF = 0
        n_sigdiff_pairwise_afterBF = 0
        
        # Tukey HSD post-hoc pairwise differences between dates for each feature
        for feature in feat_names:
            # Tukey HSD post-hoc analysis (no Bonferroni correction!)
            tukeyHSD = pairwise_tukeyhsd(CONTROL_DF[feature], CONTROL_DF[grouping_variable])
            n_sigdiff_pairwise_beforeBF += sum(tukeyHSD.reject)
            
            # Tukey HSD post-hoc analysis (Bonferroni correction)
            tukeyHSD_BF = MultiComparison(CONTROL_DF[feature], CONTROL_DF[grouping_variable])
            n_sigdiff_pairwise_afterBF += sum(tukeyHSD_BF.tukeyhsd().reject)   
            
        total_comparisons = len(feat_names) * 6
        reject_H0_percentage = n_sigdiff_pairwise_afterBF / total_comparisons * 100
        
        print("""%d / %d (%.1f%%) of pairwise-comparisons of imaging dates (%d features) 
        show significant variation for OP50 control (TukeyHSD)""" %\
        (n_sigdiff_pairwise_afterBF, total_comparisons, reject_H0_percentage, len(feat_names)))
        
        # TODO: Reverse-engineer p-values using mean/std?
        #from statsmodels.stats.libqsturng import psturng
        ##studentized range statistic
        #rs = res2[1][2] / res2[1][3]
        #pvalues = psturng(np.abs(rs), 3, 27)
        
    #%% MANOVA (date, temp, humid, etc)

#    maov = MANOVA.from_formula('' + '' + '' ~ , data=CONTROL_DF)
#    print(maov.mv_test())
    
    #%% Boxplots for most important features across days
    
    plotroot = os.path.join(DIRPATH, "Plots")
    if not os.path.exists(plotroot):
        os.makedirs(plotroot)

    pvals_corrected = TEST_RESULTS_DF.loc['pval_corrected']
    n_sigfeats = sum(pvals_corrected < p_value_threshold)
    
    if pvals_corrected.isna().all():
        print("No signficant features found across days for OP50 control!")
    elif n_sigfeats > 0:
        # Rank p-values in ascending order
        ranked_pvals = pvals_corrected.sort_values(ascending=True)
         
        # Drop NaNs
        ranked_pvals = ranked_pvals.dropna(axis=0)
        
        # Drop non-sig feats
        ranked_pvals = ranked_pvals[ranked_pvals < p_value_threshold]

        # Select the first n pvalues for plotting
        topfeats = ranked_pvals[:n_top_feats_per_food]
                
        if n_sigfeats < n_top_feats_per_food:
            print("WARNING: Only %d features found to vary significantly across days" % n_sigfeats)
            
        print("\nTop %d features for OP50 that differ significantly across days:\n" % len(topfeats))
        print(*[feat + '\n' for feat in list(topfeats.index)])
    
        # for f, feature in enumerate(feat_names[0:25]):
        for feature in topfeats.index:
            print("P-value for '%s': %s" % (feature, str(topfeats[feature])))
            OP50_feat_df = CONTROL_DF[[grouping_variable, feature]]
            
            # Plot boxplots of OP50 control across days for most significant features
            plt.close('all')
            fig = plt.figure(figsize=[10,6])
            ax = fig.add_subplot(1,1,1)
            sns.boxplot(x=grouping_variable, y=feature, data=OP50_feat_df)
            if grouping_variable == "date_recording_yyyymmdd":
                ax.set_xlabel('Imaging Date (YYYYMMDD)', fontsize=15, labelpad=12)
            ax.set_title(feature, fontsize=20, pad=20)
            
            # TODO: Add reverse-engineered pvalues to plot?
            
            # Save plot
            plots_outpath = os.path.join(plotroot, feature + '_across_days.eps')
            savefig(plots_outpath, tellme=True, saveFormat='eps')            
    
    #%% PCA of OP50 CONTROL DATA ACROSS DAYS
    
    PCAplotroot = os.path.join(plotroot, 'PCA')
    if not os.path.exists(PCAplotroot):
        os.makedirs(PCAplotroot)
    
    # Read list of important features (highlighted by previous research - see Javer, 2018 paper)
    if useTop256:
        featroot = DIRPATH.split("/Results/")[0]
        featslistpath = os.path.join(featroot,'AuxiliaryFiles','top256_tierpsy_no_blob_no_eigen_only_abs_no_norm.csv')
        top256features = pd.read_csv(featslistpath)
        
        # Take first set of 256 features (it does not matter which set is chosen)
        top256features = top256features[top256features.columns[0]]   
        top256features = [feat for feat in top256features if feat in CONTROL_DF.columns]
        print("PCA: Using existing results for %d/256 features in Top256 list (Javer 2018)" % len(top256features))
    
        # Drop all but top256 columns for PCA
        data = CONTROL_DF[top256features]
    else:
        data = CONTROL_DF[feat_names]
    
    # Normalise the data before PCA
    zscores = data.apply(zscore, axis=0)
    
    # Drop features with NaN values after normalising
    colnames_before = list(zscores.columns)
    zscores.dropna(axis=1, inplace=True)
    colnames_after = list(zscores.columns)
    nan_cols = [col for col in colnames_before if col not in colnames_after]
    if len(nan_cols) > 0:
        print("Dropped %d features with NaN values after normalization:\n%s" % (len(nan_cols), nan_cols))
    
    # Perform PCA on extracted features
    print("\nPerforming Principal Components Analysis (PCA)...")
    
    # Fit the PCA model with the normalised data
    pca = PCA()
    pca.fit(zscores)
    
    # Plot summary data from PCA: explained variance (most important features)
    important_feats, fig = pcainfo(pca, zscores, PC=1, n_feats2print=10)
    
    # Save plot of PCA explained variance
    PCAplotpath = os.path.join(PCAplotroot, 'PCA_explained.eps')
    savefig(PCAplotpath, tight_layout=True, tellme=True, saveFormat='eps')
    
    # Project data (zscores) onto PCs
    projected = pca.transform(zscores) # A matrix is produced
    # NB: Could also have used pca.fit_transform()
    
    # Store the results for first few PCs in dataframe
    projected_df = pd.DataFrame(projected[:,:PCs_to_keep],\
                                columns=['PC' + str(n+1) for n in range(PCs_to_keep)])
    
    # Add concatenate projected PC results to metadata
    projected_df.set_index(CONTROL_DF.index, inplace=True) # Do not lose video snippet index position
    CONTROL_PROJECTED_DF = pd.concat([CONTROL_DF[meta_colnames], projected_df], axis=1)
    
    #%% 2D Plot - first 2 PCs - OP50 Control (coloured by imaging date)
    
    # Plot first 2 principal components
    plt.close('all'); plt.ion()
    plt.rc('xtick',labelsize=15)
    plt.rc('ytick',labelsize=15)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=[10,10])
    
    # Create colour palette for plot loop
    group_vars = list(CONTROL_PROJECTED_DF[grouping_variable].unique())
    palette = itertools.cycle(sns.color_palette("gist_rainbow", len(group_vars)))
    
    for g_var in group_vars:
        group_projected_df = CONTROL_PROJECTED_DF[CONTROL_PROJECTED_DF[grouping_variable]==int(g_var)]
        sns.scatterplot(group_projected_df['PC1'], group_projected_df['PC2'], color=next(palette), s=100)
    ax.set_xlabel('Principal Component 1', fontsize=15, labelpad=12)
    ax.set_ylabel('Principal Component 2', fontsize=15, labelpad=12)
    if useTop256:
        ax.set_title('Top256 features 2-Component PCA', fontsize=20)
    else: 
        ax.set_title('All features 2-Component PCA', fontsize=20)
    plt.tight_layout(rect=[0.04, 0, 0.84, 0.96])
    ax.legend(group_vars, frameon=False, loc=(1, 0.65), fontsize=15)
    ax.grid()
    
    # Save scatterplot of first 2 PCs
    PCAplotpath = PCAplotpath.replace('PCA_explained', '2_component_PCA')
    savefig(PCAplotpath, tight_layout=True, tellme=True, saveFormat='eps')
    
    plt.show(); plt.pause(2)


    #%% Plot 3 PCs - OP50 across imaging dates
    rotate = True
    
    # Work-around for 3D plot colour warnings
    mpl_axes_logger.setLevel('ERROR')

    # Plot first 3 principal components
    plt.close('all')
    fig = plt.figure(figsize=[10,10])
    ax = Axes3D(fig, rect=[0.04, 0, 0.8, 0.96]) # ax = fig.add_subplot(111, projection='3d')
    
    # Re-initialise colour palette for plot loop
    palette = itertools.cycle(sns.color_palette("gist_rainbow", len(group_vars)))
    
    for g_var in group_vars:
        group_projected_df = CONTROL_PROJECTED_DF[CONTROL_PROJECTED_DF['date_recording_yyyymmdd']==int(g_var)]
        ax.scatter(xs=group_projected_df['PC1'], ys=group_projected_df['PC2'], zs=group_projected_df['PC3'],\
                   zdir='z', color=next(palette), s=50, depthshade=depthshade)
    ax.set_xlabel('Principal Component 1', fontsize=15, labelpad=12)
    ax.set_ylabel('Principal Component 2', fontsize=15, labelpad=12)
    ax.set_zlabel('Principal Component 3', fontsize=15, labelpad=12)
    if useTop256:
        ax.set_title('Top256 features 2-Component PCA', fontsize=20, pad=20)
    else: 
        ax.set_title('All features 2-Component PCA', fontsize=20, pad=20)
    ax.legend(group_vars, frameon=False, loc=(0.97, 0.65), fontsize=15)
    ax.grid()
    
    # Save scatterplot of first 3 PCs
    PCAplotpath = PCAplotpath.replace('PCA_explained', '3_component_PCA')
    savefig(PCAplotpath, tight_layout=False, tellme=True, saveFormat='eps')
    
    # Rotate the axes and update
    if rotate:
        for angle in range(0, 360):
            ax.view_init(30, angle)
            plt.draw(); plt.pause(0.001)
    else:
        plt.show()

#%% INPUT HANDLING AND GLOBAL PARAMETERS
        
# TODO: Optional: call script from terminal command line as well as use as function

if __name__ == '__main__':
    tic = time.time()
    if len(sys.argv) >= 2:
        print("\nRunning script", os.path.basename(sys.argv[0]), "...")
        path_to_control_data = sys.argv[1]
        feat_names = list(sys.argv[2:])
        
        # Analyse control variation over time
        control_variation(path_to_control_data, feat_names)   
    else:
        print("""Please provide path to control data, followed by an unpacked list 
        of feature column names as inputs.""")
        
    toc = time.time()
    print("OP50 control analysis complete.\n(Time taken: %d seconds)" % (toc-tic))
