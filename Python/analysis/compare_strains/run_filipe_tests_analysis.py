#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse Filipe's Tests - N2 vs R60K (ribosomal mutant) behaviour on E. coli OP50

@author: sm5911
@date: 20/11/2020
"""

#%% Imports

import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from scipy.stats import kruskal, ttest_ind, f_oneway, zscore
from statsmodels.stats import multitest as smm # AnovaRM
from matplotlib import pyplot as plt
from matplotlib import transforms, patches
    
# Path to Github helper functions (USER-DEFINED path to local copy of Github repo)
PATH_LIST = ['/Users/sm5911/Tierpsy_Versions/tierpsy-tools-python/']
for sysPath in PATH_LIST:
    if sysPath not in sys.path:
        sys.path.insert(0, sysPath)
        
from helper import concatenate_metadata, clean_features_summaries, ranksumtest
from tierpsytools.read_data import compile_features_summaries
from tierpsytools.read_data import hydra_metadata

#%% Globals

omit_strain_list = None # List of bacterial or worm strains to omit (depending on chosen 'grouping_var')
control_strains = ['OP50','N2'] # Control strains if worm/bacteria
useTop256 = False # Use Tierpsy Top256 features for analysis?
filter_size_related_feats = False # Drop size features from analysis?
show_plots = False
nan_threshold = 0.2 # Threshold NaN proportion to drop feature from analysis  
p_value_threshold = 0.05 # Threshold p-value for statistical significance                                                       
PCs_to_keep = 10 # PCA: Number of principle components to use
n_PC_feats2plot = 10 # PCA: Number of top features influencing PC to plot
perplexities = [5,10,20,30] # tSNE: Perplexity parameter for tSNE mapping
n_neighbours = [5,10,20,30] # UMAP: N-neighbours parameter for UMAP projections                                            
min_dist = 0.3 # Minimum distance parameter for UMAP projections    
      
#%% Functions


#%% Main

if __name__ == "__main__":
    
    # Accept command-line args
    parser = argparse.ArgumentParser(description='Analyse Tierpsy results (96-well)')
    
    # Project root directory
    parser.add_argument('--project_root_dir', help="Project root directory,\
                        containing 'AuxiliaryFiles', 'RawVideos',\
                        'MaskedVideos' and 'Results' folders,\
                        eg. /Volumes/hermes$/KeioScreen_96WP",
                        default="/Volumes/hermes$/Filipe_Tests_96WP")
    parser.add_argument('--grouping_var', help="Treatment variable by which you\
                        want to group results, eg. 'worm_strain' or 'food_type'",
                        default='worm_strain')
    args = parser.parse_args()
    
    project_root_dir = Path(args.project_root_dir)
    grouping_var = str(args.grouping_var)
    print('\nProject root directory: %s' % str(project_root_dir))
    print('Grouping variable: %s' % grouping_var)

    imaging_dates = None # Imaging dates
    timepoint = None # Timepoint to analyse
    analyse_control_variation = True # Analyse variation in control data?
     
    # Path to metadata              
    metadata_dir = project_root_dir / "AuxiliaryFiles"
    metadata_path = metadata_dir / 'metadata.csv' 
    
    # Process metadata     
    if not metadata_path.exists():
        print("Compiling metadata..")
        # Compile metadata for experiment dates
        metadata = concatenate_metadata(metadata_dir, imaging_dates)
        
        metadata.to_csv(metadata_path, index=False)
    else:
        metadata = pd.read_csv(metadata_path, header=0, dtype={"comments":str})
        print("Metadata loaded.")
        
    # Record metadata column names
    metadata_colnames = list(metadata.columns)
    
    # Path to feature summary results
    results_dir = project_root_dir / "Results"
    combined_feats_path = results_dir / "full_features.csv"
    combined_fnames_path = results_dir / "full_filenames.csv"
    full_results_path = results_dir / 'fullresults.csv'

    # Process feature summaries
    if not np.logical_and(combined_feats_path.is_file(), 
                          combined_fnames_path.is_file()):   
        print("\nProcessing feature summary results...")
        # Compile full results: metadata + featsums
        feat_files = [file for file in Path(results_dir).rglob('features_summary*.csv')]
        fname_files = [Path(str(file).replace("/features_", "/filenames_")) for file in feat_files]
        
        # Keep only features files for which matching filenames_summaries exist
        feat_files = [feat_fl for feat_fl,fname_fl in zip(feat_files, fname_files)
                      if fname_fl is not None]
        fname_files = [fname_fl for fname_fl in fname_files
                        if fname_fl is not None]
        
        # Compile feature summaries for mathed features/filename summaries
        compile_features_summaries.compile_tierpsy_summaries(feat_files=feat_files, 
                                                             compiled_feat_file=combined_feats_path,
                                                             compiled_fname_file=combined_fnames_path,
                                                             fname_files=fname_files)
    
        # Read features/filename summaries
        feature_summaries = pd.read_csv(combined_feats_path, comment='#')
        filename_summaries = pd.read_csv(combined_fnames_path, comment='#')
    
        features, metadata = hydra_metadata.read_hydra_metadata(feature_summaries, 
                                                                filename_summaries,
                                                                metadata,
                                                                add_bluelight=False)
        
        features, metadata = clean_features_summaries(features, 
                                                      metadata,
                                                      featurelist=None,
                                                      imputeNaN=True,
                                                      nan_threshold=nan_threshold,
                                                      filter_size_related_feats=filter_size_related_feats)
        # Join metadata + results
        fullresults = metadata.join(features)
    
        # Save full results to file
        fullresults.to_csv(full_results_path, index=False)
        print("Results saved to:\n '%s'" % full_results_path) 
    else:
        try:
            fullresults = pd.read_csv(full_results_path, dtype={"comments":str}) 
            print("Results loaded.")
        except Exception as EE:
            print("ERROR: %s" % EE)
            print("Please process feature summaries and provide correct path to results.")

    # Record new columns added to metadata
    for col in ['featuresN_filename', 'file_id', 'is_good_well']:
        if col not in metadata_colnames:
            metadata_colnames.append(col)
    
    # Analysis is case-sensitive - ensure that there is no confusion in strain names
    assert len(fullresults[grouping_var].unique()) == len(fullresults[grouping_var].str.upper().unique())
    
    # Record worm/bacterial strain names for which we have results
    strain_list = [strain for strain in list(fullresults[grouping_var].unique())]
    if omit_strain_list:
        strain_list = [strain for strain in strain_list if strain not in omit_strain_list]    
          
    # Subset data for worm/bacterial strains to investigate
    fullresults = fullresults[fullresults[grouping_var].isin(strain_list)]
        
    # (OPTIONAL) Subset for imaging dates provided
    if imaging_dates:
        fullresults = fullresults[fullresults['date_yyyymmdd'].isin(imaging_dates)]
    
    # # (OPTIONAL) Subset for a single timepoint only, if given
    # if timepoint:
    #     fullresults = fullresults[fullresults['time_point'].isin(timepoint)]
    
    # (OPTIONAL) Load Top256
    if useTop256:
        top256_path = project_root_dir / 'AuxiliaryFiles' /\
                      'top256_tierpsy_no_blob_no_eigen_only_abs_no_norm.csv'
    
        # Read list of important features (as shown previously by Javer, 2018) and 
        # take first set of 256 features (it does not matter which set is chosen)
        top256_df = pd.read_csv(top256_path)
        featurelist = list(top256_df[top256_df.columns[0]])
        n = len(featurelist)
        print("Feature list loaded (n=%d features)" % n)
    
        # Remove features from Top256 that are path curvature related
        featurelist = [feat for feat in featurelist if "path_curvature" not in feat]
        n_feats_after = len(featurelist)
        print("Dropped %d features from Top%d that are related to path curvature" %\
              ((n - n_feats_after), n)) 
    
        # Ensure results exist for features in featurelist
        featurelist = [feat for feat in featurelist if feat in fullresults.columns]
        assert len(featurelist) == n_feats_after
    else:
        # Determine list of features to plot
        featurelist = [feat for feat in fullresults.columns if feat not in metadata_colnames]
        
#%% Perform STATISTICS: t-tests/rank-sum tests
#   - To look for behavioural features that differ significantly on test strains vs control 
    
    # Identify control and test strains for statistics
    test_strains = [strain for strain in strain_list if strain not in control_strains]
    control_strain = [strain for strain in strain_list if strain in control_strains]
    assert len(control_strain) == 1
    control_strain = control_strain[0]

    # Record name of statistical test used (ttest/ranksumtest)
    TEST = ttest_ind
    test_name = str(TEST).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]

    # Pre-allocate dataframes for storing test statistics and p-values
    test_stats_df = pd.DataFrame(index=list(test_strains), columns=featurelist)
    test_pvalues_df = pd.DataFrame(index=list(test_strains), columns=featurelist)
    sigdifffeats_df = pd.DataFrame(index=test_pvalues_df.index, columns=['N_sigdiff_beforeBF','N_sigdiff_afterBF'])
    
    # Store control dataframe
    control_df = fullresults[fullresults[grouping_var] == control_strain]

    # Compare each strain to OP50: compute test statistics for each feature
    for t, strain in enumerate(test_strains):
        print("Computing %s tests for %s vs %s..." % (test_name, control_strain, strain))
            
        # Grab feature summary results for that strain
        test_strain_df = fullresults[fullresults[grouping_var] == strain]
        
        # Drop non-data columns
        test_data = test_strain_df.drop(columns=metadata_colnames)
        control_data = control_df.drop(columns=metadata_colnames)
                   
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
            print("Dropped %d feature summaries for %s (all zeros)" % (zero_cols, strain))
    
        # Perform rank-sum tests comparing between strains for each feature (max features = 4539)
        test_stats, test_pvalues = TEST(test_data, control_data)
    
        # Add test results to out-dataframe
        test_stats_df.loc[strain][shared_colnames] = test_stats
        test_pvalues_df.loc[strain][shared_colnames] = test_pvalues
        
        # Record the names and number of significant features 
        sigdiff_feats = test_pvalues_df.columns[np.where(test_pvalues < p_value_threshold)]
        sigdifffeats_df.loc[strain,'N_sigdiff_beforeBF'] = len(sigdiff_feats)
            
    # Benjamini/Hochberg corrections for multiple comparisons
    sigdifffeatslist = []
    test_pvalues_corrected_df = pd.DataFrame(index=test_pvalues_df.index, columns=test_pvalues_df.columns)
    for strain in test_pvalues_df.index:
        # Locate pvalue results (row) for strain
        strain_pvals = test_pvalues_df.loc[strain] # pd.Series object
        
        # Perform Benjamini/Hochberg correction for multiple comparisons on t-test pvalues
        _corrArray = smm.multipletests(strain_pvals.values, alpha=p_value_threshold, method='fdr_bh',\
                                        is_sorted=False, returnsorted=False)
        
        # Get pvalues for features that passed the Benjamini/Hochberg (non-negative) correlation test
        pvalues_corrected = _corrArray[1][_corrArray[0]]
        
        # Add pvalues to dataframe of corrected ttest pvalues
        test_pvalues_corrected_df.loc[strain, _corrArray[0]] = pvalues_corrected
        
        # Record the names and number of significant features (after Benjamini/Hochberg correction)
        sigdiff_feats = pd.Series(test_pvalues_corrected_df.columns[np.where(_corrArray[1] < p_value_threshold)])
        sigdiff_feats.name = strain
        sigdifffeatslist.append(sigdiff_feats)
        sigdifffeats_df.loc[strain,'N_sigdiff_afterBF'] = len(sigdiff_feats)

    # Concatenate into dataframe of features for each strain that differ significantly from behaviour on OP50
    sigdifffeats_strain_df = pd.concat(sigdifffeatslist, axis=1, ignore_index=True, sort=False)
    sigdifffeats_strain_df.columns = test_pvalues_corrected_df.index

    # Save test statistics to file
    stats_outpath = project_root_dir / 'Results' / 'Stats' / '{}_results.csv'.format(test_name)
    sigfeats_outpath = Path(str(stats_outpath).replace('_results.csv', '_significant_features.csv'))
    stats_outpath.parent.mkdir(exist_ok=True, parents=True) # Create save directory if it does not exist
    test_pvalues_corrected_df.to_csv(stats_outpath) # Save test results to CSV
    sigdifffeats_strain_df.to_csv(sigfeats_outpath, index=False) # Save feature list to text file
        
    # Proportion of features significantly different from OP50
    propfeatssigdiff = ((test_pvalues_corrected_df < p_value_threshold).sum(axis=1)/len(featurelist))*100
    propfeatssigdiff = propfeatssigdiff.sort_values(ascending=False)
    
    # Plot sigdiff feats
    fig = plt.figure(figsize=[7,10])
    ax = fig.add_subplot(1,1,1)
    propfeatssigdiff.plot.barh(ec='black') # fc
    ax.set_xlabel('% significantly different features', fontsize=16, labelpad=10)
    ax.set_ylabel('Strain', fontsize=17, labelpad=10)
    plt.xlim(0,100)
    plt.tight_layout(rect=[0.02, 0.02, 0.96, 1])
    plots_outpath = project_root_dir / 'Results' / 'Plots' / 'All' / 'Percentage_features_sigdiff.eps'
    plots_outpath.parent.mkdir(exist_ok=True, parents=True)
    print("Saving figure: %s" % plots_outpath.name)
    plt.savefig(plots_outpath, format='eps', dpi=600)
    if show_plots:
        plt.show(); plt.pause(2); plt.close('all')
    
#%% STATISTICS: One-way ANOVA/Kruskal-Wallis + Tukey HSD post-hoc tests 
#               for pairwise differences between strains for each feature

    # Record name of statistical test used (kruskal/f_oneway)
    TEST = f_oneway
    test_name = str(TEST).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]

    print("\nComputing %s tests between strains for each feature..." % test_name)

    # Perform 1-way ANOVAs for each feature between test strains
    test_pvalues_df = pd.DataFrame(index=['stat','pval'], columns=featurelist)
    for f, feature in tqdm(enumerate(featurelist)):
            
        # Perform test and capture outputs: test statistic + p value
        test_stat, test_pvalue = TEST(*[fullresults[fullresults[grouping_var]==strain][feature]\
                                           for strain in fullresults[grouping_var].unique()])
        test_pvalues_df.loc['stat',feature] = test_stat
        test_pvalues_df.loc['pval',feature] = test_pvalue

    # Perform Bonferroni correction for multiple comparisons on one-way ANOVA pvalues
    _corrArray = smm.multipletests(test_pvalues_df.loc['pval'], alpha=p_value_threshold, method='fdr_bh',\
                                   is_sorted=False, returnsorted=False)
    
    # Get pvalues for features that passed the Benjamini-Hochberg (non-negative) correlation test
    pvalues_corrected = _corrArray[1][_corrArray[0]]
    
    # Add pvalues to one-way ANOVA results dataframe
    test_pvalues_df = test_pvalues_df.append(pd.Series(name='pval_corrected'))
    test_pvalues_df.loc['pval_corrected', _corrArray[0]] = pvalues_corrected
    
    # Store names of features that show significant differences across the test bacteria
    sigdiff_feats = test_pvalues_df.columns[np.where(test_pvalues_df.loc['pval_corrected'] < p_value_threshold)]
    print("Complete!\n%d/%d (%.1f%%) features exhibit significant differences between strains (%s test, Benjamini-Hochberg)"\
          % (len(sigdiff_feats), len(test_pvalues_df.columns), len(sigdiff_feats)/len(test_pvalues_df.columns)*100, test_name))
    
    # Compile list to store names of significant features
    sigfeats_out = pd.Series(test_pvalues_df.columns[np.where(test_pvalues_df.loc['pval_corrected'] < p_value_threshold)])
    sigfeats_out.name = 'significant_features_' + test_name
    sigfeats_out = pd.DataFrame(sigfeats_out)
    
    # Save test statistics to file
    stats_outpath = project_root_dir / 'Results' / 'Stats' / '{}_results.csv'.format(test_name)
    sigfeats_outpath = stats_outpath.replace('_results.csv', '_significant_features.csv')
    stats_outpath.parent.mkdir(exist_ok=True, parents=True)
    test_pvalues_df.to_csv(stats_outpath) # Save test results as CSV
    sigfeats_out.to_csv(sigfeats_outpath, index=False) # Save feature list as text file
    
    topfeats = test_pvalues_df.loc['pval_corrected'].sort_values(ascending=True)[:10]
    print("Top 10 significant features by %s test:\n" % test_name)
    for feat in topfeats.index:
        print(feat)

# TODO: Perform post-hoc analyses (eg.Tukey HSD) for pairwise comparisons between strains for each feature?


#%% BOX PLOTS - INDIVIDUAL PLOTS OF TOP RANKED FEATURES (STATS) FOR EACH strain
# - Rank features by pvalue significance (lowest first) and select the Top 10 features for each strain
# - Plot boxplots of the most important features for each strain compared to OP50
# - Plot features separately with feature as title and in separate folders for each strain

    # Load test results (pvalues) for plotting
    # NB: Non-parametric ranksum test preferred over t-test as many features may not be normally distributed
    test_name = 'ttest_ind'
        
    stats_inpath = project_root_dir / 'Results' / 'Stats' / '{}_results.csv'.format(test_name)
    test_pvalues_df = pd.read_csv(stats_inpath, index_col=0)
    print("Loaded %s results." % test_name)
    
    n_top_features = 5 # Number of top-ranked features to plot
    
    plt.ioff()
    plt.close('all')
    print("\nPlotting box plots of top %d (%s) features for each strain:\n" % (n_top_features, test_name))
    for i, strain in enumerate(test_pvalues_df.index):
        pvals = test_pvalues_df.loc[strain]
        n_sigfeats = sum(pvals < p_value_threshold)
        n_nonnanfeats = np.logical_not(pvals.isna()).sum()
        if pvals.isna().all():
            print("No signficant features found for %s" % strain)
        elif n_sigfeats > 0:       
            # Rank p-values in ascending order
            ranked_pvals = pvals.sort_values(ascending=True)
            # Drop NaNs
            ranked_pvals = ranked_pvals.dropna(axis=0)
            topfeats = ranked_pvals[:n_top_features] # Select the top ranked p-values
            topfeats = topfeats[topfeats < p_value_threshold] # Drop non-sig feats   
            ## OPTIONAL: Cherry-picked (relatable) features
            #topfeats = pd.Series(index=['curvature_midbody_abs_90th'])
            
            if n_sigfeats < n_top_features:
                print("Only %d significant features found for %s" % (n_sigfeats, strain))
            #print("\nTop %d features for %s:\n" % (len(topfeats), strain))
            #print(*[feat + '\n' for feat in list(topfeats.index)])
    
            # Subset feature summary results for test-strain + OP50-control only
            plot_df = fullresults[np.logical_or(fullresults[grouping_var]==control_strain,\
                                                fullresults[grouping_var]==strain)] 
        
            # Colour/legend dictionary
            labels = list(plot_df[grouping_var].unique())
            colour_dict = {strain:'#C2FDBE', control_strain:'#0C9518'}
                                                  
            # Boxplots of OP50 vs test-strain for each top-ranked significant feature
            for f, feature in enumerate(topfeats.index):
                plt.close('all')
                sns.set_style('darkgrid')
                fig = plt.figure(figsize=[10,8])
                ax = fig.add_subplot(1,1,1)
                sns.boxplot(x=grouping_var, y=feature, data=plot_df,
                            palette=colour_dict, showfliers=False, showmeans=True,
                            meanprops={"marker":"x", "markersize":5, "markeredgecolor":"k"},
                            flierprops={"marker":"x", "markersize":15, "markeredgecolor":"r"})
                sns.swarmplot(x=grouping_var, y=feature, data=plot_df, s=10, 
                              marker=".", color='k')
                ax.set_xlabel('Strain', fontsize=15, labelpad=12)
                ax.set_ylabel(feature, fontsize=15, labelpad=12)
                ax.set_title(feature, fontsize=20, pad=40)
    
                # Add plot legend
                patch_list = []
                for l, key in enumerate(colour_dict.keys()):
                    patch = patches.Patch(color=colour_dict[key], label=key)
                    patch_list.append(patch)
                    if key == strain:
                        ylim = plot_df[plot_df[grouping_var]==key][feature].max()
                        pval = test_pvalues_df.loc[key, feature]
                        if isinstance(pval, float) and pval < p_value_threshold:
                            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                            ax.text(l - 0.1, 1, 'p={:g}'.format(float('{:.2g}'.format(pval))),\
                            fontsize=13, color='k', verticalalignment='bottom', transform=trans)
                plt.tight_layout(rect=[0.04, 0, 0.84, 0.96])
                plt.legend(handles=patch_list, labels=colour_dict.keys(), 
                           loc=(1.02, 0.8), borderaxespad=0.4, frameon=False, fontsize=15)
    
                # Save figure
                plots_outpath = project_root_dir / 'Results' / 'Plots' / strain /\
                                ('{0}_'.format(f + 1) + feature + '.eps')
                plots_outpath.parent.mkdir(exist_ok=True, parents=True)
                print("[%d] Saving figure: %s" % (i, plots_outpath.name))
                plt.savefig(plots_outpath, format='eps', dpi=300)
                if show_plots:
                    plt.show(); plt.pause(2)
                plt.close(fig) # Close plot


#     control_path = project_root_dir / 'Results' / 'Control' / 'control_results.csv'
#     check_feature_normality = True # Check if features obey Gaussian normality?
#     is_normal_threshold = 0.95 # Threshold normal features for parametric stats
#     max_sigdiff_strains_plot_cap = 60
#     # tSNE: Perplexity is similar to n-nearest neighbours, eg. expected cluster size

