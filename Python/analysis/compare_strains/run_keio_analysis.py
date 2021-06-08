#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse Keio Screen results

Please run the following scripts beforehand:
1. preprocessing/compile_keio_results.py
2. statistical_testing/perform_keio_stats.py

@author: sm5911
@date: 19/04/2021
"""

#%% IMPORTS

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from time import time
from matplotlib import pyplot as plt
from scipy.stats import zscore # levene, ttest_ind, f_oneway, kruskal
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from tierpsytools.analysis.significant_features import mRMR_feature_selection

from read_data.paths import get_save_dir
from read_data.read import load_json, load_topfeats
from analysis.control_variation import control_variation
# from write_data.write import write_list_to_file
from filter_data.clean_feature_summaries import clean_summary_results, subset_results
from statistical_testing.stats_helper import levene_f_test #shapiro_normality_test
from statistical_testing.perform_keio_stats import average_control_keio
from feature_extraction.decomposition.pca import plot_pca, remove_outliers_pca
from feature_extraction.decomposition.tsne import plot_tSNE
from feature_extraction.decomposition.umap import plot_umap
from feature_extraction.decomposition.hierarchical_clustering import plot_clustermap, plot_barcode_heatmap
from visualisation.super_plots import superplot
from visualisation.plotting_helper import boxplots_grouped, errorbar_sigfeats #boxplots_sigfeats, sig_asterix, barplot_sigfeats, plot_day_variation, 

#%% GLOBALS

JSON_PARAMETERS_PATH = "analysis/20210406_parameters_keio_screen.json"
FEATURES_PATH = "/Users/sm5911/Documents/Keio_Screen/features.csv"
METADATA_PATH = "/Users/sm5911/Documents/Keio_Screen/metadata.csv"

#%% FUNCTIONS

def compare_strains_keio(features, metadata, args):

    assert set(features.index) == set(metadata.index)

    # categorical variable to investigate, eg.'gene_name'
    GROUPING_VAR = args.grouping_variable
    assert len(metadata[GROUPING_VAR].unique()) == len(metadata[GROUPING_VAR].str.upper().unique())
    print("\nInvestigating '%s' variation" % GROUPING_VAR)    
    
    # Subset results (rows) to omit selected strains
    if args.omit_strains is not None:
        features, metadata = subset_results(features, 
                                            metadata, 
                                            column=GROUPING_VAR, 
                                            groups=args.omit_strains, 
                                            omit=True)

    STRAIN_LIST = list(metadata[GROUPING_VAR].unique())    
    CONTROL = args.control_dict[GROUPING_VAR] # control strain to use

    # Load Tierpsy Top feature set + subset (columns) for top feats only
    if args.n_top_feats is not None:
        top_feats_path = Path(args.tierpsy_top_feats_dir) / "tierpsy_{}.csv".format(str(args.n_top_feats))        
        topfeats = load_topfeats(top_feats_path, add_bluelight=True, 
                                 remove_path_curvature=True, header=None)

        # Drop features that are not in results
        top_feats_list = [feat for feat in list(topfeats) if feat in features.columns]
        features = features[top_feats_list]

    SAVE_DIR = get_save_dir(args)

    ##### Control variation #####
                
    # Subset results for control data
    control_metadata = metadata[metadata['source_plate_id']=='BW']
    control_features = features.reindex(control_metadata.index)
        
    if args.analyse_control:
        # Clean data after subset - to remove features with zero std
        control_feat_clean, control_meta_clean = clean_summary_results(control_features, 
                                                                       control_metadata, 
                                                                       max_value_cap=False,
                                                                       imputeNaN=False)
        control_variation(control_feat_clean, 
                          control_meta_clean, 
                          args,
                          variables=['date_yyyymmdd','instrument_name','imaging_run_number'])
    
    ##### STATISTICS #####

    stats_dir =  SAVE_DIR / GROUPING_VAR / "Stats"
    stats_path = stats_dir / '{}_results.csv'.format(args.test) # LMM/ANOVA/Kruskal  
        
    ### F-tests to compare variance in samples with control (and correct for multiple comparisons)
    # Sample size matters in that unequal variances don't pose a problem for a t-test with 
    # equal sample sizes. So as long as your sample sizes are equal, you don't have to worry about 
    # homogeneity of variances. If they are not equal, perform F-tests first to see if variance is 
    # equal before doing a t-test
    if args.f_test:
        # F-test for equal variances
        levene_stats_path = stats_dir / 'levene_results.csv'
        levene_stats = levene_f_test(features, 
                                     metadata,
                                     grouping_var=GROUPING_VAR, 
                                     p_value_threshold=args.pval_threshold, 
                                     multitest_method=args.fdr_method,
                                     saveto=levene_stats_path,
                                     del_if_exists=False)
    
        # if p < 0.05 then variances are not equal, and sample size matters
        prop_eqvar = (levene_stats['pval'] > args.pval_threshold).sum() / len(levene_stats['pval'])
        print("Percentage equal variance %.1f%%" % (prop_eqvar * 100))
    
    if args.collapse_control:
        print("Collapsing control data (mean of each day)")
        features, metadata = average_control_keio(features, metadata)
                            
    # Record mean sample size per group
    mean_sample_size = int(np.round(metadata.join(features).groupby([GROUPING_VAR], 
                                                                    as_index=False).size().mean()))
    print("Mean sample size: %d" % mean_sample_size)
    
    ### Load statistics results

    # Read ANOVA results and record significant features
    print("Loading statistics results")
    if len(STRAIN_LIST) > 2:
        anova_table = pd.read_csv(stats_path, index_col=0)
        pvals = anova_table.sort_values(by='pvals', ascending=True)['pvals'] # rank features by p-value
        fset = pvals[pvals < args.pval_threshold].index.to_list()
        print("%d significant features found by %s (P<%.2f)" % (len(fset), args.test, args.pval_threshold))
            
    # Compare k sigfeat and ANOVA (or t-test) significant feature set overlap
    k_sigfeats_path = stats_dir / 'k_significant_features.csv'
    ksig_table = pd.read_csv(k_sigfeats_path, index_col=0)
    fset_ksig = ksig_table[ksig_table['pvals'] < args.pval_threshold].index.to_list()
    if len(fset) > 0:
        fset_overlap = set(fset).intersection(set(fset_ksig))
        prop_overlap = len(fset_overlap) / len(fset)
        print("%.1f%% overlap with k-significant features" % (prop_overlap * 100))
        if prop_overlap < 0.5 and len(fset) > 100:
            print("WARNING: Inconsistency in statistics for feature set agreement between "
                  + "%s and k significant features!" % args.test) 
        if args.use_k_sig_feats_overlap:
            fset = list(ksig_table.loc[fset_overlap].sort_values(by='pvals', 
                                                                 ascending=True).index)
    else:
        print("No significant features found for %s by %s" % (GROUPING_VAR, args.test))

    t_test = 't-test' if args.test == 'ANOVA' else 'Mann-Whitney' # aka. Wilcoxon rank-sum
    ttest_path = stats_dir / '{}_results.csv'.format(t_test) #t-test/Mann-Whitney

    # Read t-test results and record significant features (NOT ORDERED)
    ttest_table = pd.read_csv(ttest_path, index_col=0)
    pvals_t = ttest_table[[c for c in ttest_table if "pvals_" in c]] 
    pvals_t.columns = [c.split('pvals_')[-1] for c in pvals_t.columns]       
    fset_ttest = pvals_t[(pvals_t < args.pval_threshold).sum(axis=1) > 0].index.to_list()
    print("\n%d significant features found by %s (P<%.2f)" % (len(fset_ttest), t_test, args.pval_threshold))

    ##### mRMR feature selection: minimum Redunduncy, Maximum Relevance #####

    plot_dir = SAVE_DIR / GROUPING_VAR / "Plots"    
    
    mrmr_dir = plot_dir / 'mrmr'
    mrmr_dir.mkdir(exist_ok=True, parents=True)
    mrmr_results_path = mrmr_dir / "mrmr_results.csv"

    if not mrmr_results_path.exists():
        estimator = Pipeline([('scaler', StandardScaler()), ('estimator', LogisticRegression())])
        y = metadata[GROUPING_VAR].values
        (mrmr_feat_set, 
         mrmr_scores, 
         mrmr_support) = mRMR_feature_selection(features, y_class=y, k=10,
                                                redundancy_func='pearson_corr', 
                                                relevance_func='kruskal',
                                                n_bins=10, mrmr_criterion='MID',
                                                plot=True, k_to_plot=5, 
                                                close_after_plotting=True,
                                                saveto=mrmr_dir, figsize=None)
        # save results                                        
        mrmr_table = pd.concat([pd.Series(mrmr_feat_set), pd.Series(mrmr_scores)], axis=1)
        mrmr_table.columns = ['feature','score']
        mrmr_table.to_csv(mrmr_results_path, header=True, index=False)
        
        n_cv = 5
        cv_scores_mrmr = cross_val_score(estimator, features[mrmr_feat_set], y, cv=n_cv)
        cv_scores_mrmr = pd.DataFrame(cv_scores_mrmr, columns=['cv_score'])
        cv_scores_mrmr.to_csv(mrmr_dir / "cv_scores.csv", header=True, index=False)        
        print('\nMRMR CV Score: %f (n=%d)' % (np.mean(cv_scores_mrmr), n_cv))        
    else:
        # load mrmr results
        mrmr_table = pd.read_csv(mrmr_results_path)
        
    mrmr_feat_set = mrmr_table['feature'].to_list()

    ##### PLOTTING #####

    if len(fset) > 0:
        print("\nMaking errorbar plots")
        errorbar_sigfeats(features, metadata, 
                          group_by=GROUPING_VAR, 
                          fset=fset, 
                          control=CONTROL, 
                          max_feats2plt=args.n_sig_features, 
                          figsize=[130,6], 
                          fontsize=2, 
                          saveDir=(plot_dir / 'errorbar'))
        
        print("\nMaking boxplots")
        boxplots_grouped(feat_meta_df=metadata.join(features), 
                         group_by=GROUPING_VAR,
                         control_group=CONTROL,
                         test_pvalues_df=pvals_t.T, # ranked by test pvalue significance
                         feature_set=fset,
                         max_feats2plt=args.n_sig_features, 
                         max_groups_plot_cap=None,
                         p_value_threshold=args.pval_threshold,
                         drop_insignificant=False,
                         sns_colour_palette="tab10",
                         figsize=[6,130], 
                         saveDir=(plot_dir / 'boxplots'))

    # Rank strains by number of sigfeats by t-test 
    ranked_nsig = (pvals_t < args.pval_threshold).sum(axis=0).sort_values(ascending=False)
    
    # Select top 100 hit strains by n sigfeats
    # TODO: Choose strains with > 5 sigfeats as hit strains?
    hit_strains_nsig = ranked_nsig[ranked_nsig > 0].index.to_list()
    print("%d significant strains (ie. with 1 or more sigfeats) found by t-test" % len(hit_strains_nsig))
    #hit_nuo = ranked_nsig[[i for i in ranked_nsig[ranked_nsig > 0].index if 'nuo' in i]]

    print("Plotting ranked strains by number of significant features")
    plt.ioff()
    fig, ax = plt.subplots()
    ax.plot(ranked_nsig)
    plt.xticks(rotation=90, fontsize=3)
    plt.savefig(plot_dir / 'ranked_number_sigfeats.png', dpi=600)
    plt.close()
    
    # Rank strains by lowest p-value for any feature
    ranked_pval = pvals_t.min(axis=0).sort_values(ascending=True)
    
    # Select top 100 hit strains by lowest p-value for any feature
    hit_strains_pval = ranked_pval[ranked_pval < args.pval_threshold].index.to_list()

    print("Plotting ranked strains by lowest p-value of any feature")
    plt.ioff()
    fig, ax = plt.subplots()
    ax.plot(ranked_pval)
    plt.xticks(rotation=45, fontsize=3)
    plt.savefig(plot_dir / 'ranked_lowest_pval.png', dpi=600)
    plt.close()
    
    max_n_hits = 100
       
    if len(hit_strains_nsig) > 0:
        hit_strains_nsig = hit_strains_nsig[:max_n_hits]
        hit_strains_nsig.insert(0, CONTROL)
         
        # boxplots for all sigfeats for these strains - for all strains and paired
        
        # subset for top strains only
        hit_features, hit_metadata = subset_results(features, 
                                                    metadata, 
                                                    column=GROUPING_VAR,
                                                    groups=hit_strains_nsig)
            
        # from tierpsytools.analysis.significant_features import plot_feature_boxplots
        # for f in fset:
        #     plot_feature_boxplots(feat_to_plot=fset,
        #                           y_class=meta_plot[GROUPING_VAR],
        #                           scores=pvals_t.rank(axis=1),
        #                           pvalues=np.asarray(pvals_t).flatten(),
        #                           saveto=None,
        #                           close_after_plotting=True)

        # # Individual boxplots of significant features by pairwise t-test (each group vs control)
        # boxplots_sigfeats(feat_meta_df=metadata.join(features), 
        #                   test_pvalues_df=pvals_t.T, 
        #                   group_by=GROUPING_VAR, 
        #                   control_strain=CONTROL, 
        #                   feature_set=fset, #['speed_norm_50th_bluelight'],
        #                   saveDir=plot_dir / 'paired_boxplots',
        #                   max_feats2plt=args.n_sig_features,
        #                   p_value_threshold=args.pval_threshold,
        #                   drop_insignificant=True,
        #                   verbose=False)
        
        superplot_dir = plot_dir / 'superplots'    
        for feat in fset[:args.n_sig_features]:
            # plot variation with respect to 'date_yyyymmdd'
            superplot(hit_features, hit_metadata, feat, 
                      x1=GROUPING_VAR, 
                      x2='date_yyyymmdd',
                      saveDir=superplot_dir,
                      show_points=True, 
                      plot_means=True,
                      dodge=False)
            # TODO: Add t-test/LMM pvalues to superplots       

            # # plot variation with respect to 'instrument_name'
            # superplot(features, metadata, feat, 
            #           x1=GROUPING_VAR, 
            #           x2='instrument_name',
            #           saveDir=superplot_dir,
            #           show_points=True, 
            #           plot_means=True,
            #           dodge=True)
            # # plot variation with respect to 'imaging_run_number'
            # superplot(features, metadata, feat, 
            #           x1=GROUPING_VAR, 
            #           x2='imaging_run_number',
            #           saveDir=superplot_dir,
            #           show_points=True, 
            #           plot_means=True,
            #           dodge=True)
    
    ##### Hierarchical Clustering Analysis #####
    
    # Z-normalise data
    featZ = features.apply(zscore, axis=0)
    #featZ = (features-features.mean())/features.std() # minus mean, divide by std
    
    #from tierpsytools.preprocessing.scaling_class import scalingClass
    #scaler = scalingClass(scaling='standardize')
    #featZ = scaler.fit_transform(features)

    # Drop features with NaN values after normalising
    n_cols = len(featZ.columns)
    featZ.dropna(axis=1, inplace=True)
    n_dropped = n_cols - len(featZ.columns)
    if n_dropped > 0:
        print("Dropped %d features after normalisation (NaN)" % n_dropped)

    ### Control clustermap
    
    # control data is clustered and feature order is stored and applied to full data
    print("\nPlotting control clustermap")
    control_clustermap_path = plot_dir / 'heatmaps' / (GROUPING_VAR + '_clustermap.pdf')
    cg = plot_clustermap(featZ, metadata,
                         group_by=([GROUPING_VAR] if GROUPING_VAR == 'date_yyyymmdd' 
                                   else [GROUPING_VAR, 'date_yyyymmdd']),
                         method='complete',#[linkage, complete, average, weighted, centroid]
                         figsize=[18,6],
                         saveto=control_clustermap_path)

    #col_linkage = cg.dendrogram_col.calculated_linkage
    clustered_features = np.array(featZ.columns)[cg.dendrogram_col.reordered_ind]
                
    ## Save z-normalised values
    # z_stats = featZ.join(metadata[GROUPING_VAR]).groupby(by=GROUPING_VAR).mean().T
    # z_stats.columns = ['z-mean_' + v for v in z_stats.columns.to_list()]
    # z_stats.to_csv(z_stats_path, header=True, index=None)
    
    # Clustermap of full data   
    print("\nPlotting all strains clustermap")    
    full_clustermap_path = plot_dir / 'heatmaps' / (GROUPING_VAR + '_full_clustermap.pdf')
    fg = plot_clustermap(featZ, 
                         metadata, 
                         group_by=GROUPING_VAR,
                         method='complete',
                         figsize=[20, 30],
                         saveto=full_clustermap_path)
    
    # If no control clustering (due to no day variation) then use clustered features for all 
    # strains to order barcode heatmaps
    if clustered_features is None:
        clustered_features = np.array(featZ.columns)[fg.dendrogram_col.reordered_ind]
    
    pvals_heatmap = anova_table.loc[clustered_features, 'pvals']
    pvals_heatmap.name = 'P < {}'.format(args.pval_threshold)

    assert all(f in featZ.columns for f in pvals_heatmap.index)
            
    # # Plot barcode heatmap (grouping by date)
    # if len(metadata['date_yyyymmdd'].unique()) > 1:
    #     print("\nPlotting barcode heatmap by date")
    #     heatmap_date_path = plot_dir / 'heatmaps' / (GROUPING_VAR + '_date_heatmap.pdf')
    #     plot_barcode_heatmap(featZ=featZ[clustered_features], 
    #                          meta=metadata, 
    #                          group_by=[GROUPING_VAR, 'date_yyyymmdd'],
    #                          pvalues_series=pvals_heatmap,
    #                          p_value_threshold=args.pval_threshold,
    #                          selected_feats=fset if len(fset) > 0 else None,
    #                          saveto=heatmap_date_path,
    #                          figsize=[20, 30],
    #                          sns_colour_palette="Pastel1")
    
    # Plot group-mean heatmap (averaged across days)
    print("\nPlotting barcode heatmap")
    heatmap_path = plot_dir / 'heatmaps' / (GROUPING_VAR + '_heatmap.pdf')
    plot_barcode_heatmap(featZ=featZ[clustered_features], 
                          meta=metadata, 
                          group_by=[GROUPING_VAR], 
                          pvalues_series=pvals_heatmap,
                          p_value_threshold=args.pval_threshold,
                          selected_feats=fset if len(fset) > 0 else None,
                          saveto=heatmap_path,
                          figsize=[20, 30],
                          sns_colour_palette="Pastel1")        
                    
    ##### Principal Components Analysis #####

    pca_dir = plot_dir / 'PCA'

    print("\nPerforming PCA")
    if args.remove_outliers:
        outlier_path = pca_dir / 'mahalanobis_outliers.pdf'
        features, inds = remove_outliers_pca(df=features, saveto=outlier_path)
        metadata = metadata.reindex(features.index) # reindex metadata
        featZ = features.apply(zscore, axis=0) # re-normalise data

        # Drop features with NaN values after normalising
        n_cols = len(featZ.columns)
        featZ.dropna(axis=1, inplace=True)
        n_dropped = n_cols - len(featZ.columns)
        if n_dropped > 0:
            print("Dropped %d features after normalisation (NaN)" % n_dropped)


    # plot pca coloured by Keio COG category
    COG_families = {'Information storage and processing' : ['J', 'K', 'L', 'D', 'O'], 
                    'Cellular processes' : ['M', 'N', 'P', 'T', 'C', 'G', 'E'], 
                    'Metabolism' : ['F', 'H', 'I', 'Q', 'R'], 
                    'Poorly characterised' : ['S', 'U', 'V']}
    COG_mapping_dict = {i : k for (k, v) in COG_families.items() for i in v}
    
    COG_info = []
    for i in metadata['COG_category']:
        try:
            COG_info.append(COG_mapping_dict[i])
        except:
            COG_info.append('')
    metadata['COG_info'] = COG_info
    _ = plot_pca(featZ, metadata, 
                 group_by='COG_info', 
                 control=None,
                 var_subset=list(metadata['COG_info'].dropna().unique()), 
                 saveDir=pca_dir / 'COG',
                 PCs_to_keep=10,
                 n_feats2print=10,
                 kde=False,
                 sns_colour_palette="gist_rainbow",
                 n_dims=2,
                 hypercolor=False)

    #from tierpsytools.analysis.decomposition import plot_pca
    _ = plot_pca(featZ, metadata, 
                 group_by=GROUPING_VAR, 
                 control=CONTROL,
                 var_subset=None, 
                 saveDir=pca_dir,
                 PCs_to_keep=10,
                 n_feats2print=10,
                 kde=False,
                 sns_colour_palette="plasma",
                 n_dims=2,
                 hypercolor=False)
    # TODO: Ensure sns colour palette does not plot white points for PCA

    ##### t-distributed Stochastic Neighbour Embedding #####   
    
    print("\nPerforming tSNE")
    tsne_dir = plot_dir / 'tSNE'
    perplexities = [5, 15, 30, mean_sample_size] # NB: should be roughly equal to group size    
    _ = plot_tSNE(featZ, metadata,
                  group_by=GROUPING_VAR,
                  var_subset=None,
                  saveDir=tsne_dir,
                  perplexities=perplexities,
                  sns_colour_palette="plasma")
   
    ##### Uniform Manifold Projection #####  
    
    print("\nPerforming UMAP")
    umap_dir = plot_dir / 'UMAP'
    n_neighbours = [5, 15, 30, mean_sample_size] # NB: should be roughly equal to group size
    min_dist = 0.1 # Minimum distance parameter    
    _ = plot_umap(featZ, metadata,
                  group_by=GROUPING_VAR,
                  var_subset=None,
                  saveDir=umap_dir,
                  n_neighbours=n_neighbours,
                  min_dist=min_dist,
                  sns_colour_palette="plasma")    
    
#%% MAIN
if __name__ == "__main__":
    tic = time()
    parser = argparse.ArgumentParser(description="Read clean features and etadata and find 'hit' \
                                                  Keio knockout strains that alter worm behaviour")
    parser.add_argument('-j', '--json', help="Path to JSON parameters file", 
                        default=JSON_PARAMETERS_PATH, type=str)
    parser.add_argument('--features_path', help="Path to feature summaries file", 
                        default=FEATURES_PATH, type=str)
    parser.add_argument('--metadata_path', help="Path to metadata file", 
                        default=METADATA_PATH, type=str)
    args = parser.parse_args()
    
    args = load_json(args.json)
    
    # Read clean feature summaries + metadata
    print("Loading metadata and feature summary results...")
    features = pd.read_csv(FEATURES_PATH)
    metadata = pd.read_csv(METADATA_PATH, dtype={'comments':str, 'source_plate_id':str})
    
    compare_strains_keio(features, metadata, args)
    
    toc = time()
    print("\nDone in %.1f seconds (%.1f minutes)" % (toc - tic, (toc - tic) / 60))  

#%%

# # Scale the features (necessary if you want to do PCA)
# scaler = scalingClass(scaling='standardize')
# features = scaler.fit_transform(features)

# # Check day-to-day variation
# # Get the PCA decomposition
# pca = PCA(n_components=2)
# Y = pca.fit_transform(features)

# # Plot the samples of each day in the first two PCs
# plt.figure()
# for day in metadata['date_yyyymmdd'].unique():
#     plt.scatter(*Y[metadata['date_yyyymmdd']==day, :].T, label=day)
# plt.legend()

# # Hierarchical clutering
# # Get row colors and mathcing legend data
# labels = metadata['worm_strain']
# unique_labels = labels.unique()

# # lut aka. look-up table - {label ---> color}
# palette = sns.color_palette(n_colors=unique_labels.shape[0])
# lut = dict(zip(unique_labels, palette))

# # Convert the dictionary into a Series
# row_colors = pd.DataFrame(labels)['worm_strain'].map(lut)

# # metric: try ['euclidean', 'cosine', 'correlation']
# g = sns.clustermap(features, method='complete', metric='cosine', row_colors=row_colors)
