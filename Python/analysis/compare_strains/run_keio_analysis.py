#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse Keio Screen results

Please run the following scripts prior to this one:
1. preprocessing/compile_keio_results.py
2. statistical_testing/perform_keio_stats.py

@author: sm5911
@date: 19/04/2021
"""

#%% IMPORTS

import argparse
import numpy as np
import pandas as pd
from time import time
from pathlib import Path
from scipy.stats import zscore # ttest_ind, f_oneway, kruskal
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances_argmin_min

from read_data.paths import get_save_dir
from read_data.read import load_json, load_top256
from analysis.control_variation import control_variation
# from write_data.write import write_list_to_file
from filter_data.clean_feature_summaries import clean_summary_results, subset_results
# from statistical_testing.stats_helper import shapiro_normality_test
from feature_extraction.decomposition.pca import plot_pca, remove_outliers_pca
from feature_extraction.decomposition.tsne import plot_tSNE
from feature_extraction.decomposition.umap import plot_umap
from feature_extraction.decomposition.hierarchical_clustering import plot_clustermap, plot_barcode_heatmap
from visualisation.super_plots import superplot
from visualisation.plotting_helper import (boxplots_grouped) #sig_asterix, barplot_sigfeats, boxplots_sigfeats, plot_day_variation, 

#from tierpsytools.analysis.significant_features import k_significant_feat
#from tierpsytools.analysis.statistical_tests import univariate_tests
#from tierpsytools.drug_screenings.filter_compounds import compounds_with_low_effect_univariate

#%% GLOBALS

JSON_PARAMETERS_PATH = "analysis/20210406_parameters_keio_screen.json"
FEATURES_PATH = "/Users/sm5911/Documents/Keio_Screen/features.csv"
METADATA_PATH = "/Users/sm5911/Documents/Keio_Screen/metadata.csv"

#%% FUNCTIONS

def main(features, metadata, args):

    assert set(features.index) == set(metadata.index)

    GROUPING_VAR = args.grouping_variable # categorical variable to investigate, eg.'gene_name' / 'worm_strain'
    assert len(metadata[GROUPING_VAR].unique()) == len(metadata[GROUPING_VAR].str.upper().unique())
    print("\nInvestigating '%s' variation" % GROUPING_VAR)    
    
    # Subset results (rows) to omit selected strains
    if args.omit_strains is not None:
        features, metadata = subset_results(features, metadata, GROUPING_VAR, args.omit_strains)

    STRAIN_LIST = list(metadata[GROUPING_VAR].unique())    
    CONTROL = args.control_dict[GROUPING_VAR] # control strain to use

    # Load Tierpsy Top256 feature set + subset (columns) for Top256
    if args.use_top256:
        aux_dir = Path(args.project_dir) / "AuxiliaryFiles"
        top256_path = aux_dir / 'top256_tierpsy_no_blob_no_eigen_only_abs_no_norm.csv'
        top256 = load_top256(top256_path, add_bluelight=args.align_bluelight)
        
        # Ensure results exist for features in featurelist
        top256_feat_list = [feat for feat in top256 if feat in features.columns]
        print("Dropped %d features in Top256 that are missing from results" %\
              (len(top256)-len(top256_feat_list)))
        
        # Use Top256 features for analysis
        features = features[top256_feat_list]

    SAVE_DIR = get_save_dir(args)

    ##### Control variation #####
                
    # Subset results for control data
    control_metadata = metadata[metadata['source_plate_id']=='BW']
    control_features = features.reindex(control_metadata.index)
    
    # Clean data after subset - to remove features with zero std
    control_features, control_metadata = clean_summary_results(control_features, 
                                                               control_metadata, 
                                                               max_value_cap=False,
                                                               imputeNaN=False)
    # control_variation(control_features, 
    #                   control_metadata, 
    #                   args,
    #                   variables=['date_yyyymmdd','instrument_name','imaging_run_number'])
    
    ##### STATISTICS #####

    # Stats test to use
    assert args.test in ['ANOVA','Kruskal','LMM']
    if args.test == 'LMM':
        # If 'LMM' is chosen, ensure there are multiple day replicates to compare at each timepoint
        assert all(len(metadata.loc[metadata['imaging_run_number']==timepoint, 
                   args.lmm_random_effect].unique()) > 1 for timepoint in args.runs)

    t_test = 't-test' if args.test == 'ANOVA' else 'Mann-Whitney' # aka. Wilcoxon rank-sum

    stats_dir =  SAVE_DIR / GROUPING_VAR / "Stats"
    stats_path = stats_dir / '{}_results.csv'.format(args.test) # LMM/ANOVA/Kruskal  
    ttest_path = stats_dir / '{}_results.csv'.format(t_test) #t-test/Mann-Whitney
                            
    # Record mean sample size per group
    mean_sample_size = int(np.round(metadata.join(features).groupby([GROUPING_VAR], as_index=False).size().mean()))
    print("Mean sample size: %d" % mean_sample_size)
    
    ### Load statistics results

    # Read ANOVA results and record significant features
    print("Loading statistics results")        
    pval_stats = pd.read_csv(stats_path, index_col=0)
    pvals = pval_stats.sort_values(by=args.test, ascending=True)
    fset = pvals.index[pvals[args.test] < args.pval_threshold].to_list()
    print("%d significant features found by %s (P<%.2f, %s)" % (len(fset), args.test, 
                                                                args.pval_threshold, args.fdr_method))
    # Read t-test results and record significant features
    pvals_t = pd.read_csv(ttest_path, index_col=0)
    assert all(f in pvals_t.columns for f in features.columns)
    fset_ttest = list(pvals_t.columns[(pvals_t < args.pval_threshold).sum(axis=0) > 0])
    print("%d significant features found by %s (P<%.2f, %s)" % (len(fset_ttest), t_test, 
                                                                args.pval_threshold, args.fdr_method))
        
    ##### Plotting #####

    plot_dir = SAVE_DIR / GROUPING_VAR / "Plots"
    superplot_dir = plot_dir / 'superplots'    
    
    # if len(fset) > 1:        
    #     for feat in fset[:args.n_sig_features]:
            # plot variation with respect to 'date_yyyymmdd'
            # superplot(features, metadata, feat, 
            #           x1=GROUPING_VAR, 
            #           x2='date_yyyymmdd',
            #           saveDir=superplot_dir,
            #           show_points=True, 
            #           plot_means=True,
            #           dodge=True)
            # # plot variation with respect to 'instrument_name'
            # superplot(features, metadata, feat, 
            #           x1=GROUPING_VAR, 
            #           x2='instrument_name',
            #           saveDir=superplot_dir,
            #           show_points=True, 
            #           plot_means=True,
            #           dodge=True)
            # # plot variation with respect to 'imaging_ruun_number'
            # superplot(features, metadata, feat, 
            #           x1=GROUPING_VAR, 
            #           x2='imaging_run_number',
            #           saveDir=superplot_dir,
            #           show_points=True, 
            #           plot_means=True,
            #           dodge=True)
            
            # TODO: Add t-test/LMM pvalues to superplots       

            # superplot(features, metadata, feat, x1='source_plate_id', x2='date_yyyymmdd', 
            #           saveDir=superplot_dir, show_points=False, plot_means=False)
        
            # superplot(features, metadata, feat, x1='source_plate_id', x2='imaging_run_number', 
            #           saveDir=superplot_dir, show_points=False, plot_means=False)
            
            # superplot(features, metadata, feat, x1='instrument_name', x2='imaging_run_number', 
            #           saveDir=superplot_dir, show_points=False, plot_means=False)
                
            # # strain vs date yyyymmdd
            # superplot(features, metadata, feat, 
            #           x1=GROUPING_VAR, x2='date_yyyymmdd',
            #           plot_type='box', #show_points=True, sns_colour_palettes=["plasma","viridis"]
            #           dodge=True, saveDir=superplot_dir)
    
            # # plate ID vs run number
            # superplot(features, metadata, feat, 
            #           x1=GROUPING_VAR, x2='imaging_run_number',
            #           dodge=True, saveDir=superplot_dir)

    
            
# =============================================================================
#     ### mRMR feature selection: minimum Redunduncy, Maximum Relevance
# 
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.pipeline import Pipeline
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.model_selection import cross_val_score
#     from tierpsytools.analysis.significant_features import mRMR_feature_selection
# 
#     estimator = Pipeline([('scaler', StandardScaler()), ('estimator', LogisticRegression())])
#     y = metadata[GROUPING_VAR].values
#     data = metadata.drop(columns=GROUPING_VAR).join(features)
#     
#     (mrmr_feat_set, 
#      mrmr_scores, 
#      mrmr_support) = mRMR_feature_selection(data, k=10, y_class=y,
#                                             redundancy_func='pearson_corr', 
#                                             relevance_func='kruskal',
#                                             n_bins=4, mrmr_criterion='MID',
#                                             plot=True, k_to_plot=5, 
#                                             close_after_plotting=False,
#                                             saveto=None, figsize=None)
#     
#     cv_scores_mrmr = cross_val_score(estimator, data[mrmr_feat_set], y, cv=5)
#     print('MRMR')
#     print(np.mean(cv_scores_mrmr))
# =============================================================================    
                                                           
        # # Boxplots of significant features by ANOVA/LMM (all groups)
        # boxplots_grouped(feat_meta_df=metadata.join(features), 
        #                  group_by=GROUPING_VAR,
        #                  control_group=CONTROL,
        #                  test_pvalues_df=pvals_t.T, # ranked by test pvalue significance
        #                  feature_set=fset,
        #                  saveDir=(plot_dir / 'grouped_boxplots'),
        #                  max_features_plot_cap=args.n_sig_features, 
        #                  max_groups_plot_cap=None,
        #                  p_value_threshold=args.pval_threshold,
        #                  drop_insignificant=False,
        #                  sns_colour_palette="tab10",
        #                  figsize=[6, (len(STRAIN_LIST) / 3 if len(STRAIN_LIST) > 10 else 12)])
                
        # Individual boxplots of significant features by pairwise t-test (each group vs control)
        # boxplots_sigfeats(feat_meta_df=metadata.join(features), 
        #                   test_pvalues_df=pvals_t, 
        #                   group_by=GROUPING_VAR, 
        #                   control_strain=control_group, 
        #                   feature_set=fset, #['speed_norm_50th_bluelight'],
        #                   saveDir=plot_dir / 'paired_boxplots',
        #                   max_features_plot_cap=args.n_sig_features,
        #                   p_value_threshold=args.pval_threshold,
        #                   drop_insignificant=True,
        #                   verbose=False)
            
        # from tierpsytools.analysis.significant_features import plot_feature_boxplots
        # plot_feature_boxplots(feat_to_plot=fset,
        #                       y_class=GROUPING_VAR,
        #                       scores=pvalues.rank(axis=1),
        #                       feat=features,
        #                       pvalues=np.asarray(pvalues).flatten(),
        #                       saveto=None,
        #                       close_after_plotting=False)
    
    ##### Hierarchical Clustering Analysis #####

    assert not features.isna().sum(axis=1).any()
    assert not (features.std(axis=1) == 0).any()
    
    # Z-normalise data
    featZ = features.apply(zscore, axis=1)
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

    ### Cluster analysis
    
    column_linkage = linkage(featZ.T, method='complete', metric='correlation')
    n_clusters = len(STRAIN_LIST)
    clusters = fcluster(column_linkage, n_clusters, criterion='maxclust')
    un,n = np.unique(clusters, return_counts=True)
    cluster_centres = (featZ.T).groupby(by=clusters).mean() # get cluster centres
    
    # get index of closest feature to cluster centre
    central, _ = pairwise_distances_argmin_min(cluster_centres, featZ.T, metric='cosine')
    assert(np.unique(central).shape[0] == n_clusters)
    central = featZ.columns.to_numpy()[central]

    # make cluster dataframe
    df = pd.DataFrame(index=featZ.columns, columns=['group_label', 'stat_label', 'motion_label'])
    df['group_label'] = clusters
    stats = np.array(['10th', '50th', '90th', 'IQR'])
    df['stat_label'] = [np.unique([x for x in stats if x in ft]) for ft in df.index]
    df['stat_label'] = df['stat_label'].apply(lambda x: x[0] if len(x)==1 else np.nan)
    motions = np.array(['forward', 'backward', 'paused'])
    df['motion_label'] = [[x for x in motions if x in ft] for ft in df.index]
    df['motion_label'] = df['motion_label'].apply(lambda x: x[0] if len(x)==1 else np.nan)
    df['representative_feature'] = False
    df.loc[central, 'representative_feature'] = True
    df = df.fillna('none')
    df.to_csv(stats_dir / 'feature_clusters.csv', index=True) # save cluster df to file
    
# =============================================================================
#     ### Fingerprints
#     from tierpsytools.analysis.fingerprints import tierpsy_fingerprints
#     
#     clusters = pd.read_csv(stats_dir / 'feature_clusters.csv', index_col=0)
#     fingers = {}
#     for group in group_list:
#         if group == control_group:
#             continue
#         print('Getting fingerprint of %s: %s' % (GROUPING_VAR, group))
#         (stats_dir / group).mkdir(exist_ok=True)
#         mask = metadata[GROUPING_VAR].isin([control_group, group])
#         finger = tierpsy_fingerprints(bluelight=True, 
#                                       test='Mann-Whitney', 
#                                       multitest_method='fdr_by',
#                                       significance_threshold=0.05, 
#                                       groups=clusters, 
#                                       groupby=['group_label'],
#                                       test_results=None, 
#                                       representative_feat='representative_feature')
#     
#         # Fit the fingerprint (run univariate tests and create the profile)
#         finger.fit(features[mask], metadata.loc[mask, GROUPING_VAR], control=control_group)
#         
#         # Plot and save the fingerprint for this strain
#         finger.plot_fingerprints(merge_bluelight=False, feature_names_as_xticks=True,
#                                   saveto=(stats_dir / group /'fingerprint.png'))
# 
#         # Plot and save boxplots for all the representative features
#         finger.plot_boxplots(features[mask], metadata.loc[mask, GROUPING_VAR], stats_dir / group, control=control_group)
#         
#         fingers[group] = finger # Store the fingerprint object
# =============================================================================

    ### Control clustermap
    
    # control data is clustered and feature order is stored and applied to full data
    if len(STRAIN_LIST) > 1:
        control_clustermap_path = plot_dir / 'HCA' / (GROUPING_VAR + '_clustermap.pdf')
        cg = plot_clustermap(featZ, metadata,
                             group_by=([GROUPING_VAR] if GROUPING_VAR == 'date_yyyymmdd' 
                                       else [GROUPING_VAR, 'date_yyyymmdd']),
                             col_linkage=None,
                             method='complete',#[linkage, complete, average, weighted, centroid]
                             figsize=[18,6],
                             saveto=control_clustermap_path)

        col_linkage = cg.dendrogram_col.calculated_linkage
        clustered_features = np.array(featZ.columns)[cg.dendrogram_col.reordered_ind]
    else:
        clustered_features = None
                
    ## Save z-normalised values
    # z_stats = featZ.join(metadata[GROUPING_VAR]).groupby(by=GROUPING_VAR).mean().T
    # z_stats.columns = ['z-mean_' + v for v in z_stats.columns.to_list()]
    # z_stats.to_csv(z_stats_path, header=True, index=None)
    
    # Clustermap of full data       
    full_clustermap_path = plot_dir / 'HCA' / (GROUPING_VAR + '_full_clustermap.pdf')
    fg = plot_clustermap(featZ, metadata, 
                         group_by=GROUPING_VAR,
                         col_linkage=None,
                         method='complete',
                         figsize=[20, (len(STRAIN_LIST) / 4 if len(STRAIN_LIST) > 10 else 6)],
                         saveto=full_clustermap_path)
    
    # If no control clustering (due to no day variation) then use clustered features for all 
    # strains to order barcode heatmaps
    if clustered_features is None:
        clustered_features = np.array(featZ.columns)[fg.dendrogram_col.reordered_ind]
    
    if len(STRAIN_LIST) > 2:
        #pvals_heatmap = anova_table.loc[clustered_features, 'pvals']
        pvals_heatmap = pval_stats.loc[clustered_features, args.test]
    elif len(STRAIN_LIST) == 2:
        #pvals_heatmap = pvals_t.loc[clustered_features, pvals_t.columns[0]]
        pvals_heatmap = pvals_t.loc[pvals_t.index[0], clustered_features]
    pvals_heatmap.name = 'P < {}'.format(args.pval_threshold)

    assert all(f in featZ.columns for f in pvals_heatmap.index)
            
    # Plot barcode heatmap (grouping by date)
    if len(STRAIN_LIST) > 1:
        heatmap_date_path = plot_dir / 'HCA' / (GROUPING_VAR + '_date_heatmap.pdf')
        plot_barcode_heatmap(featZ=featZ[clustered_features], 
                             meta=metadata, 
                             group_by=[GROUPING_VAR, 'date_yyyymmdd'],
                             pvalues_series=pvals_heatmap,
                             p_value_threshold=args.pval_threshold,
                             selected_feats=fset if len(fset) > 0 else None,
                             saveto=heatmap_date_path,
                             figsize=[20, (len(STRAIN_LIST) / 4 if len(STRAIN_LIST) > 10 else 6)],
                             sns_colour_palette="Pastel1")
    
    # Plot group-mean heatmap (averaged across days)
    heatmap_path = plot_dir / 'HCA' / (GROUPING_VAR + '_heatmap.pdf')
    plot_barcode_heatmap(featZ=featZ[clustered_features], 
                         meta=metadata, 
                         group_by=[GROUPING_VAR], 
                         pvalues_series=pvals_heatmap,
                         p_value_threshold=args.pval_threshold,
                         selected_feats=fset if len(fset) > 0 else None,
                         saveto=heatmap_path,
                         figsize=[20, (len(STRAIN_LIST) / 4 if len(STRAIN_LIST) > 10 else 6)],
                         sns_colour_palette="Pastel1")        
                    
    ##### Principal Components Analysis #####

    if args.remove_outliers:
        outlier_path = plot_dir / 'mahalanobis_outliers.pdf'
        features, inds = remove_outliers_pca(df=features, 
                                         features_to_analyse=None, 
                                         saveto=outlier_path)
        metadata = metadata.reindex(features.index) # reindex metadata
        featZ = features.apply(zscore, axis=1) # re-normalise data

        # Drop features with NaN values after normalising
        n_cols = len(featZ.columns)
        featZ.dropna(axis=1, inplace=True)
        n_dropped = n_cols - len(featZ.columns)
        if n_dropped > 0:
            print("Dropped %d features after normalisation (NaN)" % n_dropped)

    #from tierpsytools.analysis.decomposition import plot_pca
    pca_dir = plot_dir / 'PCA'
    projected_df = plot_pca(featZ, metadata, 
                            group_by=GROUPING_VAR, 
                            control=CONTROL,
                            var_subset=None, 
                            saveDir=pca_dir,
                            PCs_to_keep=10,
                            n_feats2print=10,
                            sns_colour_palette="plasma",
                            n_dims=2,
                            hypercolor=False)
    # TODO: Ensure sns colour palette does not plot white points for PCA

    ##### t-distributed Stochastic Neighbour Embedding #####   
    
    tsne_dir = plot_dir / 'tSNE'
    perplexities = [5, 15, 30, mean_sample_size] # NB: should be roughly equal to group size
    
    tSNE_df = plot_tSNE(featZ, metadata,
                        group_by=GROUPING_VAR,
                        var_subset=None,
                        saveDir=tsne_dir,
                        perplexities=perplexities,
                        sns_colour_palette="plasma")
   
    ##### Uniform Manifold Projection #####  
    
    umap_dir = plot_dir / 'UMAP'
    n_neighbours = [5, 15, 30, mean_sample_size] # NB: should be roughly equal to group size
    min_dist = 0.1 # Minimum distance parameter
    
    umap_df = plot_umap(featZ, metadata,
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
    features = pd.read_csv(FEATURES_PATH)
    metadata = pd.read_csv(METADATA_PATH, dtype={'comments':str, 'source_plate_id':str})
    
    main(features, metadata, args)
    
    toc = time()
    print("\nDone in %.1f seconds (%.1f minutes)" % (toc-tic, (toc-tic)/60))  

#%%

# # Scale the features (necessary if tou want to do PCA)
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
# # Get row colors that show MOA and mathcing legend data
# labels = metadata['worm_strain']
# unique_labels = labels.unique()

# # lut aka. look-up table - {label ---> color}
# palette = sns.color_palette(n_colors=unique_labels.shape[0])
# lut = dict(zip(unique_labels, palette))

# # Convert the dictionary into a Series
# row_colors = pd.DataFrame(labels)['worm_strain'].map(lut)

# # metric: try ['euclidean', 'cosine', 'correlation']
# g = sns.clustermap(features, method='complete', metric='cosine', row_colors=row_colors)
