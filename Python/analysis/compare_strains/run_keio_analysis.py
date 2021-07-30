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
from write_data.write import write_list_to_file
from analysis.control_variation import control_variation
# from write_data.write import write_list_to_file
from filter_data.clean_feature_summaries import clean_summary_results, subset_results
from statistical_testing.stats_helper import levene_f_test # shapiro_normality_test
from statistical_testing.perform_keio_stats import average_control_keio
from clustering.hierarchical_clustering import plot_clustermap # plot_barcode_heatmap
from feature_extraction.decomposition.pca import plot_pca, remove_outliers_pca
from feature_extraction.decomposition.tsne import plot_tSNE
from feature_extraction.decomposition.umap import plot_umap
from visualisation.plotting_helper import errorbar_sigfeats, boxplots_sigfeats # boxplots_grouped, barplot_sigfeats, plot_day_variations 

#%% GLOBALS

JSON_PARAMETERS_PATH = "analysis/20210719_parameters_keio_screen.json"

MAX_N_HITS = 100 

#%% FUNCTIONS

def compare_strains_keio(features, metadata, args):
    """ Compare Keio single-gene deletion mutants with wild-type BW25113 control and look to see if 
        they signfiicantly alter N2 C. elegans behaviour while feeding.
        
        Subset results to omit selected strains (optional) 
        Inputs
        ------
        features, metadata : pd.DataFrame
            Matching features summaries and metadata
        
        args : Object 
            Python object with the following attributes:
            - drop_size_features : bool
            - norm_features_only : bool
            - percentile_to_use : str
            - remove_outliers : bool
            - omit_strains : list
            - grouping_variable : str
            - control_dict : dict
            - collapse_control : bool
            - n_top_feats : int
            - tierpsy_top_feats_dir (if n_top_feats) : str
            - test : str
            - f_test : bool
            - pval_threshold : float
            - fdr_method : str
            - n_sig_features : int
    """

    assert set(features.index) == set(metadata.index)

    # categorical variable to investigate, eg.'gene_name'
    grouping_var = args.grouping_variable
    assert len(metadata[grouping_var].unique()) == len(metadata[grouping_var].str.upper().unique())
    print("\nInvestigating '%s' variation" % grouping_var)    
    
    # Subset results (rows) to omit selected strains
    if args.omit_strains is not None:
        features, metadata = subset_results(features, 
                                            metadata, 
                                            column=grouping_var, 
                                            groups=args.omit_strains, 
                                            omit=True)

    control = args.control_dict[grouping_var] # control strain to use

    # Load Tierpsy Top feature set + subset (columns) for top feats only
    if args.n_top_feats is not None:
        top_feats_path = Path(args.tierpsy_top_feats_dir) / "tierpsy_{}.csv".format(str(args.n_top_feats))        
        topfeats = load_topfeats(top_feats_path, add_bluelight=True, 
                                 remove_path_curvature=True, header=None)

        # Drop features that are not in results
        top_feats_list = [feat for feat in list(topfeats) if feat in features.columns]
        features = features[top_feats_list]

    save_dir = get_save_dir(args)
    stats_dir =  save_dir / grouping_var / "Stats_{}".format(args.fdr_method)
    plot_dir = save_dir / grouping_var / "Plots_{}".format(args.fdr_method)

# =============================================================================
#     ##### Pairplot Tierpsy 16 #####
#     if args.n_top_feats == 16:
#         g = sns.pairplot(features, height=1.5)
#         for ax in g.axes.flatten():
#             # rotate x and y axis labels
#             ax.set_xlabel(ax.get_xlabel(), rotation = 90)
#             ax.set_ylabel(ax.get_ylabel(), rotation = 0)
#         plt.subplots_adjust(left=0.3, bottom=0.3)
#         plt.show()
# =============================================================================
            
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
                                     grouping_var=grouping_var, 
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
    mean_sample_size = int(np.round(metadata.join(features).groupby([grouping_var], as_index=False).size().mean()))
    print("Mean sample size: %d" % mean_sample_size)
    
    ##### mRMR feature selection: minimum Redunduncy, Maximum Relevance #####
    
    mrmr_dir = plot_dir / 'mrmr'
    mrmr_dir.mkdir(exist_ok=True, parents=True)
    mrmr_results_path = mrmr_dir / "mrmr_results.csv"

    if not mrmr_results_path.exists():
        estimator = Pipeline([('scaler', StandardScaler()), ('estimator', LogisticRegression())])
        y = metadata[grouping_var].values
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
        print('MRMR CV Score: %f (n=%d)' % (np.mean(cv_scores_mrmr), n_cv))        
    else:
        # load mrmr results
        mrmr_table = pd.read_csv(mrmr_results_path)
        
    mrmr_feat_set = mrmr_table['feature'].to_list()
    print("\nTop %d features found by MRMR:" % len(mrmr_feat_set))
    for feat in mrmr_feat_set:
        print(feat)
    
    print("Loading statistics results")

    ### ANOVA

    if not args.use_corrected_pvals:
        anova_path = stats_dir / '{}_results_uncorrected.csv'.format(args.test)
    else:
        anova_path = stats_dir / '{}_results.csv'.format(args.test)
    
    # load results + record significant features
    anova_table = pd.read_csv(anova_path, index_col=0)            
    pvals = anova_table.sort_values(by='pvals', ascending=True)['pvals'] # rank features by p-value
    fset = pvals[pvals < args.pval_threshold].index.to_list()
    print("\n%d significant features found by %s (P<%.2f, %s)" % (len(fset), args.test, 
          args.pval_threshold, ('uncorrected' if not args.use_corrected_pvals else args.fdr_method)))
    
    ### k-significant features 
    
    if len(fset) > 0:
        # Compare k sigfeat and ANOVA significant feature set overlap
        if not args.use_corrected_pvals:
            k_sigfeats_path = stats_dir / "k_significant_features_uncorrected.csv"
        else:
            k_sigfeats_path = stats_dir / "k_significant_features.csv"
            
        ksig_table = pd.read_csv(k_sigfeats_path, index_col=0)
        fset_ksig = ksig_table[ksig_table['pvals'] < args.pval_threshold].index.to_list()

        fset_overlap = set(fset).intersection(set(fset_ksig))
        prop_overlap = len(fset_overlap) / len(fset)
        print("%.1f%% overlap with k-significant features" % (prop_overlap * 100))
        
        if prop_overlap < 0.5 and len(fset) > 100:
            print("WARNING: Inconsistency in statistics for feature set agreement between "
                  + "%s and k significant features!" % args.test)
            
        if args.use_k_sig_feats_overlap:
            fset = list(ksig_table.loc[fset_overlap].sort_values(by='pvals', ascending=True).index)
            
        ### t-test
            
        t_test = 't-test' if args.test == 'ANOVA' else 'Mann-Whitney' # aka. Wilcoxon rank-sum
    
        if not args.use_corrected_pvals:
            ttest_path = stats_dir / '{}_results_uncorrected.csv'.format(t_test)
        else:
            ttest_path = stats_dir / '{}_results.csv'.format(t_test)
         
        # read t-test results + record significant features (NOT ORDERED)
        ttest_table = pd.read_csv(ttest_path, index_col=0)
        pvals_t = ttest_table[[c for c in ttest_table if "pvals_" in c]] 
        pvals_t.columns = [c.split('pvals_')[-1] for c in pvals_t.columns]       
        fset_ttest = pvals_t[(pvals_t < args.pval_threshold).sum(axis=1) > 0].index.to_list()
        print("%d significant features found by %s (P<%.2f, %s)" % (len(fset_ttest), t_test, 
              args.pval_threshold, ('uncorrected' if not args.use_corrected_pvals else args.fdr_method)))
    
    else:
        print("No significant features found for %s by %s" % (grouping_var, args.test))

    ##### PLOTTING #####
    
    if len(fset) > 0:
        # Rank strains by number of sigfeats by t-test 
        ranked_nsig = (pvals_t < args.pval_threshold).sum(axis=0).sort_values(ascending=False)
        # Select top hit strains by n sigfeats (select strains with > 5 sigfeats as hit strains?)
        hit_strains_nsig = ranked_nsig[ranked_nsig > 0].index.to_list()
        #hit_nuo = ranked_nsig[[i for i in ranked_nsig[ranked_nsig > 0].index if 'nuo' in i]]
        # if no sigfaets, subset for top strains ranked by lowest p-value by t-test for any feature
        print("%d significant strains (with 1 or more significant features)" % len(hit_strains_nsig))
        if len(hit_strains_nsig) > 0:
            write_list_to_file(hit_strains_nsig, save_dir / 'hit_strains.txt')

        # Rank strains by lowest p-value for any feature
        ranked_pval = pvals_t.min(axis=0).sort_values(ascending=True)
        # Select top 100 hit strains by lowest p-value for any feature
        hit_strains_pval = ranked_pval[ranked_pval < args.pval_threshold].index.to_list()
        max_n_hits = max(len(hit_strains_pval), MAX_N_HITS)
        hit_strains_pval = ranked_pval.index[:max_n_hits].to_list()
        write_list_to_file(hit_strains_pval, save_dir / 'Top100_lowest_pval.txt')
        
        print("\nPlotting ranked strains by number of significant features")
        ranked_nsig_path = plot_dir / ('ranked_number_sigfeats' + '_' + 
                                       ('uncorrected' if args.fdr_method is None else 
                                        args.fdr_method) + '.png')
        plt.ioff()
        fig, ax = plt.subplots()
        ax.plot(ranked_nsig)
        ax.set_xticklabels([])
        plt.xlabel("Strains (ranked)", fontsize=10)
        plt.ylabel("Number of significant features", fontsize=10)
        plt.savefig(ranked_nsig_path, dpi=600)
        plt.close()
        
        print("Plotting ranked strains by lowest p-value of any feature")
        lowest_pval_path = plot_dir / ('ranked_lowest_pval' + '_' + 
                                       ('uncorrected' if args.fdr_method is None else 
                                        args.fdr_method) + '.png')
        plt.ioff()
        fig, ax = plt.subplots()
        ax.plot(ranked_pval)
        plt.axhline(y=args.pval_threshold, c='dimgray', ls='--')
        ax.set_xticklabels([])
        plt.xlabel("Strains (ranked)", fontsize=10)
        plt.ylabel("Lowest p-value by t-test", fontsize=10)
        plt.savefig(lowest_pval_path, dpi=600)
        plt.close()

        print("\nMaking errorbar plots")
        errorbar_sigfeats(features, metadata, 
                          group_by=grouping_var, 
                          fset=fset, 
                          control=control, 
                          rank_by='mean',
                          max_feats2plt=args.n_sig_features, 
                          figsize=[130,6], 
                          fontsize=2, 
                          saveDir=plot_dir / 'errorbar')
        
# =============================================================================
#         print("Making boxplots")
#         boxplots_grouped(feat_meta_df=metadata.join(features), 
#                           group_by=grouping_var,
#                           control_group=control,
#                           test_pvalues_df=(pvals_t.T if len(fset) > 0 else None), # ranked by test pvalue significance
#                           feature_set=fset,
#                           max_feats2plt=args.n_sig_features, 
#                           max_groups_plot_cap=None,
#                           p_value_threshold=args.pval_threshold,
#                           drop_insignificant=False,
#                           sns_colour_palette="tab10",
#                           figsize=[6,130], 
#                           saveDir=plot_dir / ('boxplots' + '_' + ('uncorrected' if args.fdr_method is None else args.fdr_method) + '.png')
# =============================================================================

        # If no sigfeats, subset for top strains ranked by lowest p-value by t-test for any feature    
        if len(hit_strains_nsig) == 0:
            print("\nSubsetting for top %d strains ranked by lowest p-value of any feature" % max_n_hits)
            hit_strains_pval.insert(0, control)
            features, metadata = subset_results(features, 
                                                metadata, 
                                                column=grouping_var,
                                                groups=hit_strains_pval, verbose=False)
            write_list_to_file(hit_strains_pval, save_dir / 'Top100_lowest_pval.txt')
        elif len(hit_strains_nsig) > 0:
            print("\nSubsetting for %d hit strains + control" % min(len(hit_strains_nsig), max_n_hits))
            hit_strains_nsig = hit_strains_nsig[:max_n_hits]
            hit_strains_nsig.insert(0, control)
            features, metadata = subset_results(features,
                                                metadata,
                                                column=grouping_var,
                                                groups=hit_strains_nsig, verbose=False)
                         
        # Individual boxplots of significant features by pairwise t-test (each group vs control)
        boxplots_sigfeats(features,
                          y_class=metadata[grouping_var],
                          control=control,
                          pvals=pvals_t, 
                          feature_set=None,
                          saveDir=plot_dir / ('paired_boxplots' + '_' + 
                                              ('uncorrected' if args.fdr_method is None else 
                                               args.fdr_method) + '.png'),
                          p_value_threshold=args.pval_threshold,
                          drop_insignificant=(True if len(hit_strains_nsig) > 0 else False),
                          max_sig_feats=args.n_sig_features,
                          max_strains=max_n_hits,
                          sns_colour_palette="tab10",
                          colour_by=None,
                          verbose=False)
        
        # # superplots of variation with respect to 'date_yyyymmdd'
        # from visualisation.super_plots import superplot
        # print("Plotting superplots of date variation for significant features")
        # for feat in tqdm(fset[:args.n_sig_features]):
        #     superplot(hit_features, hit_metadata, feat, 
        #               x1=grouping_var, 
        #               x2='date_yyyymmdd',
        #               saveDir=plot_dir / 'superplots',
        #               show_points=True, 
        #               plot_means=True,
        #               dodge=False)

        # from tierpsytools.analysis.significant_features import plot_feature_boxplots
        # plot_feature_boxplots(feat_to_plot=hit_features,
        #                       y_class=hit_metadata[grouping_var],
        #                       scores=pvals_t.rank(axis=1),
        #                       pvalues=np.asarray(pvals_t).flatten(),
        #                       saveto=None,
        #                       close_after_plotting=True)
           
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
    control_clustermap_path = plot_dir / 'heatmaps' / (grouping_var + '_clustermap.pdf')
    cg = plot_clustermap(featZ, metadata,
                         group_by=([grouping_var] if grouping_var == 'date_yyyymmdd' 
                                   else [grouping_var, 'date_yyyymmdd']),
                         method='complete', # metric=['euclidean','cosine','correlation']
                         #[linkage, complete, average, weighted, centroid]
                         figsize=[18,6],
                         sub_adj={'top':1,'bottom':0.3,'left':0,'right':0.9},
                         saveto=control_clustermap_path)

    #col_linkage = cg.dendrogram_col.calculated_linkage
    clustered_features = np.array(featZ.columns)[cg.dendrogram_col.reordered_ind]
                
    ## Save z-normalised values
    # z_stats = featZ.join(hit_metadata[grouping_var]).groupby(by=grouping_var).mean().T
    # z_stats.columns = ['z-mean_' + v for v in z_stats.columns.to_list()]
    # z_stats.to_csv(z_stats_path, header=True, index=None)
    
    # Clustermap of full data   
    print("Plotting all strains clustermap")    
    full_clustermap_path = plot_dir / 'heatmaps' / (grouping_var + '_full_clustermap.pdf')
    fg = plot_clustermap(featZ, 
                         metadata, 
                         group_by=grouping_var,
                         method='complete', # metric=['euclidean','cosine','correlation']
                         figsize=[15,30],
                         sub_adj={'top':1,'bottom':0.3,'left':0,'right':0.9},
                         saveto=full_clustermap_path)
    
    # If no control clustering (due to no day variation) then use clustered features for all 
    # strains to order barcode heatmaps
    if clustered_features is None:
        clustered_features = np.array(featZ.columns)[fg.dendrogram_col.reordered_ind]
    
    pvals_heatmap = anova_table.loc[clustered_features, 'pvals']
    pvals_heatmap.name = 'P < {}'.format(args.pval_threshold)

    assert all(f in featZ.columns for f in pvals_heatmap.index)
            
    # # Plot barcode heatmap (grouping by date)
    # if len(hit_metadata['date_yyyymmdd'].unique()) > 1:
    #     print("\nPlotting barcode heatmap by date")
    #     heatmap_date_path = plot_dir / 'heatmaps' / (grouping_var + '_date_heatmap.pdf')
    #     plot_barcode_heatmap(featZ=featZ[clustered_features], 
    #                          meta=hit_metadata, 
    #                          group_by=[grouping_var, 'date_yyyymmdd'],
    #                          pvalues_series=pvals_heatmap,
    #                          p_value_threshold=args.pval_threshold,
    #                          selected_feats=fset if len(fset) > 0 else None,
    #                          saveto=heatmap_date_path,
    #                          figsize=[20, 30],
    #                          sns_colour_palette="Pastel1")
    
    # # Plot group-mean heatmap (averaged across days)
    # print("\nPlotting barcode heatmap")
    # heatmap_path = plot_dir / 'heatmaps' / (grouping_var + '_heatmap.pdf')
    # plot_barcode_heatmap(featZ=featZ[clustered_features], 
    #                      meta=hit_metadata, 
    #                      group_by=[grouping_var], 
    #                      pvalues_series=pvals_heatmap,
    #                      p_value_threshold=args.pval_threshold,
    #                      selected_feats=fset if len(fset) > 0 else None,
    #                      saveto=heatmap_path,
    #                      figsize=[20, 30],
    #                      sns_colour_palette="Pastel1")        
                    
    ##### Principal Components Analysis #####

    pca_dir = plot_dir / 'PCA'
    
    # Z-normalise data for all strains
    featZ = features.apply(zscore, axis=0)

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

    #from tierpsytools.analysis.decomposition import plot_pca
    _ = plot_pca(featZ, metadata, 
                  group_by=grouping_var, 
                  control=control,
                  var_subset=None, 
                  saveDir=pca_dir,
                  PCs_to_keep=10,
                  n_feats2print=10,
                  kde=False,
                  sns_colour_palette="plasma",
                  n_dims=2,
                  hypercolor=False)
    # TODO: Ensure sns colour palette does not plot white points for PCA

    ######### DELETE #########
    # Add metadata information for COG
    if not 'COG_info' in metadata.columns:
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
                COG_info.append('Unknown')
        metadata['COG_info'] = COG_info
    else:
        print("COG info was already appended by compile_keio_results.py, so you may delete me!")
    ######### DELETE #########

    # plot pca coloured by Keio COG category    
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

    ##### t-distributed Stochastic Neighbour Embedding #####   
    
    print("\nPerforming tSNE")
    tsne_dir = plot_dir / 'tSNE'
    perplexities = [mean_sample_size] # NB: should be roughly equal to group size    
    _ = plot_tSNE(featZ, metadata,
                  group_by=grouping_var,
                  var_subset=None,
                  saveDir=tsne_dir,
                  perplexities=perplexities,
                  sns_colour_palette="plasma")
   
    ##### Uniform Manifold Projection #####  
    
    print("\nPerforming UMAP")
    umap_dir = plot_dir / 'UMAP'
    n_neighbours = [mean_sample_size] # NB: should be roughly equal to group size
    min_dist = 0.1 # Minimum distance parameter    
    _ = plot_umap(featZ, metadata,
                  group_by=grouping_var,
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
                        default=None, type=str)
    parser.add_argument('--metadata_path', help="Path to metadata file", 
                        default=None, type=str)
    args = parser.parse_args()

    FEATURES_PATH = args.features_path
    METADATA_PATH = args.metadata_path

    args = load_json(args.json)
    
    if FEATURES_PATH is None:
        FEATURES_PATH = Path(args.save_dir) / 'features.csv'
    if METADATA_PATH is None:
        METADATA_PATH = Path(args.save_dir) / 'metadata.csv'
        
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
