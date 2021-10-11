#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare control variation across: days, rigs, plates, wells

Stats:
    ANOVA/Kruskal - for significant features in control across day/rig/plate/well
    t-test/ranksum - for significant features between each day/rig/plate/well and a control for each
    Linear mixed models - for significant features in control across days, accounting for 
                          rig/plate/well variation
    
Plots:
    Boxplots of significant features by ANONVA/Kruskal/LMM
    Boxplots of significant features by t-test/ranksum
    Heatmaps of control across day/rig/plate/well
    PCA/tSNE/UMAP of control across day/rig/plate/well

@author: saul.moore11@lms.mrc.ac.uk
@date: 09/02/2021

"""        
        
#%% Imports

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from pathlib import Path
from scipy.stats import zscore # ttest_ind, f_oneway, kruskal

from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
#from tierpsytools.analysis.significant_features import k_significant_feat
#from tierpsytools.drug_screenings.filter_compounds import compounds_with_low_effect_univariate

from read_data.paths import get_save_dir
from read_data.read import load_json, load_topfeats
from write_data.write import write_list_to_file
from filter_data.clean_feature_summaries import clean_summary_results, subset_results
#from statistical_testing.stats_helper import shapiro_normality_test
from clustering.hierarchical_clustering import plot_clustermap, plot_barcode_heatmap
from feature_extraction.decomposition.pca import plot_pca, remove_outliers_pca
from feature_extraction.decomposition.tsne import plot_tSNE
from feature_extraction.decomposition.umap import plot_umap
from visualisation.super_plots import superplot
from visualisation.plotting_helper import sig_asterix, barplot_sigfeats
#plot_day_variation, boxplots_sigfeats, boxplots_grouped

#%% Globals

JSON_PARAMETERS_PATH = "analysis/20210406_parameters_keio_screen.json"
FEATURES_PATH = "/Users/sm5911/Documents/Keio_Screen/features.csv"
METADATA_PATH = "/Users/sm5911/Documents/Keio_Screen/metadata.csv"

METHOD = 'complete'
METRIC = 'euclidean'
       
#%% Functions

def control_variation(feat, 
                      meta, 
                      args,
                      variables=['date_yyyymmdd','instrument_name','imaging_run_number'],
                      n_sig_features=None):
    """ Analyse variation in control data with respect to each categorical variable in 'variables'
        
        Inputs
        ------
        feat, meta : pd.DataFrame
            Matching features summaries and metadata for control data
        
        args : Object 
            Python object with the following attributes:
            - remove_outliers : bool
            - grouping_variable : str
            - control_dict : dict
            - test : str
            - pval_threshold : float
            - fdr_method : str
            - n_sig_features : int
            - n_top_feats : int
            - drop_size_features : bool
            - norm_features_only : bool
            - percentile_to_use : str
            - remove_outliers : bool
            
        variables : list
            List of categorical random variables to analyse variation in control data
    """
    
    assert set(feat.index) == set(meta.index)
            
    save_dir = get_save_dir(args) / "control"

    # Stats test to use
    assert args.test in ['ANOVA','Kruskal','LMM']
    t_test = 't-test' if args.test == 'ANOVA' else 'Mann-Whitney' # aka. Wilcoxon rank-sums
    
    for grouping_var in tqdm(variables):
        
        # convert grouping variable column to factor (categorical)
        meta[grouping_var] = meta[grouping_var].astype(str)
                
        # get control group for eg. date_yyyymmdd
        control_group = str(args.control_dict[grouping_var])
        print("\nInvestigating variation in '%s' (control: '%s')" % (grouping_var, control_group))

        # Record mean sample size per group
        mean_sample_size = int(np.round(meta.groupby([grouping_var]).size().mean()))
        print("Mean sample size: %d" % mean_sample_size)
        
        group_list = list(meta[grouping_var].unique())
        stats_dir =  save_dir / "Stats" / grouping_var
        plot_dir = save_dir / "Plots" / grouping_var

        ##### STATISTICS #####
                      
        stats_path = stats_dir / '{}_results.csv'.format(args.test) # LMM/ANOVA/Kruskal  
        ttest_path = stats_dir / '{}_results.csv'.format(t_test)
    
        if not np.logical_and(stats_path.exists(), ttest_path.exists()):
            stats_path.parent.mkdir(exist_ok=True, parents=True)
            ttest_path.parent.mkdir(exist_ok=True, parents=True)
        
            ### ANOVA / Kruskal-Wallis tests for significantly different features across groups
            if (args.test == "ANOVA" or args.test == "Kruskal"):
                if len(group_list) > 2:                    
                    stats, pvals, reject = univariate_tests(X=feat, 
                                                            y=meta[grouping_var], 
                                                            control=control_group, 
                                                            test=args.test,
                                                            comparison_type='multiclass',
                                                            multitest_correction=args.fdr_method, 
                                                            alpha=0.05)
                    # get effect sizes
                    effect_sizes = get_effect_sizes(X=feat, 
                                                    y=meta[grouping_var],
                                                    control=control_group,
                                                    effect_type=None,
                                                    linked_test=args.test)
                                                
                    anova_table = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
                    anova_table.columns = ['stats','effect_size','pvals','reject']     

                    anova_table['significance'] = sig_asterix(anova_table['pvals'])
        
                    # Sort pvals + record significant features
                    anova_table = anova_table.sort_values(by=['pvals'], ascending=True)
                    fset = list(anova_table['pvals'].index[np.where(anova_table['pvals'] < 
                                                                    args.pval_threshold)[0]])
                    
                    # Save statistics results + significant feature set to file
                    anova_table.to_csv(stats_path, header=True, index=True)
            
                    if len(fset) > 0:
                        anova_sigfeats_path = Path(str(stats_path).replace('_results.csv', '_sigfeats.txt'))
                        write_list_to_file(fset, anova_sigfeats_path)
                        print("\n%d significant features found by %s for '%s' (P<%.2f, %s)" %\
                              (len(fset), args.test, grouping_var, args.pval_threshold, args.fdr_method))
                else:
                    fset = []
                    print("\nWARNING: Not enough groups for %s for '%s' (n=%d groups)" %\
                          (args.test, grouping_var, len(group_list)))

            # TODO: LMMs using compounds_with_low_effect_univariate
            
            ### t-tests / Mann-Whitney tests
            if len(fset) > 0 or len(group_list) == 2:
                stats_t, pvals_t, reject_t = univariate_tests(X=feat, 
                                                              y=meta[grouping_var], 
                                                              control=control_group, 
                                                              test=t_test,
                                                              comparison_type='binary_each_group',
                                                              multitest_correction=args.fdr_method, 
                                                              alpha=0.05)
                effect_sizes_t =  get_effect_sizes(X=feat, y=meta[grouping_var], 
                                                   control=control_group,
                                                   effect_type=None,
                                                   linked_test=t_test)
                
                stats_t.columns = ['stats_' + str(c) for c in stats_t.columns]
                pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
                reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
                effect_sizes_t.columns = ['effect_size_' + str(c) for c in effect_sizes_t.columns]
                
                ttest_table = pd.concat([stats_t, effect_sizes_t, pvals_t, reject_t], axis=1)

                # Record t-test significant feature set (NOT ORDERED)
                fset_ttest = list(pvals_t.index[(pvals_t < args.pval_threshold).sum(axis=1) > 0])
                
                # Save t-test results to file
                ttest_table.to_csv(ttest_path, header=True, index=True) # Save test results to CSV

                if len(fset_ttest) > 0:
                    ttest_sigfeats_path = Path(str(ttest_path).replace('_results.csv', '_sigfeats.txt'))
                    write_list_to_file(fset_ttest, ttest_sigfeats_path)
                    print("%d signficant features found for any %s vs %s (%s, P<%.2f)" %\
                          (len(fset_ttest), grouping_var, control_group, t_test, args.pval_threshold))
                
                # Barplot of number of significantly different features for each strain   
                barplot_sigfeats(test_pvalues_df=pvals_t, 
                                 saveDir=plot_dir,
                                 p_value_threshold=args.pval_threshold,
                                 test_name=t_test)
                                 
        ### Load statistics results
        
        # Read ANOVA results and record significant features
        print("Loading statistics results")
        if len(group_list) > 2:
            anova_table = pd.read_csv(stats_path, index_col=0)
            pvals = anova_table.sort_values(by='pvals', ascending=True)['pvals']
            fset = pvals[pvals < args.pval_threshold].index.to_list()
            print("%d significant features found by %s (P<%.2f)" % (len(fset), args.test, 
                                                                    args.pval_threshold))
        
        # Read t-test results and record significant features (NOT ORDERED)
        ttest_table = pd.read_csv(ttest_path, index_col=0)
        pvals_t = ttest_table[[c for c in ttest_table if "pvals_" in c]]             
        fset_ttest = pvals_t[(pvals_t < args.pval_threshold).sum(axis=1) > 0].index.to_list()
        print("%d significant features found by %s (P<%.2f)" % (len(fset_ttest), t_test, args.pval_threshold))
            
        # Use t-test significant feature set if comparing just 2 strains
        if len(group_list) == 2:
            fset = fset_ttest
                       
        if not n_sig_features:
            if args.n_sig_features is not None:
                n_sig_features = args.n_sig_features 
            else:
                n_sig_features = len(fset)
                                   
        ##### Plotting #####
        
        superplot_dir = plot_dir / 'superplots' 

        if len(fset) > 1:        
            for feature in tqdm(fset[:n_sig_features]):                
                # plot variation in variable with respect to 'date_yyyymmdd'
                superplot(feat, meta, feature, 
                          x1=grouping_var, 
                          x2=None if grouping_var == 'date_yyyymmdd' else 'date_yyyymmdd',
                          saveDir=superplot_dir,
                          pvals=pvals_t if grouping_var == 'date_yyyymmdd' else None,
                          pval_threshold=args.pval_threshold,
                          show_points=True, 
                          plot_means=True,
                          dodge=True)
                # plot variation in variable with respect to 'instrument_name'
                superplot(feat, meta, feature, 
                          x1=grouping_var, 
                          x2=None if grouping_var == 'instrument_name' else 'instrument_name',
                          saveDir=superplot_dir,
                          pvals=pvals_t if grouping_var == 'instrument_name' else None,
                          pval_threshold=args.pval_threshold,
                          show_points=True, 
                          plot_means=True,
                          dodge=True)
                # plot variation in variable with respect to 'imaging_ruun_number'
                superplot(feat, meta, feature, 
                          x1=grouping_var, 
                          x2=None if grouping_var == 'imaging_run_number' else 'imaging_run_number',
                          saveDir=superplot_dir,
                          pvals=pvals_t if grouping_var == 'imaging_run_number' else None,
                          pval_threshold=args.pval_threshold,
                          show_points=True, 
                          plot_means=True,
                          dodge=True)
                
                # TODO: Add t-test/LMM pvalues to superplots       
            
            # # Boxplots of significant features by ANOVA/LMM (all groups)
            # boxplots_grouped(feat_meta_df=meta.join(feat), 
            #                  group_by=grouping_var,
            #                  control_group=str(control_group),
            #                  test_pvalues_df=pvals_t.T, # ranked by test pvalue significance
            #                  feature_set=fset,
            #                  saveDir=(plot_dir / 'grouped_boxplots'),
            #                  max_feats2plt=args.n_sig_features, 
            #                  max_groups_plot_cap=None,
            #                  p_value_threshold=args.pval_threshold,
            #                  drop_insignificant=False,
            #                  sns_colour_palette="tab10",
            #                  figsize=[6, (len(group_list)/3 if len(group_list)>10 else 12)])
                    
            # Individual boxplots of significant features by pairwise t-test (each group vs control)
            # boxplots_sigfeats(feat_meta_df=meta.join(feat), 
            #                   test_pvalues_df=pvals_t, 
            #                   group_by=grouping_var, 
            #                   control_strain=control_group, 
            #                   feature_set=fset, #['speed_norm_50th_bluelight'],
            #                   saveDir=plot_dir / 'paired_boxplots',
            #                   max_feats2plt=args.n_sig_features,
            #                   p_value_threshold=args.pval_threshold,
            #                   drop_insignificant=True,
            #                   verbose=False)
                
            # from tierpsytools.analysis.significant_features import plot_feature_boxplots
            # plot_feature_boxplots(feat_to_plot=fset,
            #                       y_class=grouping_var,
            #                       scores=pvalues.rank(axis=1),
            #                       feat=feat,
            #                       pvalues=np.asarray(pvalues).flatten(),
            #                       saveto=None,
            #                       close_after_plotting=False)
        
        ##### Hierarchical Clustering Analysis #####
        print("\nPerforming hierarchical clustering analysis...")

        assert not feat.isna().sum(axis=1).any()
        assert not (feat.std(axis=1) == 0).any()
        
        # Z-normalise data
        featZ = feat.apply(zscore, axis=0)
        #featZ = (feat-feat.mean())/feat.std() # minus mean, divide by std
        
        #from tierpsytools.preprocessing.scaling_class import scalingClass
        #scaler = scalingClass(scaling='standardize')
        #featZ = scaler.fit_transform(feat)

        # NOT NEEDED?
        # # Drop features with NaN values after normalising
        # n_cols = len(featZ.columns)
        # featZ.dropna(axis=1, inplace=True)
        # n_dropped = n_cols - len(featZ.columns)
        # if n_dropped > 0:
        #     print("Dropped %d features after normalisation (NaN)" % n_dropped)
    
        ### Control clustermap
        
        # control data is clustered and feature order is stored and applied to full data
        if len(group_list) > 1 and len(group_list) < 50 and grouping_var != 'date_yyyymmdd':
            control_clustermap_path = plot_dir / 'heatmaps' / (grouping_var + '_date_clustermap.pdf')
            cg = plot_clustermap(featZ, meta,
                                 group_by=([grouping_var] if grouping_var == 'date_yyyymmdd' 
                                           else [grouping_var, 'date_yyyymmdd']),
                                 col_linkage=None,
                                 method=METHOD,#[linkage, complete, average, weighted, centroid]
                                 metric=METRIC,
                                 figsize=[15,8],
                                 sub_adj={'bottom':0.02,'left':0.02,'top':1,'right':0.85},
                                 label_size=12,
                                 show_xlabels=False,
                                 saveto=control_clustermap_path)
    
            #col_linkage = cg.dendrogram_col.calculated_linkage
            clustered_features = np.array(featZ.columns)[cg.dendrogram_col.reordered_ind]
        else:
            clustered_features = None
                    
        ## Save z-normalised values
        # z_stats = featZ.join(meta[grouping_var]).groupby(by=grouping_var).mean().T
        # z_stats.columns = ['z-mean_' + v for v in z_stats.columns.to_list()]
        # z_stats.to_csv(z_stats_path, header=True, index=None)
        
        # Clustermap of full data       
        full_clustermap_path = plot_dir / 'heatmaps' / (grouping_var + '_clustermap.pdf')
        fg = plot_clustermap(featZ, meta, 
                             group_by=grouping_var,
                             col_linkage=None,
                             method=METHOD,
                             metric=METRIC,
                             figsize=[15,8],
                             sub_adj={'bottom':0.02,'left':0.02,'top':1,'right':0.9},
                             label_size=12,
                             saveto=full_clustermap_path)
        
        # If no control clustering (due to no day variation) then use clustered features for all 
        # strains to order barcode heatmaps
        if clustered_features is None:
            clustered_features = np.array(featZ.columns)[fg.dendrogram_col.reordered_ind]
        
        if len(group_list) > 2:
            pvals_heatmap = anova_table.loc[clustered_features, 'pvals']
        elif len(group_list) == 2:
            pvals_heatmap = pvals_t.loc[clustered_features, pvals_t.columns[0]]
        pvals_heatmap.name = 'P < {}'.format(args.pval_threshold)
    
        assert all(f in featZ.columns for f in pvals_heatmap.index)
                
        # Plot barcode heatmap (grouping by date)
        if len(group_list) > 1 and len(group_list) < 50 and grouping_var != 'date_yyyymmdd':
            heatmap_date_path = plot_dir / 'heatmaps' / (grouping_var + '_date_heatmap.pdf')
            plot_barcode_heatmap(featZ=featZ[clustered_features], 
                                 meta=meta, 
                                 group_by=[grouping_var, 'date_yyyymmdd'],
                                 pvalues_series=pvals_heatmap,
                                 p_value_threshold=args.pval_threshold,
                                 selected_feats=fset if len(fset) > 0 else None,
                                 saveto=heatmap_date_path,
                                 figsize=[20,7],
                                 sns_colour_palette="Pastel1")
        
        # Plot group-mean heatmap (averaged across days)
        heatmap_path = plot_dir / 'heatmaps' / (grouping_var + '_heatmap.pdf')
        plot_barcode_heatmap(featZ=featZ[clustered_features], 
                             meta=meta, 
                             group_by=[grouping_var], 
                             pvalues_series=pvals_heatmap,
                             p_value_threshold=args.pval_threshold,
                             selected_feats=fset if len(fset) > 0 else None,
                             saveto=heatmap_path,
                             figsize=[20, (int(len(group_list) / 4) if len(group_list) > 10 else 6)],
                             sns_colour_palette="Pastel1")        
                        
        ##### Principal Components Analysis #####
        print("Performing principal components analysis")
    
        if args.remove_outliers:
            outlier_path = plot_dir / 'mahalanobis_outliers.pdf'
            feat, inds = remove_outliers_pca(df=feat, 
                                             features_to_analyse=None, 
                                             saveto=outlier_path)
            meta = meta.reindex(feat.index) # reindex metadata
            featZ = feat.apply(zscore, axis=0) # re-normalise data

            # Drop features with NaN values after normalising
            n_cols = len(featZ.columns)
            featZ.dropna(axis=1, inplace=True)
            n_dropped = n_cols - len(featZ.columns)
            if n_dropped > 0:
                print("Dropped %d features after normalisation (NaN)" % n_dropped)

        #from tierpsytools.analysis.decomposition import plot_pca
        pca_dir = plot_dir / 'PCA'
        _ = plot_pca(featZ, meta, 
                     group_by=grouping_var, 
                     control=control_group,
                     var_subset=None, 
                     saveDir=pca_dir,
                     PCs_to_keep=10,
                     n_feats2print=10,
                     sns_colour_palette="plasma",
                     n_dims=2,
                     label_size=15,
                     figsize=[9,8],
                     sub_adj={'bottom':0.13,'left':0.12,'top':0.98,'right':0.98},
                     # legend_loc='upper right',
                     # n_colours=20,
                     hypercolor=False)
        # TODO: Ensure sns colour palette does not plot white points for PCA

# =============================================================================
#         ##### t-distributed Stochastic Neighbour Embedding #####   
#         
#         tsne_dir = plot_dir / 'tSNE'
#         perplexities = [mean_sample_size] # NB: should be roughly equal to group size
#         
#         _ = plot_tSNE(featZ, meta,
#                       group_by=grouping_var,
#                       var_subset=None,
#                       saveDir=tsne_dir,
#                       perplexities=perplexities,
#                       sns_colour_palette="plasma")
# =============================================================================
       
# =============================================================================
#         ##### Uniform Manifold Projection #####  
#         
#         umap_dir = plot_dir / 'UMAP'
#         n_neighbours = [mean_sample_size] # NB: should be roughly equal to group size
#         min_dist = 0.1 # Minimum distance parameter
#         
#         _ = plot_umap(featZ, meta,
#                       group_by=grouping_var,
#                       var_subset=None,
#                       saveDir=umap_dir,
#                       n_neighbours=n_neighbours,
#                       min_dist=min_dist,
#                       sns_colour_palette="plasma")        
# =============================================================================
    
#%%
if __name__ == "__main__":
    tic = time()
    parser = argparse.ArgumentParser(description="Read clean features and metadata for control and \
                                     investigate variation")
    parser.add_argument('-j', '--json', help="Path to JSON parameters file", 
                        default=JSON_PARAMETERS_PATH, type=str)
    parser.add_argument('--features_path', help="Path to feature summaries file", 
                        default=FEATURES_PATH, type=str)
    parser.add_argument('--metadata_path', help="Path to metadata file", 
                        default=METADATA_PATH, type=str)
    args = parser.parse_args()
    
    args = load_json(args.json)
    
    # Read feature summaries + metadata
    features = pd.read_csv(FEATURES_PATH)
    metadata = pd.read_csv(METADATA_PATH, dtype={'comments':str, 'source_plate_id':str})
    
    # Subset for control data only
    control_strain = args.control_dict[args.grouping_variable] # control strain to use
    control_features, control_metadata = subset_results(features, metadata, 
                                                        column=args.grouping_variable,
                                                        groups=[control_strain])
    # Subset for imaging dates of interest    
    if args.dates is not None:
        dates = [int(d) for d in args.dates]
        control_features, control_metadata = subset_results(control_features, control_metadata, 
                                                            column='date_yyyymmdd', groups=dates)

    # Clean data after subset - to remove features with zero std
    control_features, control_metadata = clean_summary_results(control_features, 
                                                               control_metadata, 
                                                               max_value_cap=False,
                                                               imputeNaN=False)
    
    # Load Tierpsy Top feature set + subset (columns) for top feats only
    if args.n_top_feats is not None:
        top_feats_path = Path(args.tierpsy_top_feats_dir) / "tierpsy_{}.csv".format(str(args.n_top_feats))
        topfeats = load_topfeats(top_feats_path, add_bluelight=True, 
                                 remove_path_curvature=True, header=None)
        
        # Drop features that are not in results
        top_feats_list = [feat for feat in list(topfeats) if feat in control_features.columns]
        control_features = control_features[top_feats_list]

    print("Investigating variation in '%s' (control %s)" % (control_strain, args.grouping_variable))
    control_variation(control_features, 
                      control_metadata, 
                      args,
                      variables=['date_yyyymmdd','instrument_name','imaging_run_number']) # 'well_name'
    
    toc = time()
    print("\nDone in %.1f seconds (%.1f minutes)" % (toc - tic, (toc - tic) / 60))
    
