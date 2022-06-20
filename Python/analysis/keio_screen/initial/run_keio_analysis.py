#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse Keio Screen results

Please run the following scripts beforehand:
1. preprocessing/compile_keio_results.py
2. statistical_testing/perform_keio_stats.py

THRESHOLD MAX DISTANCE FOR CLUSTERING: 8 (Tierpsy16, fdr_bh, all strains)

@author: sm5911
@date: 19/04/2021
"""

#%% IMPORTS

import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from time import time
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.stats import zscore # levene, ttest_ind, f_oneway, kruskal

from read_data.paths import get_save_dir
from read_data.read import load_json
from write_data.write import write_list_to_file
from analysis.compare_strains.control_variation import control_variation
from filter_data.clean_feature_summaries import clean_summary_results, subset_results
from statistical_testing.perform_keio_stats import average_plate_control_data
from clustering.hierarchical_clustering import plot_clustermap, plot_barcode_heatmap
from feature_extraction.decomposition.pca import plot_pca, remove_outliers_pca
from feature_extraction.decomposition.tsne import plot_tSNE
from feature_extraction.decomposition.umap import plot_umap
from time_series.time_series_helper import get_strain_timeseries
from time_series.plot_timeseries import plot_timeseries_motion_mode
from visualisation.plotting_helper import errorbar_sigfeats, boxplots_sigfeats 
# from visualisation.plotting_helper import boxplots_grouped, barplot_sigfeats, plot_day_variation
# from visualisation.super_plots import superplot

from tierpsytools.preprocessing.filter_data import select_feat_set

#%% GLOBALS

# JSON_PARAMETERS_PATH = "analysis/20210406_parameters_keio_screen.json"
JSON_PARAMETERS_PATH = "analysis/20210914_parameters_keio_screen.json"

N_LOWEST_PVAL = 100
SUBSET_HIT_STRAINS = False
TOP_N_HITS = 10

METHOD = 'complete' # 'complete','linkage','average','weighted','centroid'
METRIC = 'euclidean' # 'euclidean','cosine','correlation'

COG_category_dict = {'J' : 'Translation, ribosomal structure, and biogenesis',
                     'K' : 'Transcription',
                     'L' : 'DNA replication, recombination, and repair',
                     'D' : 'Cell division and chromosome partitioning',
                     'O' : 'Post-translational modification, protein turnover, chaperones',
                     'M' : 'Cell envelope biogenesis, outer membrane',
                     'N' : 'Cell motility and secretion',
                     'P' : 'Inorganic ion transport and metabolism',
                     'T' : 'Signal transduction mechanisms',
                     'C' : 'Energy production and conversion',
                     'G' : 'Carbohydrate transport and metabolism',
                     'E' : 'Amino acid transport and metabolism',
                     'F' : 'Nucleotide transport and metabolism',
                     'H' : 'Coenzyme metabolism',
                     'I' : 'Lipid metabolism',
                     'Q' : 'Secondary metabolites biosynthesis, transport, and catabolism',
                     'R' : 'General function prediction only',
                     'S' : 'Function unknown', 
                     'U' : 'Function unknown', 
                     'V' : 'Function unknown',
                     'Unknown' : 'Function unknown'}

SELECTED_STRAIN_LIST = ['fepD','fepB','fepC','fepG','fes',
                        'nuoA','nuoB','nuoC','nuoF','nuoG','nuoH','nuoK','nuoL','nuoM','nuoN',
                        'cyoA','cyoC','cyoD','cyoE',
                        'sdhA','sdhB','sdhC','sdhD',
                        'atpA','atpB','atpC','atpD','atpE','atpF','atpH',
                        'entA','entB','entC','entE','entF']
FPS = 25

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
    n_strains = len(metadata[grouping_var].unique())
    assert n_strains == len(metadata[grouping_var].str.upper().unique()) # check case-sensitivity
    print("\nInvestigating '%s' variation (%d samples)" % (grouping_var, n_strains))
    
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
        assert args.n_top_feats in [8,16,256,'2k']
        features = select_feat_set(features, 
                                   tierpsy_set_name='tierpsy_{}'.format(args.n_top_feats), 
                                   append_bluelight=True)
            
    ##### Control variation #####

    control_metadata = metadata[metadata[grouping_var] == control]
    control_features = features.reindex(control_metadata.index)

    # Clean data after subset - to remove features with zero std
    control_features, control_metadata = clean_summary_results(control_features, 
                                                               control_metadata, 
                                                               max_value_cap=False,
                                                               imputeNaN=False)                  
    if args.analyse_control:
        control_variation(control_features, control_metadata, args,
                          variables=[k for k in args.control_dict.keys() if k != grouping_var],
                          n_sig_features=10)

    if args.collapse_control:
        print("\nCollapsing control data (mean of each day)")
        features, metadata = average_plate_control_data(features, metadata)
                            
    ##### STATISTICS #####

    # Record mean sample size per group
    mean_sample_size = int(np.round(metadata.join(features).groupby([grouping_var], 
                                                                    as_index=False).size().mean()))
    print("Mean sample size: %d" % mean_sample_size)

    save_dir = get_save_dir(args)
    stats_dir =  save_dir / grouping_var / "Stats" / args.fdr_method
    plot_dir = save_dir / grouping_var / "Plots" / args.fdr_method
       
    # TODO: Check initial keio cleaning and stats - I think its ok, but re-run and double check
    # TODO: Check that investigate control variation works
    
# =============================================================================
#     ##### Pairplot Tierpsy Features - Pairwise correlation matrix #####
#     if args.n_top_feats == 16:
#         g = sns.pairplot(features, height=1.5)
#         for ax in g.axes.flatten():
#             # rotate x and y axis labels
#             ax.set_xlabel(ax.get_xlabel(), rotation = 90)
#             ax.set_ylabel(ax.get_ylabel(), rotation = 0)
#         plt.subplots_adjust(left=0.3, bottom=0.3)
#         plt.show()
# =============================================================================

    if not args.use_corrected_pvals:
        anova_path = stats_dir / '{}_results_uncorrected.csv'.format(args.test)
    else:
        anova_path = stats_dir / '{}_results.csv'.format(args.test)
    
    # load results + record significant features
    print("\nLoading statistics results")
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
            write_list_to_file(hit_strains_nsig, stats_dir / 'hit_strains.txt')

        # Rank strains by lowest p-value for any feature
        ranked_pval = pvals_t.min(axis=0).sort_values(ascending=True)
        # Select top 100 hit strains by lowest p-value for any feature
        hit_strains_pval = ranked_pval[ranked_pval < args.pval_threshold].index.to_list()
        hit_strains_pval = ranked_pval.index.to_list()
        assert all(s in hit_strains_pval for s in hit_strains_nsig)
        write_list_to_file(hit_strains_pval[:N_LOWEST_PVAL], stats_dir /\
                           'lowest{}_pval.txt'.format(N_LOWEST_PVAL))
        
        print("\nPlotting ranked strains by number of significant features")
        ranked_nsig_path = plot_dir / ('ranked_number_sigfeats' + '_' + 
                                       ('uncorrected' if args.fdr_method is None else 
                                        args.fdr_method) + '.pdf')
        plt.close('all')
        if len(ranked_nsig.index) > 250:
            fig, ax = plt.subplots(figsize=(50,3), dpi=900)
            ax.plot(ranked_nsig)
            ax.set_xticklabels(ranked_nsig.index.to_list(), rotation=90, fontsize=1)
        else:
            fig, ax = plt.subplots(figsize=(30,5), dpi=600)
            ax.plot(ranked_nsig)
            ax.set_xticklabels(ranked_nsig.index.to_list(), rotation=90, fontsize=5)
        plt.xlabel("Strains (ranked)", fontsize=12, labelpad=10)
        plt.ylabel("Number of significant features", fontsize=12, labelpad=10)
        plt.subplots_adjust(left=0.03, right=0.99, bottom=0.15)
        plt.savefig(ranked_nsig_path)
        
        print("Plotting ranked strains by lowest p-value of any feature")
        lowest_pval_path = plot_dir / ('ranked_lowest_pval' + '_' + 
                                       ('uncorrected' if args.fdr_method is None else 
                                        args.fdr_method) + '.pdf')
        plt.close('all')
        if len(ranked_nsig.index) > 250:
            fig, ax = plt.subplots(figsize=(50,3), dpi=900)
            ax.plot(ranked_pval)
            plt.axhline(y=args.pval_threshold, c='dimgray', ls='--')
            ax.set_xticklabels(ranked_nsig.index.to_list(), rotation=90, fontsize=1)
        else:
            fig, ax = plt.subplots(figsize=(30,5), dpi=600)
            ax.plot(ranked_pval)
            plt.axhline(y=args.pval_threshold, c='dimgray', ls='--')
            ax.set_xticklabels(ranked_nsig.index.to_list(), rotation=90, fontsize=5)
        plt.xlabel("Strains (ranked)", fontsize=12, labelpad=10)
        plt.ylabel("Lowest p-value by t-test", fontsize=12, labelpad=10)
        plt.subplots_adjust(left=0.03, right=0.99, bottom=0.15)
        plt.savefig(lowest_pval_path)
        plt.close()

        print("\nMaking errorbar plots")
        errorbar_sigfeats(features, metadata, 
                          group_by=grouping_var, 
                          fset=fset, 
                          control=control, 
                          rank_by='mean',
                          max_feats2plt=args.n_sig_features, 
                          figsize=[20,10], 
                          fontsize=5,
                          ms=8,
                          elinewidth=1.5,
                          fmt='.',
                          tight_layout=[0.01,0.01,0.99,0.99],
                          saveDir=plot_dir / 'errorbar')
        
# =============================================================================
#         print("Making boxplots")
#         boxplots_grouped(feat_meta_df=metadata.join(features), 
#                           group_by=grouping_var,
#                           control_group=control,
#                           test_pvalues_df=(pvals_t.T if len(fset) > 0 else None), 
#                           feature_set=fset,
#                           max_feats2plt=args.n_sig_features, 
#                           max_groups_plot_cap=None,
#                           p_value_threshold=args.pval_threshold,
#                           drop_insignificant=False,
#                           sns_colour_palette="tab10",
#                           figsize=[6,130], 
#                           saveDir=plot_dir / ('boxplots' + '_' + (
#                                   'uncorrected' if args.fdr_method is None else args.fdr_method) + 
#                                   '.png'))
# =============================================================================

        # If no sigfeats, subset for top strains ranked by lowest p-value by t-test for any feature    
        if len(hit_strains_nsig) == 0:
            print("\Saving lowest %d strains ranked by p-value for any feature" % N_LOWEST_PVAL)
            write_list_to_file(hit_strains_pval, stats_dir / 'Top100_lowest_pval.txt')
            hit_strains = hit_strains_pval
        elif len(hit_strains_nsig) > 0:
            hit_strains = hit_strains_nsig

        # Individual boxplots of significant features by pairwise t-test (each group vs control)
        boxplots_sigfeats(features,
                          y_class=metadata[grouping_var],
                          control=control,
                          pvals=pvals_t, 
                          z_class=metadata['date_yyyymmdd'],
                          feature_set=fset, #None
                          # feature_set=['motion_mode_forward_fraction_prestim',
                          #              'motion_mode_forward_fraction_bluelight',
                          #              'motion_mode_forward_fraction_poststim',
                          #              'speed_50th_prestim',
                          #              'speed_50th_bluelight',
                          #              'speed_50th_poststim',
                          #              'curvature_midbody_norm_abs_50th_prestim',
                          #              'curvature_midbody_norm_abs_50th_bluelight',
                          #              'curvature_midbody_norm_abs_50th_poststim'],
                          # append_ranking_fname=False,
                          saveDir=plot_dir / 'paired_boxplots_nsig', # pval
                          p_value_threshold=args.pval_threshold,
                          drop_insignificant=True if len(hit_strains) > 0 else False,
                          max_sig_feats=args.n_sig_features,
                          max_strains=N_LOWEST_PVAL if len(hit_strains_nsig) == 0 else None,
                          sns_colour_palette="tab10",
                          verbose=False)
        
        if SUBSET_HIT_STRAINS:
            strain_list = [control] + hit_strains[:TOP_N_HITS]
            print("Subsetting for Top%d hit strains" % (len(strain_list)-1))
            features, metadata = subset_results(features, metadata, column=grouping_var,
                                                groups=strain_list, verbose=False)   
        else:
            strain_list = list(metadata[grouping_var].unique())
        
# =============================================================================
#         # NOT NECESSARY FOR ALL STRAINS - LOOK AT CONTROL ONLY FOR THIS
#         # superplots of variation with respect to 'date_yyyymmdd'
#         print("\nPlotting superplots of date variation for significant features")
#         for feat in tqdm(fset[:args.n_sig_features]):
#             # plot day variation
#             superplot(features, metadata, feat, 
#                       x1='date_yyyymmdd', 
#                       x2=None,
#                       saveDir=plot_dir / 'superplots',
#                       figsize=[24,6],
#                       show_points=False, 
#                       plot_means=True,
#                       dodge=False)
#             # plot run number vs day variation
#             superplot(features, metadata, feat, 
#                       x1='date_yyyymmdd', 
#                       x2='imaging_run_number',
#                       saveDir=plot_dir / 'superplots',
#                       figsize=[24,6],
#                       show_points=False, 
#                       plot_means=True,
#                       dodge=True)
#             # plot plate number variation
#             superplot(features, metadata, feat, 
#                       x1='date_yyyymmdd', 
#                       x2='source_plate_id',
#                       saveDir=plot_dir / 'superplots',
#                       figsize=[24,6],
#                       show_points=False, 
#                       plot_means=True,
#                       dodge=True)
#             # plot instrument name variation
#             superplot(features, metadata, feat, 
#                       x1='date_yyyymmdd', 
#                       x2='instrument_name',
#                       saveDir=plot_dir / 'superplots',
#                       figsize=[24,6],
#                       show_points=False, 
#                       plot_means=True,
#                       dodge=True)
# =============================================================================

# =============================================================================
#         from tierpsytools.analysis.significant_features import plot_feature_boxplots
#         plot_feature_boxplots(feat_to_plot=features,
#                               y_class=metadata[grouping_var],
#                               scores=pvals_t.rank(axis=1),
#                               pvalues=np.asarray(pvals_t).flatten(),
#                               saveto=None,
#                               close_after_plotting=True)
# =============================================================================

    ##### Hierarchical Clustering Analysis #####
        
    # Z-normalise control data
    control_featZ = control_features.apply(zscore, axis=0)
    #featZ = (features-features.mean())/features.std() # minus mean, divide by std
    #from tierpsytools.preprocessing.scaling_class import scalingClass
    #scaler = scalingClass(scaling='standardize')
    #featZ = scaler.fit_transform(features)

    ### Control clustermap
    
    # control data is clustered and feature order is stored and applied to full data
    print("\nPlotting control clustermap")
    
    control_clustermap_path = plot_dir / 'heatmaps' / 'date_clustermap.pdf'
    cg = plot_clustermap(control_featZ, control_metadata,
                         group_by=([grouping_var] if grouping_var == 'date_yyyymmdd' 
                                   else [grouping_var, 'date_yyyymmdd']),
                         method=METHOD, 
                         metric=METRIC,
                         figsize=[20,6],
                         sub_adj={'bottom':0.05,'left':0,'top':1,'right':0.85},
                         saveto=control_clustermap_path,
                         label_size=15,
                         show_xlabels=False)
    
    # save feature order to file
    control_feature_order = np.array(control_featZ.columns)[cg.dendrogram_col.reordered_ind]
    control_feature_order_df = pd.DataFrame(columns=['feature_name'], 
                                            index=range(1, len(control_feature_order) + 1),
                                            data=control_feature_order)
    control_feature_order_path = control_clustermap_path.parent / (control_clustermap_path.stem + 
                                                                   '_feature_order.csv')
    control_feature_order_df.to_csv(control_feature_order_path, header=True, index=True)

    # control clustermap with labels
    if args.n_top_feats <= 256:
        control_clustermap_path = plot_dir / 'heatmaps' / 'date_clustermap_label.pdf'
        cg = plot_clustermap(control_featZ, 
                             control_metadata,
                             group_by=([grouping_var] if grouping_var == 'date_yyyymmdd' 
                                       else [grouping_var, 'date_yyyymmdd']),
                             method=METHOD, 
                             metric=METRIC,
                             figsize=[30,10],
                             sub_adj={'bottom':(0.2 if args.n_top_feats >= 256 else 0.5),
                                      'left':0,'top':1,'right':0.85},
                             saveto=control_clustermap_path,
                             label_size=(2 if args.n_top_feats >= 256 else 15, 15),
                             show_xlabels=True)
    
    ### Full clustermap 

    # Z-normalise data for all strains
    featZ = features.apply(zscore, axis=0)
                    
    ## Save z-normalised values
    # z_stats = featZ.join(hit_metadata[grouping_var]).groupby(by=grouping_var).mean().T
    # z_stats.columns = ['z-mean_' + v for v in z_stats.columns.to_list()]
    # z_stats.to_csv(z_stats_path, header=True, index=None)
    
    # Clustermap of full data   
    print("Plotting all strains clustermap")    
    full_clustermap_path = plot_dir / 'heatmaps' / (grouping_var + '_clustermap.pdf')
    fg = plot_clustermap(featZ, metadata, 
                         group_by=grouping_var,
                         row_colours=None,
                         method=METHOD, 
                         metric=METRIC,
                         figsize=[20,30],
                         sub_adj={'bottom':0.01,'left':0,'top':1,'right':0.95},
                         saveto=full_clustermap_path,
                         label_size=8,
                         show_xlabels=False) # no feature labels
    
    if args.n_top_feats <= 256:
        full_clustermap_path = plot_dir / 'heatmaps' / (grouping_var + '_clustermap_label.pdf')
        fg = plot_clustermap(featZ, metadata, 
                             group_by=grouping_var,
                             row_colours=None,
                             method=METHOD, 
                             metric=METRIC,
                             figsize=[20,40],
                             sub_adj={'bottom':0.2,'left':0,'top':1,'right':0.95},
                             saveto=full_clustermap_path,
                             label_size=(2 if args.n_top_feats >= 256 else 15, 15),
                             show_xlabels=True)
    
    # save clustered feature order for all strains
    full_feature_order = np.array(featZ.columns)[fg.dendrogram_col.reordered_ind]
    full_feature_order_df = pd.DataFrame(columns=['feature_name'], 
                                                index=range(1, len(full_feature_order) + 1),
                                                data=full_feature_order)
    full_feature_order_path = full_clustermap_path.parent / (full_clustermap_path.stem + 
                                                                '_feature_order.csv')
    full_feature_order_df.to_csv(full_feature_order_path, header=True, index=True)

    ### Heatmap - features (x-axis columns) ordered by control clustered feature order
    
    pvals_heatmap = anova_table.loc[control_feature_order, 'pvals']
    pvals_heatmap.name = 'P < {}'.format(args.pval_threshold)
    assert all(f in featZ.columns for f in pvals_heatmap.index)
            
    # Plot heatmap (averaged for each sample)
    if len(metadata[grouping_var].unique()) < 250:
        print("\nPlotting barcode heatmap")
        heatmap_path = plot_dir / 'heatmaps' / (grouping_var + '_heatmap.pdf')
        plot_barcode_heatmap(featZ=featZ[control_feature_order], 
                             meta=metadata, 
                             group_by=[grouping_var], 
                             pvalues_series=pvals_heatmap,
                             p_value_threshold=args.pval_threshold,
                             selected_feats=None, # fset if len(fset) > 0 else None
                             saveto=heatmap_path,
                             figsize=[20,30],
                             sns_colour_palette="Pastel1",
                             label_size=10)        
                    
    ##### Principal Components Analysis #####

    pca_dir = plot_dir / 'PCA'
    
    # remove outlier samples from PCA
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

    coloured_strains_pca = [control] + hit_strains[:15]
    coloured_strains_pca = [s for s in coloured_strains_pca if s in metadata[grouping_var].unique()]

    #from tierpsytools.analysis.decomposition import plot_pca
    _ = plot_pca(featZ, metadata, 
                 group_by=grouping_var, 
                 control=control,
                 var_subset=coloured_strains_pca, 
                 saveDir=pca_dir,
                 PCs_to_keep=10,
                 n_feats2print=10,
                 kde=False,
                 sns_colour_palette="plasma",
                 n_dims=2,
                 label_size=8,
                 sub_adj={'bottom':0.13,'left':0.13,'top':0.95,'right':0.88},
                 legend_loc=[1.02,0.6],
                 hypercolor=False)

    # add details of COG category information to metadata 
    # (using hard-coded dict of info from Baba et al. 2006 paper)
    metadata['COG_category'] = metadata['COG_category'].map(COG_category_dict)
    
    # plot pca coloured by Keio COG category    
    _ = plot_pca(featZ, metadata, 
                 group_by='COG_category', 
                 control=None,
                 var_subset=list(metadata['COG_category'].dropna().unique()), 
                 saveDir=pca_dir / 'COG',
                 PCs_to_keep=10,
                 n_feats2print=10,
                 kde=False,
                 n_dims=2,
                 hypercolor=False,
                 label_size=8,
                 figsize=[12,8],
                 sub_adj={'bottom':0.1,'left':0.1,'top':0.95,'right':0.7},
                 legend_loc=[1.02,0.6],
                 sns_colour_palette="plasma")

    # # PCA of lowest 100 pval strains only
    # lowest100_pca_strain_list = [control] + hit_strains_pval
    # lowest100_meta = metadata[metadata[grouping_var].isin(lowest100_pca_strain_list)]
    # lowest100_feat = features.reindex(lowest100_meta.index)
    # lowest100_featZ = lowest100_feat.apply(zscore, axis=0)
    # _ = plot_pca(lowest100_featZ, lowest100_meta,
    #              group_by='COG_category',
    #              control=None,
    #              var_subset=None,
    #              saveDir=pca_dir / 'COG' / 'lowest100',
    #              PCs_to_keep=10,
    #              n_feats2print=10,
    #              kde=False,
    #              sns_colour_palette='plasma',
    #              n_dims=2,
    #              label_size=8,
    #              sub_adj={'bottom':0.13,'left':0.13,'top':0.95,'right':0.88},
    #              legend_loc=[1.02,0.6]
    #              # n_colours=len(lowest100_pca_strain_list)
    #              )

    ##### t-distributed Stochastic Neighbour Embedding #####   
    
    print("\nPerforming tSNE")
    tsne_dir = plot_dir / 'tSNE'
    perplexities = [mean_sample_size] # NB: should be roughly equal to group size    
    _ = plot_tSNE(featZ, metadata,
                  group_by=grouping_var,
                  var_subset=coloured_strains_pca,
                  saveDir=tsne_dir,
                  perplexities=perplexities,
                  figsize=[8,8],
                  label_size=8,
                  marker_size=20,
                  sns_colour_palette="plasma")
    
    print("\nPerforming tSNE")
    tsne_dir = plot_dir / 'tSNE'
    perplexities = [mean_sample_size] # NB: should be roughly equal to group size    
    _ = plot_tSNE(featZ, metadata,
                  group_by='COG_category',
                  var_subset=list(metadata['COG_category'].dropna().unique()),
                  saveDir=tsne_dir / 'COG_category',
                  perplexities=perplexities,
                  figsize=[8,8],
                  label_size=8,
                  marker_size=20,
                  sns_colour_palette="plasma")

    ##### Uniform Manifold Projection #####  
    
    print("\nPerforming UMAP")
    umap_dir = plot_dir / 'UMAP'
    n_neighbours = [mean_sample_size] # NB: should be roughly equal to group size
    min_dist = 0.1 # Minimum distance parameter    
    _ = plot_umap(featZ, metadata,
                  group_by=grouping_var,
                  var_subset=coloured_strains_pca,
                  saveDir=umap_dir,
                  n_neighbours=n_neighbours,
                  min_dist=min_dist,
                  figsize=[8,8],
                  label_size=8,
                  marker_size=20,
                  sns_colour_palette="plasma")   

    return

def selected_strains_timeseries(metadata, 
                                project_dir, 
                                save_dir, 
                                group_by='gene_name',
                                control='wild_type',
                                strain_list=['fepD'],
                                n_wells=96,
                                bluelight_stim_type='bluelight',
                                video_length_seconds=6*60,
                                bluelight_timepoints_seconds=[(60,70),(160,170),(260,270)],
                                motion_modes=['forwards','stationary','backwards'],
                                smoothing=10):
    """ Timeseries plots for standard imaging and bluelight delivery protocol for the initial and 
        confirmation screening of Keio Collection. Bluelight stimulation is delivered after 5 mins
        pre-stimulus, 10 secs stimulus every 60 secs, repeated 3 times (6 mins total), 
        followed by 5 mins post-stimulus (16 minutes total)
    """
            
    if strain_list is None:
        strain_list = list(metadata[group_by].unique())
    else:
        assert isinstance(strain_list, list)
        assert all(s in metadata[group_by].unique() for s in strain_list)
        strain_list = [s for s in strain_list if s != control]
    
    metadata['imgstore_name'] = metadata['imgstore_name_{}'.format(bluelight_stim_type)]
    
    bluelight_frames = [(i*FPS, j*FPS) for (i, j) in bluelight_timepoints_seconds]
    
    # get timeseries for BW
    BW_ts = get_strain_timeseries(metadata[metadata[group_by]==control], 
                                  project_dir=project_dir, 
                                  strain=control,
                                  group_by=group_by,
                                  n_wells=n_wells,
                                  save_dir=Path(save_dir) / 'timeseries' / 'Data' /\
                                      bluelight_stim_type / control)
    
    for strain in tqdm(strain_list):
        col_dict = dict(zip([control, strain], sns.color_palette("pastel", 2)))

        # get timeseries for strain
        strain_ts = get_strain_timeseries(metadata[metadata[group_by]==strain], 
                                          project_dir=project_dir, 
                                          strain=strain,
                                          group_by=group_by,
                                          n_wells=n_wells,
                                          save_dir=Path(save_dir) / 'timeseries' / 'Data' /\
                                              bluelight_stim_type / strain)
    
        for mode in motion_modes:
            print("Plotting timeseries for motion mode %s fraction for %s vs BW.." % (mode, strain))

            plt.close('all')
            fig, ax = plt.subplots(figsize=(12,5), dpi=200)
    
            ax = plot_timeseries_motion_mode(df=BW_ts,
                                             window=smoothing*FPS,
                                             error=True,
                                             mode=mode,
                                             max_n_frames=video_length_seconds*FPS,
                                             title=None,
                                             saveAs=None,
                                             ax=ax,
                                             bluelight_frames=(bluelight_frames if 
                                                               bluelight_stim_type == 'bluelight'
                                                               else None),
                                             colour=col_dict[control],
                                             alpha=0.25)
            
            ax = plot_timeseries_motion_mode(df=strain_ts,
                                             window=smoothing*FPS,
                                             error=True,
                                             mode=mode,
                                             max_n_frames=video_length_seconds*FPS,
                                             title=None,
                                             saveAs=None,
                                             ax=ax,
                                             bluelight_frames=(bluelight_frames if 
                                                               bluelight_stim_type == 'bluelight'
                                                               else None),
                                             colour=col_dict[strain],
                                             alpha=0.25)
        
            xticks = np.linspace(0, video_length_seconds*FPS, int(video_length_seconds/60)+1)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
            ax.set_xlabel('Time (minutes)', fontsize=12, labelpad=10)
            ax.set_ylabel('Fraction {}'.format(mode), fontsize=12, labelpad=10)
            ax.set_title('{0} vs {1}'.format(control, strain), fontsize=12, pad=10)
            ax.legend([control, strain], fontsize=12, frameon=False, loc='best')
            #TODO: plt.subplots_adjust(left=0.01,top=0.9,bottom=0.1,left=0.2)
    
            # save plot
            ts_plot_dir = save_dir / 'timeseries' / 'Plots' / '{0}'.format(strain)
            ts_plot_dir.mkdir(exist_ok=True, parents=True)
            save_path = ts_plot_dir / 'motion_mode_{0}_{1}.pdf'.format(mode, bluelight_stim_type)
            print("Saving to: %s" % save_path)
            plt.savefig(save_path)

    return

#%% MAIN
if __name__ == "__main__":
    tic = time()
    parser = argparse.ArgumentParser(description="Read clean features and metadata and find 'hit' \
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

    # Subset for desired imaging dates
    if args.dates is not None:
        assert type(args.dates) == list
        metadata = metadata.loc[metadata['date_yyyymmdd'].astype(str).isin(args.dates)]
        features = features.reindex(metadata.index)

    compare_strains_keio(features, metadata, args)
    
    # bluelight time-series
    selected_strains_timeseries(metadata, 
                                project_dir=Path(args.project_dir), 
                                save_dir=Path(args.save_dir), 
                                strain_list=SELECTED_STRAIN_LIST,
                                n_wells=96,
                                bluelight_stim_type='bluelight',
                                video_length_seconds=6*60,
                                smoothing=10)
    # prestim time-series
    selected_strains_timeseries(metadata,
                                project_dir=Path(args.project_dir),
                                save_dir=Path(args.save_dir),
                                strain_list=SELECTED_STRAIN_LIST,
                                n_wells=96,
                                bluelight_stim_type='prestim',
                                video_length_seconds=5*60,
                                smoothing=10)

    # poststim time-series
    selected_strains_timeseries(metadata,
                                project_dir=Path(args.project_dir),
                                save_dir=Path(args.save_dir),
                                strain_list=SELECTED_STRAIN_LIST,
                                n_wells=96,
                                bluelight_stim_type='poststim',
                                video_length_seconds=5*60,
                                smoothing=10)
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
