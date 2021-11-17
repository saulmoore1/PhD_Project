#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse Keio antioxidant rescue experiment results

Please run the following scripts beforehand:
1. preprocessing/compile_keio_results.py
2. statistical_testing/perform_rescue_stats.py

Main feature we are using as an indicator for the rescue: 'motion_mode_paused_fraction_bluelight'

@author: sm5911
@date: 13/11/2021
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
from matplotlib import transforms
from scipy.stats import zscore # levene, ttest_ind, f_oneway, kruskal

from read_data.paths import get_save_dir
from read_data.read import load_json, load_topfeats
#from analysis.control_variation import control_variation
#from filter_data.clean_feature_summaries import clean_summary_results, subset_results
from clustering.hierarchical_clustering import plot_clustermap, plot_barcode_heatmap
from feature_extraction.decomposition.pca import plot_pca, remove_outliers_pca
from feature_extraction.decomposition.tsne import plot_tSNE
from feature_extraction.decomposition.umap import plot_umap
from analysis.compare_strains.run_keio_analysis import COG_category_dict
from statistical_testing.perform_keio_stats import df_summary_stats

#%% GLOBALS

JSON_PARAMETERS_PATH = "analysis/20211102_parameters_keio_rescue.json"

STRAIN_COLNAME = 'gene_name'
TREATMENT_COLNAME = 'antioxidant'
CONTROL_STRAIN = 'wild_type'
CONTROL_TREATMENT = 'None'

FEATURE = 'motion_mode_paused_fraction_bluelight'
scale_outliers_box = True

METHOD = 'complete' # 'complete','linkage','average','weighted','centroid'
METRIC = 'euclidean' # 'euclidean','cosine','correlation'

#%% FUNCTIONS

def compare_keio_rescue(features, metadata, args):
    """ Compare Keio single-gene deletion mutants with wild-type BW25113 control under different 
        antioxidant treatment conditions, and look to see if the addition of antioxidants can rescue
        the worm phenotype on these mutant strains, effectively bringing the worms back to wild-type
        behaviour.
        
        - Plot boxplots for each strain, comparing each pairwise antioxidant condition vs the control
          (for all features)
        - 
        
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
            - control_dict : dict
            - n_top_feats : int
            - tierpsy_top_feats_dir (if n_top_feats) : str
            - test : str
            - pval_threshold : float
            - fdr_method : str
            - n_sig_features : int
    """

    assert set(features.index) == set(metadata.index)

    strain_list = list(metadata[STRAIN_COLNAME].unique())
    antioxidant_list = list(metadata[TREATMENT_COLNAME].unique())
    assert CONTROL_STRAIN in strain_list and CONTROL_TREATMENT in antioxidant_list
    strain_list = [CONTROL_STRAIN] + [s for s in sorted(strain_list) if s != CONTROL_STRAIN]
    antioxidant_list = [CONTROL_TREATMENT] + [a for a in sorted(antioxidant_list) if a != CONTROL_TREATMENT]
    n_strains = len(strain_list)
    n_antiox = len(antioxidant_list)
    
    # assert there will be no errors due to case-sensitivity
    assert len(metadata[STRAIN_COLNAME].unique()) == len(metadata[STRAIN_COLNAME].str.upper().unique())
    assert len(metadata[TREATMENT_COLNAME].unique()) == len(metadata[TREATMENT_COLNAME].str.upper().unique())

    # Load Tierpsy Top feature set + subset (columns) for top feats only
    if args.n_top_feats is not None:
        top_feats_path = Path(args.tierpsy_top_feats_dir) / "tierpsy_{}.csv".format(str(args.n_top_feats))
        topfeats = load_topfeats(top_feats_path, add_bluelight=args.align_bluelight, 
                                 remove_path_curvature=True, header=None)
        
        # Drop features that are not in results
        top_feats_list = [feat for feat in list(topfeats) if feat in features.columns]
        features = features[top_feats_list]
    
    assert not features.isna().any().any()
    n_feats = features.shape[1]

    # construct save paths
    save_dir = get_save_dir(args)
    stats_dir =  save_dir / "Stats" / args.fdr_method
    plot_dir = save_dir / "Plots" / args.fdr_method
    plot_dir.mkdir(exist_ok=True, parents=True)

    # Print mean sample size
    sample_size = df_summary_stats(metadata, columns=[STRAIN_COLNAME, TREATMENT_COLNAME])
    ss_savepath = save_dir / 'sample_sizes.csv'
    sample_size.to_csv(ss_savepath, index=False)  

    # add combined treatment column (for heatmap/PCA)
    metadata['treatment_combination'] = [str(s) + '_' + str(a) for s, a in 
                                         zip(metadata[STRAIN_COLNAME], metadata[TREATMENT_COLNAME])]

    # Subset for control data for strain and for treatment
    control_strain_meta = metadata[metadata[STRAIN_COLNAME] == CONTROL_STRAIN]
    control_strain_feat = features.reindex(control_strain_meta.index)
    
    control_antiox_meta = metadata[metadata[TREATMENT_COLNAME] == CONTROL_TREATMENT]
    control_antiox_feat = features.reindex(control_antiox_meta.index)
                
# =============================================================================
#     ##### Control variation ##### 
#     # Clean data after subset - to remove features with zero std
#     control_strain_feat, control_strain_meta = clean_summary_results(control_strain_feat, 
#                                                                      control_strain_meta, 
#                                                                      max_value_cap=False,
#                                                                      imputeNaN=False)  
#     
#     control_antiox_feat, control_antiox_meta = clean_summary_results(control_antiox_feat, 
#                                                                      control_antiox_meta, 
#                                                                      max_value_cap=False,
#                                                                      imputeNaN=False)                   
#     if args.analyse_control:
#         control_variation(control_strain_feat, control_strain_meta, args,
#                           variables=[TREATMENT_COLNAME], n_sig_features=10)
#         control_variation(control_antiox_feat, control_antiox_meta, args,
#                           variables=[STRAIN_COLNAME], n_sig_features=10)                            
# =============================================================================
       
    print("\nComparing %d %ss with %d %s treatments for %d features" %\
          (n_strains, STRAIN_COLNAME, n_antiox, TREATMENT_COLNAME, n_feats))

    t_test = 't-test' if args.test == 'ANOVA' else 'Mann-Whitney' # aka. Wilcoxon rank-sum

    ##### FOR EACH STRAIN #####
    
    for strain in tqdm(strain_list[1:]):
        print("\nPlotting results for %s:" % strain)
        strain_meta = metadata[metadata[STRAIN_COLNAME]==strain]
        strain_feat = features.reindex(strain_meta.index)
        
        # Load ANOVA results for strain
        if not args.use_corrected_pvals:
            anova_strain_path = stats_dir / '{}_uncorrected.csv'.format((args.test + '_' + strain))
        else:
            anova_strain_path = stats_dir / '{}_results.csv'.format((args.test + '_' + strain))
        anova_strain_table = pd.read_csv(anova_strain_path, index_col=0)            
        strain_pvals = anova_strain_table.sort_values(by='pvals', ascending=True)['pvals'] # rank features by p-value
        strain_fset = strain_pvals[strain_pvals < args.pval_threshold].index.to_list()  
        
        # load antioxidant t-test results
        ttest_strain_path = stats_dir / 'pairwise_ttests' / '{}_results.csv'.format(strain + 
                            "_vs_" + CONTROL_STRAIN)
        ttest_strain_table = pd.read_csv(ttest_strain_path, index_col=0)
        strain_pvals_t = ttest_strain_table[[c for c in ttest_strain_table if "pvals_" in c]] 
        strain_pvals_t.columns = [c.split('pvals_')[-1] for c in strain_pvals_t.columns]       
        strain_fset_t = strain_pvals_t[(strain_pvals_t < args.pval_threshold).sum(axis=1) > 0].index.to_list()
           
        # Plot ranked n significant features by t-test for each antioxidant treatment
        ranked_antiox_nsig = (strain_pvals_t < args.pval_threshold).sum(axis=0).sort_values(ascending=False)
        ranked_antiox_nsig_path = plot_dir / ('{}_ranked_number_sigfeats_'.format(strain) + ('uncorrected' 
                                              if args.fdr_method is None else args.fdr_method) + '.png')
        plt.close('all')
        fig, ax = plt.subplots() #figsize=(20,6)
        ax.plot(ranked_antiox_nsig)
        ax.set_xticklabels(ranked_antiox_nsig.index.to_list(), rotation=90, fontsize=5)
        plt.xlabel("Antioxidant (ranked)", fontsize=12, labelpad=10)
        plt.ylabel("Number of significant features", fontsize=12, labelpad=10)
        plt.tight_layout()
        plt.savefig(ranked_antiox_nsig_path, dpi=600)
        
        # Plot ranked lowest pval by t-test for each antioxidant treatment
        ranked_antiox_pval = strain_pvals_t.min(axis=0).sort_values(ascending=True)
        lowest_antiox_pval_path = plot_dir / ('{}_ranked_lowest_pval_'.format(strain) + ('uncorrected' 
                                              if args.fdr_method is None else args.fdr_method) + '.png')
        plt.close('all')
        fig, ax = plt.subplots()
        ax.plot(ranked_antiox_pval)
        plt.axhline(y=args.pval_threshold, c='dimgray', ls='--')
        ax.set_xticklabels(ranked_antiox_nsig.index.to_list(), rotation=90, fontsize=5)
        plt.xlabel("Antioxidant (ranked)", fontsize=12, labelpad=10)
        plt.ylabel("Lowest p-value by t-test", fontsize=12, labelpad=10)
        plt.tight_layout()
        plt.savefig(lowest_antiox_pval_path, dpi=600)
        plt.close()

# =============================================================================
#         print("\nMaking errorbar plots")
#         errorbar_sigfeats(strain_feat, strain_meta, 
#                           group_by=TREATMENT_COLNAME, 
#                           fset=strain_pvals.index, 
#                           control=CONTROL_TREATMENT, 
#                           rank_by='mean',
#                           max_feats2plt=args.n_sig_features, 
#                           figsize=[20,10], 
#                           fontsize=15,
#                           ms=20,
#                           elinewidth=7,
#                           fmt='.',
#                           tight_layout=[0.01,0.01,0.99,0.99],
#                           saveDir=plot_dir / 'errorbar' / strain)
# =============================================================================
                
        if strain != CONTROL_STRAIN:
            # stick together length-wise
            plot_meta = pd.concat([control_strain_meta, strain_meta], ignore_index=True)
            plot_feat = pd.concat([control_strain_feat, strain_feat], ignore_index=True)
            # stick together width-wise
            plot_df = plot_meta.join(plot_feat)
    
            # Plot boxplots for top 10 features comparing strain vs wild-type for each antioxidant treatment            
            for f, feature in enumerate(tqdm(strain_pvals.index)):
                            
                plt.close('all')
                fig, ax = plt.subplots(figsize=(10,8))
                ax = sns.boxplot(x=TREATMENT_COLNAME, y=feature, hue=STRAIN_COLNAME, data=plot_df,
                                 palette='Set3', dodge=True, order=antioxidant_list)
                ax = sns.swarmplot(x=TREATMENT_COLNAME, y=feature, hue=STRAIN_COLNAME, data=plot_df,
                                 color='k', alpha=0.7, size=4, dodge=True, order=antioxidant_list)
                n_labs = len(plot_df[STRAIN_COLNAME].unique())
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[:n_labs], labels[:n_labs], fontsize=15, frameon=False, loc='upper right')
                ax.set_xlabel(TREATMENT_COLNAME, fontsize=15, labelpad=10)
                ax.set_ylabel(feature.replace('_',' '), fontsize=15, labelpad=10)
                
                # scale plot to omit outliers (>2.5*IQR from mean)
                if scale_outliers_box:
                    grouped_strain = plot_df.groupby('antioxidant')
                    y_bar = grouped_strain[feature].median() # median is less skewed by outliers
                    # Computing IQR
                    Q1 = grouped_strain[feature].quantile(0.25)
                    Q3 = grouped_strain[feature].quantile(0.75)
                    IQR = Q3 - Q1
                    plt.ylim(min(y_bar) - 2.5 * max(IQR), max(y_bar) + 2.5 * max(IQR))
                    
                # annotate p-values
                for ii, antiox in enumerate(antioxidant_list):
                    try:
                        p = strain_pvals_t.loc[feature, antiox]
                        text = ax.get_xticklabels()[ii]
                        assert text.get_text() == antiox
                        p_text = 'P < 0.001' if p < 0.001 else 'P = %.3f' % p
                        y = (y_bar[antiox] + 2 * IQR[antiox]) if scale_outliers_box else plot_df[feature].max()
                        h = (max(IQR) / 10) if scale_outliers_box else (y - plot_df[feature].min()) / 50
                        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                        plt.plot([ii-.2, ii-.2, ii+.2, ii+.2], 
                                 [0.8, 0.81, 0.81, 0.8], #[y+h, y+2*h, y+2*h, y+h], 
                                 lw=1.5, c='k', transform=trans)
                        ax.text(ii, 0.82, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
                    except Exception as e:
                        print(e)
                        
                fig_savepath = plot_dir / 'antioxidant_boxplots' / strain / ('{}_'.format(f+1) + 
                                                                             feature + '.png')
                fig_savepath.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(fig_savepath)
            
    ##### FOR EACH ANTIOXIDANT #####
    
    for antiox in antioxidant_list:
        print("\nPlotting results for %s:" % antiox)
        antiox_meta = metadata[metadata[TREATMENT_COLNAME]==antiox]
        antiox_feat = features.reindex(antiox_meta.index)

        # Load ANOVA results for antioxidant
        if not args.use_corrected_pvals:
            anova_antiox_path = stats_dir / '{}_uncorrected.csv'.format((args.test + '_' + antiox))
        else:
            anova_antiox_path = stats_dir / '{}_results.csv'.format((args.test + '_' + antiox))
        anova_antiox_table = pd.read_csv(anova_antiox_path, index_col=0)            
        antiox_pvals = anova_antiox_table.sort_values(by='pvals', ascending=True)['pvals'] # rank features by p-value
        antiox_fset = antiox_pvals[antiox_pvals < args.pval_threshold].index.to_list()  
        
        # Load t-test results
        if not args.use_corrected_pvals:
            ttest_antiox_path = stats_dir / '{}_uncorrected.csv'.format((t_test + '_' + antiox))
        else:
            ttest_antiox_path = stats_dir / '{}_results.csv'.format((t_test + '_' + antiox))
        ttest_antiox_table = pd.read_csv(ttest_antiox_path, index_col=0)
        antiox_pvals_t = ttest_antiox_table[[c for c in ttest_antiox_table if "pvals_" in c]] 
        antiox_pvals_t.columns = [c.split('pvals_')[-1] for c in antiox_pvals_t.columns]       
        antiox_fset_t = antiox_pvals_t[(antiox_pvals_t < args.pval_threshold).sum(axis=1) > 0].index.to_list()
     
        # Plot ranked n significant features by t-test for each strain
        ranked_strain_nsig = (antiox_pvals_t < args.pval_threshold).sum(axis=0).sort_values(ascending=False)
        ranked_strain_nsig_path = plot_dir / ('{}_ranked_number_sigfeats_'.format(antiox) + ('uncorrected' 
                                              if args.fdr_method is None else args.fdr_method) + '.png')
        plt.close('all')
        fig, ax = plt.subplots() #figsize=(20,6)
        ax.plot(ranked_strain_nsig)
        ax.set_xticklabels(ranked_strain_nsig.index.to_list(), rotation=90, fontsize=5)
        plt.xlabel("Strain (ranked)", fontsize=12, labelpad=10)
        plt.ylabel("Number of significant features", fontsize=12, labelpad=10)
        plt.tight_layout()
        plt.savefig(ranked_strain_nsig_path, dpi=600)
        
        # Plot ranked lowest pval by t-test for each antioxidant treatment
        ranked_strain_pval = antiox_pvals_t.min(axis=0).sort_values(ascending=True)
        lowest_strain_pval_path = plot_dir / ('{}_ranked_lowest_pval_'.format(antiox) + ('uncorrected' 
                                              if args.fdr_method is None else args.fdr_method) + '.png')
        plt.close('all')
        fig, ax = plt.subplots()
        ax.plot(ranked_strain_pval)
        plt.axhline(y=args.pval_threshold, c='dimgray', ls='--')
        ax.set_xticklabels(ranked_strain_nsig.index.to_list(), rotation=90, fontsize=5)
        plt.xlabel("Strain (ranked)", fontsize=12, labelpad=10)
        plt.ylabel("Lowest p-value by t-test", fontsize=12, labelpad=10)
        plt.tight_layout()
        plt.savefig(lowest_strain_pval_path, dpi=600)
        plt.close()
        
        # Plot boxplots for top 10 features comparing antioxidant vs None for each strain
        # TODO

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

        # # If no sigfeats, subset for top strains ranked by lowest p-value by t-test for any feature    
        # if len(hit_strains_nsig) == 0:
        #     print("\Saving lowest %d strains ranked by p-value for any feature" % N_LOWEST_PVAL)
        #     write_list_to_file(hit_strains_pval, stats_dir / 'Top100_lowest_pval.txt')
        #     hit_strains = hit_strains_pval
        # elif len(hit_strains_nsig) > 0:
        #     hit_strains = hit_strains_nsig

        # # Individual boxplots of significant features by pairwise t-test (each group vs control)
        # boxplots_sigfeats(features,
        #                   y_class=metadata[grouping_var],
        #                   control=control,
        #                   pvals=pvals_t, 
        #                   z_class=metadata['date_yyyymmdd'],
        #                   feature_set=None,
        #                   saveDir=plot_dir / 'paired_boxplots',
        #                   p_value_threshold=args.pval_threshold,
        #                   drop_insignificant=True if len(hit_strains) > 0 else False,
        #                   max_sig_feats=args.n_sig_features,
        #                   max_strains=N_LOWEST_PVAL if len(hit_strains_nsig) == 0 else None,
        #                   sns_colour_palette="tab10",
        #                   verbose=False)
        
        # if SUBSET_HIT_STRAINS:
        #     strain_list = [control] + hit_strains[:TOP_N_HITS]
        #     print("Subsetting for Top%d hit strains" % (len(strain_list)-1))
        #     features, metadata = subset_results(features, metadata, column=grouping_var,
        #                                         groups=strain_list, verbose=False)   
        # else:
        #     strain_list = list(metadata[grouping_var].unique())
                   
    ##### Hierarchical Clustering Analysis #####

    # Z-normalise control data
    control_strain_featZ = control_strain_feat.apply(zscore, axis=0)
    
    ### Control clustermap
    
    # control data is clustered and feature order is stored and applied to full data
    print("\nPlotting clustermap for %s control" % CONTROL_STRAIN)
    
    control_clustermap_path = plot_dir / 'heatmaps' / 'control_clustermap.pdf'
    cg = plot_clustermap(control_strain_featZ, control_strain_meta,
                         group_by='treatment_combination',
                         method=METHOD, 
                         metric=METRIC,
                         figsize=[20,6],
                         sub_adj={'bottom':0.05,'left':0,'top':1,'right':0.85},
                         saveto=control_clustermap_path,
                         label_size=15,
                         show_xlabels=False)
    # control clustermap with labels
    if args.n_top_feats <= 256:
        control_clustermap_path = plot_dir / 'heatmaps' / 'control_clustermap_label.pdf'
        cg = plot_clustermap(control_strain_featZ, control_strain_meta,
                             group_by='treatment_combination',
                             method=METHOD, 
                             metric=METRIC,
                             figsize=[20,10],
                             sub_adj={'bottom':0.5,'left':0,'top':1,'right':0.85},
                             saveto=control_clustermap_path,
                             label_size=(15,15),
                             show_xlabels=True)

    #col_linkage = cg.dendrogram_col.calculated_linkage
    control_clustered_features = np.array(control_strain_featZ.columns)[cg.dendrogram_col.reordered_ind]

    # ### Full clustermap 
    # TODO: all strains, for each treatment
    # TODO: all treatments, for each strain
    # all strains/treatments together
    
    # Z-normalise data for all strains
    featZ = features.apply(zscore, axis=0)
                    
    ## Save z-normalised values
    # z_stats = featZ.join(hit_metadata[grouping_var]).groupby(by=grouping_var).mean().T
    # z_stats.columns = ['z-mean_' + v for v in z_stats.columns.to_list()]
    # z_stats.to_csv(z_stats_path, header=True, index=None)
    
    # Clustermap of full data   
    print("Plotting all strains clustermap")    
    full_clustermap_path = plot_dir / 'heatmaps' / 'full_clustermap.pdf'
    fg = plot_clustermap(featZ, metadata, 
                         group_by='treatment_combination',
                         row_colours=None,
                         method=METHOD, 
                         metric=METRIC,
                         figsize=[20,30],
                         sub_adj={'bottom':0.01,'left':0,'top':1,'right':0.95},
                         saveto=full_clustermap_path,
                         label_size=8,
                         show_xlabels=False)
    if args.n_top_feats <= 256:
        full_clustermap_path = plot_dir / 'heatmaps' / 'full_clustermap_label.pdf'
        fg = plot_clustermap(featZ, metadata, 
                             group_by='treatment_combination',
                             row_colours=None,
                             method=METHOD, 
                             metric=METRIC,
                             figsize=[20,40],
                             sub_adj={'bottom':0.18,'left':0,'top':1,'right':0.95},
                             saveto=full_clustermap_path,
                             label_size=(15,10),
                             show_xlabels=True)
    
    # clustered feature order for all strains
    _ = np.array(featZ.columns)[fg.dendrogram_col.reordered_ind]
    
    pvals_heatmap = anova_strain_table.loc[control_clustered_features, 'pvals']
    pvals_heatmap.name = 'P < {}'.format(args.pval_threshold)

    assert all(f in featZ.columns for f in pvals_heatmap.index)
            
    # Plot heatmap (averaged for each sample)
    if len(metadata['treatment_combination'].unique()) < 250:
        print("\nPlotting barcode heatmap")
        heatmap_path = plot_dir / 'heatmaps' / 'full_heatmap.pdf'
        plot_barcode_heatmap(featZ=featZ[control_clustered_features], 
                             meta=metadata, 
                             group_by='treatment_combination', 
                             pvalues_series=pvals_heatmap,
                             p_value_threshold=args.pval_threshold,
                             selected_feats=None, # fset if len(fset) > 0 else None
                             saveto=heatmap_path,
                             figsize=[20,30],
                             sns_colour_palette="Pastel1",
                             label_size=15,
                             sub_adj={'top':0.95,'bottom':0.01,'left':0.15,'right':0.92})        
                    
    # ##### Principal Components Analysis #####

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

    # plot PCA 
    # Total of 50 treatment combinations, so plot fes/fepD/entA/wild_type only
    treatment_list = sorted(list(metadata['treatment_combination'].unique()))
    treatment_subset = [i for i in treatment_list if i.split('_')[0] in ['fes','fepD','entA','wild']]
    _ = plot_pca(featZ, metadata, 
                 group_by='treatment_combination', 
                 control=CONTROL_STRAIN + '_' + CONTROL_TREATMENT,
                 var_subset=treatment_subset, 
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

    ##### t-distributed Stochastic Neighbour Embedding #####   
    mean_sample_size = int(sample_size['n_samples'].mean())

    print("\nPerforming tSNE")
    tsne_dir = plot_dir / 'tSNE'
    perplexities = [mean_sample_size] # NB: should be roughly equal to group size    
    _ = plot_tSNE(featZ, metadata,
                  group_by='treatment_combination',
                  var_subset=treatment_subset,
                  saveDir=tsne_dir,
                  perplexities=perplexities,
                  figsize=[8,8],
                  label_size=7,
                  marker_size=30,
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
                  label_size=7,
                  marker_size=30,
                  sns_colour_palette="plasma")

    ##### Uniform Manifold Projection #####  
    
    print("\nPerforming UMAP")
    umap_dir = plot_dir / 'UMAP'
    n_neighbours = [mean_sample_size] # NB: should be roughly equal to group size
    min_dist = 0.1 # Minimum distance parameter    
    _ = plot_umap(featZ, metadata,
                  group_by='treatment_combination',
                  var_subset=treatment_subset,
                  saveDir=umap_dir,
                  n_neighbours=n_neighbours,
                  min_dist=min_dist,
                  figsize=[8,8],
                  label_size=7,
                  marker_size=30,
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

    # Subset for desired imaging dates
    
    if args.dates is not None:
        assert type(args.dates) == list
        metadata = metadata.loc[metadata['date_yyyymmdd'].astype(str).isin(args.dates)]
        features = features.reindex(metadata.index)

    compare_keio_rescue(features, metadata, args)
    
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
