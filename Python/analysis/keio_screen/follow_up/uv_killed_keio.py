#!/usr/bin/env python3# -*- coding: utf-8 -*-"""Analyse Keio Dead Bacteria experiment results- Follow-up analysis of Keio hit strains that have been UV-irradiated to kill the bacteria and test  whether the bacteria still modify worm behaviour when dead - Does the effect require a biotic interaction between the bacteria and the worm?   Is the behaviour-modifying molecule only released by the bacteria in response to being ingested?Please run the following scripts beforehand:1. preprocessing/compile_keio_results.py2. statistical_testing/perform_dead_keio_stats.pyMain feature we are using as an indicator for the rescue: 'motion_mode_paused_fraction_bluelight'@author: sm5911@date: 19/11/2021"""#%% IMPORTSimport argparseimport numpy as npimport pandas as pdimport seaborn as snsfrom time import timefrom tqdm import tqdmfrom pathlib import Pathfrom matplotlib import pyplot as pltfrom matplotlib import transforms, patchesfrom scipy.stats import zscore # levene, ttest_ind, f_oneway, kruskalfrom read_data.paths import get_save_dirfrom read_data.read import load_json #load_topfeats#from analysis.control_variation import control_variation#from filter_data.clean_feature_summaries import clean_summary_results, subset_resultsfrom clustering.hierarchical_clustering import plot_clustermap, plot_barcode_heatmapfrom feature_extraction.decomposition.pca import plot_pca, remove_outliers_pcafrom feature_extraction.decomposition.tsne import plot_tSNEfrom feature_extraction.decomposition.umap import plot_umapfrom analysis.keio_screen.initial.run_keio_analysis import selected_strains_timeseriesfrom tierpsytools.preprocessing.filter_data import select_feat_set#%% GLOBALSJSON_PARAMETERS_PATH = "analysis/20211109_parameters_keio_dead.json"STRAIN_COLNAME = 'gene_name'CONTROL_STRAIN = 'wild_type'TREATMENT_COLNAME = 'dead' # boolean values mapped to: True='dead', False='live'CONTROL_TREATMENT = 'live'CONTROL_TREATMENT = FalseSTRAIN_SUBSET = ['wild_type','fepD']FEATURE = 'motion_mode_forward_fraction_bluelight'scale_outliers_box = TrueMETHOD = 'complete' # 'complete','linkage','average','weighted','centroid'METRIC = 'euclidean' # 'euclidean','cosine','correlation'N_WELLS = 6FPS = 25feature_set=['motion_mode_forward_fraction_prestim',             'motion_mode_forward_fraction_bluelight',             'motion_mode_forward_fraction_poststim',             'speed_50th_prestim',             'speed_50th_bluelight',             'speed_50th_poststim',             'curvature_midbody_norm_abs_50th_prestim',             'curvature_midbody_norm_abs_50th_bluelight',             'curvature_midbody_norm_abs_50th_poststim']    #%% FUNCTIONSdef compare_keio_dead(features, metadata, args, feature_set=None):    """ Compare live vs dead Keio single-gene deletion mutants with the respective wild-type         BW25113 control strain, and look to see if whether UV-killing the bacteria affects their         influence on worm behaviour.                - Boxplots for each feature, comparing each strain vs control, for live and dead bacteria          separately        - Boxplots for each feature, comparing live vs dead for each strain                Inputs        ------        features, metadata : pd.DataFrame            Matching features summaries and metadata                args : Object             Python object with the following attributes:            - drop_size_features : bool            - norm_features_only : bool            - percentile_to_use : str            - remove_outliers : bool            - omit_strains : list            - control_dict : dict            - n_top_feats : int            - tierpsy_top_feats_dir (if n_top_feats) : str            - test : str            - pval_threshold : float            - fdr_method : str            - n_sig_features : int    """    assert set(features.index) == set(metadata.index)    strain_list = list(metadata['gene_name'].unique())    assert CONTROL_STRAIN in strain_list    strain_list = [CONTROL_STRAIN] + [s for s in sorted(strain_list) if s != CONTROL_STRAIN]        # assert there will be no errors due to case-sensitivity    assert len(metadata['gene_name'].unique()) == len(metadata['gene_name'].str.upper().unique())    # Load Tierpsy feature set + subset (columns) for selected features only    if args.n_top_feats is not None:        features = select_feat_set(features, 'tierpsy_{}'.format(args.n_top_feats), append_bluelight=True)        features = features[[f for f in features.columns if 'path_curvature' not in f]]    elif feature_set is not None:        features = features[features.columns[features.columns.isin(feature_set)]]        assert not features.isna().any().any()    # construct save paths    save_dir = get_save_dir(args)    stats_dir =  save_dir / "Stats" / args.fdr_method    plot_dir = save_dir / "Plots" / args.fdr_method    plot_dir.mkdir(exist_ok=True, parents=True)           # dates = list(metadata['date_yyyymmdd'].unique())    # date_lut = dict(zip(dates, sns.color_palette('plasma', len(dates))))    ### LIVE BACTERIA        # Load t-test results for live bacteria    ttest_live_path = stats_dir / ('t-test_live_' + ('results.csv' if args.use_corrected_pvals                                                      else 'uncorrected.csv'))      ttest_live_table = pd.read_csv(ttest_live_path, index_col=0)    live_pvals = ttest_live_table[[c for c in ttest_live_table if "pvals_" in c]]     live_pvals.columns = [c.split('pvals_')[-1] for c in live_pvals.columns]           #live_fset = live_pvals[(live_pvals < args.pval_threshold).sum(axis=1) > 0].index.to_list()    live_metadata = metadata[metadata[TREATMENT_COLNAME]=='live']        live_features = features.reindex(live_metadata.index)    # Boxplots for each feature, comparing each strain vs control    for f, feature in enumerate(tqdm(features.columns)): #live_fset              live_df = live_metadata[[STRAIN_COLNAME, TREATMENT_COLNAME,                                  'date_yyyymmdd']].join(live_features[feature])        plt.close('all')        fig, ax = plt.subplots(figsize=(14,8))        ax = sns.boxplot(x=STRAIN_COLNAME,                          y=feature,                          order=strain_list,                          data=live_df,                         palette='plasma')                    ax = sns.swarmplot(x=STRAIN_COLNAME,                            y=feature,                            order=strain_list,                           hue='date_yyyymmdd',                            dodge=False,                           data=live_df,                           palette='Greys',                            alpha=0.7,                            size=4)                ax.set_xlabel('Strain (live)', fontsize=12, labelpad=15)        ax.set_ylabel(feature.replace('_',' '), fontsize=12, labelpad=15)            # add custom legend        # patch_list = []        # for s, c in zip(strain_list, sns.color_palette('Set3', n_colors=len(strain_list))):        #     patch = patches.Patch(color=c, label=s)        #     patch_list.append(patch)        # plt.tight_layout(rect=[0.04, 0, 0.84, 0.96])        # ax.legend(handles=patch_list, labels=strain_list, loc=(1.02, 0.5),\        #            borderaxespad=0.4, frameon=False, fontsize=15)                        # scale plot to omit outliers (>2.5*IQR from mean)        if scale_outliers_box:            grouped_strain = live_df.groupby(STRAIN_COLNAME)            y_bar = grouped_strain[feature].median() # median is less skewed by outliers            # Computing IQR            Q1 = grouped_strain[feature].quantile(0.25)            Q3 = grouped_strain[feature].quantile(0.75)            IQR = Q3 - Q1            plt.ylim(min(y_bar) - 2 * max(IQR), max(y_bar) + 2.5 * max(IQR))                    # annotate p-values        for ii, strain in enumerate(strain_list[1:]):            p = live_pvals.loc[feature, strain]            text = ax.get_xticklabels()[ii+1]            assert text.get_text() == strain            p_text = 'P < 0.001' if p < 0.001 else 'P = %.3f' % p            #y = (y_bar[strain] + 2 * IQR[strain]) if scale_outliers_box else plot_df[feature].max()            #h = (max(IQR) / 10) if scale_outliers_box else (y - plot_df[feature].min()) / 50            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)            # plt.plot([ii-.2, ii-.2, ii+.2, ii+.2], [y+h, y+2*h, y+2*h, y+h],             #          lw=1.5, c='k', transform=trans)            ax.text(ii+1, 1.02, p_text, fontsize=9, ha='center', va='bottom', transform=trans)                        fig_savepath = plot_dir / 'live_bacteria_boxplots' / ('{}_'.format(f+1) + feature + '.png')        fig_savepath.parent.mkdir(parents=True, exist_ok=True)        plt.savefig(fig_savepath)                        ### DEAD BACTERIA        # Load t-test results for dead bacteria    ttest_dead_path = stats_dir / ('t-test_dead_' + ('results.csv' if args.use_corrected_pvals                                                      else 'uncorrected.csv'))      ttest_dead_table = pd.read_csv(ttest_dead_path, index_col=0)    dead_pvals = ttest_dead_table[[c for c in ttest_dead_table if "pvals_" in c]]     dead_pvals.columns = [c.split('pvals_')[-1] for c in dead_pvals.columns]           #dead_fset = dead_pvals[(dead_pvals < args.pval_threshold).sum(axis=1) > 0].index.to_list()    dead_metadata = metadata[metadata[TREATMENT_COLNAME]=='dead']        dead_features = features.reindex(dead_metadata.index)    # Boxplots for each feature, comparing each strain vs control    for f, feature in enumerate(tqdm(features.columns)): #live_fset                dead_df = dead_metadata[[STRAIN_COLNAME, TREATMENT_COLNAME,                                  'date_yyyymmdd']].join(dead_features[feature])                plt.close('all')        fig, ax = plt.subplots(figsize=(14,8))        ax = sns.boxplot(x=STRAIN_COLNAME,                          y=feature,                          order=strain_list,                          data=dead_df,                         palette='plasma')        ax = sns.swarmplot(x=STRAIN_COLNAME,                            y=feature,                            order=strain_list,                           hue='date_yyyymmdd',                            dodge=False,                           data=dead_df,                           palette='Greys',                            alpha=0.7,                            size=4)        ax.set_xlabel('Strain (UV-killed)', fontsize=12, labelpad=15)        ax.set_ylabel(feature.replace('_',' '), fontsize=12, labelpad=15)                # scale plot to omit outliers (>2.5*IQR from mean)        if scale_outliers_box:            grouped_strain = dead_df.groupby(STRAIN_COLNAME)            y_bar = grouped_strain[feature].median() # median is less skewed by outliers            # Computing IQR            Q1 = grouped_strain[feature].quantile(0.25)            Q3 = grouped_strain[feature].quantile(0.75)            IQR = Q3 - Q1            plt.ylim(min(y_bar) - 2 * max(IQR), max(y_bar) + 2.5 * max(IQR))                    # annotate p-values        for ii, strain in enumerate(strain_list[1:]):            p = dead_pvals.loc[feature, strain]            text = ax.get_xticklabels()[ii+1]            assert text.get_text() == strain            p_text = 'P < 0.001' if p < 0.001 else 'P = %.3f' % p            #y = (y_bar[strain] + 2 * IQR[strain]) if scale_outliers_box else plot_df[feature].max()            #h = (max(IQR) / 10) if scale_outliers_box else (y - plot_df[feature].min()) / 50            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)            # plt.plot([ii-.2, ii-.2, ii+.2, ii+.2], [y+h, y+2*h, y+2*h, y+h],             #          lw=1.5, c='k', transform=trans)            ax.text(ii+1, 1.02, p_text, fontsize=9, ha='center', va='bottom', transform=trans)                        fig_savepath = plot_dir / 'dead_bacteria_boxplots' / ('{}_'.format(f+1) + feature + '.png')        fig_savepath.parent.mkdir(parents=True, exist_ok=True)        plt.savefig(fig_savepath)    ### LIVE VS DEAD        hue_order = ['live','dead']    # Boxplots for each feature, comparing each strain vs control    for f, feature in enumerate(tqdm(features.columns)): #live_fset                plot_df = metadata[[STRAIN_COLNAME, TREATMENT_COLNAME,                             'date_yyyymmdd']].join(features[feature])        plt.close('all')        fig, ax = plt.subplots(figsize=(20,8))        ax = sns.boxplot(x=STRAIN_COLNAME,                          y=feature,                          order=strain_list,                          hue=TREATMENT_COLNAME,                          hue_order=hue_order,                         dodge=True,                         data=plot_df,                         palette='tab10')                        # for date, colour in date_lut.items():        #     date_df = plot_df[plot_df['date_yyyymmdd']==date]        ax = sns.swarmplot(x=STRAIN_COLNAME,                            y=feature,                            order=strain_list,                           hue=TREATMENT_COLNAME,                            hue_order=hue_order,                           dodge=True,                           data=plot_df, # date_df                           color='k', # colour                           alpha=0.7,                            size=4)              # add custom legend        patch_list = []        for l, c in zip(hue_order, sns.color_palette('tab10', n_colors=len(hue_order))):            patch = patches.Patch(color=c, label=l)            patch_list.append(patch)        plt.tight_layout(rect=[0.04, 0.02, 0.9, 0.96])        ax.legend(handles=patch_list, labels=hue_order, loc=(1.01, 0.9),\                    borderaxespad=0.4, frameon=False, fontsize=15)        ax.set_xlabel('Strain', fontsize=12, labelpad=15)        ax.set_ylabel(feature.replace('_',' '), fontsize=12, labelpad=15)                # scale plot to omit outliers (>2.5*IQR from mean)        if scale_outliers_box:            grouped_strain = plot_df.groupby(STRAIN_COLNAME)            y_bar = grouped_strain[feature].median() # median is less skewed by outliers            # compute IQR            Q1 = grouped_strain[feature].quantile(0.25)            Q3 = grouped_strain[feature].quantile(0.75)            IQR = Q3 - Q1            plt.ylim(min(y_bar) - 3 * max(IQR), max(y_bar) + 3 * max(IQR))                    # annotate p-values        for ii, strain in enumerate(strain_list):            # load t-test live vs dead pvals for strain            ttest_dead_path = stats_dir / ('t-test_{}_'.format(strain) + ('results.csv' if                                            args.use_corrected_pvals else 'uncorrected.csv'))            ttest_dead_table = pd.read_csv(ttest_dead_path, index_col=0)            dead_pvals = ttest_dead_table[[c for c in ttest_dead_table if "pvals_" in c]]             p = dead_pvals.loc[feature, 'pvals_True']            text = ax.get_xticklabels()[ii]            assert text.get_text() == strain            p_text = 'P < 0.001' if p < 0.001 else 'P = %.3f' % p            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)            plt.plot([ii-.2, ii-.2, ii+.2, ii+.2],                      [0.98, 0.99, 0.99, 0.98],                     lw=1.5, c='k', transform=trans)            ax.text(ii, 1.01, p_text, fontsize=9, ha='center', va='bottom', transform=trans)                        fig_savepath = plot_dir / 'boxplots_live_vs_dead' / ('{}_'.format(f+1) + feature + '.png')        fig_savepath.parent.mkdir(parents=True, exist_ok=True)        plt.savefig(fig_savepath)        ### Hierarchical Clustering    control_strain_meta = metadata[metadata[STRAIN_COLNAME]==CONTROL_STRAIN]    control_strain_feat = features.reindex(control_strain_meta.index)        # Z-normalise control data    control_strain_featZ = control_strain_feat.apply(zscore, axis=0)        ### Control clustermap        print("\nPlotting clustermap (date) for %s control" % CONTROL_STRAIN)        control_clustermap_path = plot_dir / 'heatmaps' / 'control_date_clustermap.pdf'    cg = plot_clustermap(control_strain_featZ, control_strain_meta,                         group_by='date_yyyymmdd',                         method=METHOD,                          metric=METRIC,                         figsize=[20,6],                         sub_adj={'bottom':0.05,'left':0,'top':1,'right':0.85},                         saveto=control_clustermap_path,                         label_size=15,                         show_xlabels=False)    # control clustermap with labels    if args.n_top_feats is not None and args.n_top_feats <= 256:        control_clustermap_path = plot_dir / 'heatmaps' / 'control_date_clustermap_label.pdf'        cg = plot_clustermap(control_strain_featZ, control_strain_meta,                             group_by='date_yyyymmdd',                             method=METHOD,                              metric=METRIC,                             figsize=[20,10],                             sub_adj={'bottom':0.7,'left':0,'top':1,'right':0.85},                             saveto=control_clustermap_path,                             label_size=(15,15),                             show_xlabels=True)        # control data is clustered and feature order is stored and applied to full data    print("\nPlotting clustermap (treatment) for %s control" % CONTROL_STRAIN)    control_clustermap_path = plot_dir / 'heatmaps' / 'control_clustermap.pdf'    cg = plot_clustermap(control_strain_featZ, control_strain_meta,                         group_by='treatment_combination',                         method=METHOD,                          metric=METRIC,                         figsize=[20,6],                         sub_adj={'bottom':0.05,'left':0,'top':1,'right':0.85},                         saveto=control_clustermap_path,                         label_size=20,                         show_xlabels=False)    # control clustermap with labels    if args.n_top_feats is not None and args.n_top_feats <= 256:        control_clustermap_path = plot_dir / 'heatmaps' / 'control_clustermap_label.pdf'        cg = plot_clustermap(control_strain_featZ, control_strain_meta,                             group_by='treatment_combination',                             method=METHOD,                              metric=METRIC,                             figsize=[20,10],                             sub_adj={'bottom':0.7,'left':0,'top':1,'right':0.85},                             saveto=control_clustermap_path,                             label_size=20,                             show_xlabels=True)    #col_linkage = cg.dendrogram_col.calculated_linkage    control_clustered_features = np.array(control_strain_featZ.columns)[cg.dendrogram_col.reordered_ind]    ### Full clustermap         # Z-normalise data for all strains    featZ = features.apply(zscore, axis=0)                        # Save z-normalised values    z_stats_path = stats_dir / 'z-normalised_values.csv'    z_stats = featZ.join(metadata['treatment_combination']).groupby(by='treatment_combination').mean().T    z_stats.columns = ['z-mean_' + v for v in z_stats.columns.to_list()]    z_stats.to_csv(z_stats_path, header=True, index=None)        # Clustermap of full data       print("Plotting all strains clustermap")        full_clustermap_path = plot_dir / 'heatmaps' / 'full_clustermap.pdf'    fg = plot_clustermap(featZ, metadata,                          group_by='treatment_combination',                         row_colours=None,                         method=METHOD,                          metric=METRIC,                         figsize=[20,30],                         sub_adj={'bottom':0.01,'left':0,'top':1,'right':0.95},                         saveto=full_clustermap_path,                         label_size=20,                         show_xlabels=False)    if args.n_top_feats is not None and args.n_top_feats <= 256:        full_clustermap_path = plot_dir / 'heatmaps' / 'full_clustermap_label.pdf'        fg = plot_clustermap(featZ, metadata,                              group_by='treatment_combination',                             row_colours=None,                             method=METHOD,                              metric=METRIC,                             figsize=[20,40],                             sub_adj={'bottom':0.18,'left':0,'top':1,'right':0.95},                             saveto=full_clustermap_path,                             label_size=20,                             show_xlabels=True)        # clustered feature order for all strains    _ = np.array(featZ.columns)[fg.dendrogram_col.reordered_ind]        # load ANOVA results for significant features    test_path = stats_dir / 'ANOVA_results.csv'    anova_table = pd.read_csv(test_path, header=0, index_col=0)    pvals_heatmap = anova_table.loc[control_clustered_features, 'pvals']    pvals_heatmap.name = 'P < {}'.format(args.pval_threshold)        assert all(f in featZ.columns for f in pvals_heatmap.index)                # Plot heatmap (averaged for each sample)    if len(metadata['treatment_combination'].unique()) < 250:        print("\nPlotting barcode heatmap")        heatmap_path = plot_dir / 'heatmaps' / 'full_heatmap.pdf'        plot_barcode_heatmap(featZ=featZ[control_clustered_features],                              meta=metadata,                              group_by='treatment_combination',                              pvalues_series=pvals_heatmap,                             p_value_threshold=args.pval_threshold,                             selected_feats=None, # fset if len(fset) > 0 else None                             saveto=heatmap_path,                             figsize=[20,30],                             sns_colour_palette="Pastel1",                             label_size=20,                             sub_adj={'bottom':0.01,'left':0.15,'top':0.95,'right':0.92})            ### Principal Components Analysis    pca_dir = plot_dir / 'PCA'        treatment_subset = [CONTROL_STRAIN,'fes','fepD','fepB']    treatment_list = []    for s in sorted(treatment_subset):        treatment_list.append(s + '_live')        treatment_list.append(s + '_dead')    featZ = features.apply(zscore, axis=0) # normalise data    # plot PCA - Total of 50 treatment combinations, colour points for fes/fepD/entA/wild_type only    _ = plot_pca(featZ, metadata,                  group_by='treatment_combination',                  control=str(CONTROL_STRAIN) + '_' + str(CONTROL_TREATMENT),                 var_subset=treatment_list,                  saveDir=pca_dir,                 PCs_to_keep=10,                 n_feats2print=10,                 kde=False,                 sns_colour_palette="Paired",                 n_dims=2,                 label_size=10,                 sub_adj={'bottom':0.13,'left':0.13,'top':0.95,'right':0.82},                 legend_loc=[1.02,0.73],                 hypercolor=False,                 s=30,                 alpha=0.7)    mean_sample_size = round(metadata.groupby('treatment_combination')['well_name'].count().mean())    print("Mean sample size per treatment: %d" % mean_sample_size)        # t-distributed Stochastic Neighbour Embedding    tsne_dir = plot_dir / 'tSNE'    perplexities = [mean_sample_size] # NB: should be roughly equal to group size        _ = plot_tSNE(featZ, metadata,                  group_by='treatment_combination',                  var_subset=treatment_list,                  saveDir=tsne_dir,                  perplexities=perplexities,                  figsize=[10,10],                  label_size=10,                  sns_colour_palette="Paired",                  s=100,                  alpha=0.7)    # Uniform Manifold Projection    umap_dir = plot_dir / 'UMAP'    n_neighbours = [mean_sample_size] # NB: should be roughly equal to group size    min_dist = 0.1 # Minimum distance parameter        _ = plot_umap(featZ, metadata,                  group_by='treatment_combination',                  var_subset=treatment_list,                  saveDir=umap_dir,                  n_neighbours=n_neighbours,                  min_dist=min_dist,                  figsize=[8,8],                  label_size=10,                  sns_colour_palette="Paired",                  s=30,                  alpha=0.7)        # remove 'other' strains & re-run PCA for treatment_subset only       meta = metadata[metadata[STRAIN_COLNAME].isin(treatment_subset)]    feat = features.reindex(meta.index)    pca_dir = plot_dir / 'PCA_selected'            # remove outlier samples from PCA for selected strains    outlier_path = pca_dir / 'mahalanobis_outliers.pdf'    feat, inds = remove_outliers_pca(df=feat, saveto=outlier_path)    meta = meta.reindex(feat.index) # reindex metadata    featZ = feat.apply(zscore, axis=0) # re-normalise data    treatment_list = sorted(list(meta['treatment_combination'].unique()))            # plot PCA - Total of 50 treatment combinations, colour points for fes/fepD/entA/wild_type only    _ = plot_pca(featZ, meta,                  group_by='treatment_combination',                  control=str(CONTROL_STRAIN) + '_' + str(CONTROL_TREATMENT),                 var_subset=treatment_list,                  saveDir=pca_dir,                 PCs_to_keep=10,                 n_feats2print=10,                 kde=False,                 sns_colour_palette="Paired",                 n_dims=2,                 label_size=10,                 sub_adj={'bottom':0.13,'left':0.13,'top':0.95,'right':0.82},                 legend_loc=[1.02,0.73],                 hypercolor=False,                 s=30,                 alpha=0.7)        mean_sample_size = round(metadata.groupby('treatment_combination')['well_name'].count().mean())    print("Mean sample size per treatment: %d" % mean_sample_size)        # t-distributed Stochastic Neighbour Embedding    tsne_dir = plot_dir / 'tSNE_selected'    perplexities = [mean_sample_size] # NB: should be roughly equal to group size        _ = plot_tSNE(featZ, meta,                  group_by='treatment_combination',                  var_subset=treatment_list,                  saveDir=tsne_dir,                  perplexities=perplexities,                  figsize=[10,10],                  label_size=10,                  sns_colour_palette="Paired",                  s=100,                  alpha=0.7)    # Uniform Manifold Projection    umap_dir = plot_dir / 'UMAP_selected'    n_neighbours = [mean_sample_size] # NB: should be roughly equal to group size    min_dist = 0.1 # Minimum distance parameter        _ = plot_umap(featZ, meta,                  group_by='treatment_combination',                  var_subset=treatment_list,                  saveDir=umap_dir,                  n_neighbours=n_neighbours,                  min_dist=min_dist,                  figsize=[8,8],                  label_size=10,                  sns_colour_palette="Paired",                  s=30,                  alpha=0.7)        return#%% MAINif __name__ == "__main__":    tic = time()    parser = argparse.ArgumentParser(description="Read clean features and etadata and find 'hit' \                                                  Keio knockout strains that alter worm behaviour")    parser.add_argument('-j', '--json', help="Path to JSON parameters file",                         default=JSON_PARAMETERS_PATH, type=str)    parser.add_argument('--features_path', help="Path to feature summaries file",                         default=None, type=str)    parser.add_argument('--metadata_path', help="Path to metadata file",                         default=None, type=str)    args = parser.parse_args()    FEATURES_PATH = args.features_path    METADATA_PATH = args.metadata_path    args = load_json(args.json)        if FEATURES_PATH is None:        FEATURES_PATH = Path(args.save_dir) / 'features.csv'    if METADATA_PATH is None:        METADATA_PATH = Path(args.save_dir) / 'metadata.csv'            # Read clean feature summaries + metadata    print("Loading metadata and feature summary results...")    features = pd.read_csv(FEATURES_PATH)    metadata = pd.read_csv(METADATA_PATH, dtype={'comments':str, 'source_plate_id':str})    # Subset for desired imaging dates       if args.dates is not None:        assert type(args.dates) == list        metadata = metadata.loc[metadata['date_yyyymmdd'].astype(str).isin(args.dates)]        features = features.reindex(metadata.index)    # add combined treatment column (for heatmap/PCA/timeseries)    metadata['dead'] = ['dead' if i else 'live' for i in metadata['dead']]    metadata['treatment_combination'] = metadata[STRAIN_COLNAME] + '_' + metadata[TREATMENT_COLNAME]    compare_keio_dead(features, metadata, args, feature_set=feature_set)        if STRAIN_SUBSET is not None:        metadata = metadata[metadata['gene_name'].isin(STRAIN_SUBSET)]    timeseries_control = CONTROL_STRAIN + '_' + CONTROL_TREATMENT            selected_strains_timeseries(metadata,                                project_dir=Path(args.project_dir),                                 save_dir=Path(args.save_dir) / 'timeseries',                                 strain_list=None,                                group_by='treatment_combination',                                control=timeseries_control,                                n_wells=96,                                bluelight_stim_type='bluelight',                                video_length_seconds=360,                                bluelight_timepoints_seconds=[(60, 70),(160, 170),(260, 270)],                                motion_modes=['forwards','paused','backwards'],                                smoothing=10)        toc = time()    print("\nDone in %.1f seconds (%.1f minutes)" % (toc - tic, (toc - tic) / 60))  