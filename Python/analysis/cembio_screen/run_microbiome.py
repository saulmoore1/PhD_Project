#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of the C. elegans microbiome screen data collected in February 2020
- No peptone NGM
- 1, 3, 5 and 24 hours on food
- No blue light

@author: sm5911
@date: 13/12/2022

"""

#%% Imports

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import zscore
from matplotlib import pyplot as plt
from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from tierpsytools.preprocessing.filter_data import select_feat_set
from clustering.hierarchical_clustering import plot_clustermap, plot_barcode_heatmap
from write_data.write import write_list_to_file
from visualisation.plotting_helper import sig_asterix, errorbar_sigfeats, boxplots_sigfeats
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from time_series.plot_timeseries import plot_timeseries_feature #selected_strains_timeseries

#%% Globals

PROJECT_DIR = "/Volumes/behavgenom$/Saul/MicrobiomeScreen96WP"
SAVE_DIR = "/Users/sm5911/Documents/Microbiome_Screen"
#/Volumes/behavgenom$/Saul/MicrobiomeTests96WP

N_WELLS = 96
IMAGING_DATES = ['20200213','20200221']

NAN_THRESH_SAMPLE = 0.8
NAN_THRESH_FEATURE = 0.05
MIN_SKEL_PER_VIDEO = None
MIN_SKEL_SUM = None

FEATURE_SET = 256 #['speed_50th', 'angular_velocity_abs_50th', 'motion_mode_forward_fraction'] #256

P_VALUE_THRESHOLD = 0.05
FDR_METHOD = 'fdr_by'
N_LOWEST_PVAL = 10
MAX_N_FEATS = 10

METHOD = 'complete' # 'complete','linkage','average','weighted','centroid'
METRIC = 'euclidean' # 'euclidean','cosine','correlation'

#%% Functions

def microbiome_stats(metadata,
                     features,
                     group_by='food_type',
                     control='OP50',
                     save_dir=None,
                     feature_set=None,
                     pvalue_threshold=0.05,
                     fdr_method='fdr_by'):
    
    # check case-sensitivity
    assert len(metadata[group_by].unique()) == len(metadata[group_by].str.upper().unique())
    
    if feature_set is not None:
        feature_set = [feature_set] if isinstance(feature_set, str) else feature_set
        assert isinstance(feature_set, list)
        assert(all(f in features.columns for f in feature_set))
    else:
        feature_set = features.columns.tolist()
        
    features = features[feature_set].reindex(metadata.index)

    # print mean sample size
    sample_size = metadata.groupby(group_by).count()
    print("Mean sample size of %s: %d" % (group_by, int(sample_size[sample_size.columns[-1]].mean())))

    n = len(metadata[group_by].unique())
        
    fset = []
    if n > 2:
   
        # Perform ANOVA - is there variation among strains at each window?
        anova_path = Path(save_dir) / 'ANOVA' / 'ANOVA_results.csv'
        anova_path.parent.mkdir(parents=True, exist_ok=True)

        stats, pvals, reject = univariate_tests(X=features, 
                                                y=metadata[group_by], 
                                                control=control, 
                                                test='ANOVA',
                                                comparison_type='multiclass',
                                                multitest_correction=fdr_method,
                                                alpha=pvalue_threshold,
                                                n_permutation_test=None)

        # get effect sizes
        effect_sizes = get_effect_sizes(X=features,
                                        y=metadata[group_by],
                                        control=control,
                                        effect_type=None,
                                        linked_test='ANOVA')

        # compile + save results
        test_results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
        test_results.columns = ['stats','effect_size','pvals','reject']     
        test_results['significance'] = sig_asterix(test_results['pvals'])
        test_results = test_results.sort_values(by=['pvals'], ascending=True) # rank by p-value
        test_results.to_csv(anova_path, header=True, index=True)

        # use reject mask to find significant feature set
        fset = pvals.loc[reject['ANOVA']].sort_values(by='ANOVA', ascending=True).index.to_list()

        if len(fset) > 0:
            print("%d significant features found by ANOVA by '%s' (P<%.2f, %s)" %\
                  (len(fset), group_by, pvalue_threshold, fdr_method))
            anova_sigfeats_path = anova_path.parent / 'ANOVA_sigfeats.txt'
            write_list_to_file(fset, anova_sigfeats_path)
             
    # Perform t-tests
    stats_t, pvals_t, reject_t = univariate_tests(X=features,
                                                  y=metadata[group_by],
                                                  control=control,
                                                  test='t-test',
                                                  comparison_type='binary_each_group',
                                                  multitest_correction=fdr_method,
                                                  alpha=pvalue_threshold)
    
    effect_sizes_t = get_effect_sizes(X=features,
                                      y=metadata[group_by],
                                      control=control,
                                      linked_test='t-test')
    
    stats_t.columns = ['stats_' + str(c) for c in stats_t.columns]
    pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
    reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
    effect_sizes_t.columns = ['effect_size_' + str(c) for c in effect_sizes_t.columns]
    ttest_results = pd.concat([stats_t, pvals_t, reject_t, effect_sizes_t], axis=1)
    
    # save results
    ttest_path = Path(save_dir) / 't-test' / 't-test_results.csv'
    ttest_path.parent.mkdir(exist_ok=True, parents=True)
    ttest_results.to_csv(ttest_path, header=True, index=True)
    
    nsig = sum(reject_t.sum(axis=1) > 0)
    print("%d significant features between any %s vs %s (t-test, P<%.2f, %s)" %\
          (nsig, group_by, control, pvalue_threshold, fdr_method))

    return

def microbiome_boxplots(metadata,
                        features,
                        group_by='food_type',
                        control='OP50',
                        save_dir=None,
                        stats_dir=None,
                        feature_set=None,
                        pvalue_threshold=0.05,
                        drop_insignificant=False,
                        scale_outliers=False,
                        ylim_minmax=None):
    
    feature_set = features.columns.tolist() if feature_set is None else feature_set
    assert isinstance(feature_set, list) and all(f in features.columns for f in feature_set)
                    
    # load t-test results for window
    if stats_dir is not None:
        ttest_path = Path(stats_dir) / 't-test' / 't-test_results.csv'
        ttest_df = pd.read_csv(ttest_path, header=0, index_col=0)
        pvals = ttest_df[[c for c in ttest_df.columns if 'pval' in c]]
        pvals.columns = [c.replace('pvals_','') for c in pvals.columns]
    
    boxplots_sigfeats(features,
                      y_class=metadata[group_by],
                      control=control,
                      pvals=pvals,
                      z_class=None,
                      feature_set=feature_set,
                      saveDir=Path(save_dir),
                      drop_insignificant=drop_insignificant,
                      p_value_threshold=pvalue_threshold,
                      scale_outliers=scale_outliers,
                      ylim_minmax=ylim_minmax)
    
    return

#%% Main

if __name__ == '__main__':
    
    aux_dir = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    res_dir = Path(PROJECT_DIR) / 'Results'
    
    metadata_path_local = Path(SAVE_DIR) / 'metadata.csv'
    features_path_local = Path(SAVE_DIR) / 'features.csv'
    
    if not metadata_path_local.exists() and not features_path_local.exists():
        metadata, metadata_path = compile_metadata(aux_dir,
                                                   imaging_dates=IMAGING_DATES,
                                                   n_wells=N_WELLS, 
                                                   add_well_annotations=False,
                                                   from_source_plate=False)
        
        features, metadata = process_feature_summaries(metadata_path, 
                                                       results_dir=res_dir, 
                                                       compile_day_summaries=True, 
                                                       imaging_dates=IMAGING_DATES, 
                                                       align_bluelight=False, 
                                                       window_summaries=False,
                                                       n_wells=N_WELLS)

        # Clean results - Remove bad well data + features with too many NaNs/zero std + impute NaNs
        features, metadata = clean_summary_results(features, 
                                                   metadata,
                                                   feature_columns=None,
                                                   nan_threshold_row=NAN_THRESH_SAMPLE,
                                                   nan_threshold_col=NAN_THRESH_FEATURE,
                                                   max_value_cap=None,
                                                   imputeNaN=True,
                                                   min_nskel_per_video=MIN_SKEL_PER_VIDEO,
                                                   min_nskel_sum=MIN_SKEL_SUM,
                                                   drop_size_related_feats=False,
                                                   norm_feats_only=False)
        
        assert not metadata['worm_strain'].isna().any()
        assert not metadata['food_type'].isna().any()
        
        # save clean metadata and features
        metadata.to_csv(metadata_path_local, index=False)
        features.to_csv(features_path_local, index=False)
        
    else:
        metadata = pd.read_csv(metadata_path_local, header=0, index_col=None, dtype={'comments':str})
        features = pd.read_csv(features_path_local, header=0, index_col=None)

    assert not features.isna().sum(axis=1).any()
    assert not (features.std(axis=1) == 0).any()
 
    # Subset for 5 hours timepoint only 
    metadata = metadata[metadata['time_point']==3]
    features = features.reindex(metadata.index)
 
    # load feature set
    if FEATURE_SET is not None:
        # subset for selected feature set (and remove path curvature features)
        if isinstance(FEATURE_SET, int) and FEATURE_SET in [8,16,256]:
            features = select_feat_set(features, 'tierpsy_{}'.format(FEATURE_SET), append_bluelight=False)
            features = features[[f for f in features.columns if 'path_curvature' not in f]]
        elif isinstance(FEATURE_SET, list) or isinstance(FEATURE_SET, set):
            assert all(f in features.columns for f in FEATURE_SET)
            features = features[FEATURE_SET].copy()
    feature_list = features.columns.tolist()

    strain_list = list(metadata['food_type'].unique())
    print("Analysing %d strains..." % len(strain_list))
    
    counts = metadata.groupby('food_type').count()['file_id']
    counts.to_csv(Path(SAVE_DIR) / 'sample_counts.csv')
    
    plot_dir = Path(SAVE_DIR) / 'Plots'
    stats_dir = Path(SAVE_DIR) / 'Stats'

    ##### Hierarchical Clustering Analysis #####
        
    # Z-normalise control data
    control_metadata = metadata[metadata['food_type']=='OP50']
    control_features = features.reindex(control_metadata.index)
    control_featZ = control_features.apply(zscore, axis=0)
    
    ### Control clustermap
    
    # control data is clustered and feature order is stored and applied to full data
    print("\nPlotting control clustermap")
    
    control_clustermap_path = plot_dir / 'heatmaps' / 'date_clustermap.pdf'
    cg = plot_clustermap(control_featZ, control_metadata,
                         group_by='date_yyyymmdd',
                         method=METHOD, 
                         metric=METRIC,
                         figsize=[30,10],
                         sub_adj={'bottom':0.05,'left':0,'top':1,'right':0.85},
                         saveto=control_clustermap_path,
                         label_size=15,
                         show_xlabels=True,
                         bluelight_col_colours=False)
    
    # save feature order to file
    control_feature_order = np.array(control_featZ.columns)[cg.dendrogram_col.reordered_ind]
    control_feature_order_df = pd.DataFrame(columns=['feature_name'], 
                                            index=range(1, len(control_feature_order) + 1),
                                            data=control_feature_order)
    control_feature_order_path = control_clustermap_path.parent / (control_clustermap_path.stem + 
                                                                   '_feature_order.csv')
    control_feature_order_df.to_csv(control_feature_order_path, header=True, index=True)

    ### Full clustermap 

    # Z-normalise data for all strains
    featZ = features.apply(zscore, axis=0)
                    
    ## Save z-normalised values
    # z_stats = featZ.join(hit_metadata[grouping_var]).groupby(by=grouping_var).mean().T
    # z_stats.columns = ['z-mean_' + v for v in z_stats.columns.to_list()]
    # z_stats.to_csv(z_stats_path, header=True, index=None)
    
    # Clustermap of full data   
    print("Plotting all strains clustermap")    
    full_clustermap_path = plot_dir / 'heatmaps' / 'strain_clustermap.pdf'
    fg = plot_clustermap(featZ, metadata, 
                         group_by='food_type',
                         row_colours=None,
                         method=METHOD, 
                         metric=METRIC,
                         figsize=[20,40], # (20,40)
                         sub_adj={'bottom':0.01,'left':0,'top':1,'right':0.85},
                         saveto=full_clustermap_path,
                         label_size=(2,25), # (2,2)
                         show_xlabels=False,
                         bluelight_col_colours=False)
    
    # save clustered feature order for all strains
    full_feature_order = np.array(featZ.columns)[fg.dendrogram_col.reordered_ind]
    full_feature_order_df = pd.DataFrame(columns=['feature_name'], 
                                                index=range(1, len(full_feature_order) + 1),
                                                data=full_feature_order)
    full_feature_order_path = full_clustermap_path.parent / (full_clustermap_path.stem + 
                                                                '_feature_order.csv')
    full_feature_order_df.to_csv(full_feature_order_path, header=True, index=True)

    full_strain_order = [l.get_text() for l in fg.ax_heatmap.get_yticklabels()]

    ### Stats and boxplots ###
    
    # perform anova and t-tests comparing each treatment to control
    microbiome_stats(metadata,
                     features,
                     group_by='food_type',
                     control='OP50',
                     save_dir=Path(SAVE_DIR) / 'Stats',
                     feature_set=None,
                     pvalue_threshold=P_VALUE_THRESHOLD,
                     fdr_method=FDR_METHOD)
    
    # boxplots comparing each treatment to control for each feature
    microbiome_boxplots(metadata,
                        features,
                        group_by='food_type',
                        control='OP50',
                        save_dir=Path(SAVE_DIR) / 'Plots',
                        stats_dir=Path(SAVE_DIR) / 'Stats',
                        feature_set=None,
                        pvalue_threshold=P_VALUE_THRESHOLD,
                        drop_insignificant=True,
                        scale_outliers=False,
                        ylim_minmax=None)

    anova_path = stats_dir / 'ANOVA' / 'ANOVA_results.csv'
    
    # load results + record significant features
    print("\nLoading statistics results")
    anova_table = pd.read_csv(anova_path, index_col=0)            
    pvals = anova_table.sort_values(by='pvals', ascending=True)['pvals'] # rank features by p-value
    fset = pvals[pvals < P_VALUE_THRESHOLD].index.to_list()
    print("\n%d significant features found by ANOVA (P<0.05, %s)" % (len(fset), FDR_METHOD))
    
    ### t-test
        
    ttest_path = stats_dir / 't-test' / 't-test_results.csv'
     
    # read t-test results + record significant features (NOT ORDERED)
    ttest_table = pd.read_csv(ttest_path, index_col=0)
    pvals_t = ttest_table[[c for c in ttest_table if "pvals_" in c]] 
    pvals_t.columns = [c.split('pvals_')[-1] for c in pvals_t.columns]       
    fset_ttest = pvals_t[(pvals_t < P_VALUE_THRESHOLD).sum(axis=1) > 0].index.to_list()
    print("%d significant features found by t-test (P<0.05, %s)" % (len(fset_ttest), FDR_METHOD))
   
    if len(fset) > 0:
        # Rank strains by number of sigfeats by t-test 
        ranked_nsig = (pvals_t < P_VALUE_THRESHOLD).sum(axis=0).sort_values(ascending=False)
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
        hit_strains_pval = ranked_pval[ranked_pval < P_VALUE_THRESHOLD].index.to_list()
        hit_strains_pval = ranked_pval.index.to_list()
        assert all(s in hit_strains_pval for s in hit_strains_nsig)
        write_list_to_file(hit_strains_pval[:N_LOWEST_PVAL], stats_dir /\
                           'lowest{}_pval.txt'.format(N_LOWEST_PVAL))
        
        print("\nPlotting ranked strains by number of significant features")
        ranked_nsig_path = plot_dir / ('ranked_number_sigfeats' + '_' + FDR_METHOD + '.pdf')
        plt.close('all')
        fig, ax = plt.subplots(figsize=(15,8), dpi=900)
        ax.plot(ranked_nsig)
        ax.set_xticklabels(ranked_nsig.index.to_list(), rotation=90, fontsize=15)
        _ = [t.set_color(i) for (i,t) in zip(['red' if i.get_text() in ['MYb10','MYb330','MYb541'] 
                                              else 'black' for i in ax.xaxis.get_ticklabels()], 
                                             ax.xaxis.get_ticklabels())]
        plt.xlabel("Strains (ranked)", fontsize=15, labelpad=10)
        plt.ylabel("Number of significant features", fontsize=15, labelpad=10)
        plt.subplots_adjust(left=0.03, right=0.99, bottom=0.15)
        plt.savefig(ranked_nsig_path)
        
        print("Plotting ranked strains by lowest p-value of any feature")
        lowest_pval_path = plot_dir / ('ranked_lowest_pval' + '_' + FDR_METHOD + '.pdf')
        plt.close('all')
        fig, ax = plt.subplots(figsize=(15,8), dpi=900)
        ax.plot(ranked_pval)
        plt.axhline(y=P_VALUE_THRESHOLD, c='dimgray', ls='--')
        ax.set_xticklabels(ranked_nsig.index.to_list(), rotation=90, fontsize=15)
        plt.xlabel("Strains (ranked)", fontsize=15, labelpad=10)
        plt.ylabel("Lowest p-value by t-test", fontsize=15, labelpad=10)
        plt.subplots_adjust(left=0.03, right=0.99, bottom=0.15)
        plt.savefig(lowest_pval_path)
        plt.close()
    
        print("\nMaking errorbar plots")
        errorbar_sigfeats(features, metadata, 
                          group_by='food_type', 
                          fset=feature_list,
                          control='OP50', 
                          rank_by='mean',
                          max_feats2plt=MAX_N_FEATS,
                          highlight_subset=['MYb541','MYb330','MYb10'],
                          figsize=[20,8], 
                          fontsize=20,
                          ms=30,
                          elinewidth=10,
                          highlight_colour='red',
                          fmt='-',
                          tight_layout=[0.01,0.01,0.99,0.99],
                          saveDir=plot_dir / 'errorbar')

    # If no sigfeats, subset for top strains ranked by lowest p-value by t-test for any feature    
    if len(hit_strains_nsig) == 0:
        print("\Saving lowest %d strains ranked by p-value for any feature" % N_LOWEST_PVAL)
        write_list_to_file(hit_strains_pval, stats_dir / 'Top100_lowest_pval.txt')
        hit_strains = hit_strains_pval
    elif len(hit_strains_nsig) > 0:
        hit_strains = hit_strains_nsig

    # Individual boxplots of significant features by pairwise t-test (each group vs control)
    boxplots_sigfeats(features,
                      y_class=metadata['food_type'],
                      control='OP50',
                      pvals=pvals_t, 
                      z_class=metadata['date_yyyymmdd'],
                      #feature_set=fset, #None
                      feature_set=None,
                      # append_ranking_fname=False,
                      saveDir=plot_dir / 'paired_boxplots_nsig', # pval
                      p_value_threshold=P_VALUE_THRESHOLD,
                      drop_insignificant=True if len(hit_strains) > 0 else False,
                      max_sig_feats=MAX_N_FEATS,
                      max_strains=N_LOWEST_PVAL if len(hit_strains_nsig) == 0 else None,
                      sns_colour_palette="tab10",
                      verbose=False)


    ### Heatmap ###
    
    # features (x-axis columns) ordered by control clustered feature order
    
    # pvals_heatmap = anova_table.loc[control_feature_order, 'pvals']
    # pvals_heatmap.name = 'P < {}'.format(args.pval_threshold)
    # assert all(f in featZ.columns for f in pvals_heatmap.index)
            
    # # Plot heatmap (averaged for each sample)
    # if len(metadata[grouping_var].unique()) < 250:
    #     print("\nPlotting barcode heatmap")
    #     heatmap_path = plot_dir / 'heatmaps' / (grouping_var + '_heatmap.pdf')
    #     plot_barcode_heatmap(featZ=featZ[control_feature_order], 
    #                          meta=metadata, 
    #                          group_by=[grouping_var], 
    #                          strain_order=full_strain_order,
    #                          pvalues_series=pvals_heatmap,
    #                          p_value_threshold=args.pval_threshold,
    #                          selected_feats=None, # fset if len(fset) > 0 else None
    #                          saveto=heatmap_path,
    #                          figsize=[20,40],
    #                          sns_colour_palette="Pastel1",
    #                          label_size=2)
    
    # timeseries plots of speed for each treatment vs 'N2-BW-nan' control
    plot_timeseries_feature(metadata,
                            project_dir=Path(PROJECT_DIR),
                            save_dir=Path(SAVE_DIR) / 'timeseries-speed',
                            group_by='food_type',
                            control='OP50',
                            groups_list=hit_strains,
                            feature='speed',
                            n_wells=N_WELLS,
                            bluelight_stim_type=None,
                            video_length_seconds=15*60,
                            bluelight_timepoints_seconds=None,
                            smoothing=25,
                            fps=25,
                            ylim_minmax=None) # ylim_minmax for speed feature only

   
