#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of pan-genome screen results

@author: sm5911
@date: 02/09/2022

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
from clustering.hierarchical_clustering import plot_clustermap
from write_data.write import write_list_to_file
from visualisation.plotting_helper import sig_asterix, boxplots_sigfeats, errorbar_sigfeats
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from time_series.plot_timeseries import plot_timeseries_feature #, selected_strains_timeseries

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/PanGenome_Screen_96WP"
SAVE_DIR = "/Users/sm5911/Documents/PanGenome_Screen"

N_WELLS = 96
IMAGING_DATES = ['20191024','20191031'] # No featsums for '20191017' - dropped data

NAN_THRESH_SAMPLE = 0.8
NAN_THRESH_FEATURE = 0.05

FEATURE_SET = 256 #['speed_50th', 'angular_velocity_abs_50th', 'motion_mode_forward_fraction']

P_VALUE_THRESHOLD = 0.05
FDR_METHOD = 'fdr_by'

METHOD = 'complete' # 'complete','linkage','average','weighted','centroid'
METRIC = 'euclidean' # 'euclidean','cosine','correlation'

N_LOWEST_PVAL = 100
MAX_N_FEATS = 10

#%% Functions

def pangenome_stats(metadata,
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

def pangenome_boxplots(metadata,
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
                                                   min_nskel_per_video=None,
                                                   min_nskel_sum=None,
                                                   drop_size_related_feats=False,
                                                   norm_feats_only=False)
        
        assert not metadata['worm_strain'].isna().any()
        assert not metadata['food_type'].isna().any()
        
        # append strain info to metadata - using plate-strain mapping data (from Dani)
        mapping = pd.read_excel(aux_dir / 'PanGenome_Plate_Strain_Mapping_Dani.xlsx')
        mapping['plate_well_id'] = mapping[['Plate','Well']].astype(str).agg('_'.join, axis=1)
        metadata['plate_well_id'] = metadata[['imaging_plate_id','well_name']
                                             ].astype(str).agg('_'.join, axis=1)
        
        metadata = metadata.merge(mapping, how='left', on='plate_well_id')
        
        metadata = metadata[np.logical_or(~metadata['Strainname'].isna(), 
                                          metadata['food_type']=='OP50')]
        metadata['Strainname'] = metadata['Strainname'].fillna('OP50')
                
        features = features.reindex(metadata.index)
        
        # save clean metadata and features
        metadata.to_csv(metadata_path_local, index=False)
        features.to_csv(features_path_local, index=False)
        
    else:
        metadata = pd.read_csv(metadata_path_local, header=0, index_col=None, dtype={'comments':str})
        features = pd.read_csv(features_path_local, header=0, index_col=None)

    assert not features.isna().sum(axis=1).any()
    assert not (features.std(axis=1) == 0).any()
    
    # load feature set
    if FEATURE_SET is not None:
        # subset for selected feature set (and remove path curvature features)
        if isinstance(FEATURE_SET, int) and FEATURE_SET in [8,16,256]:
            features = select_feat_set(features, 'tierpsy_{}'.format(FEATURE_SET), 
                                       append_bluelight=False)
            features = features[[f for f in features.columns if 'path_curvature' not in f]]
        elif isinstance(FEATURE_SET, list) or isinstance(FEATURE_SET, set):
            assert all(f in features.columns for f in FEATURE_SET)
            features = features[FEATURE_SET].copy()
    feature_list = features.columns.tolist()

    strain_list = list(metadata['Strainname'].unique())

    stats_dir = Path(SAVE_DIR) / 'Stats'
    plot_dir = Path(SAVE_DIR) / 'Plots'
    
    counts = metadata.groupby('Strainname').count()['file_id']
    counts.to_csv(Path(SAVE_DIR) / 'sample_counts.csv')
    
    metadata['Broadphenotype'] = metadata['Broadphenotype'].fillna('Unknown')
    metadata['Broadphenotype'] = [i.split(' ')[0] for i in metadata['Broadphenotype']]
        
    ##### Hierarchical Clustering Analysis #####

    featZ = features.apply(zscore, axis=0)
                    
    # Clustermap of full data   
    print("Plotting all strains clustermap")    
    full_clustermap_path = plot_dir / 'heatmaps' / 'strain_clustermap.pdf'
    fg = plot_clustermap(featZ, metadata, 
                         group_by='Strainname',
                         #colour_by='Broadphenotype', # TODO: colour rows by broadphenotype
                         row_colours=True,
                         method=METHOD, 
                         metric=METRIC,
                         figsize=[20,50], # (20,40)
                         sub_adj={'bottom':0.01,'left':0,'top':1,'right':0.95},
                         saveto=full_clustermap_path,
                         label_size=(2,7), # (2,2)
                         show_xlabels=False,
                         bluelight_col_colours=False) # no feature labels
        
    # save clustered feature order for all strains
    full_feature_order = np.array(featZ.columns)[fg.dendrogram_col.reordered_ind]
    full_feature_order_df = pd.DataFrame(columns=['feature_name'], 
                                                index=range(1, len(full_feature_order) + 1),
                                                data=full_feature_order)
    full_feature_order_path = full_clustermap_path.parent / (full_clustermap_path.stem + 
                                                                '_feature_order.csv')
    full_feature_order_df.to_csv(full_feature_order_path, header=True, index=True)

    full_strain_order = [l.get_text() for l in fg.ax_heatmap.get_yticklabels()]

    # perform anova and t-tests comparing each treatment to control
    pangenome_stats(metadata,
                    features,
                    group_by='Strainname',
                    control='OP50',
                    save_dir=stats_dir,
                    feature_set=feature_list,
                    pvalue_threshold=P_VALUE_THRESHOLD,
                    fdr_method=FDR_METHOD)
    
    # boxplots comparing each treatment to control for each feature
    pangenome_boxplots(metadata,
                       features,
                       group_by='Strainname',
                       control='OP50',
                       save_dir=plot_dir,
                       stats_dir=stats_dir,
                       feature_set=feature_list,
                       pvalue_threshold=P_VALUE_THRESHOLD,
                       drop_insignificant=True,
                       scale_outliers=False,
                       ylim_minmax=None)

    
    # load results + record significant features
    print("\nLoading statistics results")
    anova_path = stats_dir / 'ANOVA' / 'ANOVA_results.csv'
    anova_table = pd.read_csv(anova_path, index_col=0)            
    pvals = anova_table.sort_values(by='pvals', ascending=True)['pvals'] # rank features by p-value
    fset = pvals[pvals < P_VALUE_THRESHOLD].index.to_list()
    print("\n%d significant features found by ANOVA (P<0.05, %s)" % (len(fset), FDR_METHOD))
    
    ### t-test
             
    # read t-test results + record significant features (NOT ORDERED)
    ttest_path = stats_dir / 't-test' / 't-test_results.csv'
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
    
        strains_with_data = counts[counts >=2].index.tolist()
        err_meta = metadata[metadata['Strainname'].isin(strains_with_data)]
        err_feats = features.reindex(err_meta.index)
        print("\nMaking errorbar plots")
        errorbar_sigfeats(err_feats, err_meta, 
                          group_by='Strainname', 
                          fset=fset, #['speed_10th','motion_mode_paused_fraction'], 
                          control='OP50', 
                          rank_by='mean',
                          max_feats2plt=MAX_N_FEATS,
                          highlight_subset=hit_strains_nsig[:10],
                          figsize=[30,15], 
                          fontsize=18,
                          ms=20,
                          elinewidth=12,
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
                      y_class=metadata['Strainname'],
                      control='OP50',
                      pvals=pvals_t, 
                      z_class=None,
                      feature_set=feature_list,
                      # append_ranking_fname=False,
                      saveDir=plot_dir / 'paired_boxplots_nsig', # pval
                      p_value_threshold=P_VALUE_THRESHOLD,
                      drop_insignificant=False, #True if len(hit_strains) > 0 else False,
                      max_sig_feats=MAX_N_FEATS,
                      max_strains=N_LOWEST_PVAL if len(hit_strains_nsig) == 0 else None,
                      sns_colour_palette="tab10",
                      verbose=False)
    
    # timeseries plots of speed for each treatment vs 'N2-BW-nan' control
    plot_timeseries_feature(metadata,
                            project_dir=Path(PROJECT_DIR),
                            save_dir=plot_dir / 'timeseries-speed',
                            group_by='Strainname',
                            control='OP50',
                            groups_list=strain_list,
                            feature='speed',
                            n_wells=N_WELLS,
                            bluelight_stim_type='bluelight',
                            video_length_seconds=360,
                            bluelight_timepoints_seconds=None,
                            smoothing=10,
                            fps=25,
                            ylim_minmax=None) # ylim_minmax for speed feature only
    
    