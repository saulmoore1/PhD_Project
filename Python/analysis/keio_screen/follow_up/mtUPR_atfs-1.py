#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse whether C. elegans atfs-1 mutant (mitochondrial UPR mutant) is 'aroused' by fepD E. coli

@author: sm5911
@date: 17/02/2023

"""

#%% Imports 

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from visualisation.plotting_helper import sig_asterix, boxplots_sigfeats, all_in_one_boxplots
from write_data.write import write_list_to_file
from time_series.plot_timeseries import plot_timeseries, get_strain_timeseries

from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from tierpsytools.preprocessing.filter_data import select_feat_set

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Keio_Worm_UPRmt"
SAVE_DIR = "/Users/sm5911/Documents/Keio_UPRmt_atfs-1"

N_WELLS = 6
FPS = 25

nan_threshold_row = 0.8
nan_threshold_col = 0.05

FEATURE_SET = ['speed_50th']

BLUELIGHT_TIMEPOINTS_SECONDS = [(60, 70),(160, 170),(260, 270)]

WINDOW_DICT = {0:(290,300)}

WINDOW_NAME_DICT = {0 : "20-30 seconds after blue light 3"}

#%% Functions

def stats(metadata,
          features,
          group_by='treatment',
          control='BW',
          save_dir=None,
          feature_set=None,
          pvalue_threshold=0.05,
          fdr_method='fdr_bh'):
    
    # check case-sensitivity
    assert len(metadata[group_by].unique()) == len(metadata[group_by].str.upper().unique())
    
    if feature_set is not None:
        feature_set = [feature_set] if isinstance(feature_set, str) else feature_set
        assert isinstance(feature_set, list)
        assert(all(f in features.columns for f in feature_set))
    else:
        feature_set = features.columns.tolist()
    features = features[feature_set]

    # print mean sample size
    sample_size = metadata.groupby(group_by).count()
    print("Mean sample size of %s/window: %d" % (group_by, 
                                                 int(sample_size[sample_size.columns[-1]].mean())))

    fset = []
    n = len(metadata[group_by].unique())
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

    return #anova_results, ttest_results

def boxplots(metadata,
             features,
             group_by='treatment',
             control='BW',
             save_dir=None,
             stats_dir=None,
             feature_set=None,
             pvalue_threshold=0.05):
        
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
                      pvals=pvals if stats_dir is not None else None,
                      z_class=None,
                      feature_set=feature_set,
                      saveDir=Path(save_dir),
                      drop_insignificant=True if feature_set is None else False,
                      p_value_threshold=pvalue_threshold,
                      scale_outliers=True,
                      append_ranking_fname=False)

    return

#%% Main

if __name__ == '__main__':
    
    aux_dir = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    res_dir = Path(PROJECT_DIR) / 'Results'
    
    metadata_path_local = Path(SAVE_DIR) / 'metadata.csv'
    features_path_local = Path(SAVE_DIR) / 'features.csv'
    
    if not metadata_path_local.exists() and not features_path_local.exists():
        metadata, metadata_path = compile_metadata(aux_dir, 
                                                   n_wells=N_WELLS, 
                                                   add_well_annotations=False,
                                                   from_source_plate=True)
        
        features, metadata = process_feature_summaries(metadata_path, 
                                                       results_dir=res_dir, 
                                                       compile_day_summaries=True, 
                                                       imaging_dates=None, 
                                                       align_bluelight=False, 
                                                       window_summaries=True,
                                                       n_wells=N_WELLS)

        # Clean results - Remove bad well data + features with too many NaNs/zero std + impute NaNs
        features, metadata = clean_summary_results(features, 
                                                   metadata,
                                                   feature_columns=None,
                                                   nan_threshold_row=nan_threshold_row,
                                                   nan_threshold_col=nan_threshold_col,
                                                   max_value_cap=1e15,
                                                   imputeNaN=True,
                                                   min_nskel_per_video=None,
                                                   min_nskel_sum=None,
                                                   drop_size_related_feats=False,
                                                   norm_feats_only=False)
        
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
            features = select_feat_set(features, 'tierpsy_{}'.format(FEATURE_SET), append_bluelight=True)
            features = features[[f for f in features.columns if 'path_curvature' not in f]]
        elif isinstance(FEATURE_SET, list) or isinstance(FEATURE_SET, set):
            assert all(f in features.columns for f in FEATURE_SET)
            features = features[FEATURE_SET].copy()
    feature_list = features.columns.tolist()

    # subset metadata results for bluelight videos only 
    bluelight_videos = [i for i in metadata['imgstore_name'] if 'bluelight' in i]
    metadata = metadata[metadata['imgstore_name'].isin(bluelight_videos)]

    # perform anova and t-tests comparing each treatment to BW control
    treatment_cols = ['worm_strain','bacteria_strain','drug_type']
    metadata['treatment'] = metadata[treatment_cols].astype(str).agg('-'.join, axis=1)
    metadata['treatment'] = [i.replace('-nan','') for i in metadata['treatment']]

    # strain list
    worm_strain_list = sorted(metadata['worm_strain'].unique())
    treatment_list = sorted(metadata['treatment'].unique())
    no_para_list = [s for s in treatment_list if not 'Paraquat' in s]
    para_list = [s for s in treatment_list if 'Paraquat' in s]

    metadata['window'] = metadata['window'].astype(int)
    assert metadata['window'].nunique() == 1

    ### without paraquat
    
    meta_no_para = metadata[metadata['treatment'].isin(no_para_list)]
    feat_no_para = features.reindex(meta_no_para.index)
           
    # compare vs N2-BW
    stats(meta_no_para,
          feat_no_para,
          group_by='treatment',
          control='N2-BW',
          save_dir=Path(SAVE_DIR) / 'Stats',
          feature_set=feature_list,
          pvalue_threshold=0.05,
          fdr_method='fdr_bh')
        
    colour_dict = dict(zip(no_para_list, sns.color_palette('tab10', len(no_para_list))))
    all_in_one_boxplots(meta_no_para,
                        feat_no_para,
                        group_by='treatment',
                        control='N2-BW',
                        sigasterix=True,
                        fontsize=15,
                        order=no_para_list,
                        colour_dict=colour_dict,
                        feature_set=feature_list,
                        save_dir=Path(SAVE_DIR) / 'Plots',
                        ttest_path=Path(SAVE_DIR) / 'Stats' / 't-test' / 't-test_results.csv',
                        pvalue_threshold=0.05,
                        figsize=(15,8),
                        ylim_minmax=(0,300),
                        subplots_adjust={'bottom':0.35,'top':0.9,'left':0.1,'right':0.95})
    
    # # compare vs N2-fepD  
    # stats(metadata,
    #       features,
    #       group_by='treatment',
    #       control='N2-fepD',
    #       save_dir=Path(SAVE_DIR) / 'Stats_N2-fepD',
    #       feature_set=feature_list,
    #       pvalue_threshold=0.05,
    #       fdr_method='fdr_bh')
        
    # colour_dict = dict(zip(strain_list, sns.color_palette('tab10', len(strain_list))))
    # all_in_one_boxplots(metadata,
    #                     features,
    #                     group_by='treatment',
    #                     control='N2-fepD',
    #                     sigasterix=True,
    #                     fontsize=15,
    #                     order=strain_list,
    #                     colour_dict=colour_dict,
    #                     feature_set=feature_list,
    #                     save_dir=Path(SAVE_DIR) / 'Plots_N2-fepD',
    #                     ttest_path=Path(SAVE_DIR) / 'Stats_N2-fepD' / 't-test' / 't-test_results.csv',
    #                     pvalue_threshold=0.05,
    #                     figsize=(15,8),
    #                     ylim_minmax=(0,300),
    #                     subplots_adjust={'bottom':0.35,'top':0.9,'left':0.1,'right':0.95})
    
    ### with paraquat
    
    meta_para = metadata[metadata['treatment'].isin(para_list)]
    feat_para = features.reindex(meta_para.index)
           
    # compare vs N2-BW
    stats(meta_para,
          feat_para,
          group_by='treatment',
          control='N2-BW-Paraquat',
          save_dir=Path(SAVE_DIR) / 'Stats_PQ',
          feature_set=feature_list,
          pvalue_threshold=0.05,
          fdr_method='fdr_bh')
        
    colour_dict = dict(zip(para_list, sns.color_palette('tab10', len(para_list))))
    all_in_one_boxplots(meta_para,
                        feat_para,
                        group_by='treatment',
                        control='N2-BW-Paraquat',
                        sigasterix=True,
                        fontsize=15,
                        order=para_list,
                        colour_dict=colour_dict,
                        feature_set=feature_list,
                        save_dir=Path(SAVE_DIR) / 'Plots_PQ',
                        ttest_path=Path(SAVE_DIR) / 'Stats_PQ' / 't-test' / 't-test_results.csv',
                        pvalue_threshold=0.05,
                        figsize=(15,8),
                        ylim_minmax=(0,300),
                        subplots_adjust={'bottom':0.35,'top':0.9,'left':0.1,'right':0.95})
    
    # timeseries - without paraquat
    
    for worm in tqdm(worm_strain_list):
        groups = ['BW', 'fepD']
        print("Plotting timeseries speed for %s" % worm)
        
        bluelight_frames = [(i*FPS, j*FPS) for (i, j) in BLUELIGHT_TIMEPOINTS_SECONDS]
        feature = 'speed'
    
        save_dir = Path(SAVE_DIR) / 'timeseries-speed'
        ts_plot_dir = save_dir / worm
        ts_plot_dir.mkdir(exist_ok=True, parents=True)
        save_path = ts_plot_dir / '{}_speed_bluelight.pdf'.format(worm)
        
        plt.close('all')
        fig, ax = plt.subplots(figsize=(15,6), dpi=300)
        col_dict = dict(zip(groups, sns.color_palette('tab10', len(groups))))
    
        for group in groups:
            
            treatment = worm + '-' + group
            
            # get control timeseries
            group_ts = get_strain_timeseries(metadata,
                                             project_dir=Path(PROJECT_DIR),
                                             strain=treatment,
                                             group_by='treatment',
                                             feature_list=[feature],
                                             save_dir=save_dir,
                                             n_wells=N_WELLS,
                                             verbose=True)
            
            ax = plot_timeseries(df=group_ts,
                                  feature=feature,
                                  error=True,
                                  max_n_frames=360*FPS, 
                                  smoothing=10*FPS, 
                                  ax=ax,
                                  bluelight_frames=bluelight_frames,
                                  colour=col_dict[group])
    
        plt.ylim(-20, 260)
        xticks = np.linspace(0, 360*FPS, int(360/60)+1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
        ax.set_xlabel('Time (minutes)', fontsize=20, labelpad=10)
        ylab = feature.replace('_50th'," (µm s$^{-1}$)")
        ax.set_ylabel(ylab, fontsize=20, labelpad=10)
        ax.legend(groups, fontsize=12, frameon=False, loc='best', handletextpad=1)
        plt.subplots_adjust(left=0.1, top=0.98, bottom=0.15, right=0.98)
    
        # save plot
        print("Saving to: %s" % save_path)
        plt.savefig(save_path)

    # timeseries - with paraquat
    
    for worm in tqdm(worm_strain_list):
        groups = ['BW-Paraquat', 'fepD-Paraquat']
        print("Plotting timeseries speed for %s" % worm)
        
        bluelight_frames = [(i*FPS, j*FPS) for (i, j) in BLUELIGHT_TIMEPOINTS_SECONDS]
        feature = 'speed'
    
        save_dir = Path(SAVE_DIR) / 'timeseries-speed'
        ts_plot_dir = save_dir / worm
        ts_plot_dir.mkdir(exist_ok=True, parents=True)
        save_path = ts_plot_dir / '{}_speed_bluelight.pdf'.format(worm + '_' + 'Paraquat')
        
        plt.close('all')
        fig, ax = plt.subplots(figsize=(15,6), dpi=300)
        col_dict = dict(zip(groups, sns.color_palette('tab10', len(groups))))
    
        for group in groups:
            
            treatment = worm + '-' + group
            
            # get control timeseries
            group_ts = get_strain_timeseries(metadata,
                                             project_dir=Path(PROJECT_DIR),
                                             strain=treatment,
                                             group_by='treatment',
                                             feature_list=[feature],
                                             save_dir=save_dir,
                                             n_wells=N_WELLS,
                                             verbose=True)
            
            ax = plot_timeseries(df=group_ts,
                                  feature=feature,
                                  error=True,
                                  max_n_frames=360*FPS, 
                                  smoothing=10*FPS, 
                                  ax=ax,
                                  bluelight_frames=bluelight_frames,
                                  colour=col_dict[group])
    
        plt.ylim(-20, 260)
        xticks = np.linspace(0, 360*FPS, int(360/60)+1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
        ax.set_xlabel('Time (minutes)', fontsize=20, labelpad=10)
        ylab = feature.replace('_50th'," (µm s$^{-1}$)")
        ax.set_ylabel(ylab, fontsize=20, labelpad=10)
        ax.legend(groups, fontsize=12, frameon=False, loc='best', handletextpad=1)
        plt.subplots_adjust(left=0.1, top=0.98, bottom=0.15, right=0.98)
    
        # save plot
        print("Saving to: %s" % save_path)
        plt.savefig(save_path)
