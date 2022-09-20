#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keio Worm Stress Mutants 1

@author: sm5911
@date: 30/07/2022

"""

#%% Imports

import pandas as pd
from tqdm import tqdm
from pathlib import Path

from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from write_data.write import write_list_to_file
from visualisation.plotting_helper import sig_asterix, boxplots_sigfeats, all_in_one_boxplots
from analysis.keio_screen.follow_up.uv_paraquat_antioxidant import masked_video_list_from_metadata
from time_series.plot_timeseries import plot_timeseries_feature, selected_strains_timeseries

from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from tierpsytools.preprocessing.filter_data import select_feat_set

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Keio_Worm_Stress_Mutants"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Worm_Stress_Mutants"

N_WELLS = 6

FPS = 25

FEATURE_SET = ['speed_50th']

nan_threshold_row = 0.8
nan_threshold_col = 0.05

THRESHOLD_FILTER_DURATION = 25 # threshold trajectory length (n frames) / 25 fps => 1 second
THRESHOLD_FILTER_MOVEMENT = 10 # threshold movement (n pixels) * 12.4 microns per pixel => 124 microns
THRESHOLD_LEAVING_DURATION = 50 # n frames a worm has to leave food for => a true leaving event

BLUELIGHT_TIMEPOINTS_SECONDS = [(60, 70),(160, 170),(260, 270)]

WINDOW_DICT = {0:(65,75),1:(90,100),
               2:(165,175),3:(190,200),
               4:(265,275),5:(290,300)}

WINDOW_NAME_DICT = {0:"blue light 1", 1: "20-30 seconds after blue light 1",
                    2:"blue light 2", 3: "20-30 seconds after blue light 2",
                    4:"blue light 3", 5: "20-30 seconds after blue light 3"}

#%% Functions

def worm_stress_mutants_stats(metadata,
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

def worm_stress_mutants_boxplots(metadata,
                                 features,
                                 group_by='treatment',
                                 control='BW',
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
                      ylim_minmax=ylim_minmax,
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
        
        assert not metadata['worm_strain'].isna().any()
        
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
    metadata = metadata[metadata['bluelight']=='bluelight']
    features = features.reindex(metadata.index)
    
    treatment_cols = ['worm_strain','bacteria_strain','drug_type']
    metadata['treatment'] = metadata.loc[:,treatment_cols].astype(str).agg('-'.join, axis=1)
    control = 'N2-BW-nan'
    
    # metadata = metadata[['bluelight' in metadata.loc[i,'imgstore_name'] for i in metadata.index]].copy()
    # metadata['imgstore_name_bluelight'] = metadata['imgstore_name']
    
    # save video file list for treatments (for manual inspection)
    video_dict = masked_video_list_from_metadata(metadata[metadata['window']==0], 
                                                 group_by='treatment', 
                                                 groups_list=None,
                                                 imgstore_col='imgstore_name',
                                                 project_dir=Path(PROJECT_DIR),
                                                 save_dir=Path(SAVE_DIR) / 'video_filenames')
    
    # boxplots comparing each treatment to control for each feature
    # fixed scale across plots for speed to 0-250 um/sec for easier comparison across conditions
    metadata['window'] = metadata['window'].astype(int)
    window_list = list(metadata['window'].unique())

    strain_list = list(metadata['treatment'].unique())
    BW_worms = [s for s in strain_list if 'BW-nan' in s]
    BW_Paraquat_worms = [s for s in strain_list if 'BW-Paraquat' in s]
    fepD_worms = [s for s in strain_list if 'fepD-nan' in s]
    fepD_Paraquat_worms = [s for s in strain_list if 'fepD-Paraquat' in s]
    
    for window in tqdm(window_list):
        meta_window = metadata[metadata['window']==window]
        feat_window = features.reindex(meta_window.index)
        
        stats_dir = Path(SAVE_DIR) / 'Stats' / WINDOW_NAME_DICT[window]
        plot_dir = Path(SAVE_DIR) / 'Plots' / WINDOW_NAME_DICT[window]
        
        # perform anova and t-tests comparing all treatments to control
        worm_stress_mutants_stats(meta_window,
                                  feat_window,
                                  group_by='treatment',
                                  control=control,
                                  save_dir=stats_dir,
                                  feature_set=feature_list,
                                  pvalue_threshold=0.05,
                                  fdr_method='fdr_bh')
    
        worm_stress_mutants_boxplots(meta_window,
                                     feat_window,
                                     group_by='treatment',
                                     control=control,
                                     save_dir=plot_dir,
                                     stats_dir=stats_dir,
                                     feature_set=feature_list,
                                     pvalue_threshold=0.05,
                                     scale_outliers=False,
                                     ylim_minmax=(-20,330)) # ylim_minmax for speed feature only 
        
        # worm strains on BW
        BW_worm_meta = meta_window.query("bacteria_strain=='BW' and drug_type!='Paraquat'")
        BW_worm_feat = feat_window.reindex(BW_worm_meta.index)
        worm_stress_mutants_stats(BW_worm_meta,
                                  BW_worm_feat,
                                  group_by='worm_strain',
                                  control='N2',
                                  save_dir=stats_dir / 'BW_worms',
                                  feature_set=feature_list,
                                  pvalue_threshold=0.05,
                                  fdr_method='fdr_bh')
        all_in_one_boxplots(BW_worm_meta,
                            BW_worm_feat,
                            group_by='worm_strain',
                            control='N2',
                            save_dir=plot_dir / 'all-in-one' / 'BW_worms',
                            ttest_path=stats_dir / 'BW_worms' / 't-test' / 't-test_results.csv',
                            feature_set=feature_list,
                            pvalue_threshold=0.05,
                            figsize=(30,10))

        # worm strains on BW + Paraquat
        BW_Paraquat_worm_meta = meta_window.query("bacteria_strain=='BW' and drug_type=='Paraquat'")
        BW_Paraquat_worm_feat = feat_window.reindex(BW_Paraquat_worm_meta.index)
        worm_stress_mutants_stats(BW_Paraquat_worm_meta,
                                  BW_Paraquat_worm_feat,
                                  group_by='worm_strain',
                                  control='N2',
                                  save_dir=stats_dir / 'BW_Paraquat_worms',
                                  feature_set=feature_list,
                                  pvalue_threshold=0.05,
                                  fdr_method='fdr_bh')
        all_in_one_boxplots(BW_Paraquat_worm_meta,
                            BW_Paraquat_worm_feat,
                            group_by='worm_strain',
                            control='N2',
                            save_dir=plot_dir / 'all-in-one' / 'BW_Paraquat_worms',
                            ttest_path=stats_dir / 'BW_Paraquat_worms' / 't-test' / 't-test_results.csv',
                            feature_set=feature_list,
                            pvalue_threshold=0.05,
                            figsize=(30,10))
        
        # worm strains on fepD
        fepD_worm_meta = meta_window.query("bacteria_strain=='fepD' and drug_type!='Paraquat'")
        fepD_worm_feat = feat_window.reindex(fepD_worm_meta.index)
        worm_stress_mutants_stats(fepD_worm_meta,
                                  fepD_worm_feat,
                                  group_by='worm_strain',
                                  control='N2',
                                  save_dir=stats_dir / 'fepD_worms',
                                  feature_set=feature_list,
                                  pvalue_threshold=0.05,
                                  fdr_method='fdr_bh')
        all_in_one_boxplots(fepD_worm_meta,
                            fepD_worm_feat,
                            group_by='worm_strain',
                            control='N2',
                            save_dir=plot_dir / 'all-in-one' / 'fepD_worms',
                            ttest_path=stats_dir / 'fepD_worms' / 't-test' / 't-test_results.csv',
                            feature_set=feature_list,
                            pvalue_threshold=0.05,
                            figsize=(30,10))

        # worm strains on fepD + Paraquat
        fepD_Paraquat_worm_meta = meta_window.query("bacteria_strain=='fepD' and drug_type=='Paraquat'")
        fepD_Paraquat_worm_feat = feat_window.reindex(fepD_Paraquat_worm_meta.index)
        worm_stress_mutants_stats(fepD_Paraquat_worm_meta,
                                  fepD_Paraquat_worm_feat,
                                  group_by='worm_strain',
                                  control='N2',
                                  save_dir=stats_dir / 'fepD_Paraquat_worms',
                                  feature_set=feature_list,
                                  pvalue_threshold=0.05,
                                  fdr_method='fdr_bh')
        all_in_one_boxplots(fepD_Paraquat_worm_meta,
                            fepD_Paraquat_worm_feat,
                            group_by='worm_strain',
                            control='N2',
                            save_dir=plot_dir / 'all-in-one' / 'fepD_Paraquat_worms',
                            ttest_path=stats_dir / 'fepD_Paraquat_worms' / 't-test' / 't-test_results.csv',
                            feature_set=feature_list,
                            pvalue_threshold=0.05,
                            figsize=(30,10))
        
    # subset for single metadata window for full bluelight video timeseries plots
    metadata = metadata[metadata['window']==0]

    BW_worms = [s for s in strain_list if 'BW-nan' in s]
    BW_Paraquat_worms = [s for s in strain_list if 'BW-Paraquat' in s]
    fepD_worms = [s for s in strain_list if 'fepD-nan' in s]
    fepD_Paraquat_worms = [s for s in strain_list if 'fepD-Paraquat' in s]
    
    # # timeseries plots of motion mode
    # selected_strains_timeseries(metadata,
    #                             project_dir=Path(PROJECT_DIR), 
    #                             save_dir=Path(SAVE_DIR) / 'timeseries-motion_mode', 
    #                             strain_list=strain_list,
    #                             group_by='treatment',
    #                             control=control,
    #                             n_wells=6,
    #                             bluelight_stim_type='bluelight',
    #                             video_length_seconds=360,
    #                             bluelight_timepoints_seconds=BLUELIGHT_TIMEPOINTS_SECONDS,
    #                             motion_modes=['forwards','paused','backwards'],
    #                             smoothing=10)
    
    # timeseries plots of speed for each treatment vs 'N2-BW-nan' control
    plot_timeseries_feature(metadata,
                            project_dir=Path(PROJECT_DIR),
                            save_dir=Path(SAVE_DIR) / 'timeseries-speed' / 'N2_control',
                            group_by='treatment',
                            control='N2-BW-nan',
                            groups_list=BW_worms,
                            feature='speed',
                            n_wells=6,
                            bluelight_stim_type='bluelight',
                            video_length_seconds=360,
                            bluelight_timepoints_seconds=BLUELIGHT_TIMEPOINTS_SECONDS,
                            smoothing=10,
                            fps=FPS,
                            ylim_minmax=(-20,330)) # ylim_minmax for speed feature only

    # timeseries plots of speed for each treatment vs 'N2-BW-nan' control
    plot_timeseries_feature(metadata,
                            project_dir=Path(PROJECT_DIR),
                            save_dir=Path(SAVE_DIR) / 'timeseries-speed' / 'N2_Paraquat_control',
                            group_by='treatment',
                            control='N2-BW-Paraquat',
                            groups_list=BW_Paraquat_worms,
                            feature='speed',
                            n_wells=6,
                            bluelight_stim_type='bluelight',
                            video_length_seconds=360,
                            bluelight_timepoints_seconds=BLUELIGHT_TIMEPOINTS_SECONDS,
                            smoothing=10,
                            fps=FPS,
                            ylim_minmax=(-20,330)) # ylim_minmax for speed feature only
    
    # timeseries plots of speed for each 'X-fepD-nan' treatment vs 'N2-fepD-nan' control
    plot_timeseries_feature(metadata,
                            project_dir=Path(PROJECT_DIR),
                            save_dir=Path(SAVE_DIR) / 'timeseries-speed' / 'fepD_control',
                            group_by='treatment',
                            control='N2-fepD-nan',
                            groups_list=fepD_worms,
                            feature='speed',
                            n_wells=6,
                            bluelight_stim_type='bluelight',
                            video_length_seconds=360,
                            bluelight_timepoints_seconds=BLUELIGHT_TIMEPOINTS_SECONDS,
                            smoothing=10,
                            fps=FPS,
                            ylim_minmax=(-20,330)) # ylim_minmax for speed feature only

    plot_timeseries_feature(metadata,
                            project_dir=Path(PROJECT_DIR),
                            save_dir=Path(SAVE_DIR) / 'timeseries-speed' / 'fepD_Paraquat_control',
                            group_by='treatment',
                            control='N2-fepD-Paraquat',
                            groups_list=fepD_Paraquat_worms,
                            feature='speed',
                            n_wells=6,
                            bluelight_stim_type='bluelight',
                            video_length_seconds=360,
                            bluelight_timepoints_seconds=BLUELIGHT_TIMEPOINTS_SECONDS,
                            smoothing=10,
                            fps=FPS,
                            ylim_minmax=(-20,330)) # ylim_minmax for speed feature only

    # TODO: Worms were NOT ON FOOD in iron(III)sulphate + enterobactin plates
    # Experiment 1: N2 and VC2591 (flp-2) worms - source plates: [2,7,11,12,13,14,15,16] 
    # Experiment 2: N2 with iron(III)sulphate + enterobactin - source plates: [1,3,4,5,6,8,9,10] 
    # with control H2O plates for iron: [7,11,14] wells [A1,A2,A3] only (from Experiment 1)

