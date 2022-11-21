#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iron + Enterobactin

# Experiment 2: N2 with iron(III)sulphate + enterobactin - source plates: [1,3,4,5,6,8,9,10] 
# with control H2O plates for iron: [7,11,14] wells [A1,A2,A3] only (from Experiment 1)   

@author: sm5911
@date: 29/09/2022

"""

#%% Imports

import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from write_data.write import write_list_to_file
from visualisation.plotting_helper import sig_asterix, boxplots_sigfeats, all_in_one_boxplots
from analysis.keio_screen.follow_up.uv_paraquat_antioxidant import masked_video_list_from_metadata
from time_series.plot_timeseries import plot_timeseries_feature#, selected_strains_timeseries

from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from tierpsytools.preprocessing.filter_data import select_feat_set

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Keio_Iron_Enterobactin"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Iron_Enterobactin"

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

treatment_cols = ['worm_strain','bacteria_strain','drug_type','drug2_type','drug2_solvent']

#%% Functions

def do_stats(metadata,
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

def make_boxplots(metadata,
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
                            
    # load control data for N2 on BW with no DMSO from + append
    control_data_dir = '/Users/sm5911/Documents/Keio_Worm_Stress_Mutants'
    control_metadata = pd.read_csv(Path(control_data_dir) / 'metadata.csv', dtype={'comments':str})    
    control_metadata = control_metadata.query("worm_strain=='N2' and " +
                                              "drug_type!='Paraquat'")
    
    control_features = pd.read_csv(Path(control_data_dir) / 'features.csv')
    control_features = control_features.reindex(control_metadata.index)
    
    missing_cols = set(metadata.columns) - set(control_metadata.columns) # no drug2 columns
    
    metadata = pd.concat([metadata, control_metadata], axis=0, ignore_index=True)
    features = pd.concat([features, control_features], axis=0, ignore_index=True)
 
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
    if not 'bluelight' in metadata.columns:
        metadata['bluelight'] = [i.split('_run')[-1].split('_')[1] for i in metadata['imgstore_name']]
    metadata = metadata[metadata['bluelight']=='bluelight']

    metadata['drug_type'] = metadata['drug_type'].fillna('None')
    metadata['drug2_type'] = metadata['drug2_type'].fillna('None')
    metadata['drug2_solvent'] = metadata['drug2_solvent'].fillna('None') 
    
    # reindex features
    features = features.reindex(metadata.index)    
    
    metadata['treatment'] = metadata.loc[:,treatment_cols].astype(str).agg('-'.join, axis=1)
    strain_list = sorted(list(metadata['treatment'].unique()))
 
    metadata['window'] = metadata['window'].astype(int)
    window_list = list(metadata['window'].unique())

    for window in tqdm(window_list):
        
        meta_window = metadata[metadata['window']==window]
        feat_window = features.reindex(meta_window.index)

        stats_dir = Path(SAVE_DIR) / 'Stats' / WINDOW_NAME_DICT[window]
        plot_dir = Path(SAVE_DIR) / 'Plots' / WINDOW_NAME_DICT[window]
        
        # DMSO control
        DMSO_window_meta = meta_window.query("worm_strain=='N2' and bacteria_strain=='BW' and " + 
                                             "drug_type!='Paraquat' and drug_type!='Fe2O12S3' and " +
                                             "drug2_type!='Enterobactin'")
        mapping_dict = {'N2-BW-None-None-None':'BW',
                        'N2-BW-None-None-DMSO':'BW-DMSO'}
        DMSO_window_meta['treatment'] = DMSO_window_meta['treatment'].map(mapping_dict)

        DMSO_window_feat = features.reindex(DMSO_window_meta.index)

        do_stats(DMSO_window_meta,
                 DMSO_window_feat,
                 group_by='treatment',
                 control='BW',
                 save_dir=stats_dir / 'DMSO_control',
                 feature_set=feature_list,
                 pvalue_threshold=0.05,
                 fdr_method='fdr_bh')
        
        make_boxplots(DMSO_window_meta,
                      DMSO_window_feat,
                      group_by='treatment',
                      control='BW',
                      save_dir=plot_dir / 'DMSO_control',
                      stats_dir=stats_dir / 'DMSO_control',
                      feature_set=feature_list,
                      pvalue_threshold=0.05,
                      scale_outliers=False,
                      ylim_minmax=(-20,330)) # ylim_minmax for speed feature only 
                
        # Enterobactin
        meta_ent = meta_window.query("drug_type!='Fe2O12S3'")
        feat_ent = feat_window.reindex(meta_ent.index)
        
        mapping_dict = {'N2-BW-None-None-None':'BW-None',
                        'N2-fepD-None-None-None':'fepD-None',
                        'N2-BW-None-None-DMSO':'BW-DMSO',
                        'N2-fepD-None-None-DMSO':'fepD-DMSO',
                        'N2-BW-None-Enterobactin-DMSO':'BW-Enterobactin', 
                        'N2-fepD-None-Enterobactin-DMSO':'fepD-Enterobactin'}
        meta_ent['treatment'] = meta_ent['treatment'].map(mapping_dict)
        meta_ent['drug'] = [t.split('-')[-1] for t in meta_ent['treatment']]
        
        do_stats(meta_ent,
                 feat_ent,
                 group_by='treatment',
                 control='BW-DMSO',
                 save_dir=stats_dir / 'Enterobactin',
                 feature_set=feature_list,
                 pvalue_threshold=0.05,
                 fdr_method='fdr_bh')
        
        colour_dict = dict(zip(['BW','fepD'], sns.color_palette('tab10',2)))
        all_in_one_boxplots(meta_ent,
                            feat_ent,
                            group_by='drug',
                            hue='bacteria_strain',
                            hue_order=['BW','fepD'],
                            order=['None','DMSO','Enterobactin'],
                            save_dir=plot_dir / 'Enterobactin',
                            ttest_path=None,
                            feature_set=feature_list,
                            pvalue_threshold=0.05,
                            sigasterix=True,
                            colour_dict=colour_dict,
                            fontsize=25,
                            figsize=(12,8),
                            vline_boxpos=None,
                            legend=False,
                            ylim_minmax=(-20,300),
                            subplots_adjust={'bottom':0.3,'top':0.95,'left':0.15,'right':0.95})
        
        # iron
        meta_iron = meta_window.query("drug2_type!='Enterobactin' and drug2_solvent!='DMSO'")
        feat_iron = feat_window.reindex(meta_iron.index)

        mapping_dict = {'N2-BW-None-None-None':'BW-None',
                        'N2-fepD-None-None-None':'fepD-None',
                        'N2-BW-Fe2O12S3-None-None':'BW-Iron', 
                        'N2-fepD-Fe2O12S3-None-None':'fepD-Iron'}
        meta_iron['treatment'] = meta_iron['treatment'].map(mapping_dict)
        meta_iron['drug'] = [t.split('-')[-1] for t in meta_iron['treatment']]

        do_stats(meta_iron,
                 feat_iron,
                 group_by='treatment',
                 control='BW-None',
                 save_dir=stats_dir / 'Iron(III)suplhate',
                 feature_set=feature_list,
                 pvalue_threshold=0.05,
                 fdr_method='fdr_bh')
        
        colour_dict = dict(zip(['BW','fepD'], sns.color_palette('tab10',2)))
        all_in_one_boxplots(meta_iron,
                            feat_iron,
                            group_by='drug',
                            hue='bacteria_strain',
                            hue_order=['BW','fepD'],
                            order=['None','Iron'],
                            save_dir=plot_dir / 'Iron(III)suplhate',
                            ttest_path=None, #stats_dir / 'Iron(III)suplhate' / 't-test' / 't-test_results.csv'
                            feature_set=feature_list,
                            pvalue_threshold=0.05,
                            sigasterix=True,
                            colour_dict=colour_dict,
                            override_palette_dict={2:'lightskyblue',3:'sandybrown'},
                            fontsize=25,
                            figsize=(12,8),
                            vline_boxpos=None,
                            legend=False,
                            ylim_minmax=(-20,300),
                            subplots_adjust={'bottom':0.2,'top':0.95,'left':0.15,'right':0.95})
    
    metadata = metadata[metadata['window']==0]
    
    mapping_dict = {'N2-BW-None-None-None':'BW',
                    'N2-fepD-None-None-None':'fepD',
                    'N2-BW-None-None-DMSO':'BW-DMSO',
                    'N2-fepD-None-None-DMSO':'fepD-DMSO',
                    'N2-BW-None-Enterobactin-DMSO':'BW-Enterobactin-DMSO', 
                    'N2-fepD-None-Enterobactin-DMSO':'fepD-Enterobactin-DMSO',
                    'N2-BW-Fe2O12S3-None-None':'BW + iron', 
                    'N2-fepD-Fe2O12S3-None-None':'fepD + iron',
                    'N2-BW-Fe2O12S3-Enterobactin-DMSO':'BW + iron + Ent',
                    'N2-fepD-Fe2O12S3-Enterobactin-DMSO':'fepD + iron + Ent'}
    metadata['treatment'] = metadata['treatment'].map(mapping_dict)

    # timeseries plots of speed for BW + iron vs BW control
    groups_list = ['BW', 'BW + iron']
    plot_timeseries_feature(metadata,
                            project_dir=Path(PROJECT_DIR),
                            save_dir=Path(SAVE_DIR) / 'timeseries-speed' / 'all',
                            group_by='treatment',
                            control='BW',
                            groups_list=groups_list,
                            feature='speed',
                            n_wells=6,
                            bluelight_stim_type='bluelight',
                            video_length_seconds=360,
                            bluelight_timepoints_seconds=BLUELIGHT_TIMEPOINTS_SECONDS,
                            smoothing=10,
                            ylim_minmax=(-20,300),
                            fps=FPS,
                            col_dict={'BW':colour_dict['BW'],'BW + iron':'lightskyblue'})         

    # timeseries plots of speed for each fepD + iron vs fepD control   
    groups_list = ['fepD','fepD + iron']
    plot_timeseries_feature(metadata,
                            project_dir=Path(PROJECT_DIR),
                            save_dir=Path(SAVE_DIR) / 'timeseries-speed' / 'all',
                            group_by='treatment',
                            control='fepD',
                            groups_list=groups_list,
                            feature='speed',
                            n_wells=6,
                            bluelight_stim_type='bluelight',
                            video_length_seconds=360,
                            bluelight_timepoints_seconds=BLUELIGHT_TIMEPOINTS_SECONDS,
                            smoothing=10,
                            ylim_minmax=(-20,300),
                            fps=FPS,
                            col_dict={'fepD':colour_dict['fepD'],'fepD + iron':'sandybrown'})
 
    # timeseries plots of speed for BW + ent vs BW DMSO control    
    groups_list = ['BW-DMSO', 'BW-Enterobactin-DMSO']
    plot_timeseries_feature(metadata,
                            project_dir=Path(PROJECT_DIR),
                            save_dir=Path(SAVE_DIR) / 'timeseries-speed' / 'all',
                            group_by='treatment',
                            control='BW-DMSO',
                            groups_list=groups_list,
                            feature='speed',
                            n_wells=6,
                            bluelight_stim_type='bluelight',
                            video_length_seconds=360,
                            bluelight_timepoints_seconds=BLUELIGHT_TIMEPOINTS_SECONDS,
                            smoothing=10,
                            ylim_minmax=(-20,300),
                            fps=FPS,
                            col_dict={'BW-DMSO':colour_dict['BW'],'BW-Enterobactin-DMSO':'lightskyblue'})         

    # timeseries plots of speed for each fepD + ent vs fepD DMSO control   
    groups_list = ['fepD-DMSO','fepD-Enterobactin-DMSO']
    plot_timeseries_feature(metadata,
                            project_dir=Path(PROJECT_DIR),
                            save_dir=Path(SAVE_DIR) / 'timeseries-speed' / 'all',
                            group_by='treatment',
                            control='fepD-DMSO',
                            groups_list=groups_list,
                            feature='speed',
                            n_wells=6,
                            bluelight_stim_type='bluelight',
                            video_length_seconds=360,
                            bluelight_timepoints_seconds=BLUELIGHT_TIMEPOINTS_SECONDS,
                            smoothing=10,
                            ylim_minmax=(-20,300),
                            fps=FPS,
                            col_dict={'fepD-DMSO':colour_dict['fepD'],'fepD-Enterobactin-DMSO':'sandybrown'})
    
    # XXX: Worms were NOT ON FOOD in iron(III)sulphate + enterobactin plates     

