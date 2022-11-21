#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keio Worm Stress Mutants 1

@author: sm5911
@date: 30/07/2022

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
from write_data.write import write_list_to_file
from visualisation.plotting_helper import sig_asterix, boxplots_sigfeats, all_in_one_boxplots
from analysis.keio_screen.follow_up.uv_paraquat_antioxidant import masked_video_list_from_metadata
from time_series.plot_timeseries import plot_timeseries_feature, plot_timeseries #, selected_strains_timeseries
from time_series.time_series_helper import get_strain_timeseries

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

omit_strains = ["RID(RNAi)::unc-31","PY7505"]

strain_name_mapping_dict = {"N2":"N2",
                            # neuron ablation / RNAi
                            "ZD763":"ASJ-ablated",
                            "PY7505":"ASI-ablated",
                            "RID(RNAi)::unc-31":"RID(RNAi)",
                            "RID(RNAi)_lite-1(wt)":"RID(RNAi) lite-1(wt)",
                            # Antioxidant pathways
                            "prdx-2":"prdx-2",
                            "GA187":"sod-1",
                            "GA476":"sod-1; sod-5",
                            "GA184":"sod-2",
                            "GA813":"sod-1; sod-2",
                            "GA186":"sod-3",
                            "GA480":"sod-2; sod-3",
                            "GA416":"sod-4",
                            "GA822":"sod-4; sod-5",
                            "GA503":"sod-5",
                            "GA814":"sod-3; sod-5",
                            "GA800":"OE ctl-1+ctl-2+ctl-3",
                            "GA801":"OE sod-1",
                            "GA804":"OE sod-2",
                            "clk-1":"clk-1",
                            "gas-1":"gas-1",
                            "msrA":"msra-1",
                            # Neuropeptide pathways
                            "pdfr-1":"pdfr-1",
                            "PS8997":"flp-1",
                            "VC2490":"W07E11.1_&_flp-2",
                            "VC2591":"flp-2",
                            # Neurotransmitters
                            "eat-4":"eat-4",
                            # Mitochondria
                            "nuo-6":"nuo-6"}

worm_list_dict = {'neuron_ablation':["N2","ASJ-ablated","ASI-ablated","RID(RNAi)","RID(RNAi) lite-1(wt)"],
                  'neuropeptides_&_neurotransmitters':["N2","pdfr-1","flp-1","flp-2","W07E11.1_&_flp-2","eat-4"],
                  'antioxidant':["N2","prdx-2","sod-1","sod-1; sod-5","sod-2","sod-1; sod-2",
                                 "sod-3","sod-2; sod-3","sod-4","sod-4; sod-5","sod-5",
                                 "sod-3; sod-5","OE sod-1","OE sod-2","OE ctl-1+ctl-2+ctl-3",
                                 "clk-1","gas-1","msra-1","nuo-6"]}

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
    if not 'bluelight' in metadata.columns:
        metadata['bluelight'] = [i.split('_run')[-1].split('_')[1] for i in metadata['imgstore_name']]
    metadata = metadata[metadata['bluelight']=='bluelight']

    # omit results for unwanted strains from metadata
    metadata = metadata[~metadata['worm_strain'].isin(omit_strains)]
    metadata['worm_strain'] = metadata['worm_strain'].map(strain_name_mapping_dict)
    # reindex features
    features = features.reindex(metadata.index)

    treatment_cols = ['worm_strain','bacteria_strain','drug_type']
    metadata['treatment'] = metadata.loc[:,treatment_cols].astype(str).agg('-'.join, axis=1)
    control = 'N2-BW-nan'
    
    # save video file list for treatments (for manual inspection)
    video_dict = masked_video_list_from_metadata(metadata[metadata['window']==0], 
                                                 group_by='treatment', 
                                                 groups_list=None,
                                                 imgstore_col='imgstore_name',
                                                 project_dir=Path(PROJECT_DIR),
                                                 save_dir=Path(SAVE_DIR) / 'video_filenames')

        
    # metadata = metadata[['bluelight' in metadata.loc[i,'imgstore_name'] for i in metadata.index]].copy()
    # metadata['imgstore_name_bluelight'] = metadata['imgstore_name']
        
    # boxplots comparing each treatment to control for each feature
    # fixed scale across plots for speed to 0-250 um/sec for easier comparison across conditions
    metadata['window'] = metadata['window'].astype(int)
    window_list = list(metadata['window'].unique())

    strain_list = sorted(list(metadata['treatment'].unique()))
    no_paraquat_strains = [s for s in strain_list if not 'paraquat' in s.lower()]
    paraquat_strains = [s for s in strain_list if 'paraquat' in s.lower()]
        
    for window in tqdm(window_list):
        meta_window = metadata[metadata['window']==window]
        feat_window = features.reindex(meta_window.index)
        
        stats_dir = Path(SAVE_DIR) / 'Stats' / WINDOW_NAME_DICT[window]
        plot_dir = Path(SAVE_DIR) / 'Plots' / WINDOW_NAME_DICT[window]

        # all treatments
        treatment_cols = ['worm_strain','bacteria_strain','drug_type']
        meta_window['treatment'] = meta_window.loc[:,treatment_cols].astype(str).agg('-'.join, axis=1)
        control = 'N2-BW-nan'
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

        treatment_cols = ['worm_strain','bacteria_strain']
        control = 'N2-BW'        

        # without paraquat
        no_paraquat_meta = meta_window[meta_window['treatment'].isin(no_paraquat_strains)]
        no_paraquat_meta['treatment'] = no_paraquat_meta.loc[:,treatment_cols].astype(str).agg('-'.join, axis=1)
        no_paraquat_feat = feat_window.reindex(no_paraquat_meta.index)
 
        # worm strains vs N2 (without paraquat)
        worm_stress_mutants_stats(no_paraquat_meta,
                                  no_paraquat_feat,
                                  group_by='treatment',
                                  control=control,
                                  save_dir=stats_dir / 'no_paraquat',
                                  feature_set=feature_list,
                                  pvalue_threshold=0.05,
                                  fdr_method='fdr_bh')
        
        for worm_list in worm_list_dict.keys():
            no_para_meta = no_paraquat_meta[no_paraquat_meta['worm_strain'].isin(worm_list_dict[worm_list])]
            no_para_feat = no_paraquat_feat.reindex(no_para_meta.index)
            order = sorted(no_para_meta['worm_strain'].unique(), key=str.casefold)
            if worm_list == 'neuropeptides_&_neurotransmitters':
                order = [w for w in order if w != 'pdfr-1'] + ['pdfr-1']
            all_in_one_boxplots(no_para_meta,
                                no_para_feat,
                                group_by='worm_strain',
                                hue='bacteria_strain',
                                control='N2',
                                control_hue='BW',
                                order=['N2'] + [w for w in order if w != 'N2'],
                                hue_order=['BW','fepD'],
                                colour_dict=None,
                                save_dir=plot_dir / 'all-in-one' / 'no_paraquat' / worm_list,
                                ttest_path=stats_dir / 'no_paraquat' / 't-test' / 't-test_results.csv',
                                feature_set=feature_list,
                                pvalue_threshold=0.05,
                                sigasterix=True,
                                fontsize=40,
                                figsize=(max(len(worm_list_dict[worm_list])*2,10),16),
                                vline_boxpos=None,
                                ylim_minmax=(-50,260),
                                legend=False,
                                subplots_adjust={'bottom':0.4,'top':0.95,'left':0.1,'right':0.95})

        # with paraquat
        paraquat_meta = meta_window[meta_window['treatment'].isin(paraquat_strains)]
        paraquat_meta['treatment'] = paraquat_meta.loc[:,treatment_cols].astype(str).agg('-'.join, axis=1)
        paraquat_feat = feat_window.reindex(paraquat_meta.index)
 
        # worm strains vs N2 (with paraquat)
        worm_stress_mutants_stats(paraquat_meta,
                                  paraquat_feat,
                                  group_by='treatment',
                                  control=control,
                                  save_dir=stats_dir / 'paraquat',
                                  feature_set=feature_list,
                                  pvalue_threshold=0.05,
                                  fdr_method='fdr_bh')
        
        for worm_list in worm_list_dict.keys():
            para_meta = paraquat_meta[paraquat_meta['worm_strain'].isin(worm_list_dict[worm_list])]
            para_feat = paraquat_feat.reindex(para_meta.index)
            order = sorted(para_meta['worm_strain'].unique(), key=str.casefold)
            if worm_list == 'neuropeptides_&_neurotransmitters':
                order = [w for w in order if w != 'pdfr-1'] + ['pdfr-1']
            all_in_one_boxplots(para_meta,
                                para_feat,
                                group_by='worm_strain',
                                hue='bacteria_strain',
                                control='N2',
                                control_hue='BW',
                                order=['N2'] + [w for w in order if w != 'N2'],
                                hue_order=['BW','fepD'],
                                colour_dict=None,
                                save_dir=plot_dir / 'all-in-one' / 'paraquat' / worm_list,
                                ttest_path=stats_dir / 'paraquat' / 't-test' / 't-test_results.csv',
                                feature_set=feature_list,
                                pvalue_threshold=0.05,
                                sigasterix=True,
                                fontsize=40,
                                figsize=(max(len(worm_list_dict[worm_list])*2,10),16),
                                vline_boxpos=None,
                                ylim_minmax=(-50,260),
                                legend=False,
                                subplots_adjust={'bottom':0.4,'top':0.95,'left':0.1,'right':0.95})
                    
    # subset for single metadata window for full bluelight video timeseries plots
    metadata = metadata[metadata['window']==0]
    
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
                            save_dir=Path(SAVE_DIR) / 'timeseries-speed' / 'BW_control',
                            group_by='treatment',
                            control='N2-BW-nan',
                            groups_list=[s for s in strain_list if 'BW' in s and not 'paraquat' in s.lower()],
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
                            save_dir=Path(SAVE_DIR) / 'timeseries-speed' / 'BW_Paraquat_control',
                            group_by='treatment',
                            control='N2-BW-Paraquat',
                            groups_list=[s for s in strain_list if 'BW' in s and 'paraquat' in s.lower()],
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
                            groups_list=[s for s in strain_list if 'fepD' in s and not 'paraquat' in s.lower()],
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
                            groups_list=[s for s in strain_list if 'fepD' in s and 'paraquat' in s.lower()],
                            feature='speed',
                            n_wells=6,
                            bluelight_stim_type='bluelight',
                            video_length_seconds=360,
                            bluelight_timepoints_seconds=BLUELIGHT_TIMEPOINTS_SECONDS,
                            smoothing=10,
                            fps=FPS,
                            ylim_minmax=(-20,330)) # ylim_minmax for speed feature only
    
    # bespoke timeseries
    rescue_list = ['OE sod-2','clk-1','gas-1','sod-1; sod-2','sod-2','sod-2; sod-3','eat-4',
                   'RID(RNAi) lite-1(wt)','flp-2','W07E11.1_&_flp-2','pdfr-1']
    for strain in tqdm(rescue_list):
        groups = ['N2-BW-nan', strain + '-BW-nan', strain + '-fepD-nan']
        print("Plotting timeseries speed for %s" % strain)
                
        bluelight_frames = [(i*FPS, j*FPS) for (i, j) in BLUELIGHT_TIMEPOINTS_SECONDS]
        feature = 'speed'

        save_dir = Path(SAVE_DIR) / 'timeseries-speed' / 'rescues'
        ts_plot_dir = save_dir / 'Plots' / strain
        ts_plot_dir.mkdir(exist_ok=True, parents=True)
        save_path = ts_plot_dir / 'speed_bluelight.pdf'
        
        plt.close('all')
        fig, ax = plt.subplots(figsize=(15,6), dpi=300)
        col_dict = dict(zip(groups, sns.color_palette('tab10', len(groups))))

        for group in groups:
            
            # get control timeseries
            group_ts = get_strain_timeseries(metadata,
                                             project_dir=Path(PROJECT_DIR),
                                             strain=group,
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

        plt.ylim(-20, 300)
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
    
