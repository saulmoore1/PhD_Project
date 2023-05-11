#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse Proteomics Mutants

Investigate whether the arousal phenotype is present in any of the double-knockout and 
over-expression mutants prepared to investigate the differentially regulated genes between 
fepD and BW highlighted by proteomics analysis

@author: sm5911
@date: 28/06/2022

"""

#%% Imports

import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from visualisation.plotting_helper import sig_asterix, boxplots_sigfeats, all_in_one_boxplots
from write_data.write import write_list_to_file
from time_series.plot_timeseries import plot_timeseries_feature #selected_strains_timeseries
# from analysis.keio_screen.check_keio_screen_worm_trajectories import check_tracked_objects

from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from tierpsytools.preprocessing.filter_data import select_feat_set

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Proteomics_Mutants"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Proteomics_Mutants"

N_WELLS = 6
FPS = 25

nan_threshold_row = 0.8
nan_threshold_col = 0.05

FEATURE_SET = ['speed_50th']

BLUELIGHT_TIMEPOINTS_SECONDS = [(60, 70),(160, 170),(260, 270)]

WINDOW_DICT = {0:(65,75),1:(90,100),
               2:(165,175),3:(190,200),
               4:(265,275),5:(290,300)}

WINDOW_NAME_DICT = {0:"blue light 1", 1: "20-30 seconds after blue light 1",
                    2:"blue light 2", 3: "20-30 seconds after blue light 2",
                    4:"blue light 3", 5: "20-30 seconds after blue light 3"}

#%% Functions

def proteomics_mutants_stats(metadata,
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

def proteomics_mutants_boxplots(metadata,
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


def main():
    
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
    metadata['treatment'] = metadata[['food_type','drug_type']].astype(str).agg('-'.join, axis=1)
    metadata['treatment'] = [i.replace('-nan','') for i in metadata['treatment']]

    strain_list = list(metadata['treatment'].unique())
    fepD_strains = [s for s in strain_list if 'fepD' in s]
    OE_strains = [s for s in fepD_strains if 'iptg' in s]
    KO_strains = [s for s in fepD_strains if s not in OE_strains]

    metadata['window'] = metadata['window'].astype(int)
    window_list = list(metadata['window'].unique())

    # boxplots comparing each treatment to control for each feature
    # fixed scale across plots for speed to 0-250 um/sec for easier comparison across conditions
    for window in tqdm(window_list):
        meta_window = metadata[metadata['window']==window]
        feat_window = features.reindex(meta_window.index)
    
        proteomics_mutants_stats(meta_window,
                                 feat_window,
                                 group_by='treatment',
                                 control='BW',
                                 save_dir=Path(SAVE_DIR) / 'Stats' / WINDOW_NAME_DICT[window],
                                 feature_set=feature_list,
                                 pvalue_threshold=0.05,
                                 fdr_method='fdr_bh')

        # boxplots comparing each treatment to BW control for each feature
        proteomics_mutants_boxplots(meta_window,
                                    feat_window,
                                    group_by='treatment',
                                    control='BW',
                                    feature_set=feature_list,
                                    save_dir=Path(SAVE_DIR) / 'Plots' / WINDOW_NAME_DICT[window],
                                    stats_dir=Path(SAVE_DIR) / 'Stats' / WINDOW_NAME_DICT[window],
                                    pvalue_threshold=0.05)
        
        # compare OE mutants to fepD-iptg        
        meta_OE = meta_window[meta_window['treatment'].isin(OE_strains)]
        feat_OE = feat_window.reindex(meta_OE.index)
        proteomics_mutants_stats(meta_OE,
                                 feat_OE,
                                 group_by='treatment',
                                 control='fepD-iptg',
                                 save_dir=Path(SAVE_DIR) / 'Stats_OE' / WINDOW_NAME_DICT[window],
                                 feature_set=feature_list,
                                 pvalue_threshold=0.05,
                                 fdr_method='fdr_bh')
        
        colour_dict = dict(zip(sorted(OE_strains), sns.color_palette('Set2', len(OE_strains))))
        all_in_one_boxplots(meta_OE,
                            feat_OE,
                            group_by='treatment',
                            control='fepD-iptg',
                            sigasterix=True,
                            fontsize=15,
                            order=sorted(OE_strains),
                            colour_dict=colour_dict,
                            feature_set=feature_list,
                            save_dir=Path(SAVE_DIR) / 'Plots_OE' / WINDOW_NAME_DICT[window] / 'all-in-one',
                            ttest_path=Path(SAVE_DIR) / 'Stats_OE' / WINDOW_NAME_DICT[window] /\
                                't-test' / 't-test_results.csv',
                            pvalue_threshold=0.05,
                            figsize=(6,9),
                            ylim_minmax=(0,260),
                            subplots_adjust={'bottom':0.35,'top':0.95,'left':0.2,'right':0.95})
        
        # compare KO mutants to fepD
        meta_KO = meta_window[meta_window['treatment'].isin(KO_strains)]
        feat_KO = feat_window.reindex(meta_KO.index)
        proteomics_mutants_stats(meta_KO,
                                 feat_KO,
                                 group_by='treatment',
                                 control='fepD',
                                 save_dir=Path(SAVE_DIR) / 'Stats_KO' / WINDOW_NAME_DICT[window],
                                 feature_set=feature_list,
                                 pvalue_threshold=0.05,
                                 fdr_method='fdr_bh')
        
        colour_dict = dict(zip(sorted(KO_strains), sns.color_palette('Set2', len(KO_strains))))
        all_in_one_boxplots(meta_KO,
                            feat_KO,
                            group_by='treatment',
                            control='fepD',
                            sigasterix=True,
                            fontsize=15,
                            order=sorted(KO_strains),
                            colour_dict=colour_dict,
                            feature_set=feature_list,
                            save_dir=Path(SAVE_DIR) / 'Plots_KO' / WINDOW_NAME_DICT[window] / 'all-in-one',
                            ttest_path=Path(SAVE_DIR) / 'Stats_KO' / WINDOW_NAME_DICT[window] /\
                                't-test' / 't-test_results.csv',
                            pvalue_threshold=0.05,
                            figsize=(15,8),
                            ylim_minmax=(0,260),
                            subplots_adjust={'bottom':0.25,'top':0.95,'left':0.1,'right':0.95})
                        
    # # timeseries motion mode fraction for each treatment vs BW control
    # selected_strains_timeseries(metadata,
    #                             project_dir=Path(PROJECT_DIR), 
    #                             save_dir=Path(SAVE_DIR) / 'timeseries', 
    #                             strain_list=strain_list,
    #                             group_by='treatment',
    #                             control='BW',
    #                             n_wells=6,
    #                             bluelight_stim_type='bluelight',
    #                             video_length_seconds=360,
    #                             bluelight_timepoints_seconds=BLUELIGHT_TIMEPOINTS_SECONDS,
    #                             motion_modes=['forwards','paused','backwards'],
    #                             smoothing=10,
    #                             fps=FPS)    
    
    metadata = metadata[metadata['window']==0]
    
    # # timeseries plots of speed for each treatment vs control
    # plot_timeseries_feature(metadata,
    #                         project_dir=Path(PROJECT_DIR),
    #                         save_dir=Path(SAVE_DIR) / 'timeseries-speed',
    #                         group_by='treatment',
    #                         control='BW',
    #                         groups_list=strain_list,
    #                         feature='speed',
    #                         n_wells=6,
    #                         bluelight_stim_type='bluelight',
    #                         video_length_seconds=360,
    #                         bluelight_timepoints_seconds=BLUELIGHT_TIMEPOINTS_SECONDS,
    #                         smoothing=10,
    #                         fps=FPS,
    #                         ylim_minmax=(-20,330)) # fixed the scale across plots to -10 to 310 um/sec
  
    # timeseries plot of speed for OE strains vs fepD-iptg
    plot_timeseries_feature(metadata,
                            project_dir=Path(PROJECT_DIR),
                            save_dir=Path(SAVE_DIR) / 'timeseries-speed' / 'OE_strains',
                            group_by='treatment',
                            control='fepD-iptg',
                            groups_list=OE_strains,
                            feature='speed',
                            n_wells=6,
                            bluelight_stim_type='bluelight',
                            video_length_seconds=360,
                            bluelight_timepoints_seconds=BLUELIGHT_TIMEPOINTS_SECONDS,
                            smoothing=10,
                            fps=FPS,
                            ylim_minmax=(-20,330)) # fixed the scale across plots to -10 to 310 um/sec

    # timeseries plot of speed for KO strains vs fepD
    plot_timeseries_feature(metadata,
                            project_dir=Path(PROJECT_DIR),
                            save_dir=Path(SAVE_DIR) / 'timeseries-speed' / 'KO_strains',
                            group_by='treatment',
                            control='fepD',
                            groups_list=KO_strains,
                            feature='speed',
                            n_wells=6,
                            bluelight_stim_type='bluelight',
                            video_length_seconds=360,
                            bluelight_timepoints_seconds=BLUELIGHT_TIMEPOINTS_SECONDS,
                            smoothing=10,
                            fps=FPS,
                            ylim_minmax=(-20,330),
                            col_dict=colour_dict) # fixed the scale across plots to -10 to 310 um/sec
    
    return


#%% Main

if __name__ == '__main__':
    main()
    