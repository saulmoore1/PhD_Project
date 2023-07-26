#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keio fepD vs BW: supplementation with enterobactin/iron/paraquat

Adding supplements BW and fepD lawns
- 

@author: sm5911
@date: 11/05/2022

"""

#%% Imports

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from visualisation.plotting_helper import sig_asterix, all_in_one_boxplots
from write_data.write import write_list_to_file
from time_series.plot_timeseries import plot_window_timeseries_feature # plot_timeseries_motion_mode
# from analysis.keio_screen.check_keio_screen_worm_trajectories import check_tracked_objects

from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from tierpsytools.preprocessing.filter_data import select_feat_set

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Iron_Ent_Paraquat"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Iron_Ent_Paraquat"
N_WELLS = 6

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
MIN_NSKEL_PER_VIDEO = None
MIN_NSKEL_SUM = 500

FEATURE_SET = ['speed_50th']

# Featsums 10 seconds centred on end of each BL pulse and also 20-30 seconds after end of each BL pulse
WINDOW_DICT_SECONDS = {0:(1805,1815), 1:(1830,1840), 2:(1865,1875),
                       3:(1890,1900), 4:(1925,1935), 5:(1950,1960)}

WINDOW_NAME_DICT = {0:"blue light 1", 1: "20-30 seconds after blue light 1",
                    2:"blue light 2", 3: "20-30 seconds after blue light 2",
                    4:"blue light 3", 5: "20-30 seconds after blue light 3"}

food_type_list = ['BW', 'fepD']
drug_type_list = ['none', 'enterobactin', 'feCl3', 'fe2O12S3', 'paraquat']
paraquat_conc_mM = [0.5, 1, 2, 4]
iron_conc_mM = [1, 4]
iron_treatment_list = ['BW-none-nan', 'BW-feCl3-1.0', 'BW-feCl3-4.0',
                       'BW-fe2O12S3-1.0', 'BW-fe2O12S3-4.0', 
                       'fepD-none-nan', 'fepD-feCl3-1.0', 'fepD-feCl3-4.0', 
                       'fepD-fe2O12S3-1.0', 'fepD-fe2O12S3-4.0']
paraquat_treatment_list = ['BW-none-nan', 'BW-paraquat-0.5', 'BW-paraquat-1.0', 
                           'BW-paraquat-2.0', 'BW-paraquat-4.0', 
                           'fepD-none-nan', 'fepD-paraquat-0.5', 'fepD-paraquat-1.0',
                           'fepD-paraquat-2.0', 'fepD-paraquat-4.0']
    
BLUELIGHT_TIMEPOINTS_MINUTES = [30,31,32]
FPS = 25
VIDEO_LENGTH_SECONDS = 38*60
SMOOTH_WINDOW_SECONDS = 5
BLUELIGHT_WINDOWS_ONLY_TS = True

#%% Functions

def supplements_stats(metadata,
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


def main():
    
    AUX_DIR = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    RES_DIR = Path(PROJECT_DIR) / 'Results'

    META_PATH = Path(SAVE_DIR) / 'metadata.csv'
    FEAT_PATH = Path(SAVE_DIR) / 'features.csv'
    
    if not META_PATH.exists() or not FEAT_PATH.exists():
        Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)

        # compile metadata
        metadata, metadata_path = compile_metadata(aux_dir=AUX_DIR, 
                                                   imaging_dates=None, 
                                                   n_wells=N_WELLS,
                                                   add_well_annotations=False,
                                                   from_source_plate=True)
        
        # compile window summaries
        features, metadata = process_feature_summaries(metadata_path=metadata_path, 
                                                       results_dir=RES_DIR, 
                                                       compile_day_summaries=True,
                                                       imaging_dates=None, 
                                                       align_bluelight=False,
                                                       window_summaries=True,
                                                       n_wells=N_WELLS)
                        
        # clean results
        features, metadata = clean_summary_results(features, 
                                                   metadata,
                                                   feature_columns=None,
                                                   nan_threshold_row=NAN_THRESHOLD_ROW,
                                                   nan_threshold_col=NAN_THRESHOLD_COL,
                                                   max_value_cap=1e15,
                                                   imputeNaN=True,
                                                   min_nskel_per_video=MIN_NSKEL_PER_VIDEO,
                                                   min_nskel_sum=MIN_NSKEL_SUM,
                                                   drop_size_related_feats=False,
                                                   norm_feats_only=False,
                                                   percentile_to_use=None)

        assert not features.isna().sum(axis=1).any()
        assert not (features.std(axis=1) == 0).any()
    
        # save features
        metadata.to_csv(META_PATH, index=False)
        features.to_csv(FEAT_PATH, index=False) 
    
    else:
        # load clean metadata and features
        metadata = pd.read_csv(META_PATH, dtype={'comments':str, 'source_plate_id':str})
        features = pd.read_csv(FEAT_PATH, index_col=None)

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

    treatment_cols = ['food_type','drug_type','imaging_plate_drug_conc','solvent']
    metadata['treatment'] = metadata[treatment_cols].astype(str).agg('-'.join, axis=1) 
    order = ['BW-none-nan-nan','BW-none-nan-DMSO',
             'fepD-none-nan-nan','fepD-none-nan-DMSO',
             'BW-paraquat-0.5-H2O','BW-paraquat-1.0-H2O',
             'BW-paraquat-2.0-H2O','BW-paraquat-4.0-H2O',
             'fepD-paraquat-0.5-H2O','fepD-paraquat-1.0-H2O',
             'fepD-paraquat-2.0-H2O','fepD-paraquat-4.0-H2O',
             'BW-enterobactin-nan-DMSO','fepD-enterobactin-nan-DMSO',
             'BW-feCl3-1.0-NGM','BW-feCl3-4.0-NGM','BW-feCl3-4.0-H2O',
             'fepD-feCl3-1.0-NGM','fepD-feCl3-4.0-NGM','fepD-feCl3-4.0-H2O',
             'BW-fe2O12S3-1.0-NGM','BW-fe2O12S3-4.0-NGM','BW-fe2O12S3-4.0-H2O',
             'fepD-fe2O12S3-1.0-NGM','fepD-fe2O12S3-4.0-NGM','fepD-fe2O12S3-4.0-H2O']
    control = 'BW-none-nan-nan'

    # subset metadata results for bluelight videos only 
    bluelight_videos = [i for i in metadata['imgstore_name'] if 'bluelight' in i]
    metadata = metadata[metadata['imgstore_name'].isin(bluelight_videos)]

    metadata['window'] = metadata['window'].astype(int)
    window_list = list(metadata['window'].unique())

    for window in tqdm(window_list):
        meta_window = metadata[metadata['window']==window]
        feat_window = features.reindex(meta_window.index)
        
        stats_dir = Path(SAVE_DIR) / 'Stats' / WINDOW_NAME_DICT[window]
        plot_dir = Path(SAVE_DIR) / 'Plots' / WINDOW_NAME_DICT[window]

        # T-tests comparing supplementation of BW and fepD lawns with enterobactin, iron and paraquat
        # against BW and fepD without drugs added: 
        # - BW vs BW + enterobactin
        # - BW vs BW + iron
        # - BW vs BW + paraquat
        # - fepD vs fepD + enterobactin
        # - fepD vs fepD + iron
        # - fepD vs fepD + paraquat        
        # paraquat: fepD vs BW
        
        meta_paraquat = meta_window[np.logical_or(meta_window['drug_type']=='paraquat',
                                                  np.logical_and(meta_window['drug_type']=='none',
                                                                 meta_window['solvent']!='DMSO'))]
        feat_paraquat = feat_window.reindex(meta_paraquat.index)
        supplements_stats(meta_paraquat,
                          feat_paraquat,
                          group_by='treatment',
                          control=control,
                          save_dir=stats_dir / 'paraquat',
                          feature_set=FEATURE_SET,
                          pvalue_threshold=0.05,
                          fdr_method='fdr_bh')
        colour_labels = sns.color_palette('tab10', 2)
        groups = meta_paraquat['treatment'].unique()
        colours = [colour_labels[0] if 'BW' in s else colour_labels[1] for s in groups]
        colour_dict = {key:col for (key,col) in zip(groups, colours)}
        all_in_one_boxplots(meta_paraquat,
                            feat_paraquat,
                            group_by='treatment',
                            control=control,
                            save_dir=plot_dir / 'all-in-one' / 'paraquat',
                            ttest_path=stats_dir / 'paraquat' / 't-test' / 't-test_results.csv',
                            feature_set=feature_list,
                            pvalue_threshold=0.05,
                            sigasterix=True,
                            order=[s for s in order if s in groups],
                            colour_dict=colour_dict,
                            figsize=(30,10),
                            # ylim_minmax=(-20,130),
                            vline_boxpos=[1],
                            fontsize=20,
                            subplots_adjust={'bottom':0.4,'top':0.95,'left':0.05,'right':0.98})
        
        # enterobactin: fepD vs BW
        meta_ent = meta_window[np.logical_or(meta_window['drug_type']=='enterobactin',
                                             meta_window['drug_type']=='none')]
        feat_ent = feat_window.reindex(meta_ent.index)
        supplements_stats(meta_ent,
                          feat_ent,
                          group_by='treatment',
                          control=control,
                          save_dir=stats_dir / 'enterobactin',
                          feature_set=FEATURE_SET,
                          pvalue_threshold=0.05,
                          fdr_method='fdr_bh')
        colour_labels = sns.color_palette('tab10', 2)
        groups = meta_ent['treatment'].unique()
        colours = [colour_labels[0] if 'BW' in s else colour_labels[1] for s in groups]
        colour_dict = {key:col for (key,col) in zip(groups, colours)}
        all_in_one_boxplots(meta_ent,
                            feat_ent,
                            group_by='treatment',
                            control=control,
                            save_dir=plot_dir / 'all-in-one' / 'enterobactin',
                            ttest_path=stats_dir / 'enterobactin' / 't-test' / 't-test_results.csv',
                            feature_set=feature_list,
                            pvalue_threshold=0.05,
                            sigasterix=True,
                            order=[s for s in order if s in groups],
                            colour_dict=colour_dict,
                            figsize=(30,10),
                            # ylim_minmax=(-20,130),
                            vline_boxpos=[3],
                            fontsize=20,
                            subplots_adjust={'bottom':0.4,'top':0.95,'left':0.05,'right':0.98})
        
        # iron: fepD vs BW (H2O solvent only)
        meta_iron = meta_window[np.logical_or(np.logical_or(meta_window['drug_type']=='feCl3',
                                                            meta_window['drug_type']=='fe2O12S3'),
                                              np.logical_and(meta_window['drug_type']=='none',
                                                             meta_window['solvent']!='DMSO'))]
        feat_iron = feat_window.reindex(meta_iron.index)
        supplements_stats(meta_iron,
                          feat_iron,
                          group_by='treatment',
                          control=control,
                          save_dir=stats_dir / 'iron',
                          feature_set=FEATURE_SET,
                          pvalue_threshold=0.05,
                          fdr_method='fdr_bh')
        colour_labels = sns.color_palette('tab10', 2)
        groups = meta_iron['treatment'].unique()
        colours = [colour_labels[0] if 'BW' in s else colour_labels[1] for s in groups]
        colour_dict = {key:col for (key,col) in zip(groups, colours)}
        all_in_one_boxplots(meta_iron,
                            feat_iron,
                            group_by='treatment',
                            control=control,
                            sigasterix=True,
                            save_dir=plot_dir / 'all-in-one' / 'iron',
                            ttest_path=stats_dir / 'iron' / 't-test' / 't-test_results.csv',
                            feature_set=feature_list,
                            pvalue_threshold=0.05,
                            order=[s for s in order if s in groups],
                            colour_dict=colour_dict,
                            figsize=(20,10),
                            # ylim_minmax=(-20,130),
                            vline_boxpos=[1,4,7,10],
                            fontsize=20,
                            subplots_adjust={'bottom':0.4,'top':0.95,'left':0.05,'right':0.98})

    metadata = metadata[metadata['window']==0]
        
    # timeseries plots of speed for fepD vs BW control
    BLUELIGHT_TIMEPOINTS_SECONDS = [(i*60,i*60+10) for i in BLUELIGHT_TIMEPOINTS_MINUTES]

    plot_window_timeseries_feature(metadata=metadata,
                                   project_dir=Path(PROJECT_DIR),
                                   save_dir=Path(SAVE_DIR) / 'timeseries-speed',
                                   group_by='treatment',
                                   control=control,
                                   groups_list=None,
                                   feature='speed',
                                   n_wells=6,
                                   bluelight_timepoints_seconds=BLUELIGHT_TIMEPOINTS_SECONDS,
                                   bluelight_windows_separately=True,
                                   smoothing=10,
                                   figsize=(15,5),
                                   fps=FPS,
                                   ylim_minmax=(-20,220),
                                   xlim_crop_around_bluelight_seconds=(120,300),
                                   video_length_seconds=VIDEO_LENGTH_SECONDS)

    # # Check length/area of tracked objects - prop bad skeletons
    # results_df = check_tracked_objects(metadata, 
    #                                    length_minmax=(200, 2000), 
    #                                    width_minmax=(20, 500),
    #                                    save_to=Path(SAVE_DIR) / 'tracking_checks.csv')

    return

#%% Main

if __name__ == '__main__':
    main()
    
    