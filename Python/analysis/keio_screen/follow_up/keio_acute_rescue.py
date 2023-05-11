#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse Keio Follow-up Acute Effect Antioxidant Rescue experiment
- window feature summaries for Ziwei's optimal windows around each bluelight stimulus
- Bluelight delivered for 10 seconds every 5 minutes, for a total of 45 minutes

When do we start to see an effect on worm behaviour? At which timepoint/window? 
Do we still see arousal of worms on siderophore mutants, even after a short period of time?

@author: sm5911
@date: 20/01/2022

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
from time_series.plot_timeseries import plot_window_timeseries_feature

from tierpsytools.preprocessing.filter_data import select_feat_set
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes

#%% Globals

PROJECT_DIR = '/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Acute_Rescue'
SAVE_DIR = '/Users/sm5911/Documents/Keio_Acute_Rescue'

N_WELLS = 6

FEATURE_SET = ['speed_50th']

nan_threshold_row = 0.8
nan_threshold_col = 0.05

scale_outliers_box = False

ALL_WINDOWS = False
WINDOW_LIST = None # if ALL_WINDOWS is False

# Featsums 10 seconds centred on end of each BL pulse and also 20-30 seconds after end of each BL pulse
WINDOW_DICT = {0:(305,315),1:(330,340),
               2:(605,615),3:(630,640),
               4:(905,915),5:(930,940),
               6:(1205,1215),7:(1230,1240),
               8:(1505,1515),9:(1530,1540),
               10:(1805,1815),11:(1830,1840),
               12:(2105,2115),13:(2130,2140),
               14:(2405,2415),15:(2430,2440)}

WINDOW_NAME_DICT = {0:"blue light 1", 1: "20-30 seconds after blue light 1",
                    2:"blue light 2", 3: "20-30 seconds after blue light 2",
                    4:"blue light 3", 5: "20-30 seconds after blue light 3",
                    6:"blue light 4", 7: "20-30 seconds after blue light 4",
                    8:"blue light 5", 9: "20-30 seconds after blue light 5",
                    10:"blue light 6", 11: "20-30 seconds after blue light 6",
                    12:"blue light 7", 13: "20-30 seconds after blue light 7",
                    14:"blue light 8", 15: "20-30 seconds after blue light 8"}

BLUELIGHT_TIMEPOINTS_MINUTES = [5,10,15,20,25,30,35,40]
FPS = 25
VIDEO_LENGTH_SECONDS = 45*60

#%% Functions

def acute_rescue_stats(metadata,
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

def acute_rescue_boxplots(metadata,
                          features,
                          control,
                          group_by='treatment',
                          feature_set=None,
                          save_dir=None,
                          stats_dir=None,
                          pvalue_threshold=0.05,
                          fdr_method='fdr_by'):
    
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
                      scale_outliers=True)
    
    return

def main():
    
    aux_dir = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    res_dir = Path(PROJECT_DIR) / 'Results'
    
    metadata_local_path = Path(SAVE_DIR) / 'metadata.csv'
    features_local_path = Path(SAVE_DIR) / 'features.csv'
    
    if not metadata_local_path.exists() or not features_local_path.exists():

        # load metadata    
        metadata, metadata_path = compile_metadata(aux_dir, 
                                                   imaging_dates=None, 
                                                   add_well_annotations=False, 
                                                   n_wells=N_WELLS)
        
        features, metadata = process_feature_summaries(metadata_path,
                                                       res_dir, 
                                                       compile_day_summaries=True, 
                                                       imaging_dates=None, 
                                                       align_bluelight=False, 
                                                       window_summaries=True,
                                                       n_wells=N_WELLS)
     
        # Clean results - Remove bad well data + features with too many NaNs/zero std + impute
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
                                                   norm_feats_only=False,
                                                   percentile_to_use=None)
    
        assert not features.isna().sum(axis=1).any()
        assert not (features.std(axis=1) == 0).any()
                
        # save clean metadata and features
        metadata.to_csv(metadata_local_path, index=False)
        features.to_csv(features_local_path, index=False) 
    
    else:
        # load clean metadata and features
        metadata = pd.read_csv(metadata_local_path, dtype={'comments':str, 'source_plate_id':str})
        features = pd.read_csv(features_local_path, index_col=None)

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
    
    # Drop results missing antioxidant name
    metadata = metadata[~metadata['antioxidant'].isna()]
    
    treatment_cols = ['gene_name','antioxidant']
    metadata['treatment'] = metadata[treatment_cols].astype(str).agg('-'.join, axis=1)
    control = 'BW-None'

    metadata['window'] = metadata['window'].astype(int)
    window_list = list(metadata['window'].unique())

    # drop results for trolox and resveratrol
    metadata = metadata.query("antioxidant=='None' or antioxidant=='vitC' or antioxidant=='NAC'")

    for window in tqdm(window_list):
        window_meta = metadata[metadata['window']==window]
        window_feat = features.reindex(window_meta.index)
        
        stats_dir = Path(SAVE_DIR) / 'Stats' / WINDOW_NAME_DICT[window]
        plot_dir = Path(SAVE_DIR) / 'Plots' / WINDOW_NAME_DICT[window]
        
        # statistics save path
        acute_rescue_stats(window_meta, 
                           window_feat,
                           group_by='treatment',
                           control=control,
                           save_dir=stats_dir,
                           feature_set=feature_list,
                           pvalue_threshold=0.05,
                           fdr_method='fdr_bh')
        order = ['BW-None','BW-vitC','BW-NAC','fepD-None','fepD-vitC','fepD-NAC']
        colour_labels = sns.color_palette('tab10', 2)
        colours = [colour_labels[0] if 'BW' in s else colour_labels[1] for s in order]
        colour_dict = {key:col for (key,col) in zip(order, colours)}
        all_in_one_boxplots(window_meta,
                            window_feat,
                            group_by='treatment',
                            control=control,
                            save_dir=plot_dir,
                            ttest_path=stats_dir / 't-test' / 't-test_results.csv',
                            feature_set=feature_list,
                            pvalue_threshold=0.05,
                            order=order,
                            colour_dict=colour_dict,
                            figsize=(20, 10),
                            ylim_minmax=None,
                            vline_boxpos=[2],
                            fontsize=15,
                            subplots_adjust={'bottom': 0.2, 'top': 0.95, 'left': 0.05, 'right': 0.98})
        
    
    # timeseries plots of speed for fepD vs BW control for each window
    
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
                                   ylim_minmax=(-20,280),
                                   video_length_seconds=VIDEO_LENGTH_SECONDS)

    return
    
#%% Main

if __name__ == '__main__':
    main()
    
    