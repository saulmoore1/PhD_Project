#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse fast-effect (acute response) videos
- Bluelight delivered for 10 seconds every 30 minutes, for a total of 5 hours
- window feature summaries +30 -> +60 seconds after each bluelight stimulus

When do we start to see an effect on worm behaviour? At which timepoint/window? 
Do we still see arousal of worms on siderophore mutants, even after a short period of time?

@author: sm5911
@date: 24/11/2021

"""

#%% Imports

import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from time_series.plot_timeseries import plot_timeseries_feature # plot_timeseries_motion_mode
from visualisation.plotting_helper import sig_asterix, all_in_one_boxplots
from write_data.write import write_list_to_file

from tierpsytools.preprocessing.filter_data import select_feat_set
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from tierpsytools.analysis.statistical_tests import _multitest_correct

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Acute_Effect"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Acute_Effect"

FEATURE_SET = ['speed_50th']

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05

# Featsums 10 seconds centred on end of each BL pulse and also 20-30 seconds after end of each BL pulse
WINDOW_DICT = {0:(1805,1815),1:(1830,1840),
               2:(3605,3615),3:(3630,3640),
               4:(5405,5415),5:(5430,5440),
               6:(7205,7215),7:(7230,7240),
               8:(9005,9015),9:(9030,9040),
               10:(10805,10815),11:(10830,10840),
               12:(12605,12615),13:(12630,12640),
               14:(14405,14415),15:(14430,14440)}

WINDOW_NAME_DICT = {0:"blue light 1", 1: "20-30 seconds after blue light 1",
                    2:"blue light 2", 3: "20-30 seconds after blue light 2",
                    4:"blue light 3", 5: "20-30 seconds after blue light 3",
                    6:"blue light 4", 7: "20-30 seconds after blue light 4",
                    8:"blue light 5", 9: "20-30 seconds after blue light 5",
                    10:"blue light 6", 11: "20-30 seconds after blue light 6",
                    12:"blue light 7", 13: "20-30 seconds after blue light 7",
                    14:"blue light 8", 15: "20-30 seconds after blue light 8"}

WINDOW_LIST = [1,3,5,7,9,11,13,15]

OMIT_STRAINS_LIST = ['trpD']

FPS = 25
BLUELIGHT_TIMEPOINTS_MINUTES = [30,60,90,120,150,180,210,240]

VIDEO_LENGTH_SECONDS = 5*60*60

#%% Functions

def fast_effect_stats(metadata,
                      features,
                      group_by='gene_name',
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
        if save_dir is not None:
            anova_path = Path(save_dir) / 'ANOVA' / 'ANOVA_results.csv'
            anova_path.parent.mkdir(parents=True, exist_ok=True)
            test_results.to_csv(anova_path, header=True, index=True)

        # use reject mask to find significant feature set
        fset = pvals.loc[reject['ANOVA']].sort_values(by='ANOVA', ascending=True).index.to_list()

        if len(fset) > 0:
            print("%d significant features found by ANOVA by '%s' (P<%.2f, %s)" %\
                  (len(fset), group_by, pvalue_threshold, fdr_method))
            if save_dir is not None:
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
    if save_dir is not None:
        ttest_path = Path(save_dir) / 't-test' / 't-test_results.csv'
        ttest_path.parent.mkdir(exist_ok=True, parents=True)
        ttest_results.to_csv(ttest_path, header=True, index=True)
    
    nsig = sum(reject_t.sum(axis=1) > 0)
    print("%d significant features between any %s vs %s (t-test, P<%.2f, %s)" %\
          (nsig, group_by, control, pvalue_threshold, fdr_method))

    if save_dir is not None:
        return
    elif n > 2:
        return test_results, ttest_results
    else:
        return ttest_results
    

def main():
    
    aux_dir = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    results_dir =  Path(PROJECT_DIR) / 'Results'
    
    metadata_path_local = Path(SAVE_DIR) / 'metadata.csv'
    features_path_local = Path(SAVE_DIR) / 'features.csv'
    
    if not metadata_path_local.exists() or not features_path_local.exists():
        # load metadata    
        metadata, metadata_path = compile_metadata(aux_dir, 
                                                   imaging_dates=None, 
                                                   add_well_annotations=False, 
                                                   n_wells=6)
        
        features, metadata = process_feature_summaries(metadata_path, 
                                                       results_dir, 
                                                       compile_day_summaries=False, 
                                                       imaging_dates=None, 
                                                       align_bluelight=False, 
                                                       window_summaries=True,
                                                       n_wells=6)
     
        # # Subset results (rows) to remove entries for wells with unknown strain data for 'gene_name'
        # n = metadata.shape[0]
        # metadata = metadata.loc[~metadata['gene_name'].isna(),:]
        # features = features.reindex(metadata.index)
        # print("%d entries removed with no gene name metadata" % (n - metadata.shape[0]))
     
        # update gene names for mutant strains
        # metadata['gene_name'] = [args.control_dict['gene_name'] if s == 'BW' else s 
        #                          for s in metadata['gene_name']]
        #['BW\u0394'+g if not g == 'BW' else 'wild_type' for g in metadata['gene_name']]
        
        # Clean results - Remove features with too many NaNs/zero std + impute remaining NaNs
        features, metadata = clean_summary_results(features, 
                                                   metadata,
                                                   feature_columns=None,
                                                   nan_threshold_row=NAN_THRESHOLD_ROW,
                                                   nan_threshold_col=NAN_THRESHOLD_COL,
                                                   max_value_cap=1e15,
                                                   imputeNaN=True,
                                                   min_nskel_per_video=None,
                                                   min_nskel_sum=None,
                                                   drop_size_related_feats=False,
                                                   norm_feats_only=False,
                                                   percentile_to_use=None)

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
    
    control = 'BW'
    
    if OMIT_STRAINS_LIST is not None:
        metadata = metadata[~metadata['gene_name'].isin(OMIT_STRAINS_LIST)]

    metadata['window'] = metadata['window'].astype(int)
    if WINDOW_LIST is not None:
        metadata = metadata[metadata['window'].isin(WINDOW_LIST)]
        
    window_list = list(metadata['window'].unique())   
    
    features = features.reindex(metadata.index)
 
    # stats and boxplots
    for window in tqdm(window_list):
        meta_window = metadata[metadata['window']==window]
        feat_window = features.reindex(meta_window.index)

        stats_dir = Path(SAVE_DIR) / 'Stats' / WINDOW_NAME_DICT[window]
        plot_dir = Path(SAVE_DIR) / 'Plots' / WINDOW_NAME_DICT[window]
        
        fast_effect_stats(meta_window, 
                          feat_window, 
                          group_by='gene_name',
                          control=control,
                          save_dir=stats_dir,
                          feature_set=feature_list,
                          pvalue_threshold=0.05,
                          fdr_method='fdr_bh')

        order = ['BW'] + [s for s in meta_window['gene_name'].unique() if s != 'BW']
        colour_dict = dict(zip(order, sns.color_palette('tab10', len(order))))
        all_in_one_boxplots(meta_window,
                            feat_window,
                            group_by='gene_name',
                            control=control,
                            save_dir=plot_dir / 'all-in-one',
                            ttest_path=stats_dir / 't-test' / 't-test_results.csv',
                            feature_set=feature_list,
                            pvalue_threshold=0.05,
                            order=order,
                            colour_dict=colour_dict,
                            figsize=(10,8),
                            ylim_minmax=(-20,350),
                            vline_boxpos=None,
                            fontsize=20,
                            subplots_adjust={'bottom':0.15,'top':0.9,'left':0.15,'right':0.95})        
        
    pvalues_dict = {}
    for window in window_list:
        stats_dir = Path(SAVE_DIR) / 'Stats' / WINDOW_NAME_DICT[window]

        # read p-values for each strain and correct for multiple comparisons (fdr_bh)
        pvals_df = pd.read_csv(stats_dir / 't-test' / 't-test_results.csv', index_col=0)
        pvalues_dict[window] = pvals_df.loc['speed_50th','pvals_fepD']
    
    reject, corrected_pvals = _multitest_correct(pd.Series(list(pvalues_dict.values())), 
                                                 multitest_method='fdr_bh', fdr=0.05)
    pvalues_dict = dict(zip(window_list, corrected_pvals))
    pvals = pd.DataFrame.from_dict(pvalues_dict, orient='index', columns=['pvals'])
    ttest_corrected_savepath = Path(SAVE_DIR) / 'Stats' / 't-test_corrected' / 't-test_window_results.csv'
    ttest_corrected_savepath.parent.mkdir(exist_ok=True, parents=True)
    pvals.to_csv(ttest_corrected_savepath)

    colour_dict = dict(zip(['BW','fepD'], sns.color_palette('tab10', len(order))))
    all_in_one_boxplots(metadata,
                        features,
                        group_by='window',
                        hue='gene_name',
                        order=window_list,
                        hue_order=['BW','fepD'],
                        control='BW',
                        save_dir=Path(SAVE_DIR) / 'Plots',
                        ttest_path=None,
                        feature_set=feature_list,
                        pvalue_threshold=0.05,
                        colour_dict=colour_dict,
                        figsize=(15,8),
                        ylim_minmax=(-70,370),
                        vline_boxpos=None,
                        fontsize=20,
                        legend=False,
                        subplots_adjust={'bottom':0.15,'top':0.9,'left':0.15,'right':0.95})        
    
    # timeseries plots of speed for fepD vs BW control
    
    strain_list = sorted(metadata['gene_name'].unique())
    BLUELIGHT_TIMEPOINTS_SECONDS = [(i*60,i*60+10) for i in BLUELIGHT_TIMEPOINTS_MINUTES]
    
    plot_timeseries_feature(metadata=metadata,
                            project_dir=Path(PROJECT_DIR),
                            save_dir=Path(SAVE_DIR) / 'timeseries-speed',
                            group_by='gene_name',
                            control='BW',
                            groups_list=strain_list,
                            feature='speed',
                            n_wells=6,
                            bluelight_stim_type='bluelight',
                            bluelight_timepoints_seconds=BLUELIGHT_TIMEPOINTS_SECONDS,
                            smoothing=120,
                            fps=FPS,
                            ylim_minmax=(-20,370))    
    
    return

#%% Main

if __name__ == '__main__':   
    main()

    