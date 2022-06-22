#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keio Antioxidants - Experiments adding antioxidants to E. coli BW25113 (control) and fepD mutant
bacteria of the Keio Collection

@author: sm5911
@date: 18/05/2022

"""

#%% Imports

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from preprocessing.compile_hydra_data import process_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from visualisation.plotting_helper import sig_asterix
from write_data.write import write_list_to_file
from time_series.time_series_helper import get_strain_timeseries
from time_series.plot_timeseries import plot_timeseries_motion_mode
# from analysis.keio_screen.check_keio_screen_worm_trajectories import check_tracked_objects

from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from tierpsytools.preprocessing.filter_data import select_feat_set

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Keio_Antioxidants_6WP"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Antioxidants"

BLUELIGHT_WINDOWS_ONLY_TS = True
BLUELIGHT_TIMEPOINTS_MINUTES = [30,31,32]
FPS = 25
VIDEO_LENGTH_SECONDS = 38*60
SMOOTH_WINDOW_SECONDS = 5

N_TOP_FEATS = None # Tierpsy feature set to use: 16, 256, '2k', None
IMAGING_DATES = ['20220418']

nan_threshold_row = 0.8
nan_threshold_col = 0.05
motion_modes = ['forwards','backwards','stationary']

WINDOW_DICT_SECONDS = {0:(1790,1800), 1:(1805,1815), 2:(1815,1825),
                       3:(1850,1860), 4:(1865,1875), 5:(1875,1885),
                       6:(1910,1920), 7:(1925,1935), 8:(1935,1945)}

feature_set = ['motion_mode_forward_fraction']

#%% Functions

def antioxidant_stats(metadata, 
                      features,
                      control,
                      group_by='treatment', # drug_typwe, food_type, etc
                      feature_set=None, 
                      save_dir=None, 
                      window_list=None,
                      fdr_method='fdr_by',
                      p_value_threshold=0.05):
    
    if window_list is None:
        window_list = list(WINDOW_DICT_SECONDS.keys())
    assert(all(w in metadata['window'].unique() for w in window_list))
    
    assert not features.isna().any().any()
    
    if feature_set is None:
        feature_set = list(features.columns)
    else:
        assert(all(f in features.columns for f in feature_set))
        features = features[feature_set]
        
    treatment_list = sorted(metadata[group_by].unique())
    assert control in treatment_list

    # construct save paths (args.save_dir / topfeats? etc)
    fdr_method = 'uncorrected' if fdr_method is None else fdr_method
    
    for window in window_list:
        window_meta = metadata[metadata['window']==window]
        window_feat = features.reindex(window_meta.index)
        
        ##### ANOVA #####
    
        # make path to save ANOVA results
        test_path = save_dir / window / fdr_method / 'ANOVA_results.csv'
        test_path.parent.mkdir(exist_ok=True, parents=True)
    
        # ANOVA across strains for significant feature differences
        if len(metadata[group_by].unique()) > 2:   
            stats, pvals, reject = univariate_tests(X=window_feat, 
                                                    y=window_meta[group_by], 
                                                    test='ANOVA',
                                                    control=control,
                                                    comparison_type='multiclass',
                                                    multitest_correction=fdr_method, # uncorrected
                                                    alpha=p_value_threshold,
                                                    n_permutation_test=None) # 'all'
        
            # get effect sizes
            effect_sizes = get_effect_sizes(X=window_feat, 
                                            y=window_meta[group_by], 
                                            control=control,
                                            effect_type=None,
                                            linked_test='ANOVA')
                                            
            # compile + save results (corrected)
            test_results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
            test_results.columns = ['stats','effect_size','pvals','reject']     
            test_results['significance'] = sig_asterix(test_results['pvals'])
            test_results = test_results.sort_values(by=['pvals'], ascending=True) # rank pvals
            test_results.to_csv(test_path, header=True, index=True)
            
            nsig = test_results['reject'].sum()
            print("%d features (%.f%%) signficantly different among '%s'" % (nsig, 
                  (len(test_results.index)/nsig)*100, group_by))
            
        ##### t-tests #####
    
        if nsig > 0:
                         
            # t-tests for each treatment combination vs control
            stats_t, pvals_t, reject_t = univariate_tests(X=window_feat, 
                                                          y=window_meta[group_by], 
                                                          control=control, 
                                                          test='t-test',
                                                          comparison_type='binary_each_group',
                                                          multitest_correction=fdr_method, 
                                                          alpha=p_value_threshold)
            # get effect sizes for comparisons
            effect_sizes_t =  get_effect_sizes(X=window_feat, 
                                               y=window_meta[group_by], 
                                               control=control,
                                               effect_type=None,
                                               linked_test='t-test')
            
            # compile
            stats_t.columns = ['stats_' + str(c) for c in stats_t.columns]
            pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
            reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
            effect_sizes_t.columns = ['effect_size_' + str(c) for c in effect_sizes_t.columns]
            ttest_df = pd.concat([stats_t, effect_sizes_t, pvals_t, reject_t], axis=1)
    
            # save t-test results
            ttest_path = test_path.parent / 'ttest_results.csv'
            ttest_df.to_csv(ttest_path, header=True, index=True)
    
            # record t-test significant features (not ordered)
            fset_ttest = pvals_t[np.asmatrix(reject_t)].index.unique().to_list()
            #assert set(fset_ttest) == set(pvals_t.index[(pvals_t < args.pval_threshold).sum(axis=1) > 0])
            
            print("%d significant features found (t-test %s, P<%.2f)" % (len(fset_ttest), fdr_method, 
                                                                         p_value_threshold))
    
            if len(fset_ttest) > 0:
                ttest_sigfeats_path = test_path.parent / 'significant_features_t_test.txt'
                write_list_to_file(fset_ttest, ttest_sigfeats_path)

    return

def antioxidants_boxplots(metadata,
                          features,
                          control,
                          group_by='treatment',
                          save_dir=SAVE_DIR):
    # TODO: boxplots for 30, 31, 32 min BL pulses
    
    return


def antioxidants_timeseries(metadata, 
                            control='BW-none-nan-H2O',
                            group_by='treatment',
                            project_dir=PROJECT_DIR, 
                            save_dir=SAVE_DIR):
    """ Timeseries plots for worm motion mode on BW and fepD bacteria with and without the 
        addition of antioxidants: Trolox (vitamin E), trans-resveratrol, vitamin C and NAC
    """
    
    # Each treatment vs BW control    
    # control_list = np.repeat(control, metadata[group_by].nunique()-1))
    # treatment_list = [t for t in sorted(metadata[group_by].unique()) if t!= control]
    control_list = list(np.concatenate([np.array(['fepD-none-nan-H2O']),
                                        np.repeat(control, 12)]))
    treatment_list = ['fepD-none-nan-EtOH','fepD-none-nan-H2O','BW-none-nan-EtOH',
                      'BW-vitC-500.0-H2O','fepD-vitC-500.0-H2O',
                      'BW-NAC-500.0-H2O','BW-NAC-1000.0-H2O',
                      'fepD-NAC-500.0-H2O','fepD-NAC-1000.0-H2O',
                      'BW-trolox-500.0-EtOH','fepD-trolox-500.0-EtOH',
                      'BW-trans-resveratrol-500.0-EtOH','fepD-trans-resveratrol-500.0-EtOH']
    title_list = ['fepD: H2O vs EtOH','BW vs fepD','BW: H2O vs EtOH',
                  'BW vs BW + vitC','BW vs fepD + vitC',
                  'BW vs BW + NAC','BW vs BW + NAC',
                  'BW vs fepD + NAC','BW vs fepD + NAC',
                  'BW vs BW + trolox','BW vs fepD + trolox',
                  'BW vs BW + trans-resveratrol','BW vs fepD + trans-resveratrol']
    labs = [('fepD + H2O', 'fepD + EtOH'),('BW', 'fepD'),('BW + H2O', 'BW + EtOH'),
            ('BW', 'BW + vitC (500ug/mL)'),('BW', 'fepD + vitC (500ug/mL)'),
            ('BW', 'BW + NAC (500ug/mL)'),('BW', 'BW + NAC (1000ug/mL)'),
            ('BW', 'fepD + NAC (500ug/mL)'),('BW', 'fepD + NAC (1000ug/mL)'),
            ('BW', 'BW + trolox (500ug/mL in EtOH)'),('BW', 'fepD + trolox (500ug/mL in EtOH)'),
            ('BW', 'BW + trans-resveratrol (500ug/mL in EtOH)'),
            ('BW', 'fepD + trans-resveratrol (500ug/mL in EtOH)')]
    
    
    for control, treatment, title, lab in tqdm(zip(control_list, treatment_list, title_list, labs)):
        
        # get timeseries for control data
        control_ts = get_strain_timeseries(metadata[metadata[group_by]==control], 
                                           project_dir=project_dir, 
                                           strain=control,
                                           group_by=group_by,
                                           n_wells=6,
                                           save_dir=Path(save_dir) / 'Data' / control,
                                           verbose=False,
                                           return_error_log=False)
        
        # get timeseries for treatment data
        treatment_ts = get_strain_timeseries(metadata[metadata[group_by]==treatment], 
                                             project_dir=project_dir, 
                                             strain=treatment,
                                             group_by=group_by,
                                             n_wells=6,
                                             save_dir=Path(save_dir) / 'Data' / treatment,
                                             verbose=False)
 
        colour_dict = dict(zip([control, treatment], sns.color_palette("pastel", 2)))
        bluelight_frames = [(i*60*FPS, i*60*FPS+10*FPS) for i in BLUELIGHT_TIMEPOINTS_MINUTES]

        for mode in motion_modes:
                    
            print("Plotting timeseries '%s' fraction for '%s' vs '%s'..." %\
                  (mode, treatment, control))
    
            plt.close('all')
            fig, ax = plt.subplots(figsize=(15,5), dpi=200)
    
            ax = plot_timeseries_motion_mode(df=control_ts,
                                             window=SMOOTH_WINDOW_SECONDS*FPS,
                                             error=True,
                                             mode=mode,
                                             max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
                                             title=None,
                                             saveAs=None,
                                             ax=ax,
                                             bluelight_frames=bluelight_frames,
                                             colour=colour_dict[control],
                                             alpha=0.25)
            
            ax = plot_timeseries_motion_mode(df=treatment_ts,
                                             window=SMOOTH_WINDOW_SECONDS*FPS,
                                             error=True,
                                             mode=mode,
                                             max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
                                             title=None,
                                             saveAs=None,
                                             ax=ax,
                                             bluelight_frames=bluelight_frames,
                                             colour=colour_dict[treatment],
                                             alpha=0.25)
        
            xticks = np.linspace(0, VIDEO_LENGTH_SECONDS*FPS, int(VIDEO_LENGTH_SECONDS/60)+1)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
            ax.set_xlabel('Time (minutes)', fontsize=12, labelpad=10)
            ax.set_ylabel('Fraction {}'.format(mode), fontsize=12, labelpad=10)
            ax.set_title(title, fontsize=12, pad=10)
            ax.legend([lab[0], lab[1]], fontsize=12, frameon=False, loc='best')
    
            if BLUELIGHT_WINDOWS_ONLY_TS:
                ts_plot_dir = Path(save_dir) / 'Plots' / 'timeseries_bluelight' / treatment
                ax.set_xlim([min(BLUELIGHT_TIMEPOINTS_MINUTES)*60*FPS-60*FPS, 
                             max(BLUELIGHT_TIMEPOINTS_MINUTES)*60*FPS+70*FPS])
            else:   
                ts_plot_dir = Path(save_dir) / 'Plots' / 'timeseries' / treatment
    
            #plt.tight_layout()
            ts_plot_dir.mkdir(exist_ok=True, parents=True)
            save_path = ts_plot_dir / '{0}_{1}.pdf'.format(treatment, mode)
            print("Saving to: %s" % save_path)
            plt.savefig(save_path)  

    return


#%% Main
if __name__ == '__main__':
    
    aux_dir = Path(PROJECT_DIR) / 'AuxiliaryFiles'    
    results_dir =  Path(PROJECT_DIR) / 'Results'
    
    # load metadata    
    metadata, metadata_path = process_metadata(aux_dir, 
                                               imaging_dates=IMAGING_DATES, 
                                               add_well_annotations=False, 
                                               n_wells=6)
    
    features, metadata = process_feature_summaries(metadata_path, 
                                                   results_dir, 
                                                   compile_day_summaries=True, 
                                                   imaging_dates=IMAGING_DATES, 
                                                   align_bluelight=False, 
                                                   window_summaries=True,
                                                   n_wells=6)
 
    # Subset results (rows) to remove entries for wells with unknown strain data for 'gene_name'
    if metadata['food_type'].isna().any():
        n = metadata.shape[0]
        metadata = metadata.loc[~metadata['food_type'].isna(),:]
        features = features.reindex(metadata.index)
        print("%d entries removed with no gene name metadata" % (n - metadata.shape[0]))
 
    # Create is_bad_well column - refer to manual metadata for bad 35mm petri plates
    metadata['is_bad_well'] = False

    # Clean results - Remove bad well data + features with too many NaNs/zero std 
    #                                      + impute remaining NaNs
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

    assert not features.isna().sum(axis=1).any()
    assert not (features.std(axis=1) == 0).any()
            
    # Load Tierpsy feature set + subset (columns) for selected features only
    if N_TOP_FEATS is not None:
        features = select_feat_set(features, 'tierpsy_{}'.format(N_TOP_FEATS), append_bluelight=True)
        features = features[[f for f in features.columns if 'path_curvature' not in f]]
    
    metadata['imaging_plate_drug_conc'] = metadata['imaging_plate_drug_conc'].astype(str)
    metadata['treatment'] = metadata[['food_type','drug_type','imaging_plate_drug_conc','solvent']
                                     ].agg('-'.join, axis=1)
    
    control_treatment = 'BW-none-nan-H2O'
    
    antioxidant_stats(metadata, 
                      features, 
                      group_by='treatment',
                      control=control_treatment,
                      feature_set=feature_set, 
                      save_dir=Path(SAVE_DIR) / 'Stats')
    
    # plot timeseries for each treatment vs control
    antioxidants_timeseries(metadata, 
                            control=control_treatment,
                            group_by='treatment',
                            project_dir=Path(PROJECT_DIR),
                            save_dir=Path(SAVE_DIR))
    
    # # Check length/area of tracked objects - prop bad skeletons
    # results_df = check_tracked_objects(metadata, 
    #                                    length_minmax=(200, 2000), 
    #                                    width_minmax=(20, 500),
    #                                    save_to=Path(SAVE_DIR) / 'tracking_checks.csv')
