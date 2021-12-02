#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse fast-effect (acute response) videos
- window feature summaries for Ziwei's optimal windows around each bluelight stimulus
- Bluelight delivered for 10 seconds every 30 minutes, for a total of 5 hours

When do we start to see an effect on worm behaviour? At which timepoint/window? 
Do we still see arousal of worms on siderophore mutants, even after a short period of time?

@author: sm5911
@date: 24/11/2021

"""

#%% Imports

import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import transforms
from matplotlib import pyplot as plt
from read_data.read import load_json
from read_data.paths import get_save_dir
from preprocessing.compile_hydra_data import process_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from statistical_testing.stats_helper import pairwise_ttest
from statistical_testing.perform_keio_stats import df_summary_stats

#%% Globals

JSON_PARAMETERS_PATH = 'analysis/20211102_parameters_keio_fast_effect.json'

FEATURE = 'motion_mode_paused_fraction'

scale_outliers_box = True

# windows summary window number to corresponding frame number mapping dictionary
WINDOW_FRAME_DICT = {2:(1805,1815),
                     5:(3605,3615),
                     8:(5405,5415),
                     11:(7205,7215),
                     14:(9005,9015),
                     17:(10805,10815),
                     20:(12605,12615),
                     23:(14405,14415)} #26:(16205,16215)

#%% Functions

def perform_fast_effect_stats(features, metadata, args):
    """ Pairwise t-tests for each window comparing worm 'motion mode paused fraction' on 
        Keio mutants vs BW control 
    """

    # categorical variables to investigate: 'gene_name' and 'window'
    print("\nInvestigating variation in fraction of worms paused between hit strains and control " +
          "(for each window)")    

    # assert there will be no errors due to case-sensitivity
    assert len(metadata['gene_name'].unique()) == len(metadata['gene_name'].str.upper().unique())
        
    # subset for windows in window_frame_dict
    window_list = list(WINDOW_FRAME_DICT.keys())
    assert all(w in metadata['window'] for w in window_list)
    metadata = metadata[metadata['window'].isin(window_list)]
    features = features.reindex(metadata.index)

    control_strain = args.control_dict['gene_name']
    strain_list = list([s for s in metadata['gene_name'].unique() if s != control_strain])    

    # print mean sample size
    sample_size = df_summary_stats(metadata, columns=['gene_name', 'window'])
    print("Mean sample size of strain/window: %d" % (int(sample_size['n_samples'].mean())))
    
    # construct save paths (args.save_dir / topfeats? etc)
    save_dir = get_save_dir(args)
    stats_dir =  save_dir / "Stats" / args.fdr_method
        
    control_meta = metadata[metadata['gene_name']==control_strain]
    control_feat = features.reindex(control_meta.index)
    control_df = control_meta.join(control_feat[[FEATURE]])
    
    for strain in strain_list:
        print("\nPairwise t-tests for each window comparing fraction of worms paused " +
              "on %s vs control" % strain)
        strain_meta = metadata[metadata['gene_name']==strain]
        strain_feat = features.reindex(strain_meta.index)
        strain_df = strain_meta.join(strain_feat[[FEATURE]])
         
        stats, pvals, reject = pairwise_ttest(control_df, 
                                              strain_df, 
                                              feature_list=[FEATURE], 
                                              group_by='window', 
                                              fdr_method='fdr_by',
                                              fdr=0.05)
 
        # compile table of results
        stats.columns = ['stats_' + str(c) for c in stats.columns]
        pvals.columns = ['pvals_' + str(c) for c in pvals.columns]
        reject.columns = ['reject_' + str(c) for c in reject.columns]
        test_results = pd.concat([stats, pvals, reject], axis=1)
        
        # save results
        ttest_strain_path = stats_dir / 'pairwise_ttests' / '{}_window_results.csv'.format(strain)
        ttest_strain_path.parent.mkdir(parents=True, exist_ok=True)
        test_results.to_csv(ttest_strain_path, header=True, index=True)
        
        for window in window_list:
            print("%s difference in '%s' between %s vs %s in window %s (paired t-test, P=%.3f, %s)" %\
                  (("SIGNIFICANT" if reject.loc[FEATURE, 'reject_{}'.format(window)] else "No"), 
                  FEATURE, strain, control_strain, window, pvals.loc[FEATURE, 'pvals_{}'.format(window)], 
                  args.fdr_method))
                
    return

def analyse_fast_effect(features, metadata, args):
    
    # categorical variables to investigate: 'gene_name' and 'window'
    print("\nInvestigating variation in fraction of worms paused between hit strains and control " +
          "(for each window)")
    
    # assert there will be no errors due to case-sensitivity
    assert len(metadata['gene_name'].unique()) == len(metadata['gene_name'].str.upper().unique())
    
    # subset for windows in window_frame_dict
    window_list = list(WINDOW_FRAME_DICT.keys())
    assert all(w in metadata['window'] for w in window_list)
    metadata = metadata[metadata['window'].isin(window_list)]
    features = features.reindex(metadata.index)

    control_strain = args.control_dict['gene_name']
    strain_list = list([s for s in metadata['gene_name'].unique() if s != control_strain])    

    # print mean sample size
    sample_size = df_summary_stats(metadata, columns=['gene_name', 'window'])
    print("Mean sample size of strain/window: %d" % (int(sample_size['n_samples'].mean())))
    
    # construct save paths (args.save_dir / topfeats? etc)
    save_dir = get_save_dir(args)
    stats_dir =  save_dir / "Stats" / args.fdr_method
    plot_dir = save_dir / "Plots" / args.fdr_method
        
    # plotting
    date_lut = dict(zip(list(metadata['date_yyyymmdd'].unique()), 
                        sns.color_palette('Set1', n_colors=len(metadata['date_yyyymmdd'].unique()))))
    for strain in strain_list:
        print("Plotting windows for %s vs control" % strain)
        
        plot_meta = metadata[np.logical_or(metadata['gene_name']==strain, 
                                         metadata['gene_name']==control_strain)]
        plot_feat = features.reindex(plot_meta.index)
        plot_df = plot_meta.join(plot_feat[[FEATURE]])
        
        # plot control/strain for all windows
        plt.close('all')
        fig, ax = plt.subplots(figsize=(12,6))
        ax = sns.boxplot(x='window', y=FEATURE, hue='gene_name', hue_order=[control_strain, strain],
                         data=plot_df, palette='Set3', dodge=True, ax=ax)
        for date in date_lut.keys():
            date_df = plot_df[plot_df['date_yyyymmdd']==date]
            ax = sns.swarmplot(x='window', y=FEATURE, hue='gene_name', hue_order=[control_strain, strain], 
                               data=date_df, color=date_lut[date], alpha=0.7, size=4, dodge=True, ax=ax)
        n_labs = len(plot_df['gene_name'].unique())
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:n_labs], labels[:n_labs], fontsize=15, frameon=False, loc='upper right')
        ax.set_xlabel('Window number', fontsize=15, labelpad=10)
        ax.set_ylabel(FEATURE.replace('_',' '), fontsize=15, labelpad=10)
        
        # scale plot to omit outliers (>2.5*IQR from mean)
        if scale_outliers_box:
            grouped_strain = plot_df.groupby('window')
            y_bar = grouped_strain[FEATURE].median() # median is less skewed by outliers
            # Computing IQR
            Q1 = grouped_strain[FEATURE].quantile(0.25)
            Q3 = grouped_strain[FEATURE].quantile(0.75)
            IQR = Q3 - Q1
            plt.ylim(min(y_bar) - 2.5 * max(IQR), max(y_bar) + 2.5 * max(IQR))
            
        # load t-test results + annotate p-values on plot
        for ii, window in enumerate(window_list):
            ttest_strain_path = stats_dir / 'pairwise_ttests' / '{}_window_results.csv'.format(strain)
            ttest_strain_table = pd.read_csv(ttest_strain_path, index_col=0, header=0)
            strain_pvals_t = ttest_strain_table[[c for c in ttest_strain_table if "pvals_" in c]] 
            strain_pvals_t.columns = [c.split('pvals_')[-1] for c in strain_pvals_t.columns] 
            p = strain_pvals_t.loc[FEATURE, str(window)]
            text = ax.get_xticklabels()[ii]
            assert text.get_text() == str(window)
            p_text = 'P < 0.001' if p < 0.001 else 'P = %.3f' % p
            #y = (y_bar[antiox] + 2 * IQR[antiox]) if scale_outliers_box else plot_df[feature].max()
            #h = (max(IQR) / 10) if scale_outliers_box else (y - plot_df[feature].min()) / 50
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            plt.plot([ii-.2, ii-.2, ii+.2, ii+.2], 
                     [0.8, 0.81, 0.81, 0.8], #[y+h, y+2*h, y+2*h, y+h], 
                     lw=1.5, c='k', transform=trans)
            ax.text(ii, 0.82, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
                
        fig_savepath = plot_dir / 'window_boxplots' / strain / (FEATURE + '.png')
        fig_savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_savepath)

    return
    
#%% Main

if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description="Analyse acute response videos to investigate how \
    fast the food takes to influence worm behaviour")
    parser.add_argument('-j','--json', help="Path to JSON parameters file", default=JSON_PARAMETERS_PATH)
    args = parser.parse_args()
    args = load_json(args.json)

    aux_dir = Path(args.project_dir) / 'AuxiliaryFiles'
    results_dir =  Path(args.project_dir) / 'Results'
    
    # load compiled window summaries results
    compiled_filenames = pd.read_csv(results_dir / 'full_window_filenames.csv', index_col=False)
    compiled_features = pd.read_csv(results_dir / 'full_window_features.csv', index_col=False)

    # subset summaries for specific windows
    windows_list = list(WINDOW_FRAME_DICT.keys())
    compiled_filenames = compiled_filenames[compiled_filenames['window'].isin(windows_list)]
    compiled_features = compiled_features[compiled_features['window'].isin(windows_list)]
    
    # load metadata    
    metadata, metadata_path = process_metadata(aux_dir, 
                                               imaging_dates=args.dates, 
                                               add_well_annotations=args.add_well_annotations, 
                                               n_wells=6)
    
    features, metadata = process_feature_summaries(metadata_path, 
                                                   results_dir, 
                                                   compile_day_summaries=True, 
                                                   imaging_dates=args.dates, 
                                                   align_bluelight=False, 
                                                   window_summaries=True,
                                                   n_wells=6)
 
    # Subset results (rows) to remove entries for wells with unknown strain data for 'gene_name'
    n = metadata.shape[0]
    metadata = metadata.loc[~metadata['gene_name'].isna(),:]
    features = features.reindex(metadata.index)
    print("%d entries removed with no gene name metadata" % (n - metadata.shape[0]))
 
    # update gene names for mutant strains
    metadata['gene_name'] = [args.control_dict['gene_name'] if s == 'BW' else s 
                             for s in metadata['gene_name']]
    #['BW\u0394'+g if not g == 'BW' else 'wild_type' for g in metadata['gene_name']]

    # Create is_bad_well column - refer to manual metadata for bad 35mm petri plates
    metadata['is_bad_well'] = False

    # Clean results - Remove bad well data + features with too many NaNs/zero std + impute remaining NaNs
    features, metadata = clean_summary_results(features, 
                                               metadata,
                                               feature_columns=None,
                                               nan_threshold_row=args.nan_threshold_row,
                                               nan_threshold_col=args.nan_threshold_col,
                                               max_value_cap=args.max_value_cap,
                                               imputeNaN=args.impute_nans,
                                               min_nskel_per_video=args.min_nskel_per_video,
                                               min_nskel_sum=args.min_nskel_sum,
                                               drop_size_related_feats=args.drop_size_features,
                                               norm_feats_only=args.norm_features_only,
                                               percentile_to_use=args.percentile_to_use)

    assert not features.isna().sum(axis=1).any()
    assert not (features.std(axis=1) == 0).any()
    
    perform_fast_effect_stats(features, metadata, args)
    
    analyse_fast_effect(features, metadata, args)
    
    
    