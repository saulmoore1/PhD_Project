#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse plate drop experiment
- Plates were dropped from a height of 10 cm (either once or 5 times in a row) and then recorded 
  immediately. N2 worm behaviour on fepD vs BW is compared in each case to see if plate drop 
  triggers behavioural arousal on fepD E. coli but not BW.

@author: sm5911
@date: 22/04/2023 (updated: 11/11/2024)

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
from visualisation.plotting_helper import sig_asterix
from time_series.plot_timeseries import plot_timeseries
from time_series.time_series_helper import get_strain_timeseries
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes, _multitest_correct

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Plate_Drop"
SAVE_DIR = "/Users/sm5911/Documents/PhD_DLBG/34_Keio_Plate_Drop"

N_WELLS = 6
NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
P_VALUE_THRESHOLD = 0.05
FDR_METHOD = 'fdr_bh'

FEATURE_LIST = ['speed_50th']
ABSOLUTE_SPEED = False
FPS = 25
DPI = 600

#%% Functions

def stats(metadata,
          features,
          group_by='treatment',
          control='BW',
          save_dir=None,
          feature_list=['speed_50th'],
          p_value_threshold=0.05,
          fdr_method='fdr_bh'):
        
    assert all(metadata.index == features.index)
    features = features[feature_list]

    # print mean sample size
    sample_size = metadata.groupby(group_by).count()
    print("Mean sample size of treatment group: %d" % int(sample_size[sample_size.columns[-1]].mean()))
    n = len(metadata[group_by].unique())
    
    if n > 2:
        
        # Perform ANOVA - is there variation among strains?
        stats, pvals, reject = univariate_tests(X=features, 
                                                y=metadata[group_by], 
                                                control=control, 
                                                test='ANOVA',
                                                comparison_type='multiclass',
                                                multitest_correction=fdr_method,
                                                alpha=p_value_threshold,
                                                n_permutation_test=None)

        # get effect sizes
        effect_sizes = get_effect_sizes(X=features,
                                        y=metadata[group_by],
                                        control=control,
                                        effect_type=None,
                                        linked_test='ANOVA')

        # compile results
        anova_results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
        anova_results.columns = ['stats','effect_size','pvals','reject']     
        anova_results['significance'] = sig_asterix(anova_results['pvals'])
        anova_results = anova_results.sort_values(by=['pvals'], ascending=True) # rank by p-value

        if save_dir is not None:
            anova_path = Path(save_dir) / 'ANOVA_results.csv'
            anova_path.parent.mkdir(parents=True, exist_ok=True)
            anova_results.to_csv(anova_path, header=True, index=True)
             
    # Perform t-tests
    stats_t, pvals_t, reject_t = univariate_tests(X=features,
                                                  y=metadata[group_by],
                                                  control=control,
                                                  test='t-test',
                                                  comparison_type='binary_each_group',
                                                  multitest_correction=fdr_method,
                                                  alpha=p_value_threshold)
    
    effect_sizes_t = get_effect_sizes(X=features,
                                      y=metadata[group_by],
                                      control=control,
                                      linked_test='t-test')
    
    stats_t.columns = ['stats_' + str(c) for c in stats_t.columns]
    pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
    reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
    effect_sizes_t.columns = ['effect_size_' + str(c) for c in effect_sizes_t.columns]
    ttest_results = pd.concat([stats_t, pvals_t, reject_t, effect_sizes_t], axis=1)
       
    if save_dir is not None:
        ttest_path = Path(save_dir) / 't-test_results.csv'
        ttest_path.parent.mkdir(exist_ok=True, parents=True)
        ttest_results.to_csv(ttest_path, header=True, index=True)
    
    nsig = sum(reject_t.sum(axis=1) > 0)
    print("%d significant features between any %s vs %s (t-test, P<%.2f, %s)" %\
          (nsig, group_by, control, p_value_threshold, fdr_method))

    if n > 2:
        return anova_results, ttest_results
    else:
        return ttest_results      


def main():
    
    aux_dir = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    res_dir = Path(PROJECT_DIR) / 'Results'
    
    metadata_path_local = Path(SAVE_DIR) / 'metadata.csv'
    features_path_local = Path(SAVE_DIR) / 'features.csv'
    
    if not metadata_path_local.exists() or not features_path_local.exists():
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
    
        # clean results: remove bad wells or features with too many NaNs/zero std and impute NaNs
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
                                                   norm_feats_only=False)
        
        # save results
        metadata.to_csv(metadata_path_local, index=False)
        features.to_csv(features_path_local, index=False)
    
    else:
        metadata = pd.read_csv(metadata_path_local, header=0, index_col=None, dtype={'comments':str})
        features = pd.read_csv(features_path_local, header=0, index_col=None)

    assert metadata['window'].nunique() == 1
    
    treatment_cols = ['bacteria_strain','plate_drop_number']
    metadata['treatment'] = metadata[treatment_cols].astype(str).agg('-'.join, axis=1)
    treatment_list = sorted(metadata['treatment'].unique())
    strain_list = sorted(metadata['bacteria_strain'].unique())
    drop_list = sorted(metadata['plate_drop_number'].unique())
    
    if ABSOLUTE_SPEED:
        for feature in FEATURE_LIST:
            features[feature] = np.abs(features[feature])
            
    # subset features for just 'speed_50th'
    features = features[FEATURE_LIST]
    
    stats_dir = Path(SAVE_DIR) / 'Stats'
    plots_dir = Path(SAVE_DIR) / 'Plots'
            
    # pairwise t-tests comparing fepD vs BW for each given number of drops (multiple test corrected)
    pvals_list = []
    for n_drop in drop_list:
        meta_n_drop = metadata[metadata['plate_drop_number']==n_drop]
        feat_n_drop = features.reindex(meta_n_drop.index)
        
        ttest_results = stats(meta_n_drop,
                              feat_n_drop,
                              group_by='bacteria_strain',
                              control='BW',
                              save_dir=None,
                              feature_list=FEATURE_LIST,
                              p_value_threshold=P_VALUE_THRESHOLD,
                              fdr_method=None)
        
        pvals = ttest_results[[c for c in ttest_results.columns if 'pvals' in c]]
        pvals['n_drop'] = n_drop
        pvals_list.append(pvals)        
    pvals_df = pd.concat(pvals_list, axis=0)

    # apply multiple testing correction + save 
    reject, pvals_df['pvals_fepD'] = _multitest_correct(pvals_df['pvals_fepD'], 
                                                        multitest_method=FDR_METHOD,
                                                        fdr=0.05)
    stats_dir.mkdir(parents=True, exist_ok=True)
    pvals_df.to_csv(stats_dir / 't-test_results.csv', header=True, index=False)
    
    # boxplots    
    plot_df = metadata[['bacteria_strain','plate_drop_number']].join(features[FEATURE_LIST])
    colour_dict = dict(zip(strain_list, sns.color_palette(palette='tab10', n_colors=2)))

    for feature in tqdm(FEATURE_LIST):
        plt.close('all')
        sns.set_style('ticks')
        fig = plt.figure(figsize=[18,12])
        ax = fig.add_subplot(1,1,1)
        sns.boxplot(x='plate_drop_number',
                    y=feature,
                    hue='bacteria_strain',
                    order=drop_list,
                    hue_order=strain_list,
                    dodge=True,
                    data=plot_df, 
                    palette=colour_dict,
                    showfliers=False, 
                    showmeans=False,
                    meanprops={"marker":"x", 
                               "markersize":5,
                               "markeredgecolor":"k"},
                    flierprops={"marker":"x", 
                                "markersize":15, 
                                "markeredgecolor":"r"})
        sns.stripplot(x='plate_drop_number',
                      y=feature,
                      hue='bacteria_strain',
                      order=drop_list,
                      hue_order=strain_list,
                      dodge=True,
                      data=plot_df,
                      s=12,
                      palette=None,
                      color='dimgray',
                      marker=".",
                      edgecolor='k',
                      linewidth=0.3)
        
        ax.axes.set_xlabel('Number of Plate Drops', fontsize=30, labelpad=25)
        ax.axes.set_xticklabels([l.get_text() for l in ax.axes.get_xticklabels()], fontsize=25)
        ax.tick_params(axis='x', which='major', pad=15)
        ax.axes.set_ylabel(('abs_' if ABSOLUTE_SPEED else '')
                           + feature.replace('_',' ') 
                           + ' (µm s$^{-1}$)', fontsize=30, labelpad=25)                             
        plt.yticks(fontsize=25)
        
        # scale y-axis
        ymax, ymin = plot_df[feature].max(), plot_df[feature].min()
        d = (ymax - ymin) / 10
        plt.ylim(ymin - d, ymax + d)
        
        # tidy up legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:len(strain_list)], labels[:len(strain_list)], loc='upper left', fontsize=20)
        
        # scale y axis for annotations    
        #trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled
        
        ymax_drop = plot_df.groupby('plate_drop_number')[feature].max() # max y for each plate drop
        
        # add pvalues to plot
        for i, n_drop in enumerate(drop_list):
            n_drop_pvals = pvals_df[pvals_df['n_drop']==n_drop]
            p = n_drop_pvals.loc[feature, 'pvals_fepD']
            text = ax.get_xticklabels()[i]
            assert text.get_text() == str(n_drop)
            p_text = sig_asterix([p])[0]
            
            y = ymax_drop[n_drop]
            ax.text(i, y+15, p_text, fontsize=30, ha='center', va='bottom')
            
            # Plot the bar: [x1,x1,x2,x2],[bar_tips,bar_height,bar_height,bar_tips]
            plt.plot([i-0.2, i-0.2, i+0.2, i+0.2], [y+10, y+12, y+12, y+10], lw=1, c='k')
            
        # save plot
        # plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plots_dir / ('plate_drop_' + ('abs_' if ABSOLUTE_SPEED else '') 
                                 + '{}.svg'.format(feature)))


    # speed time-series plots
    col_dict = dict(zip(strain_list, sns.color_palette(palette='tab10', n_colors=2)))
    colour_dict = dict(zip(treatment_list, [col_dict[i] for i in 
                                            [ii.split('-')[-1] for ii in treatment_list]]))

    for n_drop in tqdm(drop_list):
        groups = ['{}-BW'.format(n_drop),'{}-fepD'.format(n_drop)]
            
        save_dir = Path(SAVE_DIR) / 'timeseries-speed'
        save_dir.mkdir(exist_ok=True, parents=True)
        save_path = save_dir / (('abs_' if ABSOLUTE_SPEED else '')
                                + 'speed_{}_plate_drops.pdf'.format(n_drop))

        print("Plotting timeseries speed")
        plt.close('all')
        fig, ax = plt.subplots(figsize=(15,6), dpi=300)

        for group in groups:
            
            # get control timeseries
            group_ts = get_strain_timeseries(metadata,
                                             project_dir=Path(PROJECT_DIR),
                                             strain=group,
                                             group_by='treatment',
                                             feature_list=['speed'],
                                             save_dir=save_dir,
                                             n_wells=N_WELLS,
                                             verbose=True)
            if ABSOLUTE_SPEED:
                group_ts['speed'] = np.abs(group_ts['speed'])
            
            ax = plot_timeseries(df=group_ts,
                                 feature='speed',
                                 error=True,
                                 max_n_frames=360*FPS, 
                                 smoothing=10*FPS, 
                                 ax=ax,
                                 bluelight_frames=None,
                                 colour=colour_dict[group])
    
        plt.ylim(-10, 250)
        xticks = np.linspace(0, 360*FPS, int(360/60)+1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
        ax.set_xlabel('Time (minutes)', fontsize=20, labelpad=15)
        ylab = ('Absolute ' if ABSOLUTE_SPEED else '') + "Speed (µm s$^{-1}$)"
        ax.set_ylabel(ylab, fontsize=20, labelpad=15)
        ax.legend(groups, fontsize=12, frameon=False, loc='best', handletextpad=1)
        plt.subplots_adjust(left=0.1, top=0.98, bottom=0.15, right=0.98)
    
        # save plot
        print("Saving to: %s" % save_path)
        plt.savefig(save_path)
        
    return


#%% Main

if __name__ == '__main__':
    main()
    
    