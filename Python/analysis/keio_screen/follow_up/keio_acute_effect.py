#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse fast-effect (acute response) videos
- Bluelight delivered for 10 seconds every 30 minutes, for a total of 5 hours
- Window feature summaries 20-30 seconds after each bluelight stimulus

When do we start to see an effect on worm behaviour? At which timepoint/window? 
Do we see a fast-acting C. elegans arousal on siderophore mutants?

@author: sm5911
@date: 24/11/2021 (updated: 28/10/2024)

"""

#%% Imports

import pandas as pd
import seaborn as sns
#from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import transforms
from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from time_series.plot_timeseries import plot_timeseries_feature # plot_timeseries_motion_mode
from visualisation.plotting_helper import sig_asterix #, all_in_one_boxplots

#from tierpsytools.preprocessing.filter_data import select_feat_set
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from tierpsytools.analysis.statistical_tests import _multitest_correct

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Acute_Effect"
SAVE_DIR = "/Users/sm5911/Documents/PhD_DLBG/Keio_Acute_Effect"

FEATURE = 'speed_50th'

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
BLUELIGHT_TIMEPOINTS_MINUTES = [30,60,90,120,150,180,210,240]
WINDOW_BLUELIGHT_DICT = dict(zip(WINDOW_LIST, BLUELIGHT_TIMEPOINTS_MINUTES))
VIDEO_LENGTH_SECONDS = 5*60*60
FPS = 25

OMIT_STRAINS_LIST = ['trpD']

#%% Functions

def multi_window_stats(metadata,
                       features, 
                       group_by='gene_name',
                       control='BW',
                       strain_list=['BW','fepD'],
                       save_dir=None,
                       feature='speed_50th',
                       pvalue_threshold=0.05,
                       fdr_method='fdr_by'):

    # check case-sensitivity of items in 'group_by' column
    assert len(metadata[group_by].unique()) == len(metadata[group_by].str.upper().unique())
    
    window_list = sorted(metadata['window'].unique())

    # sample_size = metadata.groupby(group_by).count() # print mean sample size per group
    # print("Mean sample size per %s: %d" % (group_by, int(sample_size[sample_size.columns[-1]].mean())))

    if strain_list:
        strain_list = [control] + [i for i in strain_list if i != control]
        metadata = metadata[metadata['gene_name'].isin(strain_list)]
        features = features.reindex(metadata.index)
        
    pvalues_dict = {}

    # Perform t-tests for each window separately and then perform multiple test correction
    for window in window_list:
        
        meta_window = metadata[metadata['window']==window]
        feat_window = features.reindex(meta_window.index)
             
        stats_t, pvals_t, reject_t = univariate_tests(X=feat_window,
                                                      y=meta_window[group_by],
                                                      control=control,
                                                      test='t-test',
                                                      comparison_type='binary_each_group',
                                                      multitest_correction=fdr_method,
                                                      alpha=pvalue_threshold)
    
        effect_sizes_t = get_effect_sizes(X=feat_window,
                                          y=meta_window[group_by],
                                          control=control,
                                          linked_test='t-test')
    
        stats_t.columns = ['stats_' + str(c) for c in stats_t.columns]
        pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
        reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
        effect_sizes_t.columns = ['effect_size_' + str(c) for c in effect_sizes_t.columns]
        window_ttest_results = pd.concat([stats_t, pvals_t, reject_t, effect_sizes_t], axis=1)
        
        pvalues_dict[window] = window_ttest_results.loc[feature, pvals_t.columns[0]]

    # multiple test correction - account for separate testing across windows
    reject, corrected_pvals = _multitest_correct(pd.Series(list(pvalues_dict.values())), 
                                                 multitest_method=fdr_method, fdr=0.05)
    
    pvalues_dict = dict(zip([WINDOW_BLUELIGHT_DICT[w] for w in window_list], corrected_pvals))
    pvals = pd.DataFrame.from_dict(pvalues_dict, orient='index', columns=strain_list[1:])
    
    if save_dir is not None:
        ttest_corrected_savepath = Path(save_dir) / 't-test_window_results.csv'
        ttest_corrected_savepath.parent.mkdir(exist_ok=True, parents=True)
        pvals.to_csv(ttest_corrected_savepath)
    
    return pvals  
  
def stats(metadata,
          features,
          group_by='treatment',
          control='BW',
          feat='speed_50th',
          pvalue_threshold=0.05,
          fdr_method='fdr_bh'):
    
    """ Perform ANOVA and t-tests to compare worm speed on each treatment vs control """
        
    assert all(metadata.index == features.index)
    features = features[[feat]]

    # print mean sample size
    sample_size = metadata.groupby(group_by).count()
    print("Mean sample size of %s/window: %d" % (group_by, 
                                                 int(sample_size[sample_size.columns[-1]].mean())))

    n = len(metadata[group_by].unique())
    if n > 2:
   
        # perform ANOVA - is there variation among strains?
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

        # compile results
        anova_results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
        anova_results.columns = ['stats','effect_size','pvals','reject']     
        anova_results['significance'] = sig_asterix(anova_results['pvals'])
        anova_results = anova_results.sort_values(by=['pvals'], ascending=True) # rank by p-value
             
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
        
    nsig = sum(reject_t.sum(axis=1) > 0)
    print("%d significant features between any %s vs %s (t-test, P<%.2f, %s)" %\
          (nsig, group_by, control, pvalue_threshold, fdr_method))

    return anova_results, ttest_results


def main():
    
    aux_dir = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    results_dir = Path(PROJECT_DIR) / 'Results'
    
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
    
    # # load feature set
    # if FEATURE_SET is not None:
    #     # subset for selected feature set (and remove path curvature features)
    #     if isinstance(FEATURE_SET, int) and FEATURE_SET in [8,16,256]:
    #         features = select_feat_set(features, 'tierpsy_{}'.format(FEATURE_SET), append_bluelight=True)
    #         features = features[[f for f in features.columns if 'path_curvature' not in f]]
    #     elif isinstance(FEATURE_SET, list) or isinstance(FEATURE_SET, set):
    #         assert all(f in features.columns for f in FEATURE_SET)
    #         features = features[FEATURE_SET].copy()
    # feature_list = features.columns.tolist()

    # subset metadata results for bluelight videos only 
    bluelight_videos = [i for i in metadata['imgstore_name'] if 'bluelight' in i]
    metadata = metadata[metadata['imgstore_name'].isin(bluelight_videos)]
        
    if OMIT_STRAINS_LIST is not None:
        metadata = metadata[~metadata['gene_name'].isin(OMIT_STRAINS_LIST)]

    metadata['window'] = metadata['window'].astype(int)
    if WINDOW_LIST is not None:
        metadata = metadata[metadata['window'].isin(WINDOW_LIST)]
            
    features = features[[FEATURE]].reindex(metadata.index)
    
    strain_list = ['BW','fepD']
    strain_lut = dict(zip(strain_list, sns.color_palette(palette="tab10", n_colors=len(strain_list))))
        
    # boxplot
    plot_df = metadata.join(features)
    plot_df = plot_df[plot_df['gene_name'].isin(strain_list)] # subset for BW and fepD only
    
    plt.close('all')
    sns.set_style('ticks')
    fig = plt.figure(figsize=[15,6])
    ax = fig.add_subplot(1,1,1)
    sns.boxplot(x='window',
                y='speed_50th',
                data=plot_df, 
                order=WINDOW_LIST,
                hue='gene_name',
                hue_order=strain_list,
                palette=strain_lut,
                dodge=True,
                showfliers=False, 
                showmeans=False,
                meanprops={"marker":"x", 
                           "markersize":5,
                           "markeredgecolor":"k"},
                flierprops={"marker":"x", 
                            "markersize":15, 
                            "markeredgecolor":"r"})
    dates = sorted(plot_df['date_yyyymmdd'].unique())
    date_lut = dict(zip(dates, sns.color_palette(palette="Greys", n_colors=len(dates))))
    for date in dates:
        date_df = plot_df[plot_df['date_yyyymmdd']==date]
        sns.stripplot(x='window',
                      y='speed_50th',
                      data=date_df,
                      s=8,
                      order=WINDOW_LIST,
                      hue='gene_name',
                      hue_order=strain_list,
                      dodge=True,
                      palette=[date_lut[date]] * 2,
                      color=None,
                      marker=".",
                      edgecolor='k',
                      linewidth=0.3) #facecolors="none"
    
    # do stats
    pvals = multi_window_stats(metadata,
                               features,
                               group_by='gene_name',
                               control='BW',
                               save_dir=None,
                               feature='speed_50th',
                               pvalue_threshold=0.05,
                               fdr_method='fdr_bh') 
    # NB: Benjamini-Hochberg correction used instead of Benjamini-Yekutieli 
    #     only 2 strains (incl. BW control) / 1 feature 
    #     - correction only needed for t-test across multiple windows   
    
    # scale y axis for annotations    
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled

    # ax.axes.get_xaxis().get_label().set_visible(False) # remove x axis label
    ax.axes.set_xticklabels(BLUELIGHT_TIMEPOINTS_MINUTES, fontsize=20)
    ax.axes.set_xlabel('Time (minutes)', fontsize=25, labelpad=20)                           
    ax.tick_params(axis='y', which='major', pad=15)
    plt.yticks(fontsize=20)
    ax.axes.set_ylabel('Speed (µm s$^{-1}$)', fontsize=25, labelpad=20)  
    #plt.ylim(-50, 350)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(strain_list)], labels[:len(strain_list)], loc='best', frameon=False)
    #plt.axhline(y=0, c='grey')
    
    # add pvalues to plot
    for i, tp in enumerate(BLUELIGHT_TIMEPOINTS_MINUTES):
        p = pvals.loc[tp, 'fepD']
        text = ax.get_xticklabels()[i]
        assert text.get_text() == str(tp)
        p_text = sig_asterix([p])[0]
        ax.text(i, 1.03, p_text, fontsize=35, ha='center', va='center', transform=trans)
            
    #plt.subplots_adjust(left=0.01, right=0.9)
    boxplot_path = Path(SAVE_DIR) / 'Plots' / 'speed_50th_vs_BW.svg'
    boxplot_path.parent.mkdir(exist_ok=True, parents=True)
    #plt.subplots_adjust(left=0.125,right=0.9,bottom=0.1,top=0.9)
    plt.savefig(boxplot_path, bbox_inches='tight', transparent=True)
 
    
    # boxplots for blue light timepoint 30 minutes
    meta_window = metadata[metadata['window']==1]
    feat_window = features.reindex(meta_window.index)
    plot_df = meta_window.join(feat_window)
    strain_list = ['BW'] + [s for s in sorted(metadata['gene_name'].unique()) if s != 'BW']
    strain_lut = dict(zip(strain_list, sns.color_palette('tab10', len(strain_list))))

    plt.close('all')
    sns.set_style('ticks')
    fig = plt.figure(figsize=[12,6])
    ax = fig.add_subplot(1,1,1)
    sns.boxplot(x='gene_name',
                y='speed_50th',
                data=plot_df, 
                order=strain_list,
                palette=strain_lut,
                showfliers=False, 
                showmeans=False,
                meanprops={"marker":"x", 
                           "markersize":5,
                           "markeredgecolor":"k"},
                flierprops={"marker":"x", 
                            "markersize":15, 
                            "markeredgecolor":"r"})
    dates = sorted(plot_df['date_yyyymmdd'].unique())
    date_lut = dict(zip(dates, sns.color_palette(palette="Greys", n_colors=len(dates))))
    for date in dates:
        date_df = plot_df[plot_df['date_yyyymmdd']==date]
        sns.stripplot(x='gene_name',
                      y='speed_50th',
                      data=date_df,
                      s=8,
                      order=strain_list,
                      palette=[date_lut[date]] * len(strain_list),
                      color=None,
                      marker=".",
                      edgecolor='k',
                      linewidth=0.3) #facecolors="none"

    ax.axes.get_xaxis().get_label().set_visible(False) # remove x axis label
    # ax.axes.set_xticklabels(BLUELIGHT_TIMEPOINTS_MINUTES, fontsize=20)
    # ax.axes.set_xlabel('Time (minutes)', fontsize=25, labelpad=20)   
    ax.tick_params(axis='x', which='major', pad=15)
    plt.xticks(fontsize=20)                        
    ax.tick_params(axis='y', which='major', pad=15)
    plt.yticks(fontsize=20)
    ax.axes.set_ylabel('Speed (µm s$^{-1}$)', fontsize=25, labelpad=20)  
    plt.ylim(0, 320)
    
    # add p-values to plot (stats for all strains, first BL window = 30 minutes)
    _, ttest_results = stats(meta_window,
                             feat_window,
                             group_by='gene_name',
                             control='BW',
                             feat='speed_50th',
                             pvalue_threshold=0.05,
                             fdr_method='fdr_bh')
    ttest_results.to_csv(Path(SAVE_DIR) / 'Stats' / 't-test_all_strains_window_1_results.csv', 
                         header=True, index=False)
        
    # add pvalues to plot
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled
    pvals = ttest_results[[c for c in ttest_results.columns if 'pval' in c]]
    for i, strain in enumerate(strain_list[1:], start=1):
        p = pvals.loc['speed_50th', 'pvals_' + strain]
        text = ax.get_xticklabels()[i]
        assert text.get_text() == strain
        p_text = sig_asterix([p])[0]
        ax.text(i, 1.03, p_text, fontsize=25, ha='center', va='center', transform=trans)
            
    #plt.subplots_adjust(left=0.01, right=0.9)
    boxplot_path = Path(SAVE_DIR) / 'Plots' / 'all_strains_speed_50th_vs_BW_window_1.svg'
    boxplot_path.parent.mkdir(exist_ok=True, parents=True)
    #plt.subplots_adjust(left=0.125,right=0.9,bottom=0.1,top=0.9)
    plt.savefig(boxplot_path, bbox_inches='tight', transparent=True)

    
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

    