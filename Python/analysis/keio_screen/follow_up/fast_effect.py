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

from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from tierpsytools.preprocessing.filter_data import select_feat_set

#%% Globals

# JSON_PARAMETERS_PATH = 'analysis/20211102_parameters_keio_fast_effect.json'
PROJECT_DIR = "/Volumes/hermes$/Keio_Fast_Effect"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Fast_Effect"

FEATURE_SET = ['speed_50th']

nan_threshold_row = 0.8
nan_threshold_col = 0.05

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

# def perform_fast_effect_stats(features,
#                               metadata, 
#                               group_by='gene_name', 
#                               control='wild_type', 
#                               window_list=None, 
#                               save_dir=None,
#                               feature_set=None,
#                               pvalue_threshold=0.05,
#                               fdr_method='fdr_by'):
#     """ T-tests comparing worms on Keio mutants vs BW control (for each window separately) """

#     print("\nInvestigating variation between hit strains and control (for each window separately)")    
        
#     # subset for windows in window_frame_dict
#     if window_list is not None:
#         assert all(w in metadata['window'] for w in window_list)
#     else:
#         window_list = metadata['window'].unique().tolist()
        
#     if feature_set is not None:
#         assert isinstance(feature_set, list) and all(f in features.columns for f in feature_set)
#     else:
#         feature_set = features.columns.tolist()
       
#     strain_list = metadata[group_by].unique().tolist()

#     for window in window_list:
#         window_meta = metadata[metadata['window']==window]
#         window_feat = features.reindex(window_meta.index)
    
#         # ANOVA
#         if len(strain_list) > 2:
            
#             test_path = Path(save_dir) / 'window_{}'.format(window) / 'ANOVA_results.csv'
#             test_path.parent.mkdir(exist_ok=True, parents=True)
    
#             stats, pvals, reject = univariate_tests(X=window_feat,
#                                                     y=window_meta[group_by],
#                                                     test='ANOVA',
#                                                     control=control,
#                                                     comparison_type='multiclass',
#                                                     multitest_correction=fdr_method,
#                                                     alpha=pvalue_threshold,
#                                                     n_permutation_test=None)
            
#             effect_sizes = get_effect_sizes(X=window_feat,
#                                             y=window_meta[group_by],
#                                             control=control,
#                                             effect_type=None,
#                                             linked_test='ANOVA')
            
#             # compile and save results
#             test_results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
#             test_results.columns = ['stats','effect_size','pvals','reject']     
#             test_results['significance'] = sig_asterix(test_results['pvals'])
#             test_results = test_results.sort_values(by=['pvals'], ascending=True) # rank pvals
#             test_results.to_csv(test_path, header=True, index=True)
            
#             nsig = test_results['reject'].sum()
#             print("%d features (%.1f%%) signficantly different among '%s'" % (nsig, 
#                   (nsig/len(test_results.index))*100, group_by))
    
#         # t-tests
#         ttest_path = Path(save_dir) / 'window_{}'.format(window) / 't-test_results.csv'
    
#         stats_t, pvals_t, reject_t = univariate_tests(X=window_feat,
#                                                       y=window_meta[group_by],
#                                                       test='t-test',
#                                                       control=control,
#                                                       comparison_type='binary_each_group',
#                                                       multitest_correction=fdr_method,
#                                                       alpha=pvalue_threshold,
#                                                       n_permutation_test=None)
        
#         effect_sizes_t = get_effect_sizes(X=window_feat,
#                                           y=window_meta[group_by],
#                                           control=control,
#                                           effect_type=None,
#                                           linked_test='t-test')
        
#         # compile and save results
#         stats_t.columns = ['stats_' + str(c) for c in stats_t.columns]
#         pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
#         reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
#         effect_sizes_t.columns = ['effect_size_' + str(c) for c in effect_sizes_t.columns]
#         ttest_results = pd.concat([stats_t, effect_sizes_t, pvals_t, reject_t], axis=1)
#         ttest_results.to_csv(ttest_path, header=True, index=True)
        
#         # record t-test significant features (not ordered)
#         fset_ttest = pvals_t[np.asmatrix(reject_t)].index.unique().to_list()
#         #assert set(fset_ttest) == set(pvals_t.index[(pvals_t < args.pval_threshold).sum(axis=1) > 0])
#         print("%d significant features for any %s vs %s (t-test, %s, P<%.2f)" % (len(fset_ttest),
#               group_by, control, fdr_method, pvalue_threshold))
    
#         if len(fset_ttest) > 0:
#             ttest_sigfeats_path = Path(save_dir) / 'window_{}'.format(window) / 't-test_sigfeats.txt'
#             write_list_to_file(fset_ttest, ttest_sigfeats_path)
                
#     return

# def analyse_fast_effect(features, 
#                         metadata, 
#                         group_by='gene_name',
#                         control='wild_type',
#                         window_list=None,
#                         save_dir=None,
#                         stats_dir=None,
#                         feature_set=None,
#                         pvalue_threshold=0.05,
#                         fdr_method='fdr_by'):
        
#     if window_list is not None:
#         assert all(w in metadata['window'] for w in window_list)
#     else:
#         window_list = metadata['window'].unique().tolist()
        
#     if feature_set is not None:
#         assert isinstance(feature_set, list) and all(f in features.columns for f in feature_set)
#     else:
#         feature_set = features.columns.tolist()
        
#     # subset for windows in window_list
#     if window_list is not None:
#         assert isinstance(window_list, list) and all(w in metadata['window'] for w in window_list)
#         metadata = metadata[metadata['window'].isin(window_list)]
#         features = features.reindex(metadata.index)
#     else:
#         window_list = metadata['window'].unique().tolist()

#     strain_list = [control] + [s for s in sorted(metadata[group_by].unique()) if s != control]
        
#     # plot dates as different colours (in loop)
#     date_lut = dict(zip(list(metadata['date_yyyymmdd'].unique()), 
#                         sns.color_palette('Greys', n_colors=len(metadata['date_yyyymmdd'].unique()))))
    
#     for window in window_list:
#         print("Plotting for window %d" % window)
        
#         plot_meta = metadata[metadata['window']==window]
#         plot_feat = features.reindex(plot_meta.index)
#         plot_df = plot_meta.join(plot_feat[feature_set])
        
#         # plot control/strain for all windows
#         for feature in feature_set:
            
#             plt.close('all')
#             fig, ax = plt.subplots(figsize=((len(strain_list) if len(strain_list) >= 20 else 12),8))
#             ax = sns.boxplot(x=group_by, y=feature, order=strain_list, data=plot_df, 
#                              palette='tab10', ax=ax)
#             for date in date_lut.keys():
#                 date_df = plot_df[plot_df['date_yyyymmdd']==date]   
#                 ax = sns.stripplot(x=group_by, y=feature, order=strain_list, data=date_df, 
#                                    color=date_lut[date], alpha=0.7, size=4, ax=ax)
#             # n_labs = len(plot_df['date_yyyymmdd'].unique())
#             # handles, labels = ax.get_legend_handles_labels()
#             # ax.legend(handles[:n_labs], labels[:n_labs], fontsize=15, frameon=False, loc='lower left')
                    
#             # # scale plot to omit outliers (>2.5*IQR from mean)
#             # if scale_outliers_box:
#             #     grouped_strain = plot_df.groupby(group_by)
#             #     y_bar = grouped_strain[feature].median() # median is less skewed by outliers
#             #     # Computing IQR
#             #     Q1 = grouped_strain[feature].quantile(0.25)
#             #     Q3 = grouped_strain[feature].quantile(0.75)
#             #     IQR = Q3 - Q1
#             #     plt.ylim(-0.01, (max(y_bar) + 3 * max(IQR) if max(y_bar) + 3 * max(IQR) < 1 else 1.05))
                
#             # load t-test results + annotate p-values on plot
#             ttest_path = Path(stats_dir) / 'window_{}'.format(window) / 't-test_results.csv'
#             ttest_df = pd.read_csv(ttest_path, index_col=0)
#             pvals = ttest_df[[c for c in ttest_df.columns if 'pvals_' in c]]
#             pvals.columns = [c.split('pvals_')[-1] for c in pvals.columns]
            
#             for ii, strain in enumerate(strain_list[1:]):
#                 text = ax.get_xticklabels()[ii+1]
#                 assert text.get_text() == strain
#                 p = pvals.loc[feature, str(strain)]
#                 p_text = 'P<0.001' if p < 0.001 else 'P=%.3f' % p
#                 #y = (y_bar[antiox] + 2 * IQR[antiox]) if scale_outliers_box else plot_df[feature].max()
#                 #h = (max(IQR) / 10) if scale_outliers_box else (y - plot_df[feature].min()) / 50
#                 trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
#                 ax.text(ii+1, 1.02, p_text, fontsize=10, ha='center', va='bottom', transform=trans)

#             ax.set_xlabel('Time (minutes)', fontsize=15, labelpad=10)
#             ax.set_ylabel(feature.replace('_',' '), fontsize=15, labelpad=10)
            
#             save_path = Path(save_dir) / 'window_{}'.format(window) / (feature + '.pdf')
#             save_path.parent.mkdir(parents=True, exist_ok=True)
#             plt.savefig(save_path, dpi=300)

#     return
    
# def fast_effect_timeseries_motion_mode(metadata, 
#                                        project_dir, 
#                                        save_dir,
#                                        group_by='gene_name',
#                                        control='wild_type',
#                                        strain_list=['fepD'],
#                                        bluelight_windows_separately=False,
#                                        smoothing=120):
#     """ Timeseries plots (10 seconds BL delivered every 30 minutes, for 5 hours total) """
        
#     # get timeseries for BW
#     assert control in metadata[group_by].unique()
#     control_ts = get_strain_timeseries(metadata[metadata[group_by]==control], 
#                                        project_dir=project_dir, 
#                                        strain=control,
#                                        group_by=group_by,
#                                        n_wells=6,
#                                        save_dir=save_dir / 'Data' / control,
#                                        verbose=True)
    
#     if strain_list is not None:
#         assert isinstance(strain_list, list)
#         assert all(s in metadata[group_by].unique() for s in strain_list)
#         strain_list = [s for s in strain_list if s != control]
#     else:
#         strain_list = [s for s in metadata[group_by].unique().tolist() if s != control]
    
#     for strain in tqdm(strain_list):
#         # get timeseries for strain
#         strain_ts = get_strain_timeseries(metadata[metadata[group_by]==strain], 
#                                           project_dir=project_dir, 
#                                           strain=strain,
#                                           group_by=group_by,
#                                           n_wells=6,
#                                           save_dir=save_dir / 'Data' / strain,
#                                           verbose=True)
     
#         colour_dict = dict(zip([control, strain], sns.color_palette("tab10", 2)))
#         bluelight_frames = [(i*60*FPS, i*60*FPS+10*FPS) for i in BLUELIGHT_TIMEPOINTS_MINUTES]
    
#         for mode in motion_modes:
#             print("Plotting timeseries '%s' fraction for %s vs %s..." % (mode, strain, control))
    
#             if bluelight_windows_separately:
                
#                 for pulse, timepoint in enumerate(tqdm(BLUELIGHT_TIMEPOINTS_MINUTES), start=1):
    
#                     plt.close('all')
#                     fig, ax = plt.subplots(figsize=(15,5), dpi=150)
            
#                     ax = plot_timeseries_motion_mode(df=control_ts,
#                                                      window=smoothing*FPS,
#                                                      error=True,
#                                                      mode=mode,
#                                                      max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
#                                                      title=None,
#                                                      saveAs=None,
#                                                      ax=ax,
#                                                      bluelight_frames=bluelight_frames,
#                                                      colour=colour_dict[control],
#                                                      alpha=0.25)
                    
#                     ax = plot_timeseries_motion_mode(df=strain_ts,
#                                                      window=smoothing*FPS,
#                                                      error=True,
#                                                      mode=mode,
#                                                      max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
#                                                      title=None,
#                                                      saveAs=None,
#                                                      ax=ax,
#                                                      bluelight_frames=bluelight_frames,
#                                                      colour=colour_dict[strain],
#                                                      alpha=0.25)
                
#                     xticks = np.linspace(0, VIDEO_LENGTH_SECONDS*FPS, int(VIDEO_LENGTH_SECONDS/60)+1)
#                     ax.set_xticks(xticks)
#                     ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
                    
#                     # -30secs before to +2mins after each pulse
#                     xlim_range = (timepoint*60-30, timepoint*60+120)
#                     ax.set_xlim([xlim_range[0]*FPS, xlim_range[1]*FPS])
    
#                     ax.set_xlabel('Time (minutes)', fontsize=12, labelpad=10)
#                     ax.set_ylabel('Fraction {}'.format(mode), fontsize=12, labelpad=10)
#                     ax.set_title('{0} vs {1} (bluelight pulse {2} = {3} min)'.format(
#                         strain, control, pulse, timepoint), fontsize=12, pad=10)
#                     ax.legend([control, strain], fontsize=12, frameon=False, loc='best')
            
#                     # save plot
#                     ts_plot_dir = save_dir / 'Plots' / strain
#                     ts_plot_dir.mkdir(exist_ok=True, parents=True)
#                     save_path = ts_plot_dir /\
#                         'motion_mode_{0}_bluelight_pulse{1}_{2}min.pdf'.format(mode, pulse, timepoint)
#                     print("Saving to: %s" % save_path)
#                     plt.savefig(save_path)
                    
#             else:    
#                 plt.close('all')
#                 fig, ax = plt.subplots(figsize=(30,5), dpi=150)
        
#                 ax = plot_timeseries_motion_mode(df=control_ts,
#                                                  window=smoothing*FPS,
#                                                  error=True,
#                                                  mode=mode,
#                                                  max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
#                                                  title=None,
#                                                  saveAs=None,
#                                                  ax=ax,
#                                                  bluelight_frames=bluelight_frames,
#                                                  colour=colour_dict[control],
#                                                  alpha=0.25)
                
#                 ax = plot_timeseries_motion_mode(df=strain_ts,
#                                                  window=smoothing*FPS,
#                                                  error=True,
#                                                  mode=mode,
#                                                  max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
#                                                  title=None,
#                                                  saveAs=None,
#                                                  ax=ax,
#                                                  bluelight_frames=bluelight_frames,
#                                                  colour=colour_dict[strain],
#                                                  alpha=0.25)
            
#                 xticks = np.linspace(0, VIDEO_LENGTH_SECONDS*FPS, int(VIDEO_LENGTH_SECONDS/60)+1)
#                 ax.set_xticks(xticks)
#                 ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
#                 ax.set_xlabel('Time (minutes)', fontsize=12, labelpad=10)
#                 ax.set_ylabel('Fraction {}'.format(mode), fontsize=12, labelpad=10)
#                 ax.set_title('%s vs %s' % (strain, control), fontsize=12, pad=10)
#                 ax.legend([control, strain], fontsize=12, frameon=False, loc='best')
        
#                 # save plot
#                 ts_plot_dir = save_dir / 'Plots' / strain
#                 ts_plot_dir.mkdir(exist_ok=True, parents=True)
#                 save_path = ts_plot_dir / 'motion_mode_{}.pdf'.format(mode)
#                 print("Saving to: %s" % save_path)
#                 plt.savefig(save_path)

#     return
    
#%% Main

if __name__ == '__main__':   
    # parser = argparse.ArgumentParser(description="Analyse acute response videos to investigate how \
    # fast the food takes to influence worm behaviour")
    # parser.add_argument('-j','--json', help="Path to JSON parameters file", default=JSON_PARAMETERS_PATH)
    # args = parser.parse_args()
    # args = load_json(args.json)

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
                                                   nan_threshold_row=nan_threshold_row,
                                                   nan_threshold_col=nan_threshold_col,
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

    metadata['window'] = metadata['window'].astype(int)
    window_list = list(metadata['window'].unique())
 
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

        order = ['BW','fepD']
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
                            ylim_minmax=(-20,500),
                            vline_boxpos=None,
                            fontsize=20,
                            subplots_adjust={'bottom':0.15,'top':0.9,'left':0.15,'right':0.95})        
        
    # fast_effect_timeseries_motion_mode(metadata=metadata.query("window == 0"), # avoid plotting multiple times
    #                                    project_dir=Path(args.project_dir),
    #                                    save_dir=Path(args.save_dir) / 'timeseries',
    #                                    group_by='gene_name',
    #                                    control=control,
    #                                    bluelight_windows_separately=False,
    #                                    smoothing=120) # moving window of 2 minutes for smoothing

    # fast_effect_timeseries_motion_mode(metadata=metadata, # around each window in turn
    #                                    project_dir=Path(args.project_dir),
    #                                    save_dir=Path(args.save_dir) / 'timeseries',
    #                                    group_by='gene_name',
    #                                    control=control,
    #                                    bluelight_windows_separately=True,
    #                                    smoothing=10) # moving window of 10 seconds for smoothing
    
    # timeseries plots of speed for fepD vs BW control
    
    strain_list = ['BW','fepD']
    
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


    