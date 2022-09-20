#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keio ubiC vs BW

@author: sm5911
@date: 11/05/2022

"""

#%% Imports

import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from time_series.plot_timeseries import plot_timeseries_feature
from write_data.write import write_list_to_file
from visualisation.plotting_helper import sig_asterix, all_in_one_boxplots

from tierpsytools.preprocessing.filter_data import select_feat_set
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes

#%% Globals

PROJECT_DIR = '/Volumes/hermes$/Keio_ubiC_6WP'
SAVE_DIR = '/Users/sm5911/Documents/Keio_ubiC'
N_WELLS = 6

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
MIN_NSKEL_PER_VIDEO = None
MIN_NSKEL_SUM = 500

FEATURE_SET = ['speed_50th']

WINDOW_DICT = {0:(1805,1815),1:(1830,1840),
               2:(1865,1875),3:(1890,1900),
               4:(1925,1935),5:(1950,1960)}

WINDOW_NAME_DICT = {0:"blue light 1", 1: "20-30 seconds after blue light 1",
                    2:"blue light 2", 3: "20-30 seconds after blue light 2",
                    4:"blue light 3", 5: "20-30 seconds after blue light 3"}

BLUELIGHT_TIMEPOINTS_SECONDS = [(1800,1810),(1860,1870),(1920,1930)]
FPS = 25
VIDEO_LENGTH_SECONDS = 38*60
SMOOTH_WINDOW_SECONDS = 10
  
#%% Functions

def ubic_stats(metadata,
               features,
               group_by='food_type',
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

# def ubiC_stats(metadata, 
#                features,                       
#                save_dir,
#                window=WINDOW_NUMBER,
#                feature_list=[FEATURE],
#                pvalue_threshold=0.05,
#                fdr_method='fdr_by'):
#     """ T-tests comparing worm motion mode on ubiC and fepD vs BW control """
    
#     # subset for window of interest
#     window_meta = metadata.query("window==@window")
    
#     # testing difference in motion mode forwards on ubiC or fepD vs BW
#     do_stats(metadata=window_meta,
#              features=features.reindex(window_meta.index),
#              group_by='food_type',
#              control='BW',
#              save_dir=save_dir / 'ubiC_vs_BW',
#              feat=feature_list,
#              pvalue_threshold=pvalue_threshold,
#              fdr_method=fdr_method,
#              ttest_if_nonsig=True)

#     return
    
# def ubiC_plots(metadata,
#                features,
#                plot_dir,
#                stats_dir,
#                window=WINDOW_NUMBER,
#                feature_list=[FEATURE]):
#     """ Plots of worm motion mode on ubiC and fepD vs BW control """
    
#     assert metadata.shape[0] == features.shape[0]
        
#     window_meta = metadata.query("window==@window")
#     plot_df = window_meta.join(features.reindex(window_meta.index))

#     for feature in tqdm(feature_list):
#         plt.close('all')
#         fig, ax = plt.subplots(figsize=(12,8))
#         sns.boxplot(x='food_type', y=feature, order=food_type_list, ax=ax, data=plot_df,
#                     palette='tab10', showfliers=False) 
#                     #hue='is_dead', hue_order=is_dead_list, dodge=True
#         sns.stripplot(x='food_type', y=feature, order=food_type_list, ax=ax, data=plot_df,
#                       s=5, marker='D', color='k')
#         ax.set_xlabel('')
#         ax.set_ylabel(feature.replace('_',' '), fontsize=12, labelpad=10)
#         ax.set_title('BW control vs fepD and ubiC', pad=30, fontsize=18)
#         # annotate p-values - load t-test results for each treatment vs BW control
#         ttest_path = stats_dir / 'ubiC_vs_BW' / 'food_type_ttest_results.csv'
#         ttest_df = pd.read_csv(ttest_path, index_col=0)
#         for i, food in enumerate(food_type_list):
#             if food == 'BW':
#                 continue
#             assert ax.get_xticklabels()[i].get_text() == food
#             p = ttest_df.loc[feature, 'pvals_' + food]
#             p_text = '***\nP < 0.001' if p < 0.001 else sig_asterix([p])[0] + '\nP = %.3f' % p
#             trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
#             ax.text(i, 1.01, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
#         save_path = Path(plot_dir) / 'ubiC_vs_BW' / '{}.pdf'.format(feature)
#         save_path.parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(save_path, dpi=300)

#     return

# def ubiC_timeseries(metadata, project_dir=PROJECT_DIR, save_dir=SAVE_DIR, window=WINDOW_NUMBER):
#     """ Timeseries plots for addition of enterobactin, paraquat, and iron to BW and fepD """
    
#     metadata = metadata.query("window==@window")
        
#     control = 'BW'        
#     treatment_order = [t for t in sorted(metadata['food_type'].unique()) if t != control]

#     # get timeseries for control data
#     control_timeseries = get_strain_timeseries(metadata[metadata['food_type']==control], 
#                                                project_dir=project_dir, 
#                                                strain=control,
#                                                group_by='food_type',
#                                                n_wells=6,
#                                                save_dir=Path(save_dir) / 'Data' / control,
#                                                verbose=False)

#     for treatment in tqdm(treatment_order):
        
#         test_treatments = [control, treatment]
#         motion_modes = ['forwards','backwards','stationary']

#         for mode in motion_modes:
                    
#             # get timeseries data for treatment data
#             strain_metadata = metadata[metadata['food_type']==treatment]
#             strain_timeseries = get_strain_timeseries(strain_metadata, 
#                                                       project_dir=project_dir, 
#                                                       strain=treatment,
#                                                       group_by='food_type',
#                                                       n_wells=6,
#                                                       save_dir=Path(save_dir) / 'Data' / treatment,
#                                                       verbose=False)

#             print("Plotting timeseries '%s' fraction for '%s' vs '%s'..." %\
#                   (mode, treatment, control))

#             plt.close('all')
#             fig, ax = plt.subplots(figsize=(15,5), dpi=200)
#             colour_dict = dict(zip(test_treatments, sns.color_palette("pastel", len(test_treatments))))
#             bluelight_frames = [(i*60*FPS, i*60*FPS+10*FPS) for i in BLUELIGHT_TIMEPOINTS_MINUTES]

#             ax = plot_timeseries_motion_mode(df=control_timeseries,
#                                              window=SMOOTH_WINDOW_SECONDS*FPS,
#                                              error=True,
#                                              mode=mode,
#                                              max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
#                                              title=None,
#                                              saveAs=None,
#                                              ax=ax,
#                                              bluelight_frames=bluelight_frames,
#                                              colour=colour_dict[control],
#                                              alpha=0.25)
            
#             ax = plot_timeseries_motion_mode(df=strain_timeseries,
#                                              window=SMOOTH_WINDOW_SECONDS*FPS,
#                                              error=True,
#                                              mode=mode,
#                                              max_n_frames=VIDEO_LENGTH_SECONDS*FPS,
#                                              title=None,
#                                              saveAs=None,
#                                              ax=ax,
#                                              bluelight_frames=bluelight_frames,
#                                              colour=colour_dict[treatment],
#                                              alpha=0.25)
        
#             xticks = np.linspace(0, VIDEO_LENGTH_SECONDS*FPS, int(VIDEO_LENGTH_SECONDS/60)+1)
#             ax.set_xticks(xticks)
#             ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
#             ax.set_xlabel('Time (minutes)', fontsize=12, labelpad=10)
#             ax.set_ylabel('Fraction {}'.format(mode), fontsize=12, labelpad=10)
#             ax.legend(test_treatments, fontsize=12, frameon=False, loc='best')
    
#             if BLUELIGHT_WINDOWS_ONLY_TS:
#                 ts_plot_dir = Path(save_dir) / 'Plots' / 'timeseries_bluelight' / treatment
#                 ax.set_xlim([min(BLUELIGHT_TIMEPOINTS_MINUTES)*60*FPS-60*FPS, 
#                              max(BLUELIGHT_TIMEPOINTS_MINUTES)*60*FPS+70*FPS])
#             else:
#                 ts_plot_dir = Path(save_dir) / 'Plots' / 'timeseries' / treatment
    
#             plt.tight_layout()
#             ts_plot_dir.mkdir(exist_ok=True, parents=True)
#             plt.savefig(ts_plot_dir / '{0}_{1}.pdf'.format(treatment, mode))  
    
#     return
    
#%% Main

if __name__ == '__main__':
    
    AUX_DIR = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    RES_DIR = Path(PROJECT_DIR) / 'Results'

    META_PATH = Path(SAVE_DIR) / 'metadata.csv'
    FEAT_PATH = Path(SAVE_DIR) / 'features.csv'
    
    if not META_PATH.exists() and not FEAT_PATH.exists():
    
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
        Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)
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

        ubic_stats(meta_window,
                   feat_window,
                   group_by='food_type',
                   control=control,
                   save_dir=stats_dir,
                   feature_set=FEATURE_SET,
                   pvalue_threshold=0.05,
                   fdr_method='fdr_bh')
        
        order = sorted(meta_window['food_type'].unique())
        colour_dict = dict(zip(order, sns.color_palette('tab10', len(order))))
        all_in_one_boxplots(meta_window,
                            feat_window,
                            group_by='food_type',
                            control=control,
                            save_dir=plot_dir,
                            ttest_path=stats_dir / 't-test' / 't-test_results.csv',
                            feature_set=FEATURE_SET,
                            pvalue_threshold=0.05,
                            order=order,
                            colour_dict=colour_dict,
                            figsize=(15,8),
                            ylim_minmax=None,
                            vline_boxpos=None,
                            fontsize=15,
                            subplots_adjust={'bottom': 0.2, 'top': 0.95, 'left': 0.05, 'right': 0.98})

    # plot speed timeseries
    plot_timeseries_feature(metadata,
                            project_dir=Path(PROJECT_DIR),
                            save_dir=Path(SAVE_DIR) / 'timeseries-speed',
                            group_by='food_type',
                            control='BW',
                            groups_list=None,
                            feature='speed',
                            n_wells=6,
                            bluelight_stim_type='bluelight',
                            video_length_seconds=VIDEO_LENGTH_SECONDS,
                            bluelight_timepoints_seconds=BLUELIGHT_TIMEPOINTS_SECONDS,
                            smoothing=SMOOTH_WINDOW_SECONDS,
                            fps=FPS,
                            ylim_minmax=(-20,330))
    
    