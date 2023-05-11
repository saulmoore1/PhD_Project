#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keio Mutant Worm Screen - Response of several C. elegans mutants to fepD mutants E. coli

@author: sm5911
@date: 30/03/2022

"""

#%% Imports

import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from time_series.plot_timeseries import plot_window_timeseries_feature
from write_data.write import write_list_to_file
from visualisation.plotting_helper import sig_asterix, all_in_one_boxplots
from tierpsytools.preprocessing.filter_data import select_feat_set
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Worm_Mutants"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Worm_Mutants"

N_WELLS = 6

FEATURE_SET = ['speed_50th'] #'motion_mode_forward_fraction'

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
MIN_NSKEL_PER_VIDEO = None
MIN_NSKEL_SUM = 500
PVAL_THRESH = 0.05
FPS = 25
VIDEO_LENGTH_SECONDS = 38*60
BLUELIGHT_TIMEPOINTS_MINUTES = [30,31,32]
BLUELIGHT_WINDOWS_ONLY_TS = True

BIN_SIZE_SECONDS = 5
SMOOTH_WINDOW_SECONDS = 5

WINDOW_DICT = {0:(1805,1815),1:(1830,1840),
               2:(1865,1875),3:(1890,1900),
               4:(1925,1935),5:(1950,1960)}

WINDOW_NAME_DICT = {0:"blue light 1", 1: "20-30 seconds after blue light 1",
                    2:"blue light 2", 3: "20-30 seconds after blue light 2",
                    4:"blue light 3", 5: "20-30 seconds after blue light 3"}

#%% Functions

def mutant_worm_stats(metadata,
                      features,
                      group_by='treatment',
                      control='N2-BW',
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

def mutant_worm_boxplots(metadata, 
                         features,
                         save_dir=None,
                         ttest_path=None,
                         feature_list=None):
    
    from matplotlib import pyplot as plt
    from matplotlib import transforms
    
    if feature_list is not None:
        assert isinstance(feature_list, list) and all(f in features.columns for f in feature_list)
    else:
        feature_list = features.columns.tolist()
            
    plot_df = metadata.join(features)

    dates = plot_df['date_yyyymmdd'].unique()
    date_lut = dict(zip(dates, sns.color_palette('Greys', n_colors=len(dates))))
    
    worm_strain_list = ['N2'] + [w for w in sorted(plot_df['worm_strain'].unique()) if w != 'N2']
    bacteria_strain_list = ['BW', 'fepD']
    
    # load t-test results
    ttest_df = pd.read_csv(ttest_path, index_col=0)
    pvals = ttest_df[[c for c in ttest_df.columns if 'pvals_' in c]]
    pvals.columns = [c.split('pvals_')[-1] for c in pvals.columns]

    for feat in tqdm(feature_list):
        
        plt.close('all')
        fig, ax = plt.subplots(figsize=(12,8), dpi=300)
        sns.boxplot(x='worm_strain', 
                    y=feat, 
                    order=worm_strain_list, 
                    hue='bacteria_strain',
                    hue_order=bacteria_strain_list, 
                    dodge=True,
                    data=plot_df, 
                    palette='tab10',
                    ax=ax, 
                    showfliers=False)
        for date in date_lut.keys():
            date_df = plot_df.query("date_yyyymmdd == @date")
            ax = sns.stripplot(x='worm_strain', 
                                y=feat, 
                                order=worm_strain_list, 
                                hue='bacteria_strain', 
                                hue_order=bacteria_strain_list,
                                dodge=True,
                                data=date_df, 
                                ax=ax, 
                                color=sns.set_palette(palette=[date_lut[date]], 
                                                      n_colors=len(worm_strain_list)),
                                marker='.',
                                alpha=0.7,
                                size=4)
        bacteria_lut = dict(zip(bacteria_strain_list, 
                                sns.color_palette('tab10', len(bacteria_strain_list))))
        markers = [plt.Line2D([0,0],[0,0], color=color, marker='o', linestyle='') 
                    for color in bacteria_lut.values()]
        plt.legend(markers, bacteria_lut.keys(), numpoints=1, frameon=False, 
                    loc='best', markerscale=0.75, fontsize=8, handletextpad=0.2)
        
        # annotate p-values on plot
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

        for ii, worm in enumerate(worm_strain_list):                        
            text = ax.get_xticklabels()[ii]
            assert text.get_text() == worm
            if worm == 'N2':
                p = pvals.loc[feat, 'N2-fepD']
                p_text = 'P<0.001' if p < 0.001 else 'P=%.3f' % p
                ax.text(ii+0.2, 1.02, p_text, fontsize=6, ha='center', va='bottom', transform=trans)
                continue
            else:
                p1 = pvals.loc[feat, str(worm) + '-' + bacteria_strain_list[0]]
                p2 = pvals.loc[feat, str(worm) + '-' + bacteria_strain_list[1]]
                p1_text = 'P<0.001' if p1 < 0.001 else 'P=%.3f' % p1
                p2_text = 'P<0.001' if p2 < 0.001 else 'P=%.3f' % p2
                ax.text(ii-0.2, 1.02, p1_text, fontsize=6, ha='center', va='bottom', transform=trans)
                ax.text(ii+0.2, 1.02, p2_text, fontsize=6, ha='center', va='bottom', transform=trans)

        ax.set_xlabel('Worm - Bacteria', fontsize=15, labelpad=10)
        ax.set_ylabel(feat.replace('_',' '), fontsize=15, labelpad=10)
        
        save_path = Path(save_dir) / '{}.pdf'.format(feat)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    return


def main():
    
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
                                                   from_source_plate=False)
        
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

    # subset metadata results for bluelight videos only 
    bluelight_videos = [i for i in metadata['imgstore_name'] if 'bluelight' in i]
    metadata = metadata[metadata['imgstore_name'].isin(bluelight_videos)]
    
    metadata['treatment'] = metadata[['worm_strain','bacteria_strain']].agg('-'.join, axis=1)
    control = 'N2-BW'

    metadata['window'] = metadata['window'].astype(int)
    window_list = list(metadata['window'].unique())
    
    # worm_strain_list = list(metadata['worm_strain'].unique())
    # bacteria_strain_list = sorted(metadata['bacteria_strain'].unique())

    # ANOVA and t-tests comparing mutant worms on fepD/BW vs N2 on BW (for each window)
    for window in tqdm(window_list):
        meta_window = metadata[metadata['window']==window]
        feat_window = features.reindex(meta_window.index)

        stats_dir = Path(SAVE_DIR) / 'Stats' / WINDOW_NAME_DICT[window]
        plot_dir = Path(SAVE_DIR) / 'Plots' / WINDOW_NAME_DICT[window]

        mutant_worm_stats(meta_window,
                          feat_window,
                          group_by='treatment',
                          control=control,
                          save_dir=stats_dir,
                          feature_set=feature_list,
                          pvalue_threshold=0.05,
                          fdr_method='fdr_bh')
        order = sorted(meta_window['treatment'].unique())
        colour_labels = sns.color_palette('tab10', 2)
        colours = [colour_labels[0] if 'BW' in s else colour_labels[1] for s in order]
        colour_dict = {key:col for (key,col) in zip(order, colours)}
        all_in_one_boxplots(meta_window,
                            feat_window,
                            group_by='treatment',
                            control=control,
                            save_dir=plot_dir,
                            ttest_path=stats_dir / 't-test' / 't-test_results.csv',
                            feature_set=feature_list,
                            pvalue_threshold=0.05,
                            order=order,
                            colour_dict=colour_dict,
                            figsize=(30, 10),
                            ylim_minmax=None,
                            vline_boxpos=[1,3,5,7,9,11],
                            fontsize=15,
                            subplots_adjust={'bottom': 0.2, 'top': 0.95, 'left': 0.05, 'right': 0.98})
    
        # boxplots for each bacteria strain comparing mutants with control at each window
        mutant_worm_boxplots(metadata, 
                             features,
                             save_dir=plot_dir,
                             ttest_path=stats_dir / 't-test' / 't-test_results.csv',
                             feature_list=feature_list)
     
    metadata = metadata[metadata['window']==0]

    # timeseries plots of speed for fepD vs BW control
    BLUELIGHT_TIMEPOINTS_SECONDS = [(i*60,i*60+10) for i in BLUELIGHT_TIMEPOINTS_MINUTES]
    plot_window_timeseries_feature(metadata=metadata,
                                   project_dir=Path(PROJECT_DIR),
                                   save_dir=Path(SAVE_DIR) / 'timeseries-speed',
                                   group_by='treatment',
                                   control='N2-BW',
                                   groups_list=None,
                                   feature='speed',
                                   n_wells=6,
                                   bluelight_timepoints_seconds=BLUELIGHT_TIMEPOINTS_SECONDS,
                                   bluelight_windows_separately=True,
                                   smoothing=10,
                                   figsize=(15,5),
                                   fps=FPS,
                                   ylim_minmax=(-20,320),
                                   xlim_crop_around_bluelight_seconds=(120,300),
                                   video_length_seconds=VIDEO_LENGTH_SECONDS)
    
    return
    
#%% Main

if __name__ == '__main__':
    main()
    
