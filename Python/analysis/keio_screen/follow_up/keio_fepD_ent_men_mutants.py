#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: sm5911
"""

#%% Imports

import pandas as pd
from pathlib import Path
from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
#from tierpsytools.preprocessing.filter_data import select_feat_set
from visualisation.plotting_helper import sig_asterix
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
import seaborn as sns
import colorcet as cc
from matplotlib import pyplot as plt
from matplotlib import transforms

#%% Globals

SAVE_DIR = "/Users/sm5911/Documents/PhD_DLBG/Other/Keio_fepD_ent_men_mutants"

N_WELLS = 6

BLUELIGHT_TIMEPOINTS_SECONDS = [(60, 70),(160, 170),(260, 270)]
WINDOW_DICT = {0:(290,300)}
WINDOW_NAME_DICT = {0:"20-30 seconds after blue light 3"}

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
IMPUTE_NAN = True
MIN_NSKEL_PER_VIDEO = None
MIN_NSKEL_SUM = 500
P_VALUE_THRESHOLD = 0.05
FDR_METHOD = 'fdr_bh'

RENAME_DICT = {"control": "wild_type"}

FEATURE = 'speed_50th'

#%% Functions

def stats(metadata,
          features,
          group_by='treatment',
          control='BW',
          feat='speed_50th',
          save_dir=None,
          pvalue_threshold=0.05,
          fdr_method='fdr_bh'):
        
    assert all(metadata.index == features.index)
    if feat is not None:
        features = features[[feat]]
    
    # check case-sensitivity
    assert len(metadata[group_by].unique()) == len(metadata[group_by].str.upper().unique())

    # print mean sample size
    sample_size = metadata.groupby(group_by).count()
    print("Mean sample size per %s: %d" % (group_by, int(sample_size.max(axis=1).mean())))

    n = len(metadata[group_by].unique())
    if n > 2:
   
        # Perform ANOVA - is there variation among strains?
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

        # compile ANOVA results
        anova_results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
        anova_results.columns = ['stats','effect_size','pvals','reject']     
        anova_results['significance'] = sig_asterix(anova_results['pvals'])
        anova_results = anova_results.sort_values(by=['pvals'], ascending=True)

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
        ttest_path = Path(save_dir) / 't-test_results.csv'
        ttest_path.parent.mkdir(exist_ok=True, parents=True)
        ttest_results.to_csv(ttest_path, header=True, index=True)
    
    return ttest_results

def main():
    
    metadata_path_local = Path(SAVE_DIR) / 'metadata_clean.csv'
    features_path_local = Path(SAVE_DIR) / 'features_clean.csv'
    
    aux_dir = Path(SAVE_DIR) / 'AuxiliaryFiles'
    res_dir = Path(SAVE_DIR) / 'Results'
        
    if not metadata_path_local.exists() or not features_path_local.exists():

        # compile metadata
        metadata, metadata_path = compile_metadata(aux_dir, 
                                                   n_wells=N_WELLS, 
                                                   add_well_annotations=False,
                                                   from_source_plate=False)
        
        # compile feature summaries
        features, metadata = process_feature_summaries(metadata_path, 
                                                       results_dir=res_dir, 
                                                       compile_day_summaries=False, 
                                                       imaging_dates=None, 
                                                       align_bluelight=False, 
                                                       window_summaries=True,
                                                       n_wells=N_WELLS)

        # clean results - remove bad well data + features with too many NaNs or 
        # zero std and impute NaNs
        features, metadata = clean_summary_results(features, 
                                                   metadata,
                                                   feature_columns=None,
                                                   nan_threshold_row=NAN_THRESHOLD_ROW,
                                                   nan_threshold_col=NAN_THRESHOLD_COL,
                                                   max_value_cap=1e15,
                                                   imputeNaN=IMPUTE_NAN,
                                                   min_nskel_per_video=MIN_NSKEL_PER_VIDEO,
                                                   min_nskel_sum=MIN_NSKEL_SUM,
                                                   drop_size_related_feats=False,
                                                   norm_feats_only=False)
        
        # save clean metadata and features            
        metadata.to_csv(metadata_path_local, index=False)
        features.to_csv(features_path_local, index=False)
        
    else:
        metadata = pd.read_csv(metadata_path_local, header=0, index_col=None, 
                               dtype={'comments':str, 
                                      'date_yyyymmdd':int})
        features = pd.read_csv(features_path_local, header=0, index_col=None)
    
    assert not features.isna().sum(axis=1).any()
    assert not (features.std(axis=1) == 0).any()
    assert set(features.index) == set(metadata.index)    
    assert (len(metadata['mutation'].unique()) ==
            len(metadata['mutation'].str.upper().unique())) # check case-sensitivity

    # subset metadata results for bluelight videos only 
    bluelight_videos = [i for i in metadata['imgstore_name'] if 'bluelight' in i]
    metadata = metadata[metadata['imgstore_name'].isin(bluelight_videos)]

    features = features.reindex(metadata.index)

    # rename bacterial mutant names in metadata
    for k, v in RENAME_DICT.items():
        metadata.loc[metadata['mutation'] == k, 'mutation'] = v
                 
    # # subset features for selected tierpsy feature set
    # features = select_feat_set(features, tierpsy_set_name='tierpsy_256', 
    #                            append_bluelight=True)
        
    bwfep = ['wild_type','fepD']
    strain_list = bwfep + [s for s in sorted(metadata['mutation'].unique())
                                          if s not in bwfep]
    
    ttest_results = stats(metadata, 
                          features, 
                          group_by='mutation', 
                          control='wild_type', 
                          feat='speed_50th', 
                          save_dir=Path(SAVE_DIR),
                          pvalue_threshold=0.05,
                          fdr_method='fdr_bh')
    
    # extract t-test pvals
    pvals = ttest_results[[c for c in ttest_results.columns if 'pval' in c]]
    pvals.columns = [c.replace('pvals_','') for c in pvals.columns]

    # save plot data to file
    plot_df = metadata[['mutation','date_yyyymmdd']
                       ].join(features[['speed_50th']]
                              ).sort_values(by='mutation', ascending=True)
    save_path = Path(SAVE_DIR) / "boxplot_data.csv"
    plot_df.to_csv(save_path, header=True, index=False)

    # boxplots
    colour_dict1 = dict(zip(bwfep, sns.color_palette(palette='tab10',
                                                      n_colors=2)))
    colour_dict2 = dict(zip([s for s in strain_list if s not in bwfep], 
                            sns.color_palette(palette=cc.glasbey, 
                                              n_colors=len(strain_list)-2)))
    colour_dict = {**colour_dict1, **colour_dict2}
        
    plt.close('all')
    sns.set_style('ticks')
    fig = plt.figure(figsize=[21,10])
    ax = fig.add_subplot(1,1,1)
    sns.boxplot(x='mutation',
                y='speed_50th',
                data=plot_df, 
                order=strain_list,
                hue='mutation',
                hue_order=strain_list,
                dodge=False,
                palette=colour_dict,
                showfliers=False, 
                showmeans=False,
                meanprops={"marker":"x", 
                           "markersize":5,
                           "markeredgecolor":"k"},
                flierprops={"marker":"x", 
                            "markersize":15, 
                            "markeredgecolor":"r"})
    dates = sorted(plot_df['date_yyyymmdd'].unique())
    date_lut = dict(zip(dates, sns.color_palette(palette="Greys", 
                                                 n_colors=len(dates))))
    for date in dates:
        date_df = plot_df[plot_df['date_yyyymmdd']==date]
        sns.stripplot(x='mutation',
                      y='speed_50th',
                      data=date_df,
                      order=strain_list,
                      hue='mutation',
                      hue_order=strain_list,
                      dodge=False,
                      palette=[date_lut[date]] * plot_df['mutation'].nunique(),
                      color='dimgray',
                      marker=".",
                      edgecolor='k',
                      linewidth=0.3,
                      s=12,
                      label=str(int(date))) #facecolors="none"
    
    ax.axes.get_xaxis().get_label().set_visible(False) # remove y axis label
    ax.axes.set_xticklabels([l.get_text() for l in ax.axes.get_xticklabels()], 
                            fontsize=20, rotation=90)
    ax.tick_params(axis='x', which='major', pad=15)
    ax.axes.set_ylabel('Speed (µm s$^{-1}$)', fontsize=25, labelpad=25)                             
    plt.yticks(fontsize=20)
            
    # scale y axis for annotations    
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled
    
    # add pvalues to plot
    for i, text in enumerate(ax.axes.get_xticklabels()):
        strain = text.get_text()
        if strain == 'wild_type':
            pass
        else:
            p = pvals.loc['speed_50th',strain]
            p_text = sig_asterix([p])[0]
            ax.text(i, 1.03, p_text, fontsize=20, ha='center', va='center', 
                    transform=trans)

    boxplot_path = Path(SAVE_DIR) / 'boxplot_fepD_ent_men_mutants_speed_50th_bluelight.svg'
    plt.savefig(boxplot_path, bbox_inches='tight', transparent=True)

    return

#%% Main

if __name__ == "__main__":
    main()

