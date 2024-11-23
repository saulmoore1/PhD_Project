#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of N2 worms on fepD supplemented with citrate, uric acid or NaOH

@author: sm5911
@date: 13/12/2022 (updated: 19/11/2024)

"""

#%% Imports 

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from matplotlib import transforms
from matplotlib import pyplot as plt

from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from visualisation.plotting_helper import sig_asterix#, boxplots_sigfeats, all_in_one_boxplots
#from write_data.write import write_list_to_file
from time_series.plot_timeseries import plot_timeseries, get_strain_timeseries

from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from tierpsytools.preprocessing.filter_data import select_feat_set

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_FepD_Citrate"
SAVE_DIR = "/Users/sm5911/Documents/PhD_DLBG/24_Keio_FepD_Citrate"

N_WELLS = 6
FPS = 25

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05

P_VALUE_THRESHOLD = 0.05
FDR_METHOD = 'fdr_bh'

FEATURE_SET = ['speed_50th']

BLUELIGHT_TIMEPOINTS_SECONDS = [(60,70),(160,170),(260,270)]

WINDOW_DICT = {0:(290,300)}

WINDOW_NAME_DICT = {0:"20-30 seconds after blue light 3"}

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

# =============================================================================
# def boxplots(metadata,
#              features,
#              group_by='treatment',
#              control='BW',
#              save_dir=None,
#              stats_dir=None,
#              feature_set=None,
#              pvalue_threshold=0.05):
#         
#     feature_set = features.columns.tolist() if feature_set is None else feature_set
#     assert isinstance(feature_set, list) and all(f in features.columns for f in feature_set)
#                     
#     # load t-test results for window
#     if stats_dir is not None:
#         ttest_path = Path(stats_dir) / 't-test' / 't-test_results.csv'
#         ttest_df = pd.read_csv(ttest_path, header=0, index_col=0)
#         pvals = ttest_df[[c for c in ttest_df.columns if 'pval' in c]]
#         pvals.columns = [c.replace('pvals_','') for c in pvals.columns]
#     
#     boxplots_sigfeats(features,
#                       y_class=metadata[group_by],
#                       control=control,
#                       pvals=pvals if stats_dir is not None else None,
#                       z_class=None,
#                       feature_set=feature_set,
#                       saveDir=Path(save_dir),
#                       drop_insignificant=True if feature_set is None else False,
#                       p_value_threshold=pvalue_threshold,
#                       scale_outliers=True,
#                       append_ranking_fname=False)
# 
#     return
# =============================================================================

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

        # Clean results - Remove bad well data + features with too many NaNs/zero std + impute NaNs
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
    assert metadata['window'].nunique() == 1 and 0 in metadata['window'].unique()
    
    features = features.reindex(metadata.index)
    
    treatment_cols = ['bacteria_strain','drug_type','drug_imaging_plate_conc']
    metadata['treatment'] = metadata[treatment_cols].astype(str).agg('-'.join, axis=1)
    metadata['treatment'] = [i.replace('-NaOH-1.0','-NaOH') for i in metadata['treatment']]
    metadata['treatment'] = [i.replace('-nan','') for i in metadata['treatment']]
    metadata['treatment'] = [i.replace('.0','') if i.endswith('.0') else i for i in metadata['treatment']]

    # strain list
    treatment_list = list(metadata['treatment'].unique())
    citrate_strains = ['BW','BW-citrate-1','BW-citrate-5','BW-citrate-10','fepD',
                       'fepD-citrate-1','fepD-citrate-5','fepD-citrate-10']
    uric_acid_strains = ['BW','BW-NaOH','BW-uric acid-0.01','BW-uric acid-0.005','fepD','fepD-NaOH',
                         'fepD-uric acid-0.01','fepD-uric acid-0.005']
    assert all(c in treatment_list for c in citrate_strains)
    assert all(u in treatment_list for u in uric_acid_strains)
    
    stats_dir = Path(SAVE_DIR) / 'Stats'
    plots_dir = Path(SAVE_DIR) / 'Plots'
    
    # citrate
    meta_citrate = metadata[metadata['treatment'].isin(citrate_strains)]
    feat_citrate = features.reindex(meta_citrate.index)
    
    # stats - compare citrate treatments to BW
    _, ttest_results = stats(meta_citrate,
                             feat_citrate,
                             group_by='treatment',
                             control='BW',
                             save_dir=stats_dir / 'citrate_vs_BW',
                             feature_list=FEATURE_SET,
                             p_value_threshold=P_VALUE_THRESHOLD,
                             fdr_method=FDR_METHOD)

    # extract t-test pvals
    pvals = ttest_results[[c for c in ttest_results.columns if 'pval' in c]]
    pvals.columns = [c.replace('pvals_','') for c in pvals.columns]
    
    # boxplot
    plot_df = meta_citrate.join(feat_citrate)        
    lut = dict(zip(['BW','fepD'], sns.color_palette(palette='tab10', n_colors=2)))
    colour_dict = {i:(lut['fepD'] if 'fepD' in i else lut['BW']) for i in citrate_strains}        

    plt.close('all')
    sns.set_style('ticks')
    fig = plt.figure(figsize=[12,8])
    ax = fig.add_subplot(1,1,1)
    sns.boxplot(x='treatment',
                y='speed_50th',
                data=plot_df, 
                order=citrate_strains,
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
    date_lut = dict(zip(dates, sns.color_palette(palette="Greys", n_colors=len(dates))))
    for date in dates:
        date_df = plot_df[plot_df['date_yyyymmdd']==date]
        sns.stripplot(x='treatment',
                      y='speed_50th',
                      data=date_df,
                      order=citrate_strains,
                      palette=[date_lut[date]] * len(citrate_strains),
                      color='dimgray',
                      marker=".",
                      edgecolor='k',
                      linewidth=0.3,
                      s=12) #facecolors="none"
                    
    # scale y axis for annotations    
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled
    
    # add pvalues to plot
    for i, text in enumerate(ax.axes.get_xticklabels()):
        treatment = text.get_text()
        if treatment == 'BW':
            continue
        else:
            p = pvals.loc['speed_50th', treatment]
            p_text = sig_asterix([p])[0]
            ax.text(i, 1.03, p_text, fontsize=20, ha='center', va='center', transform=trans)

    ax.axes.get_xaxis().get_label().set_visible(False) # remove y axis label
    ax.axes.set_xticklabels([l.get_text().replace('-','\n') + 'mM' if l.get_text() not in 
                             ['BW','fepD'] else l for l in ax.axes.get_xticklabels()], fontsize=20)
    ax.tick_params(axis='x', which='major', pad=15)
    ax.axes.set_ylabel('Speed (µm s$^{-1}$)', fontsize=25, labelpad=25)                             
    plt.yticks(fontsize=20)
    plt.ylim(-20, 270)

    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[:2], labels[:2], loc='best', frameon=False, fontsize=15)
    boxplot_path = plots_dir / 'Citrate_treatment_vs_BW.svg'
    boxplot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(boxplot_path, bbox_inches='tight', transparent=True)
    
    
    # stats - compare citrate treatments to fepD
    _, ttest_results = stats(meta_citrate,
                             feat_citrate,
                             group_by='treatment',
                             control='fepD',
                             save_dir=stats_dir / 'citrate_vs_fepD',
                             feature_list=FEATURE_SET,
                             p_value_threshold=P_VALUE_THRESHOLD,
                             fdr_method=FDR_METHOD)

    # extract t-test pvals
    pvals = ttest_results[[c for c in ttest_results.columns if 'pval' in c]]
    pvals.columns = [c.replace('pvals_','') for c in pvals.columns]
    
    # boxplot
    plot_df = meta_citrate.join(feat_citrate)        
    lut = dict(zip(['BW','fepD'], sns.color_palette(palette='tab10', n_colors=2)))
    colour_dict = {i:(lut['fepD'] if 'fepD' in i else lut['BW']) for i in citrate_strains}        

    plt.close('all')
    sns.set_style('ticks')
    fig = plt.figure(figsize=[12,8])
    ax = fig.add_subplot(1,1,1)
    sns.boxplot(x='treatment',
                y='speed_50th',
                data=plot_df, 
                order=citrate_strains,
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
    date_lut = dict(zip(dates, sns.color_palette(palette="Greys", n_colors=len(dates))))
    for date in dates:
        date_df = plot_df[plot_df['date_yyyymmdd']==date]
        sns.stripplot(x='treatment',
                      y='speed_50th',
                      data=date_df,
                      order=citrate_strains,
                      palette=[date_lut[date]] * len(citrate_strains),
                      color='dimgray',
                      marker=".",
                      edgecolor='k',
                      linewidth=0.3,
                      s=12) #facecolors="none"
                    
    # scale y axis for annotations    
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled
    
    # add pvalues to plot
    for i, text in enumerate(ax.axes.get_xticklabels()):
        treatment = text.get_text()
        if treatment == 'fepD':
            continue
        else:
            p = pvals.loc['speed_50th', treatment]
            p_text = sig_asterix([p])[0]
            ax.text(i, 1.03, p_text, fontsize=20, ha='center', va='center', transform=trans)

    ax.axes.get_xaxis().get_label().set_visible(False) # remove y axis label
    ax.axes.set_xticklabels([l.get_text().replace('-','\n') + 'mM' if l.get_text() not in 
                             ['BW','fepD'] else l for l in ax.axes.get_xticklabels()], fontsize=20)
    ax.tick_params(axis='x', which='major', pad=15)
    ax.axes.set_ylabel('Speed (µm s$^{-1}$)', fontsize=25, labelpad=25)                             
    plt.yticks(fontsize=20)
    plt.ylim(-20, 270)

    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[:2], labels[:2], loc='best', frameon=False, fontsize=15)
    boxplot_path = plots_dir / 'Citrate_treatment_vs_fepD.svg'
    boxplot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(boxplot_path, bbox_inches='tight', transparent=True)

    
    # uric acid
    meta_UA = metadata[metadata['treatment'].isin(uric_acid_strains)]
    feat_UA = features.reindex(meta_UA.index)
    
    # stats - compare uric acid treatments to BW
    _, ttest_results = stats(meta_UA,
                             feat_UA,
                             group_by='treatment',
                             control='BW',
                             save_dir=stats_dir / 'uric_acid_vs_BW',
                             feature_list=FEATURE_SET,
                             p_value_threshold=P_VALUE_THRESHOLD,
                             fdr_method=FDR_METHOD)

    # extract t-test pvals
    pvals = ttest_results[[c for c in ttest_results.columns if 'pval' in c]]
    pvals.columns = [c.replace('pvals_','') for c in pvals.columns]
    
    # boxplot
    plot_df = meta_UA.join(feat_UA)        
    lut = dict(zip(['BW','fepD'], sns.color_palette(palette='tab10', n_colors=2)))
    colour_dict = {i:(lut['fepD'] if 'fepD' in i else lut['BW']) for i in uric_acid_strains}        

    plt.close('all')
    sns.set_style('ticks')
    fig = plt.figure(figsize=[12,8])
    ax = fig.add_subplot(1,1,1)
    sns.boxplot(x='treatment',
                y='speed_50th',
                data=plot_df, 
                order=uric_acid_strains,
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
    date_lut = dict(zip(dates, sns.color_palette(palette="Greys", n_colors=len(dates))))
    for date in dates:
        date_df = plot_df[plot_df['date_yyyymmdd']==date]
        sns.stripplot(x='treatment',
                      y='speed_50th',
                      data=date_df,
                      order=uric_acid_strains,
                      palette=[date_lut[date]] * len(uric_acid_strains),
                      color='dimgray',
                      marker=".",
                      edgecolor='k',
                      linewidth=0.3,
                      s=12) #facecolors="none"
                    
    # scale y axis for annotations    
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled
    
    # add pvalues to plot
    for i, text in enumerate(ax.axes.get_xticklabels()):
        treatment = text.get_text()
        if treatment == 'BW':
            continue
        else:
            p = pvals.loc['speed_50th', treatment]
            p_text = sig_asterix([p])[0]
            ax.text(i, 1.03, p_text, fontsize=20, ha='center', va='center', transform=trans)

    ax.axes.get_xaxis().get_label().set_visible(False) # remove y axis label
    ax.axes.set_xticklabels([l.get_text().replace('-','\n') + ' mg/mL' if l.get_text() not in 
                             ['BW','fepD','BW-NaOH','fepD-NaOH'] else l.get_text().replace('-','\n')
                             for l in ax.axes.get_xticklabels()], fontsize=15)
    ax.tick_params(axis='x', which='major', pad=15)
    ax.axes.set_ylabel('Speed (µm s$^{-1}$)', fontsize=25, labelpad=25)                             
    plt.yticks(fontsize=20)
    plt.ylim(-20, 270)

    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[:2], labels[:2], loc='best', frameon=False, fontsize=15)
    boxplot_path = plots_dir / 'Uric_acid_treatment_vs_BW.svg'
    boxplot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(boxplot_path, bbox_inches='tight', transparent=True)

    
    # stats - compare uric acid treatments to BW-NaOH
    _, ttest_results = stats(meta_UA,
                             feat_UA,
                             group_by='treatment',
                             control='BW-NaOH',
                             save_dir=stats_dir / 'uric_acid_vs_BW-NaOH',
                             feature_list=FEATURE_SET,
                             p_value_threshold=P_VALUE_THRESHOLD,
                             fdr_method=FDR_METHOD)

    # extract t-test pvals
    pvals = ttest_results[[c for c in ttest_results.columns if 'pval' in c]]
    pvals.columns = [c.replace('pvals_','') for c in pvals.columns]
    
    # boxplot
    plot_df = meta_UA.join(feat_UA)        
    lut = dict(zip(['BW','fepD'], sns.color_palette(palette='tab10', n_colors=2)))
    colour_dict = {i:(lut['fepD'] if 'fepD' in i else lut['BW']) for i in uric_acid_strains}        

    plt.close('all')
    sns.set_style('ticks')
    fig = plt.figure(figsize=[12,8])
    ax = fig.add_subplot(1,1,1)
    sns.boxplot(x='treatment',
                y='speed_50th',
                data=plot_df, 
                order=uric_acid_strains,
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
    date_lut = dict(zip(dates, sns.color_palette(palette="Greys", n_colors=len(dates))))
    for date in dates:
        date_df = plot_df[plot_df['date_yyyymmdd']==date]
        sns.stripplot(x='treatment',
                      y='speed_50th',
                      data=date_df,
                      order=uric_acid_strains,
                      palette=[date_lut[date]] * len(uric_acid_strains),
                      color='dimgray',
                      marker=".",
                      edgecolor='k',
                      linewidth=0.3,
                      s=12) #facecolors="none"
                    
    # scale y axis for annotations    
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled
    
    # add pvalues to plot
    for i, text in enumerate(ax.axes.get_xticklabels()):
        treatment = text.get_text()
        if treatment == 'BW-NaOH':
            continue
        else:
            p = pvals.loc['speed_50th', treatment]
            p_text = sig_asterix([p])[0]
            ax.text(i, 1.03, p_text, fontsize=20, ha='center', va='center', transform=trans)

    ax.axes.get_xaxis().get_label().set_visible(False) # remove y axis label
    ax.axes.set_xticklabels([l.get_text().replace('-','\n') + ' mg/mL' if l.get_text() not in 
                             ['BW','fepD','BW-NaOH','fepD-NaOH'] else l.get_text().replace('-','\n')
                             for l in ax.axes.get_xticklabels()], fontsize=15)
    ax.tick_params(axis='x', which='major', pad=15)
    ax.axes.set_ylabel('Speed (µm s$^{-1}$)', fontsize=25, labelpad=25)                             
    plt.yticks(fontsize=20)
    plt.ylim(-20, 270)

    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[:2], labels[:2], loc='best', frameon=False, fontsize=15)
    boxplot_path = plots_dir / 'Uric_acid_treatment_vs_BW-NaOH.svg'
    boxplot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(boxplot_path, bbox_inches='tight', transparent=True)
    
    
# =============================================================================
#     # compare uric acid treatments vs BW-NaOH
#     meta_UA = metadata[metadata['treatment'].isin(uric_acid_strains)]
#     feat_UA = features.reindex(meta_UA.index)
# 
#     # perform anova and t-tests comparing each treatment to BW control    
#     stats(meta_UA,
#           feat_UA,
#           group_by='treatment',
#           control='BW-NaOH',
#           save_dir=Path(SAVE_DIR) / 'Stats_Uric_Acid',
#           feature_set=feature_list,
#           pvalue_threshold=0.05,
#           fdr_method='fdr_bh')
# 
#     #### boxplots comparing uric acid treatments vs BW-NaOH
#     colour_dict = dict(zip(uric_acid_strains, sns.color_palette('tab10', len(uric_acid_strains))))
#     all_in_one_boxplots(meta_UA,
#                         feat_UA,
#                         group_by='treatment',
#                         control='BW-NaOH',
#                         sigasterix=True,
#                         fontsize=15,
#                         order=uric_acid_strains,
#                         colour_dict=colour_dict,
#                         feature_set=feature_list,
#                         save_dir=Path(SAVE_DIR) / 'Plots_Uric_Acid' / 'all-in-one',
#                         ttest_path=Path(SAVE_DIR) / 'Stats_Uric_Acid' / 't-test' /\
#                             't-test_results.csv',
#                         pvalue_threshold=0.05,
#                         figsize=(15,8),
#                         ylim_minmax=(0,300),
#                         subplots_adjust={'bottom':0.35,'top':0.9,'left':0.1,'right':0.95})
# 
#     # compare uric acid treatments vs fepD-NaOH
#     meta_UA = metadata[metadata['treatment'].isin(uric_acid_strains)]
#     feat_UA = features.reindex(meta_UA.index)
#     
#     stats(meta_UA,
#           feat_UA,
#           group_by='treatment',
#           control='fepD-NaOH',
#           save_dir=Path(SAVE_DIR) / 'Stats_Uric_Acid_vs_fepD',
#           feature_set=feature_list,
#           pvalue_threshold=0.05,
#           fdr_method='fdr_bh')
# 
#     colour_dict = dict(zip(uric_acid_strains, sns.color_palette('tab10', len(uric_acid_strains))))
#     all_in_one_boxplots(meta_UA,
#                         feat_UA,
#                         group_by='treatment',
#                         control='fepD-NaOH',
#                         sigasterix=True,
#                         fontsize=15,
#                         order=uric_acid_strains,
#                         colour_dict=colour_dict,
#                         feature_set=feature_list,
#                         save_dir=Path(SAVE_DIR) / 'Plots_Uric_Acid_vs_fepD' / 'all-in-one',
#                         ttest_path=Path(SAVE_DIR) / 'Stats_Uric_Acid_vs_fepD' / 't-test' /\
#                             't-test_results.csv',
#                         pvalue_threshold=0.05,
#                         figsize=(15,8),
#                         ylim_minmax=(0,300),
#                         subplots_adjust={'bottom':0.35,'top':0.9,'left':0.1,'right':0.95})
# 
# 
#     # compare citrate treatments vs BW
#     meta_citrate = metadata[metadata['treatment'].isin(citrate_strains)]
#     feat_citrate = features.reindex(meta_citrate.index)
#     
#     stats(meta_citrate,
#           feat_citrate,
#           group_by='treatment',
#           control='BW',
#           save_dir=Path(SAVE_DIR) / 'Stats_Citrate_vs_BW',
#           feature_set=feature_list,
#           pvalue_threshold=0.05,
#           fdr_method='fdr_bh')
#     
#     # boxplot - treatments vs BW
#     
#     
# 
#     colour_dict = dict(zip(citrate_strains, sns.color_palette('tab10', len(citrate_strains))))
#     all_in_one_boxplots(meta_citrate,
#                         feat_citrate,
#                         group_by='treatment',
#                         control='BW',
#                         sigasterix=True,
#                         fontsize=15,
#                         order=citrate_strains,
#                         colour_dict=colour_dict,
#                         feature_set=feature_list,
#                         save_dir=Path(SAVE_DIR) / 'Plots_Citrate_vs_BW' / 'all-in-one',
#                         ttest_path=Path(SAVE_DIR) / 'Stats_Citrate_vs_BW' / 't-test' /\
#                             't-test_results.csv',
#                         pvalue_threshold=0.05,
#                         figsize=(15,8),
#                         ylim_minmax=(0,300),
#                         subplots_adjust={'bottom':0.35,'top':0.9,'left':0.1,'right':0.95})
# 
#     # compare citrate treatments vs fepD
#     stats(meta_citrate,
#           feat_citrate,
#           group_by='treatment',
#           control='fepD',
#           save_dir=Path(SAVE_DIR) / 'Stats_Citrate_vs_fepD',
#           feature_set=feature_list,
#           pvalue_threshold=0.05,
#           fdr_method='fdr_bh')
#     
#     colour_dict = dict(zip(citrate_strains, sns.color_palette('tab10', len(citrate_strains))))
#     all_in_one_boxplots(meta_citrate,
#                         feat_citrate,
#                         group_by='treatment',
#                         control='fepD',
#                         sigasterix=True,
#                         fontsize=15,
#                         order=citrate_strains,
#                         colour_dict=colour_dict,
#                         feature_set=feature_list,
#                         save_dir=Path(SAVE_DIR) / 'Plots_Citrate_vs_fepD' / 'all-in-one',
#                         ttest_path=Path(SAVE_DIR) / 'Stats_Citrate_vs_fepD' / 't-test' /\
#                             't-test_results.csv',
#                         pvalue_threshold=0.05,
#                         figsize=(15,8),
#                         ylim_minmax=(0,300),
#                         subplots_adjust={'bottom':0.35,'top':0.9,'left':0.1,'right':0.95})
# =============================================================================
    
    ### uric acid timeseries ###
    
    for conc in ['0.01','0.005']:
      
        treatment_list = ['BW-NaOH','fepD-NaOH',
                          'BW-uric acid-{0}'.format(conc),'fepD-uric acid-{0}'.format(conc)]
        bluelight_frames = [(i*FPS, j*FPS) for (i, j) in BLUELIGHT_TIMEPOINTS_SECONDS]
        feature = 'speed'
    
        save_dir = Path(SAVE_DIR) / 'timeseries-speed' / 'Uric_Acid'
        save_dir.mkdir(exist_ok=True, parents=True)
        save_path = save_dir / 'speed_bluelight_uric_acid-{0}.pdf'.format(conc)
        
        plt.close('all')
        fig, ax = plt.subplots(figsize=(15,6), dpi=300)
        col_dict = {'BW-NaOH': sns.color_palette('tab10',2)[0],
                    'fepD-NaOH':sns.color_palette('tab10',2)[1],
                    'BW-uric acid-{0}'.format(conc):'lightskyblue',
                    'fepD-uric acid-{0}'.format(conc):'sandybrown'}
    
        for group in tqdm(treatment_list):
            
            # get control timeseries
            group_ts = get_strain_timeseries(metadata,
                                             project_dir=Path(PROJECT_DIR),
                                             strain=group,
                                             group_by='treatment',
                                             feature_list=[feature],
                                             save_dir=save_dir,
                                             n_wells=N_WELLS,
                                             verbose=True)
            
            ax = plot_timeseries(df=group_ts,
                                 feature=feature,
                                 error=True,
                                 max_n_frames=360*FPS, 
                                 smoothing=10*FPS, 
                                 ax=ax,
                                 bluelight_frames=bluelight_frames,
                                 colour=col_dict[group])
    
        plt.ylim(-20, 300)
        xticks = np.linspace(0, 360*FPS, int(360/60)+1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
        ax.set_xlabel('Time (minutes)', fontsize=20, labelpad=10)
        ylab = feature.replace('_50th'," (µm s$^{-1}$)")
        ax.set_ylabel(ylab, fontsize=20, labelpad=10)
        ax.legend(treatment_list, fontsize=12, frameon=False, loc='best', handletextpad=1)
        plt.subplots_adjust(left=0.1, top=0.98, bottom=0.15, right=0.98)
    
        # save plot
        print("Saving to: %s" % save_path)
        plt.savefig(save_path)


    ### citrate timeseries ###
     
    for conc in ['1.0','5.0','10.0']:
    
        treatment_list = ['BW','fepD','fepD-citrate-{0}'.format(conc),'BW-citrate-{0}'.format(conc)]
        bluelight_frames = [(i*FPS, j*FPS) for (i, j) in BLUELIGHT_TIMEPOINTS_SECONDS]
        feature = 'speed'
    
        save_dir = Path(SAVE_DIR) / 'timeseries-speed' / 'Citrate'
        save_dir.mkdir(exist_ok=True, parents=True)
        save_path = save_dir / 'speed_bluelight_citrate-{0}.pdf'.format(conc)
        
        plt.close('all')
        fig, ax = plt.subplots(figsize=(15,6), dpi=300)
        col_dict = {'BW': sns.color_palette('tab10',2)[0],
                    'fepD':sns.color_palette('tab10',2)[1],
                    'BW-citrate-{0}'.format(conc):'lightskyblue',
                    'fepD-citrate-{0}'.format(conc):'sandybrown'}
    
        for group in tqdm(treatment_list):
            
            # get control timeseries
            group_ts = get_strain_timeseries(metadata,
                                             project_dir=Path(PROJECT_DIR),
                                             strain=group,
                                             group_by='treatment',
                                             feature_list=[feature],
                                             save_dir=save_dir,
                                             n_wells=N_WELLS,
                                             verbose=True)
            
            ax = plot_timeseries(df=group_ts,
                                 feature=feature,
                                 error=True,
                                 max_n_frames=360*FPS, 
                                 smoothing=10*FPS, 
                                 ax=ax,
                                 bluelight_frames=bluelight_frames,
                                 colour=col_dict[group])
    
        plt.ylim(-20, 300)
        xticks = np.linspace(0, 360*FPS, int(360/60)+1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
        ax.set_xlabel('Time (minutes)', fontsize=20, labelpad=10)
        ylab = feature.replace('_50th'," (µm s$^{-1}$)")
        ax.set_ylabel(ylab, fontsize=20, labelpad=10)
        ax.legend(treatment_list, fontsize=12, frameon=False, loc='best', handletextpad=1)
        plt.subplots_adjust(left=0.1, top=0.98, bottom=0.15, right=0.98)
    
        # save plot
        print("Saving to: %s" % save_path)
        plt.savefig(save_path)

    return

#%% Main

if __name__ == '__main__':
    main()
    
