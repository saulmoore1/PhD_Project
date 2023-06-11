#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper Figure 3a - Combined plots of all mutants in fepD E. coli background created based on results of
proteomics analysis

@author: sm5911
@date: 06/02/2023

"""

#%% Imports

import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import transforms

from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from visualisation.plotting_helper import sig_asterix
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes

#%% Globals

PROJECT_DIR_LIST = ['/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Proteomics_Mutants',
                    '/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_FepD_Oxidative_Stress_Mutants',
                    '/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_FepD_Mutants',
                    '/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_FepD_Mutants_2']

SAVE_DIR = '/Users/sm5911/Documents/Keio_FepD_Mutants_Combined'
#SAVE_DIR = "/Users/sm5911/OneDrive - Imperial College London/Publications/Keio_Paper/Figures/Fig3"

N_WELLS = 6
NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
FEATURE = 'speed_50th'
P_VALUE_THRESHOLD = 0.05
FDR_METHOD = 'fdr_bh'
DPI = 900
FPS = 25
BLUELIGHT_TIMEPOINTS_SECONDS = [(60, 70),(160, 170),(260, 270)]

OMIT_STRAINS_LIST = ['deferoxamine']

#%% Functions

def stats(metadata,
          features,
          group_by='treatment',
          control='BW',
          feat='speed_50th',
          pvalue_threshold=0.05,
          fdr_method='fdr_bh'):
        
    assert all(metadata.index == features.index)
    features = features[[feat]]

    # print mean sample size
    sample_size = metadata.groupby(group_by).count()
    print("Mean sample size of %s/window: %d" % (group_by, 
                                                 int(sample_size[sample_size.columns[-1]].mean())))

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
    
    metadata_combined_path = Path(SAVE_DIR) / 'metadata.csv'
    features_combined_path = Path(SAVE_DIR) / 'features.csv'
    
    # compile combined metadata & feature summaries, clean data and save
    if not (metadata_combined_path.exists() or features_combined_path.exists()):

        metadata_list = []
        features_list = []
        for project_dir in tqdm(PROJECT_DIR_LIST):
            
            aux_dir = Path(project_dir) / 'AuxiliaryFiles'
            res_dir = Path(project_dir) / 'Results'

            _metadata, _metadata_path = compile_metadata(aux_dir, 
                                                         n_wells=N_WELLS, 
                                                         add_well_annotations=False,
                                                         from_source_plate=True)
            
            _features, _metadata = process_feature_summaries(_metadata_path, 
                                                             results_dir=res_dir, 
                                                             compile_day_summaries=True, 
                                                             imaging_dates=None, 
                                                             align_bluelight=False, 
                                                             window_summaries=True,
                                                             n_wells=N_WELLS)
            
            if 'food_type' in _metadata.columns:
                _metadata = _metadata.rename({'food_type':'bacteria_strain'}, axis=1)
                
            metadata_list.append(_metadata)
            features_list.append(_features)
            
        # subset for window (290:300 seconds)
        meta_list = []
        feat_list = []
        for i, (meta, feat) in enumerate(zip(metadata_list, features_list)):
            if meta['window'].nunique() == 6:
                meta = meta[meta['window']==5] # corresponds to 20-30 seconds after final BL pulse
                feat = feat.reindex(meta.index)
            elif meta['window'].nunique() == 1: # if summaries were only generated for final BL window
                assert meta['window'].unique()[0] == 0
            meta_list.append(meta)
            feat_list.append(feat)
            
        metadata = pd.concat(meta_list, axis=0).reset_index(drop=True)
        features = pd.concat(feat_list, axis=0).reset_index(drop=True)
        
        # clean combined results - remove bad wells + features with too many NaNs/zero std + impute NaNs
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
        metadata.to_csv(metadata_combined_path, index=False)
        features.to_csv(features_combined_path, index=False)          
        
    else:
        # load metadata and features summaries
        metadata = pd.read_csv(metadata_combined_path, header=0, index_col=None, dtype={'comments':str})
        features = pd.read_csv(features_combined_path, header=0, index_col=None)

    # subset metadata results for bluelight videos only 
    bluelight_videos = [i for i in metadata['imgstore_name'] if 'bluelight' in i]
    metadata = metadata[metadata['imgstore_name'].isin(bluelight_videos)]
    
    # omit strains
    metadata = metadata[~metadata['drug_type'].isin(OMIT_STRAINS_LIST)]
    
    metadata['treatment'] = metadata[['bacteria_strain','drug_type']
                                     ].astype(str).agg('-'.join, axis=1)
    metadata['treatment'] = [i.replace('OE_','OE ').replace('_','; ').replace('-nan','') for 
                             i in metadata['treatment']]
    
    cs = ['BW', 'fepD']
    ts = [i for i in sorted(metadata['treatment'].unique()) if i not in cs]
    fs = [i for i in ts if i.startswith('fepD')]
    bs = [i for i in ts if not i.startswith('fepD')]
    strain_list_fepD = cs + fs
    strain_list_BW = cs + bs
    
    # reindex features after subsetting metadata
    features = features.reindex(metadata.index)
 
    # BW deletion/OE mutants
    meta_bs = metadata[metadata['treatment'].isin(strain_list_BW)]
    feat_bs = features.reindex(meta_bs.index)
    
    # boxplot
    plot_df = meta_bs.join(feat_bs[[FEATURE]])
    colour_dict = dict(zip(strain_list_BW, sns.color_palette(palette='tab10',
                                                             n_colors=len(strain_list_BW))))

    for control in tqdm(cs):
        plt.close('all')
        sns.set_style('ticks')
        fig = plt.figure(figsize=[18,20])
        ax = fig.add_subplot(1,1,1)
        sns.boxplot(x='speed_50th',
                    y='treatment',
                    data=plot_df, 
                    order=strain_list_BW,
                    palette=colour_dict,
                    showfliers=False, 
                    showmeans=False,
                    meanprops={"marker":"x", 
                               "markersize":5,
                               "markeredgecolor":"k"},
                    flierprops={"marker":"x", 
                                "markersize":15, 
                                "markeredgecolor":"r"})
        dates = metadata['date_yyyymmdd'].unique()
        date_cols = dict(zip(dates, sns.color_palette(palette='Set1', n_colors=len(dates))))
        for date in dates:
            date_df = plot_df[plot_df['date_yyyymmdd'] == date]
            sns.stripplot(x='speed_50th',
                          y='treatment',
                          data=date_df,
                          s=10,
                          order=strain_list_BW,
                          hue=None,
                          palette=None,
                          color=date_cols[date],
                          marker=".",
                          edgecolor='k',
                          linewidth=0.3,
                          ax=ax)
        
        ax.axes.get_yaxis().get_label().set_visible(False) # remove y axis label
        ax.axes.set_yticklabels([l.get_text() for l in ax.axes.get_yticklabels()], fontsize=18)
        ax.tick_params(axis='y', which='major', pad=15)
        ax.axes.set_xlabel('Speed (µm s$^{-1}$)', fontsize=25, labelpad=20)                             
        plt.xticks(fontsize=18)
        plt.xlim(-20, 300)
            
        # do stats
        anova_results, ttest_results = stats(metadata,
                                             features,
                                             group_by='treatment',
                                             control=control,
                                             feat='speed_50th',
                                             pvalue_threshold=P_VALUE_THRESHOLD,
                                             fdr_method=FDR_METHOD)
        
        # t-test pvals
        pvals = ttest_results[[c for c in ttest_results.columns if 'pval' in c]]
        pvals.columns = [c.replace('pvals_','') for c in pvals.columns]
        
        # scale x axis for annotations    
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData) #x=scaled
        
        # add pvalues to plot
        for i, treatment in enumerate(strain_list_BW, start=0):
            if treatment == control:
                continue
            else:
                p = pvals.loc['speed_50th', treatment]
                text = ax.get_yticklabels()[i]
                assert text.get_text() == treatment
                p_text = sig_asterix([p])[0]
                ax.text(1.03, i, p_text, fontsize=20, ha='left', va='center', transform=trans)
                
        plt.subplots_adjust(left=0.4, right=0.9, top=0.95, bottom=0.1)
        save_path = Path(SAVE_DIR) / 'BW' / 'BW_mutants_combined_boxplot_vs_{}.png'.format(control)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=DPI)


    # fepD deletion/OE mutants
    meta_fs = metadata[metadata['treatment'].isin(strain_list_fepD)]
    feat_fs = features.reindex(meta_fs.index)
    
    # boxplot
    plot_df = meta_fs.join(feat_fs[[FEATURE]])
    colour_dict = dict(zip(strain_list_fepD, sns.color_palette(palette='tab10',
                                                               n_colors=len(strain_list_fepD))))

    for control in tqdm(cs):
        plt.close('all')
        sns.set_style('ticks')
        fig = plt.figure(figsize=[18,20])
        ax = fig.add_subplot(1,1,1)
        sns.boxplot(x='speed_50th',
                    y='treatment',
                    data=plot_df, 
                    order=strain_list_fepD,
                    palette=colour_dict,
                    showfliers=False, 
                    showmeans=False,
                    meanprops={"marker":"x", 
                               "markersize":5,
                               "markeredgecolor":"k"},
                    flierprops={"marker":"x", 
                                "markersize":15, 
                                "markeredgecolor":"r"})
        dates = metadata['date_yyyymmdd'].unique()
        date_cols = dict(zip(dates, sns.color_palette(palette='Set1', n_colors=len(dates))))
        for date in dates:
            date_df = plot_df[plot_df['date_yyyymmdd'] == date]
            sns.stripplot(x='speed_50th',
                          y='treatment',
                          data=date_df,
                          s=10,
                          order=strain_list_fepD,
                          hue=None,
                          palette=None,
                          color=date_cols[date],
                          marker=".",
                          edgecolor='k',
                          linewidth=0.3,
                          ax=ax)
        
        ax.axes.get_yaxis().get_label().set_visible(False) # remove y axis label
        ax.axes.set_yticklabels([l.get_text() for l in ax.axes.get_yticklabels()], fontsize=18)
        ax.tick_params(axis='y', which='major', pad=15)
        ax.axes.set_xlabel('Speed (µm s$^{-1}$)', fontsize=25, labelpad=20)                             
        plt.xticks(fontsize=18)
        plt.xlim(-20, 300)
            
        # do stats
        anova_results, ttest_results = stats(metadata,
                                             features,
                                             group_by='treatment',
                                             control=control,
                                             feat='speed_50th',
                                             pvalue_threshold=P_VALUE_THRESHOLD,
                                             fdr_method=FDR_METHOD)
        
        # t-test pvals
        pvals = ttest_results[[c for c in ttest_results.columns if 'pval' in c]]
        pvals.columns = [c.replace('pvals_','') for c in pvals.columns]
        
        # scale x axis for annotations    
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData) #x=scaled
        
        # add pvalues to plot
        for i, treatment in enumerate(strain_list_fepD, start=0):
            if treatment == control:
                continue
            else:
                p = pvals.loc['speed_50th', treatment]
                text = ax.get_yticklabels()[i]
                assert text.get_text() == treatment
                p_text = sig_asterix([p])[0]
                ax.text(1.03, i, p_text, fontsize=20, ha='left', va='center', transform=trans)
                
        plt.subplots_adjust(left=0.4, right=0.9, top=0.95, bottom=0.1)
        save_path = Path(SAVE_DIR) / 'fepD' / 'fepD_mutants_combined_boxplot_vs_{}.png'.format(control)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=DPI)  
        
    return

#%% Main

if __name__ == '__main__':
    main()
    
    
    