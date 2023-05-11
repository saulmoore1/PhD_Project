#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary 3 - Effect of UV-killing E. coli mutants on C. elegans arousal behaviour 

@author: sm5911
@date: 24/04/2023

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
from tierpsytools.analysis.statistical_tests import _multitest_correct

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Keio_UV_Paraquat_Antioxidant_6WP"
SAVE_DIR = "/Users/sm5911/OneDrive - Imperial College London/Publications/Keio_Paper/Figures/Supp3"

FEATURE = 'speed_50th'

N_WELLS = 6
NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
OMIT_STRAINS_LIST = ['trpD']
P_VALUE_THRESHOLD = 0.05

VIDEO_LENGTH_SECONDS = 5*60*60
BLUELIGHT_TIMEPOINTS_MINUTES = [30,60,90,120,150,180,210,240]
FPS = 25

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

#%% Functions

def stats(metadata,
          features,
          group_by='treatment',
          control='BW',
          feat='speed_50th',
          p_value_threshold=0.05,
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
        
    nsig = sum(reject_t.sum(axis=1) > 0)
    print("%d significant features between any %s vs %s (t-test, P<%.2f, %s)" %\
          (nsig, group_by, control, p_value_threshold, fdr_method))

    if n > 2:
        return anova_results, ttest_results
    else:
        return ttest_results

#%% Main

if __name__ == '__main__':   

    aux_dir = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    res_dir = Path(PROJECT_DIR) / 'Results'
    
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

    # clean results to remove bad well data + features with too many NaNs/zero std + impute NaNs
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

    # subset for arousal windows 20-30 seconds after blue light
    arousal_windows = [1,3,5,7,9,11,13,15]
    metadata = metadata[metadata['window'].isin(arousal_windows)]
    
    # omit unwanted bacterial strains
    metadata = metadata[~metadata['gene_name'].isin(OMIT_STRAINS_LIST)]
    
    # reindex features for new metadata
    features = features.reindex(metadata.index)
    
    # treatment column - gene_name-window
    metadata['treatment'] = metadata[['gene_name','window']].astype(str).agg('-'.join, axis=1)
    
    # stats - t-tests comparing BW vs fepD at each window
    pvals_list = []
    for window in arousal_windows:
        meta_window = metadata[metadata['window']==window]
        feat_window = features.reindex(meta_window.index)
        
        anova_results, ttest_results = stats(meta_window,
                                             feat_window,
                                             group_by='gene_name',
                                             control='BW',
                                             feat=FEATURE,
                                             p_value_threshold=P_VALUE_THRESHOLD,
                                             fdr_method=None)
        
        pvals = ttest_results[[c for c in ttest_results.columns if 'pvals' in c]]
        pvals.columns = [c.split('pvals_')[-1] for c in pvals.columns]
        pvals.index = [window]
        pvals_list.append(pvals)
        
    pvals_df = pd.concat(pvals_list, axis=0)

    # apply multiple testing correction across    
    reject, pvals_df = _multitest_correct(pvals_df, multitest_method='fdr_by', fdr=0.05)
        
    strain_list = sorted(metadata['gene_name'].unique())
    colour_dict = dict(zip(['BW','fepD','atpB','nuoC','sdhD'], sns.color_palette('tab10', 5)))

    # boxplots    
    for strain in tqdm(strain_list[1:]):
        
        groups = ['BW', strain]
        
        save_dir=Path(SAVE_DIR) / 'Plots'
        save_path = save_dir / '{0}_{1}.pdf'.format(strain, FEATURE)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        
        plt.close('all')
        sns.set_theme(style='white')
        fig, ax = plt.subplots(figsize=(15,8))
        plot_df = metadata.join(features[[FEATURE]])
        sns.boxplot(x='window',
                    y=FEATURE,
                    hue='gene_name',
                    hue_order=groups,
                    dodge=True,
                    showfliers=False,
                    showmeans=False,
                    order=arousal_windows, 
                    data=plot_df,
                    palette=colour_dict)
        sns.stripplot(x='window',
                      y=FEATURE,
                      hue='gene_name',
                      hue_order=groups,
                      dodge=True,
                      order=arousal_windows,
                      data=plot_df,
                      s=10,
                      color='gray',
                      marker=".",
                      edgecolor='k',
                      linewidth=.3)
        
        # Add p-value to plot
        for i, window in enumerate(arousal_windows):
            pval = pvals_df.loc[window, groups[-1]]
            p_text = sig_asterix([pval])[0]
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled
            text = ax.get_xticklabels()[i]
            assert text.get_text() == str(window)
            ax.text(i, 0.98, p_text, fontsize=30, ha='center', va='bottom', transform=trans)

        plt.xticks(rotation=0, fontsize=25)
        plt.yticks(fontsize=25)
        ax.tick_params(axis='y', which='major', pad=15)
        plt.ylabel('Speed (µm s$^{-1}$)', labelpad=30, fontsize=25)
        ax.axes.set_xticklabels([int(WINDOW_DICT[int(i.get_text())][0]/60) 
                                 for i in ax.axes.get_xticklabels()], fontsize=25)
        plt.xlabel('Time (minutes)', labelpad=30, fontsize=25)
        plt.ylim(-60, 360)
                                                
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:len(groups)], labels[:len(groups)], loc='upper left', 
                  fontsize=20, frameon=False, handletextpad=1)

        plt.subplots_adjust(bottom=0.15,top=0.9,left=0.15,right=0.95)
        plt.savefig(save_path, dpi=600)
    


