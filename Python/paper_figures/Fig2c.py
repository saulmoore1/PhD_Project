#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2c - Boxplots of worm neuron ablation and neuropeptide mutants on fepD vs BW

@author: sm5911
@date: 10/06/2023

"""

#%% Imports 

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import transforms

from visualisation.plotting_helper import sig_asterix
from time_series.plot_timeseries import plot_timeseries, get_strain_timeseries
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes

#%% Globals 

PROJECT_DIR = "/Users/sm5911/Documents/Keio_Worm_Stress_Mutants_Combined"
SAVE_DIR = "/Users/sm5911/OneDrive - Imperial College London/Publications/Keio_Paper/Figures/Fig2"

FEATURE = 'speed_50th'
FDR_METHOD = 'fdr_bh'
P_VALUE_THRESHOLD = 0.05
DPI = 900

WORM_STRAIN_MAPPING_DICT = {"N2":"N2",
                            "ZD763":"ASJ-ablated",
                            "QD71":"AIY-ablated",
                            # "PY7505":"ASI-ablated",
                            # "RID(RNAi)::unc-31":"RID(RNAi)",
                            "RID(RNAi)_lite-1(wt)":"RID(RNAi) [OMG94]",
                            "PS8997":"flp-1(sy1599)",
                            "VC2490":"flp-2(gk1039)+W07E11.1",
                            "VC2591":"flp-2(ok3351)",
                            "frpr-3":"frpr-3(ok3302)",
                            "frpr-18":"frpr-18(ok2698)",
                            "pdfr-1":"pdfr-1(ok3425)",
                            "eat-4":"eat-4(n2474)"}

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


#%% Main

if __name__ == '__main__':
    
    metadata_combined_path = Path(PROJECT_DIR) / 'metadata.csv'
    features_combined_path = Path(PROJECT_DIR) / 'features.csv'
    
    # load metadata and features summaries
    metadata = pd.read_csv(metadata_combined_path, header=0, index_col=None, dtype={'comments':str})
    features = pd.read_csv(features_combined_path, header=0, index_col=None)

    # subset metadata results for bluelight videos only 
    bluelight_videos = [i for i in metadata['imgstore_name'] if 'bluelight' in i]
    metadata = metadata[metadata['imgstore_name'].isin(bluelight_videos)]
    
    # drop paraquat results
    metadata = metadata[metadata['drug_type']!='Paraquat']
    
    # subset metadata for selected strains
    metadata = metadata[metadata['worm_strain'].isin(WORM_STRAIN_MAPPING_DICT.keys())]
    
    # strain name mapping
    metadata['worm_strain'] = metadata['worm_strain'].map(WORM_STRAIN_MAPPING_DICT)
    
    # combine worm+food as new column 'treatment' 
    metadata['treatment'] = metadata[['worm_strain','bacteria_strain']
                                     ].astype(str).agg('-'.join, axis=1)

    features = features.reindex(metadata.index)

    worm_strain_list = list(WORM_STRAIN_MAPPING_DICT.values())
    treatment_list = [worm + '-BW' if i % 2 == 0 else worm + '-fepD' for 
                      i, worm in enumerate(np.repeat(worm_strain_list, 2))]

    # stats: compare worm speed vs N2 worms on BW for each treatment (worm+food) combination
    anova_results, ttest_results = stats(metadata,
                                         features,
                                         group_by='treatment',
                                         control='N2-BW',
                                         feat='speed_50th',
                                         pvalue_threshold=P_VALUE_THRESHOLD,
                                         fdr_method=FDR_METHOD)
    
    # t-test pvals
    pvals = ttest_results[[c for c in ttest_results.columns if 'pval' in c]]
    pvals.columns = [c.replace('pvals_','') for c in pvals.columns]

    # boxplots

    plot_df = metadata.join(features)
    cols = sns.color_palette(palette='Greys', n_colors=2)
    colour_dict = {i:(cols[0] if '-BW' in i else cols[1]) for i in treatment_list}

    plt.close('all')
    sns.set_style('ticks')
    fig = plt.figure(figsize=[14,18])
    ax = fig.add_subplot(1,1,1)
    sns.boxplot(x='speed_50th',
                y='treatment',
                data=plot_df, 
                order=treatment_list,
                palette=colour_dict,
                showfliers=False, 
                showmeans=False,
                meanprops={"marker":"x", 
                           "markersize":5,
                           "markeredgecolor":"k"},
                flierprops={"marker":"x", 
                            "markersize":15, 
                            "markeredgecolor":"r"})
    sns.stripplot(x='speed_50th',
                  y='treatment',
                  data=plot_df,
                  s=12,
                  order=treatment_list,
                  hue=None,
                  palette=None,
                  color='dimgray',
                  marker=".",
                  edgecolor='k',
                  linewidth=0.3) #facecolors="none"
    
    ax.axes.get_yaxis().get_label().set_visible(False) # remove y axis label
    ax.axes.set_yticklabels([l.get_text() for l in ax.axes.get_yticklabels()], fontsize=25)
    ax.tick_params(axis='y', which='major', pad=15)
    ax.axes.set_xlabel('Speed (µm s$^{-1}$)', fontsize=30, labelpad=25)                             
    plt.xticks(fontsize=20)
    plt.xlim(-20, 250)
            
    # scale x axis for annotations    
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData) #x=scaled
    
    # add pvalues to plot
    for i, treatment in enumerate(treatment_list, start=0):
        if treatment == 'N2-BW':
            continue
        else:
            p = pvals.loc['speed_50th', treatment]
            text = ax.get_yticklabels()[i]
            assert text.get_text() == treatment
            p_text = sig_asterix([p])[0]
            ax.text(1.03, i, p_text, fontsize=30, ha='left', va='center', transform=trans)
            
    plt.subplots_adjust(left=0.4, right=0.9, bottom=0.15, top=0.95)
    plt.savefig(Path(SAVE_DIR) / 'Fig2c.png', dpi=DPI)  

