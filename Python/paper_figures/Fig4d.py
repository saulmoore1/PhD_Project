#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4d - Supplementation with Vitamin C and NAC + fepD antioxidant mutants

@author: sm5911
@date: 16/06/2023

"""

#%% Imports

# import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt

from visualisation.plotting_helper import sig_asterix
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes

#%% Globals

PROJECT_DIR = "/Users/sm5911/Documents/Keio_UV_Paraquat_Antioxidant"
SAVE_DIR = "/Users/sm5911/OneDrive - Imperial College London/Publications/Keio_Paper/Figures/Fig4"

FEATURE = 'speed_50th'
P_VALUE_THRESHOLD = 0.05
FDR_METHOD = 'fdr_bh'
DPI=900

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
    print("Mean sample size of %s: %d" % (group_by, int(sample_size['well_name'].mean())))

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
             
    # perform t-tests
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

    if n > 2:
        return anova_results, ttest_results
    else:
        return ttest_results


def main():
    
    # load metadata and features summary results
    metadata = pd.read_csv(Path(PROJECT_DIR) / 'metadata.csv', header=0, index_col=None,
                           dtype={'comments':str})
    features = pd.read_csv(Path(PROJECT_DIR) / 'features.csv', header=0, index_col=None)
    
    # subset for bluelight videos only 
    bluelight_videos = [i for i in metadata['imgstore_name'] if 'bluelight' in i]
    metadata = metadata[metadata['imgstore_name'].isin(bluelight_videos)]

    # subset for window 5 (20-30 seconds after blue light 3)
    metadata = metadata[metadata['window']==5]
    
    # subset for live bacteria only
    metadata = metadata[metadata['is_dead']=='N']
    
    # subset to remove paraquat results
    metadata = metadata[metadata['drug_type']!='Paraquat']    
    
    # reindex features
    features = features.reindex(metadata.index)
    
    treatment_cols = ['food_type','drug_type','drug_imaging_plate_conc']
    metadata['treatment'] = metadata[treatment_cols].astype(str).agg('-'.join, axis=1)
    metadata['treatment'] = [i.replace('-nan','').replace('.0','') for i in metadata['treatment']]

    metadata['treatment_plot'] = metadata[['drug_type','drug_imaging_plate_conc']
                                          ].astype(str).agg('-'.join, axis=1)
    metadata['treatment_plot'] = [i.replace('nan-nan','None').replace('.0','') for i in 
                                  metadata['treatment_plot']]
    order = ['None','Vitamin C-0.5','Vitamin C-1','NAC-0.5','NAC-1']
    
    # boxplots
    plot_df = metadata.join(features[[FEATURE]])
    colour_dict = dict(zip(['BW','fepD'], sns.color_palette(palette='tab10', n_colors=2)))
    
    plt.close('all')
    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=(12,10))
    sns.boxplot(x='treatment_plot', 
                y='speed_50th',
                hue='food_type',
                order=order,
                hue_order=['BW','fepD'],
                dodge=True,
                data=plot_df, 
                palette=colour_dict,
                showfliers=False, 
                showmeans=False,
                meanprops={"marker":"x", 
                           "markersize":5,
                           "markeredgecolor":"k"},
                medianprops={'color': 'black'},
                flierprops={"marker":"x", 
                            "markersize":15, 
                            "markeredgecolor":"r"},
                width=0.75)
    sns.stripplot(x='treatment_plot', 
                  y='speed_50th',
                  hue='food_type',
                  order=order,
                  hue_order=['BW','fepD'],
                  dodge=True,
                  data=plot_df,
                  s=10,
                  palette=[sns.color_palette('Greys',2)[1]],
                  marker=".",
                  edgecolor='k',
                  linewidth=0.3)

    ax.axes.get_xaxis().get_label().set_visible(False) # remove x axis label
    xlabs = [l.get_text().replace('-','\n') for l in ax.axes.get_xticklabels()]
    xlabs = [l + ' mM' if l != 'None' else l for l in xlabs]
    ax.axes.set_xticklabels(xlabs, fontsize=25, ha='center') #rotation=45
    ax.tick_params(axis='x', which='major', pad=5)
    ax.axes.set_ylabel('Speed (µm s$^{-1}$)', fontsize=30, labelpad=30)                             
    plt.yticks(fontsize=30)
    plt.ylim(-20, 300)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    linewidth = 2
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.xaxis.set_tick_params(width=linewidth, length=linewidth*2)
    ax.yaxis.set_tick_params(width=linewidth, length=linewidth*2)
    
    # do stats - compare all treatments to fepD no citrate control
    anova_results, ttest_results = stats(metadata,
                                         features,
                                         group_by='treatment',
                                         control='BW',
                                         feat=FEATURE,
                                         pvalue_threshold=P_VALUE_THRESHOLD,
                                         fdr_method=FDR_METHOD)
    
    # load t-test results
    pvals = ttest_results[[c for c in ttest_results.columns if 'pval' in c]]
    pvals.columns = [c.replace('pvals_','') for c in pvals.columns]

    # Add p-value to plot  
    pval_label_offset = 40
    fontsize_label = 25
    meta_grouped = metadata.groupby('treatment')
    
    for i, drug in enumerate(order):
        text = ax.get_xticklabels()[i]
        assert text.get_text() == (drug.replace('-','\n') + ' mM' if drug!='None' else drug)
        
        if drug == 'None':
            meta_fepD = meta_grouped.get_group('fepD')
            feat_fepD = features[[FEATURE]].reindex(meta_fepD.index)
            y_pos = feat_fepD[FEATURE].max() + pval_label_offset
            p = pvals.loc['speed_50th','fepD']
            p_text = sig_asterix([p],ns=True)[0]
            ax.text(i+0.19, y_pos, p_text, fontsize=fontsize_label, ha='center', va='top')
        else:
            meta_BW = meta_grouped.get_group('BW-'+drug)
            feat_BW = features[[FEATURE]].reindex(meta_BW.index)
            y1_pos = feat_BW[FEATURE].max() + pval_label_offset
            p1 = pvals.loc['speed_50th','BW-'+drug]
            p1_text = sig_asterix([p1],ns=True)[0]
            ax.text(i-0.19, y1_pos, p1_text, fontsize=fontsize_label, ha='center', va='top')
            
            meta_fepD = meta_grouped.get_group('fepD-'+drug)
            feat_fepD = features[[FEATURE]].reindex(meta_fepD.index)
            y2_pos = feat_fepD[FEATURE].max() + pval_label_offset
            p2 = pvals.loc['speed_50th','fepD-'+drug]
            p2_text = sig_asterix([p2],ns=True)[0]
            ax.text(i+0.19, y2_pos, p2_text, fontsize=fontsize_label, ha='center', va='top')
            
    ax.get_legend().remove()
    
    plt.subplots_adjust(left=0.15, bottom=0.2, right=0.99)
    plt.savefig(Path(SAVE_DIR) / 'Fig4d.png', dpi=DPI)      
    
    
    return

#%% Main

if __name__ == '__main__':
    main()
