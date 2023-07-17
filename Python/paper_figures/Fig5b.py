#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 5a - Boxplots of worm antioxidant mutants on BW and fepD

@author: sm5911
@date: 16/06/2023

"""

#%% Imports 

import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt

from visualisation.plotting_helper import sig_asterix
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes

#%% Globals 

PROJECT_DIR = "/Users/sm5911/Documents/Keio_Worm_Stress_Mutants_Combined"
SAVE_DIR = "/Users/sm5911/OneDrive - Imperial College London/Publications/Keio_Paper/Figures/Fig5"

FEATURE = 'speed_50th'
FDR_METHOD = 'fdr_bh'
P_VALUE_THRESHOLD = 0.05
DPI = 900

WORM_STRAIN_MAPPING_DICT = {"N2":"N2",
                            "prdx-2":"prdx-2(gk169)",
                            "GA187":"sod-1",
                            "GA476":"sod-1; sod-5",
                            "GA184":"sod-2",
                            "GA813":"sod-1; sod-2",
                            "GA186":"sod-3",
                            "GA480":"sod-2; sod-3",
                            "GA416":"sod-4",
                            "GA822":"sod-4; sod-5",
                            "GA503":"sod-5",
                            "GA814":"sod-3; sod-5",
                            "GA800":"OE ctl-1+ctl-2+ctl-3",
                            "GA801":"OE sod-1",
                            "GA804":"OE sod-2",
                            "clk-1":"clk-1(qm30)",
                            "gas-1":"gas-1(fc21)",
                            "msrA":"msra-1(tm1421)"}

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
    
    metadata_path = Path(PROJECT_DIR) / 'metadata.csv'
    features_path = Path(PROJECT_DIR) / 'features.csv'
    
    # load metadata and features summaries
    metadata = pd.read_csv(metadata_path, header=0, index_col=None, dtype={'comments':str})
    features = pd.read_csv(features_path, header=0, index_col=None)

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

    # record worm strains (drop duplicates)
    worm_strain_list = list(dict.fromkeys(WORM_STRAIN_MAPPING_DICT.values()))

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
    colour_dict = dict(zip(['BW','fepD'], sns.color_palette(palette='tab10', n_colors=2)))
    
    plt.close('all')
    sns.set_style('ticks')
    fig = plt.figure(figsize=[25,9])
    ax = fig.add_subplot(1,1,1)
    sns.boxplot(x='worm_strain',
                y='speed_50th',
                dodge=True,
                hue='bacteria_strain',
                hue_order=['BW','fepD'],
                order=worm_strain_list,            
                data=plot_df, 
                palette=colour_dict,
                showfliers=False, 
                showmeans=False,
                meanprops={"marker":"x", 
                           "markersize":5,
                           "markeredgecolor":"k"},
                flierprops={"marker":"x", 
                            "markersize":15, 
                            "markeredgecolor":"r"},
                width=0.8)
    sns.stripplot(x='worm_strain',
                  y='speed_50th',
                  data=plot_df,
                  s=10,
                  order=worm_strain_list,
                  hue='bacteria_strain',
                  hue_order=['BW','fepD'],
                  dodge=True,
                  palette=[sns.color_palette('Greys',2)[1]],
                  color=None,
                  marker=".",
                  edgecolor='k',
                  linewidth=0.2) #facecolors="none"
    
    linewidth = 2
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.xaxis.set_tick_params(width=linewidth, length=linewidth*3)
    ax.yaxis.set_tick_params(width=linewidth, length=linewidth*2)
    ax.axes.get_xaxis().get_label().set_visible(False) # remove x axis label
    ax.axes.set_xticklabels([l.get_text().replace(' ','\n').replace('(','\n(').replace('+','+\n') 
                             for l in ax.axes.get_xticklabels()], fontsize=25, rotation=0, ha='center')
    ax.tick_params(axis='x', which='major', pad=10)
    ax.axes.set_ylabel('Speed (µm s$^{-1}$)', fontsize=30, labelpad=30)
    plt.yticks(fontsize=25)
    plt.ylim(-20, 300)
            
    # add pvalues to plot
    y_offset_label = 40
    fontsize_label = 25
    for i, worm in enumerate(worm_strain_list, start=0):
        text = ax.get_xticklabels()[i]
        assert text.get_text() == worm.replace(' ','\n').replace('(','\n(').replace('+','+\n') 
        
        meta_grouped = metadata.groupby('treatment')
        if worm == 'N2':
            meta_fepD = meta_grouped.get_group('N2-fepD')
            feat_fepD = features[[FEATURE]].reindex(meta_fepD.index)
            y_pos = feat_fepD[FEATURE].max() + y_offset_label
            
            p = pvals.loc['speed_50th','N2-fepD']
            p_text = sig_asterix([p],ns=False)[0]
            ax.text(i+0.19, y_pos, p_text, fontsize=fontsize_label, ha='center', va='top') # transform=trans
        else:
            meta_BW = meta_grouped.get_group(worm+'-BW')
            feat_BW = features[[FEATURE]].reindex(meta_BW.index)
            y1_pos = feat_BW[FEATURE].max() + y_offset_label

            meta_fepD = meta_grouped.get_group(worm+'-fepD')
            feat_fepD = features[[FEATURE]].reindex(meta_fepD.index)
            y2_pos = feat_fepD[FEATURE].max() + y_offset_label
            
            p1 = pvals.loc['speed_50th',worm+'-BW']
            p2 = pvals.loc['speed_50th',worm+'-fepD']
            p1_text = sig_asterix([p1],ns=True)[0]
            p2_text = sig_asterix([p2],ns=True)[0]
            ax.text(i-0.19, y1_pos, p1_text, fontsize=fontsize_label, ha='center', va='top') # transform=trans
            ax.text(i+0.19, y2_pos, p2_text, fontsize=fontsize_label, ha='center', va='top') # transform=trans
            
    ax.get_legend().remove()
    
    plt.subplots_adjust(left=0.12, right=0.98, bottom=0.25, top=0.95)
    plt.savefig(Path(SAVE_DIR) / 'Fig5b.png', dpi=DPI)  

    return

#%% Main

if __name__ == '__main__':
    main()
