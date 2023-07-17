#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2e - Plate tap and drop experiments to see if mechanical stimulation can also elicit arousal
similar to fepD
 
@author: sm5911
@date: 13/06/2023

"""

#%% Imports

import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import transforms

from visualisation.plotting_helper import sig_asterix
from tierpsytools.analysis.statistical_tests import (univariate_tests, 
                                                     get_effect_sizes, 
                                                     _multitest_correct)

#%% Globals

PROJECT_DIR_LIST = ["/Users/sm5911/Documents/Keio_Plate_Drop",
                    "/Users/sm5911/Documents/Keio_Plate_Tap"]
SAVE_DIR = "/Users/sm5911/OneDrive - Imperial College London/Publications/Keio_Paper/Figures/Fig2"

FEATURE = 'speed_50th'
FDR_METHOD = 'fdr_bh'
P_VALUE_THRESHOLD = 0.05
DPI = 900

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
    
    metadata_list = []
    features_list = []
    
    for project_dir in PROJECT_DIR_LIST:
        _metadata = pd.read_csv(Path(project_dir) / 'metadata.csv', header=0, index_col=None, 
                               dtype={'comments':str})
        _features = pd.read_csv(Path(project_dir) / 'features.csv', header=0, index_col=None)
        metadata_list.append(_metadata)
        features_list.append(_features)
        
    metadata = pd.concat(metadata_list, axis=0).reset_index(drop=True)
    features = pd.concat(features_list, axis=0).reset_index(drop=True)
        
    # drop paraquat results
    metadata = metadata[metadata['drug_type']!='Paraquat']
    
    # fill in bacteria strain names
    metadata['bacteria_strain'] = metadata['bacteria_strain'].copy().fillna(metadata['food_type'])
    assert metadata['bacteria_strain'].nunique() == 2
    
    # combine existing columns for plate tap and drop experiments
    metadata['stimulus_type'] = metadata['tap_stimulus'].copy().replace('mechanism','tap').fillna('drop')
    metadata['n_stimuli'] = metadata['n_stimuli'].copy().fillna(metadata['plate_drop_number'])
    
    # n_stimuli, stimulus_type, plate_drop_height_cm
        
    # combine bacteria-stimulus_type-n_stimuli as new column 'treatment' 
    metadata['treatment'] = metadata[['bacteria_strain','stimulus_type','n_stimuli']
                                     ].astype(str).agg('-'.join, axis=1)
    metadata['treatment'] = [i.replace('.0','').replace('-nan','') for i in metadata['treatment']]
    metadata['treatment'] = [i.replace('-drop-0','') for i in metadata['treatment']]

    features = features.reindex(metadata.index)
    assert all(metadata.index == features.index)
    
    metadata['treatment_plot'] = metadata[['stimulus_type','n_stimuli']
                                          ].astype(str).agg('-'.join, axis=1)
    metadata['treatment_plot'] = [i.replace('.0','').replace('-nan','') for i in 
                                  metadata['treatment_plot']]
    order = sorted(metadata['treatment_plot'].unique())
    
    # boxplots
    
    plot_df = metadata.join(features[[FEATURE]])
    colour_dict = dict(zip(['BW','fepD'], sns.color_palette(palette='tab10', n_colors=2)))

    plt.close('all')
    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=(10,10))
    sns.boxplot(x='treatment_plot', 
                y='speed_50th',
                hue='bacteria_strain',
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
                  hue='bacteria_strain',
                  order=order,
                  hue_order=['BW','fepD'],
                  dodge=True,
                  data=plot_df,
                  s=10,
                  palette=[sns.color_palette('Greys',2)[1]],
                  # color=None,
                  marker=".",
                  edgecolor='k',
                  linewidth=0.3)

    ax.axes.get_xaxis().get_label().set_visible(False) # remove x axis label
    ax.axes.set_xticklabels([l.get_text().replace('drop-0','no\nstimulus').replace('-','\nx ') 
                             for l in ax.axes.get_xticklabels()], fontsize=30)
    ax.tick_params(axis='x', which='major', pad=15)
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
            
    # do stats - compare all treatments to BW no stimulus control
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
    for i, treatment in enumerate(order[1:], start=1):
        text = ax.get_xticklabels()[i]
        assert text.get_text() == treatment.replace('drop-0','no\nstimulus').replace('-','\nx ')

        meta_BW = meta_grouped.get_group('BW-'+treatment)
        feat_BW = features[[FEATURE]].reindex(meta_BW.index)
        y1_pos = feat_BW[FEATURE].max() + pval_label_offset

        meta_fepD = meta_grouped.get_group('fepD-'+treatment)
        feat_fepD = features[[FEATURE]].reindex(meta_fepD.index)
        y2_pos = feat_fepD[FEATURE].max() + pval_label_offset
        
        p1 = pvals.loc['speed_50th','BW-'+treatment]
        p2 = pvals.loc['speed_50th','fepD-'+treatment]
        p1_text = sig_asterix([p1],ns=True)[0]
        p2_text = sig_asterix([p2],ns=True)[0]
        ax.text(i-0.19, y1_pos, p1_text, fontsize=fontsize_label, ha='center', va='top') # transform=trans
        ax.text(i+0.19, y2_pos, p2_text, fontsize=fontsize_label, ha='center', va='top') # transform=trans
            
    # do stats - compare BW vs fepD for each treatment_plot separately (apply multiple test correct)    
    pvals_dict = {}
    meta_grouped = metadata.groupby('treatment_plot')
    for group in order:
        _meta = meta_grouped.get_group(group)
        _feat = features.reindex(_meta.index)
        
        ttest_df = stats(_meta, 
                         _feat, 
                         group_by='bacteria_strain',
                         control='BW',
                         feat=FEATURE,
                         pvalue_threshold=P_VALUE_THRESHOLD,
                         fdr_method=FDR_METHOD)
        
        pvals_dict[group] = ttest_df.loc[FEATURE, 'pvals_fepD']
    
    # apply correction for multiple testing
    reject, corrected_pvals = _multitest_correct(pd.Series(list(pvals_dict.values())), 
                                                 multitest_method=FDR_METHOD, fdr=P_VALUE_THRESHOLD)
        
    pvals_dict = dict(zip(order, corrected_pvals))
    pvals = pd.DataFrame.from_dict(pvals_dict, orient='index', columns=['pvals'])
    pvals.index = [i.get_text() for i in ax.get_xticklabels()]

    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled
    for i, group in enumerate(pvals.index):
        p = pvals.loc[group,'pvals']
        p_text = sig_asterix([p],ns=True)[0]
        #[x1, x1, x2, x2], [y, y+h, y+h, y]
        ax.plot([i-0.2,i-0.2,i+0.2,i+0.2], [0.85,0.87,0.87,0.85], lw=1.5, c='k', transform=trans)
        ax.text(i, 0.88, p_text, fontsize=fontsize_label, ha='center', va='bottom', transform=trans)

    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[:2], labels[:2], fontsize=fontsize_label, loc='upper right',
    #           frameon=False, handletextpad=0.75)
    ax.get_legend().remove()

    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.99)
    plt.savefig(Path(SAVE_DIR) / 'Fig2e.png', dpi=DPI)  
    
    return

#%% Main

if __name__ == '__main__':
    main()

