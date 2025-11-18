#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: sm5911    
"""

#%% Imports 

import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import transforms
from matplotlib import pyplot as plt
from visualisation.plotting_helper import sig_asterix
from tierpsytools.analysis.statistical_tests import univariate_tests
from tierpsytools.analysis.statistical_tests import get_effect_sizes

#%% Globals

SAVE_DIR = "/Users/sm5911/Documents/PhD_DLBG/Fig2h"

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


def main():
    
    metadata_path_local = Path(SAVE_DIR) / 'metadata.csv'
    features_path_local = Path(SAVE_DIR) / 'features.csv'
    
    metadata = pd.read_csv(metadata_path_local, header=0, index_col=None, dtype={'comments':str})
    features = pd.read_csv(features_path_local, header=0, index_col=None)

    assert not features.isna().sum(axis=1).any()
    assert not (features.std(axis=1) == 0).any()    
    assert not metadata['bacteria_strain'].isna().any()

    # subset metadata results for bluelight videos only 
    bluelight_videos = [i for i in metadata['imgstore_name'] if 'bluelight' in i]
    metadata = metadata[metadata['imgstore_name'].isin(bluelight_videos)]
        
    # subset features for metadata subset
    features = features.reindex(metadata.index)
            
    treatment_cols = ['bacteria_strain','supplement']
    metadata['treatment'] = metadata[treatment_cols].astype(str).agg('-'.join, axis=1)
    metadata['treatment'] = [i.replace('-none','') for i in metadata['treatment']]

    treatment_list = ['BW','fepD'] + [i for i in sorted(metadata['treatment'].unique()) if i not in 
                                      ['BW','fepD']]

    # stats
    _, ttest_results = stats(metadata,
                             features,
                             group_by='treatment',
                             control='BW',
                             save_dir=Path(SAVE_DIR),
                             feature_list=['speed_50th'],
                             p_value_threshold=0.05,
                             fdr_method='fdr_bh')

    # t-test pvals
    pvals = ttest_results[[c for c in ttest_results.columns if 'pval' in c]]
    pvals.columns = [c.replace('pvals_','') for c in pvals.columns]

    # save plot data
    plot_df = metadata[['treatment','date_yyyymmdd']].join(features[['speed_50th']])
    plot_df.to_csv(Path(SAVE_DIR) / 'Fig2h_data.csv', header=True, index=False)
    
    # boxplot
    lut = dict(zip(['BW','fepD'], sns.color_palette(palette='tab10', n_colors=2)))
    colour_dict = {i:(lut['fepD'] if 'fepD' in i else lut['BW']) for i in treatment_list}

    plt.close('all')
    sns.set_style('ticks')
    fig = plt.figure(figsize=[15,18])
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
    dates = sorted(plot_df['date_yyyymmdd'].unique())
    date_lut = dict(zip(dates, sns.color_palette(palette="Greys", n_colors=len(dates))))
    for date in dates:
        date_df = plot_df[plot_df['date_yyyymmdd']==date]
        sns.stripplot(x='speed_50th',
                      y='treatment',
                      data=date_df,
                      order=treatment_list,
                      palette=[date_lut[date]] * len(treatment_list),
                      color='dimgray',
                      marker=".",
                      edgecolor='k',
                      linewidth=0.3,
                      s=12) #facecolors="none"
    ax.axes.get_yaxis().get_label().set_visible(False) # remove y axis label
    ax.axes.set_yticklabels([l.get_text() for l in ax.axes.get_yticklabels()], fontsize=25)
    ax.tick_params(axis='y', which='major', pad=15)
    ax.axes.set_xlabel('Speed (µm s$^{-1}$)', fontsize=30, labelpad=25)                             
    plt.xticks(fontsize=20)
    plt.xlim(-20, 320)
    
    # scale x axis for annotations    
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData) #x=scaled
    
    # add pvalues to plot
    for i, strain in enumerate(treatment_list, start=0):
        if strain == 'BW':
            continue
        else:
            p = pvals.loc['speed_50th', strain]
            text = ax.get_yticklabels()[i]
            assert text.get_text() == strain
            p_text = sig_asterix([p])[0]
            ax.text(1.03, i, p_text, fontsize=20, ha='left', va='center', transform=trans)

    # save plot            
    plt.subplots_adjust(left=0.5, right=0.9)
    boxplot_path = Path(SAVE_DIR) / 'Fig2h_boxplot.svg'
    boxplot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(boxplot_path, bbox_inches='tight', transparent=True)
    
    return

#%% Main

if __name__ == '__main__':
    main()