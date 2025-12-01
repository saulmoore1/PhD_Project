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
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes

#%% Globals

SAVE_DIR = "/Users/sm5911/Documents/PhD_DLBG/FigS3a worm NHR mutants"

#%% Functions

def stats(metadata,
          features,
          group_by='treatment',
          control='BW',
          feat='speed_50th',
          pvalue_threshold=0.05,
          fdr_method='fdr_bh'):
    
    """ Perform t-tests to compare worm speed on each treatment vs control """
        
    assert all(metadata.index == features.index)
    features = features[[feat]]

    # print mean sample size
    sample_size = metadata.groupby(group_by).count()
    print("Mean sample size of %s/window: %d" % (group_by, 
                                                 int(sample_size[sample_size.columns[-1]].mean())))
             
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

    return ttest_results


def main():
    
    metadata_path_local = Path(SAVE_DIR) / 'metadata.csv'
    features_path_local = Path(SAVE_DIR) / 'features.csv'
    
    metadata = pd.read_csv(metadata_path_local, header=0, index_col=None, dtype={'comments':str})
    features = pd.read_csv(features_path_local, header=0, index_col=None)

    assert not features.isna().sum(axis=1).any()
    assert not (features.std(axis=1) == 0).any()    
    assert not metadata['bacteria_strain'].isna().any()
    assert not metadata['worm_gene'].isna().any()    

    # subset metadata results for bluelight videos only 
    bluelight_videos = [i for i in metadata['imgstore_name'] if 'bluelight' in i]
    metadata = metadata[metadata['imgstore_name'].isin(bluelight_videos)]
        
    # standardise gene names in metadata
    metadata['bacteria_strain'] = ['fepD' if i=='FepD' else i for i in 
                                   metadata['bacteria_strain'].copy()]
    metadata['bacteria_strain'] = ['BW' if i.upper().startswith('BW') else i for i in
                                   metadata['bacteria_strain'].copy()]
    
    # reindex features for new metadata subset
    features = features[['speed_50th']].reindex(metadata.index)

    # combine into single list of treatment combinations
    treatment_cols = ['worm_gene','bacteria_strain']
    metadata['treatment'] = metadata[treatment_cols].astype(str).agg('-'.join, axis=1)
    metadata['treatment'] = [i.replace('-none','') for i in metadata['treatment']]
    
    # stats
    control = 'N2-BW'
    ttest_results = stats(metadata,
                          features,
                          group_by='treatment',
                          control=control,
                          feat='speed_50th',
                          pvalue_threshold=0.05,
                          fdr_method='fdr_bh')
    
    ttest_path = Path(SAVE_DIR) / 't-test_results_vs_{}.csv'.format(control)
    ttest_path.parent.mkdir(exist_ok=True, parents=True)
    ttest_results.to_csv(ttest_path, header=True, index=True)
    
    # extract t-test pvals
    pvals = ttest_results[[c for c in ttest_results.columns if 'pval' in c]]
    pvals.columns = [c.replace('pvals_','') for c in pvals.columns]

    # save plot data to file
    plot_df = metadata[['worm_gene','bacteria_strain','treatment','date_yyyymmdd']
                       ].join(features).sort_values(by='treatment', ascending=True)
    plot_df.to_csv(Path(SAVE_DIR) / 'FigS3a_data.csv', header=True, index=False)
    
    worm_genes = sorted(metadata['worm_gene'].unique())
    bacteria_strains = sorted(metadata['bacteria_strain'].unique())        
    colour_dict = dict(zip(['BW','fepD'], sns.color_palette(palette='tab10', n_colors=2)))

    # boxplot
    plt.close('all')
    sns.set_style('ticks')
    fig = plt.figure(figsize=[25,10])
    ax = fig.add_subplot(1,1,1)
    sns.boxplot(x='worm_gene',
                y='speed_50th',
                data=plot_df, 
                order=worm_genes,
                hue='bacteria_strain',
                hue_order=bacteria_strains,
                palette=colour_dict,
                dodge=True,
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
        sns.stripplot(x='worm_gene',
                      y='speed_50th',
                      data=date_df,
                      s=8,
                      order=worm_genes,
                      hue='bacteria_strain',
                      hue_order=bacteria_strains,
                      dodge=True,
                      palette=[date_lut[date]] * len(bacteria_strains),
                      color=None,
                      marker=".",
                      edgecolor='k',
                      linewidth=0.3) #facecolors="none"

    # scale y axis for annotations    
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled

    ax.axes.get_xaxis().get_label().set_visible(False) # remove x axis label
    ax.axes.set_xticklabels(worm_genes, fontsize=20)
    # ax.axes.set_xlabel('Time (minutes)', fontsize=25, labelpad=20)                           
    ax.tick_params(axis='y', which='major', pad=15)
    plt.yticks(fontsize=20)
    ax.axes.set_ylabel('Speed (µm s$^{-1}$)', fontsize=25, labelpad=20)  
    plt.ylim(-20, 320)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc='upper left', frameon=False, fontsize=15)
    #plt.axhline(y=0, c='grey')
    
    # add pvalues to plot
    for i, text in enumerate(ax.axes.get_xticklabels()):
        worm = text.get_text()
        if worm == 'N2':
            p = pvals.loc['speed_50th','N2-fepD']
            p_text = sig_asterix([p])[0]
            ax.text(i+0.2, 1.03, p_text, fontsize=20, ha='center', va='center', transform=trans)
        else:
            p1 = pvals.loc['speed_50th',worm+'-BW']
            p2 = pvals.loc['speed_50th',worm+'-fepD']
            p1_text = sig_asterix([p1])[0]
            p2_text = sig_asterix([p2])[0]
            ax.text(i-0.2, 1.03, p1_text, fontsize=20, ha='center', va='center', transform=trans)
            ax.text(i+0.2, 1.03, p2_text, fontsize=20, ha='center', va='center', transform=trans)
    
    # save plot
    boxplot_path = Path(SAVE_DIR) / 'FigS3a_worm_nhr_mutants.svg'
    boxplot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(boxplot_path, bbox_inches='tight', transparent=True)
            
    return

#%% Main

if __name__ == '__main__':
    main()