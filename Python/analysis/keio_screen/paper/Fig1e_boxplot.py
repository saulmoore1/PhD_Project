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
from tierpsytools.analysis.statistical_tests import univariate_tests 
from tierpsytools.analysis.statistical_tests import get_effect_sizes
from visualisation.plotting_helper import sig_asterix

#%% Globals

SAVE_DIR = "/Users/sm5911/Documents/PhD_DLBG/Fig1e"
RENAME_DICT = {"BW" : "wild_type"}

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
BLUELIGHT_TIMEPOINTS_MINUTES = [30,60,90,120,150,180,210,240]
WINDOW_BLUELIGHT_DICT = dict(zip(WINDOW_LIST, BLUELIGHT_TIMEPOINTS_MINUTES))

OMIT_STRAINS_LIST = ['trpD']

#%% Functions

def stats(metadata,
          features,
          group_by='treatment',
          control='BW',
          feat='speed_50th',
          pvalue_threshold=0.05,
          fdr_method='fdr_bh'):
    
    """ Perform ANOVA and t-tests to compare worm speed on each treatment vs control """
        
    assert all(metadata.index == features.index)
    features = features[[feat]]

    # print mean sample size
    sample_size = metadata.groupby(group_by).count()
    print("Mean sample size of %s/window: %d" % (group_by, 
                                                 int(sample_size[sample_size.columns[-1]].mean())))

    n = len(metadata[group_by].unique())
    if n > 2:
   
        # perform ANOVA - is there variation among strains?
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
    
    metadata_path_local = Path(SAVE_DIR) / 'metadata.csv'
    features_path_local = Path(SAVE_DIR) / 'features.csv'

    metadata = pd.read_csv(metadata_path_local, header=0, index_col=None, 
                           dtype={'comments':str})
    features = pd.read_csv(features_path_local, header=0, index_col=None)

    assert not features.isna().sum(axis=1).any()
    assert not (features.std(axis=1) == 0).any()
    assert set(features.index) == set(metadata.index)    
    assert (len(metadata['gene_name'].unique()) ==
            len(metadata['gene_name'].str.upper().unique()))
            
    if OMIT_STRAINS_LIST is not None:
        metadata = metadata[~metadata['gene_name'].isin(OMIT_STRAINS_LIST)]

    # subset metadata for first blue light timepoint (30 minutes)
    metadata = metadata[metadata['window']==1]
    
    # rename gene names in metadata
    for k, v in RENAME_DICT.items():
        metadata.loc[metadata['gene_name'] == k, 'gene_name'] = v
           
    features = features.reindex(metadata.index)
    
    # save plot data to file
    plot_df = metadata[['gene_name','date_yyyymmdd']].join(features[['speed_50th']])
    plot_df.to_csv(Path(SAVE_DIR) / 'Fig1e_data.csv', header=True, index=False)
    
    strain_list = ['wild_type','fepD'] + [s for s in 
        sorted(metadata['gene_name'].unique()) if s not in ['wild_type','fepD']]
    strain_lut = dict(zip(strain_list, sns.color_palette('tab10', len(strain_list))))

    # boxplots for blue light timepoint 30 minutes
    plt.close('all')
    sns.set_style('ticks')
    fig = plt.figure(figsize=[12,6])
    ax = fig.add_subplot(1,1,1)
    sns.boxplot(x='gene_name',
                y='speed_50th',
                data=plot_df, 
                order=strain_list,
                palette=strain_lut,
                hue='gene_name',
                legend=False,
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
        sns.stripplot(x='gene_name',
                      y='speed_50th',
                      data=date_df,
                      s=8,
                      order=strain_list,
                      palette=[date_lut[date]] * len(strain_list),
                      hue='gene_name',
                      legend=False,
                      color=None,
                      marker=".",
                      edgecolor='k',
                      linewidth=0.3) #facecolors="none"

    ax.axes.get_xaxis().get_label().set_visible(False) # remove x axis label
    # ax.axes.set_xticklabels(BLUELIGHT_TIMEPOINTS_MINUTES, fontsize=20)
    # ax.axes.set_xlabel('Time (minutes)', fontsize=25, labelpad=20)   
    ax.tick_params(axis='x', which='major', pad=15)
    plt.xticks(fontsize=20)                        
    ax.tick_params(axis='y', which='major', pad=15)
    plt.yticks(fontsize=20)
    ax.axes.set_ylabel('Speed (µm s$^{-1}$)', fontsize=25, labelpad=20)  
    plt.ylim(0, 320)
    
    # add p-values to plot (stats for all strains, first BL window = 30 minutes)
    _, ttest_results = stats(metadata,
                             features,
                             group_by='gene_name',
                             control='wild_type',
                             feat='speed_50th',
                             pvalue_threshold=0.05,
                             fdr_method='fdr_bh')
    
    ttest_results.to_csv(Path(SAVE_DIR) / 'Fig1e_t-test_fdr_bh_BL_30min.csv', 
                         header=True, index=False)
        
    # add pvalues to plot
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled
    pvals = ttest_results[[c for c in ttest_results.columns if 'pval' in c]]
    for i, strain in enumerate(strain_list[1:], start=1):
        p = pvals.loc['speed_50th', 'pvals_' + strain]
        text = ax.get_xticklabels()[i]
        assert text.get_text() == strain
        p_text = sig_asterix([p])[0]
        ax.text(i, 1.03, p_text, fontsize=25, ha='center', va='center', transform=trans)
            
    #plt.subplots_adjust(left=0.01, right=0.9)
    boxplot_path = Path(SAVE_DIR) / 'Fig1e_speed_50th_BL_30min.svg'
    boxplot_path.parent.mkdir(exist_ok=True, parents=True)
    #plt.subplots_adjust(left=0.125,right=0.9,bottom=0.1,top=0.9)
    plt.savefig(boxplot_path, bbox_inches='tight', transparent=True)
    
    return

#%% Main

if __name__ == "__main__":
    main()

