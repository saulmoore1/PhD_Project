#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combinbed results for worm mutant experiments so far on fepD / BW both with/without paraquat 
(boxplots and timeseries)

@author: sm5911
@date: 10/06/2023

"""

#%% Imports 

import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import transforms
from visualisation.plotting_helper import sig_asterix
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes

#%% Globals 

SAVE_DIR = '/Users/sm5911/Documents/PhD_DLBG/Fig3c worm antioxidant mutants'

OMIT_STRAINS_LIST = ["RID(RNAi)::unc-31", "PY7505", "MQ1333", "nuo-6", 
                     "clk-1", "gas-1"]

WORM_STRAIN_DICT = {"N2":"N2",
                    ### neuron ablation / RNAi
                    "ZD763":"ASJ-ablated",
                    "QD71":"AIY-ablated",
                    # "PY7505":"ASI-ablated",
                    # "RID(RNAi)::unc-31":"RID(RNAi)",
                    "RID(RNAi)_lite-1(wt)":"RID(RNAi)[OMG94]",
                    ### Antioxidant pathways
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
                    "msrA":"msra-1(tm1421)",
                    ### Neuropeptide pathways
                    "PS8997":"flp-1(sy1599)",
                    "VC2490":"flp-2(gk1039)+W07E11.1",
                    "VC2591":"flp-2(ok3351)",
                    "frpr-3":"frpr-3(ok3302)",
                    "frpr-18":"frpr-18(ok2698)",
                    "pdfr-1":"pdfr-1(ok3425)",
                    ### Neurotransmitters
                    "eat-4":"eat-4(n2474)",
                    ### Mitochondria
                    "FGC49":"nuo-6[FGC49]",
                    #"MQ1333":"nuo-6(qm200)",
                    #"nuo-6":"nuo-6(qm200)", # MQ1333, same as above (alternative name)
                    "MQ887":"isp-1(qm150)",
                    'isp-1':"isp-1(qm150)",
                    'TM1420':"sdha-2(tm1420)"}

WORM_GROUP_DICT = {'neuron_ablation':["N2",
                                      "ASJ-ablated",
                                      "AIY-ablated",
                                      "RID(RNAi)[OMG94]"],
                   'neuropeptides_&_neurotransmitters':["N2",
                                                        "pdfr-1(ok3425)",
                                                        "flp-1(sy1599)",
                                                        "flp-2(ok3351)",
                                                        "flp-2(gk1039)+W07E11.1",
                                                        "frpr-3(ok3302)",
                                                        "frpr-18(ok2698)", 
                                                        "eat-4(n2474)"],
                   'antioxidant':["N2",
                                  "clk-1(qm30)",
                                  "gas-1(fc21)",
                                  "msra-1(tm1421)",
                                  "prdx-2(gk169)",
                                  "sod-1",
                                  "sod-2",
                                  "sod-3",
                                  "sod-4",
                                  "sod-5",
                                  "sod-1; sod-2",
                                  "sod-1; sod-5",
                                  "sod-2; sod-3",
                                  "sod-3; sod-5",
                                  "sod-4; sod-5",
                                  "OE sod-1",
                                  "OE sod-2",
                                  "OE ctl-1+ctl-2+ctl-3"],
                   'mitochondria':["N2",
                                   "nuo-6[FGC49]",
                                   #"nuo-6(qm200)",
                                   "isp-1(qm150)",
                                   "sdha-2(tm1420)"]}

FIGSIZE_DICT = {'neuron_ablation':[12,10],
                'neuropeptides_&_neurotransmitters':[15,10],
                'antioxidant':[22,10],
                'mitochondria':[12,10]}

#%% Functions

def stats(metadata,
          features,
          group_by='treatment',
          control='BW',
          feat='speed_50th',
          save_dir=None,
          pvalue_threshold=0.05,
          fdr_method='fdr_bh'):
        
    assert all(metadata.index == features.index)
    features = features[[feat]]
    
    # check case-sensitivity
    assert len(metadata[group_by].unique()) == len(metadata[group_by].str.upper().unique())

    # print mean sample size
    sample_size = metadata.groupby(group_by).count()
    print("Mean sample size per %s: %d" % (group_by, int(sample_size.max(axis=1).mean())))

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

        # compile ANOVA results
        anova_results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
        anova_results.columns = ['stats','effect_size','pvals','reject']     
        anova_results['significance'] = sig_asterix(anova_results['pvals'])
        anova_results = anova_results.sort_values(by=['pvals'], ascending=True)

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

    # save results
    if save_dir is not None:
        ttest_path = Path(save_dir) / 't-test_results.csv'
        ttest_path.parent.mkdir(exist_ok=True, parents=True)
        ttest_results.to_csv(ttest_path, header=True, index=True)
    
    return anova_results, ttest_results


def main():
    
    metadata_combined_path = Path(SAVE_DIR) / 'metadata.csv'
    features_combined_path = Path(SAVE_DIR) / 'features.csv'
    
    # load metadata and features summaries
    metadata = pd.read_csv(metadata_combined_path, header=0, index_col=None, 
                           dtype={'comments':str})
    features = pd.read_csv(features_combined_path, header=0, index_col=None)

    # subset metadata results for bluelight videos only 
    bluelight_videos = [i for i in metadata['imgstore_name'] if 'bluelight' in i]
    metadata = metadata[metadata['imgstore_name'].isin(bluelight_videos)]
    
    # subset for final blue light window
    metadata = metadata[metadata['window']==5]
    
    # omit strains from OMIT_STRAINS_LIST and rows with missing 'worm_strain' data
    n_samples =  metadata.shape[0]
    metadata = metadata[~metadata['worm_strain'].isin(OMIT_STRAINS_LIST)]
    print("%d samples dropped (omitted strains)" % (n_samples - metadata.shape[0]))

    # map worm strain names + subset for mitochondrial worm mutants only
    metadata['worm_strain'] = metadata['worm_strain'].map(WORM_STRAIN_DICT)
    metadata = metadata[metadata['worm_strain'].isin(WORM_GROUP_DICT['antioxidant'])]
    
    # subset to remove paraquat treatment results
    metadata = metadata[metadata['drug_type']!='Paraquat']

    # reset index for features
    features = features[['speed_50th']].reindex(metadata.index)
    
    # aggregate worm strain, bacteria strain and drug type to form single treatment column
    metadata['treatment'] = metadata[['worm_strain','bacteria_strain','drug_type']
                                     ].astype(str).agg('-'.join, axis=1)
    metadata['treatment'] = [i.replace('-nan','') for i in metadata['treatment']]

    # save plot data to file
    plot_df = metadata[['worm_strain','bacteria_strain','treatment','date_yyyymmdd']
                       ].join(features).sort_values(by='treatment', ascending=True)
    plot_df.to_csv(Path(SAVE_DIR) / 'Fig3c_data.csv', header=True, index=False)

    # do stats - compare each mutant-treatment combination to N2-BW without paraquat
    control = 'N2-BW'
    anova_results, ttest_results = stats(metadata,
                                         features,
                                         group_by='treatment',
                                         control=control,
                                         feat='speed_50th',
                                         save_dir=Path(SAVE_DIR),
                                         pvalue_threshold=0.05,
                                         fdr_method='fdr_bh')

    # extract t-test pvals
    pvals = ttest_results[[c for c in ttest_results.columns if 'pval' in c]]
    pvals.columns = [c.replace('pvals_','') for c in pvals.columns]
        
    # boxplot (without paraquat)
    order = [w for w in WORM_GROUP_DICT['antioxidant'] if w in 
             metadata['worm_strain'].unique()]
    colour_dict = dict(zip(['BW','fepD'], sns.color_palette(palette='tab10', 
                                                            n_colors=2)))

    plt.close('all')
    sns.set_style('ticks')
    fig = plt.figure(figsize=FIGSIZE_DICT['antioxidant'])
    ax = fig.add_subplot(1,1,1)
    sns.boxplot(x='worm_strain',
                y='speed_50th',
                data=plot_df, 
                order=order,
                hue='bacteria_strain',
                hue_order=['BW','fepD'],
                dodge=True,
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
    date_lut = dict(zip(dates, sns.color_palette(palette="Greys", 
                                                 n_colors=len(dates))))
    for date in dates:
        date_df = plot_df[plot_df['date_yyyymmdd']==date]
        sns.stripplot(x='worm_strain',
                      y='speed_50th',
                      data=date_df,
                      order=order,
                      hue='bacteria_strain',
                      hue_order=['BW','fepD'],
                      dodge=True,
                      palette=[date_lut[date]] * plot_df['bacteria_strain'].nunique(),
                      color='dimgray',
                      marker=".",
                      edgecolor='k',
                      linewidth=0.3,
                      s=12) #facecolors="none"
    
    ax.axes.get_xaxis().get_label().set_visible(False) # remove y axis label
    ax.axes.set_xticklabels([l.get_text() for l in ax.axes.get_xticklabels()], 
                            fontsize=20, rotation=90)
    ax.tick_params(axis='x', which='major', pad=15)
    ax.axes.set_ylabel('Speed (µm s$^{-1}$)', fontsize=25, labelpad=25)                             
    plt.yticks(fontsize=20)
    plt.ylim(-40, 270)
            
    # scale y axis for annotations    
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled
    
    # add pvalues to plot
    for i, text in enumerate(ax.axes.get_xticklabels()):
        worm = text.get_text()
        if worm == 'N2':
            p = pvals.loc['speed_50th','N2-fepD']
            p_text = sig_asterix([p])[0]
            ax.text(i+0.2, 1.03, p_text, fontsize=20, ha='center', va='center', 
                    transform=trans)
        else:
            p1 = pvals.loc['speed_50th',worm+'-BW']
            p2 = pvals.loc['speed_50th',worm+'-fepD']
            p1_text = sig_asterix([p1])[0]
            p2_text = sig_asterix([p2])[0]
            ax.text(i-0.2, 1.03, p1_text, fontsize=20, ha='center', va='center', 
                    transform=trans)
            ax.text(i+0.2, 1.03, p2_text, fontsize=20, ha='center', va='center', 
                    transform=trans)                

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc='best', frameon=False, fontsize=15)
    boxplot_path = Path(SAVE_DIR) / 'worm_antioxidant_mutants.svg'
    boxplot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(boxplot_path, bbox_inches='tight', transparent=True)
            
    return

#%% Main

if __name__ == '__main__':
    main()
    