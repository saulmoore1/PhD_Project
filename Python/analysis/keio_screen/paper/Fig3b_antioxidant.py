#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: sm5911
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

SAVE_DIR = "/Users/sm5911/Documents/PhD_DLBG/Fig3b"

#%% Functions

def stats(metadata,
          features,
          group_by='treatment',
          control='BW',
          save_dir=None,
          feature_set=None,
          pvalue_threshold=0.05,
          fdr_method='fdr_bh'):
    
    # check case-sensitivity
    assert len(metadata[group_by].unique()) == len(metadata[group_by].str.upper().unique())
    
    if feature_set is not None:
        feature_set = [feature_set] if isinstance(feature_set, str) else feature_set
        assert isinstance(feature_set, list)
        assert(all(f in features.columns for f in feature_set))
    else:
        feature_set = features.columns.tolist()
        
    features = features[feature_set].reindex(metadata.index)

    # print mean sample size
    sample_size = metadata.groupby(group_by).count()
    print("Mean sample size of %s: %d" % (group_by, int(sample_size[sample_size.columns[-1]].mean())))
             
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
    
    # subset metadata results for bluelight videos only 
    bluelight_videos = [i for i in metadata['imgstore_name'] if 'bluelight' in i]
    metadata = metadata[metadata['imgstore_name'].isin(bluelight_videos)]

    # subset for window after final BL pulse
    metadata = metadata[metadata['window']==5]
    
    # subset for live bacteria only
    metadata = metadata[metadata['is_dead']=='N']
    
    # fill NaN values with 'None'
    metadata['drug_type'] = metadata['drug_type'].fillna('None')
    metadata['drug_imaging_plate_conc'] = metadata['drug_imaging_plate_conc'].fillna('None')
    
    # subset for antioxidant results only
    metadata = metadata[metadata['drug_type'].isin(['None','NAC','Vitamin C'])]
    features = features.reindex(metadata.index)
    
    # reindex features
    features = features[['speed_50th']].reindex(metadata.index)

    # aggregate antioxidant type and conc into single treatment column  
    antioxidant_cols = ['drug_type','drug_imaging_plate_conc']
    metadata['treatment'] = metadata[antioxidant_cols].astype(str).agg('-'.join, axis=1)
    metadata['treatment'] = [i.split('-None')[0] for i in metadata['treatment']]
    
    antioxidant_list = ['None'] + [i for i in sorted(metadata['treatment'].unique()) 
                                   if i != 'None']
    strain_list = sorted(metadata['food_type'].unique())
        
    # save plot data to file
    plot_df = metadata[['food_type','date_yyyymmdd','treatment']].join(features)
    plot_df.to_csv(Path(SAVE_DIR) / 'Fig3a_data.csv', header=True, index=False)
    
    # boxplot
    strain_lut = dict(zip(strain_list, sns.color_palette('tab10',len(strain_list))))
    
    plt.close('all')
    sns.set_style('ticks')
    fig = plt.figure(figsize=[12,6])
    ax = fig.add_subplot(1,1,1)
    sns.boxplot(x='treatment',
                y='speed_50th',
                data=plot_df, 
                order=antioxidant_list,
                hue='food_type',
                hue_order=strain_list,
                palette=strain_lut,
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
        sns.stripplot(x='treatment',
                      y='speed_50th',
                      data=date_df,
                      s=8,
                      order=antioxidant_list,
                      hue='food_type',
                      hue_order=strain_list,
                      dodge=True,
                      palette=[date_lut[date]] * len(strain_list),
                      color=None,
                      marker=".",
                      edgecolor='k',
                      linewidth=0.3) #facecolors="none"
    
    # scale y axis for annotations    
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled

    ax.axes.get_xaxis().get_label().set_visible(False) # remove x axis label
    ax.tick_params(axis='x', which='major', pad=15)
    plt.xticks(fontsize=15)
    ax.tick_params(axis='y', which='major', pad=15)
    plt.yticks(fontsize=20)
    ax.axes.set_ylabel('Speed (µm s$^{-1}$)', fontsize=25, labelpad=20)  
    plt.ylim(-20, 280)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc='upper right', frameon=False, fontsize=15)
    #plt.axhline(y=0, c='grey')

    # stats
    metadata['strain-treatment'] = metadata[['food_type','treatment']
                                            ].astype(str).agg('-'.join, axis=1)
    metadata['strain-treatment'] = [i.split('-None')[0] for i in metadata['strain-treatment']]
    ttest_results = stats(metadata,
                          features,
                          group_by='strain-treatment',
                          control='BW',
                          save_dir=Path(SAVE_DIR),
                          feature_set=['speed_50th'],
                          pvalue_threshold=0.05,
                          fdr_method='fdr_bh')
    pvals = ttest_results[[c for c in ttest_results.columns if 'pvals' in c]]
    pvals.columns = [c.split('pvals_')[-1] for c in pvals.columns]

    # add pvalues to plot - all treatments vs BW-live
    for i, text in enumerate(ax.axes.get_xticklabels()):
        treatment = text.get_text()
        if treatment == 'None':
            p = pvals.loc['speed_50th','fepD']
            p_text = sig_asterix([p])[0]
            ax.text(i+0.2, 1.03, p_text, fontsize=20, ha='center', va='center', transform=trans)
        else:
            p1 = pvals.loc['speed_50th','BW-'+treatment]
            p2 = pvals.loc['speed_50th','fepD-'+treatment]
            p1_text = sig_asterix([p1])[0]
            p2_text = sig_asterix([p2])[0]
            ax.text(i-0.2, 1.03, p1_text, fontsize=20, ha='center', va='center', transform=trans)
            ax.text(i+0.2, 1.03, p2_text, fontsize=20, ha='center', va='center', transform=trans)
            
    # save plot
    boxplot_path = Path(SAVE_DIR) / 'Fig3b_antioxidant.svg'
    plt.savefig(boxplot_path, bbox_inches='tight', transparent=True)    

    return
    
#%% Main

if __name__ == '__main__':
    main()
