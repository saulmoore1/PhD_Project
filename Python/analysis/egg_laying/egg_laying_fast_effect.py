#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Egg Laying - Keio Acute Effect Screen

Plots of number of eggs recorded on plates +1hr after picking 10 worms onto 60mm plates seeded 
with either BW background or fepD knockout mutant bacteria and recording without delay
(with bluelight stimulus delivered every 30 minutes)

@author: sm5911
@date: 20/01/2022

"""

#%% Imports
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import transforms

from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Keio_Fast_Effect"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Fast_Effect/egg_counting"
IMAGING_DATES = [20211102, 20211109] # 20211116
CONTROL_STRAIN = "BW"

#%% Functions

def get_egg_counts(metadata, masked_dir, regex='metadata_eggs.csv', group_name='/full_data'):
    """ Search MaskedVideos directory for egg counter GUI save files 
        and update metadata with egg count data for matched files
    """

    # find egg_counter GUI 'metadata_eggs.csv' files
    egg_files = list(masked_dir.rglob('metadata_eggs.csv'))
    
    # assemble paths to egg files
    metadata['egg_filepath'] = [masked_dir / i / 'metadata_eggs.csv' for i in 
                                metadata['imgstore_name']]
    assert all(f in metadata['egg_filepath'].unique() for f in egg_files)
    
    n_missing_egg_files = sum([f not in egg_files for f in metadata['egg_filepath'].unique()])
    if n_missing_egg_files > 0:
        print("\nWARNING: %d entries in metadata are missing egg count data!" % n_missing_egg_files)
   
    # get egg counts + append to metadata
    for eggfile in egg_files:
        egg_data = pd.read_csv(eggfile)
        egg_counts = egg_data.groupby('frame_number').count()['group_name']
        
        # find matching entry in metadata
        meta_rowidx = np.where(metadata['egg_filepath']==eggfile)[0]
        assert len(meta_rowidx) == 1
        
        # append egg counts for each frame analysed
        for frame in list(egg_counts.index):
            metadata.loc[meta_rowidx[0],'n_eggs_frame_{}'.format(frame)] = egg_counts[frame]
    
    return metadata

#%% Main

if __name__ == "__main__":   
    
    # load metadata
    metadata_path = Path(PROJECT_DIR) / "AuxiliaryFiles" / "metadata.csv"
    metadata = pd.read_csv(metadata_path, dtype={"comments":str})
    
    # subset for imaging dates
    metadata = metadata[metadata['date_yyyymmdd'].isin(IMAGING_DATES)]
        
    # get egg data
    masked_dir = Path(PROJECT_DIR) / "MaskedVideos"
    metadata = get_egg_counts(metadata, masked_dir)
            
    # calculate number of eggs laid in first hour on-food
    metadata['number_eggs_1hr'] = metadata['n_eggs_frame_12'] - metadata['n_eggs_frame_0']
    
    # compute mean/std number of eggs on each food
    eggs = metadata[['gene_name','number_eggs_1hr']]
    
    # drop NaN entries
    eggs = eggs.dropna(subset=['gene_name','number_eggs_1hr'])
    
    strain_list = [CONTROL_STRAIN] + [s for s in eggs['gene_name'].unique() if s != CONTROL_STRAIN]
    
    # 1. perform chi-sq tests to see if number of eggs laid is significantly different from control 
    #    for any strain
    
    # perform ANOVA (correct for multiple comparisons) - is there variation in egg count across strains?           
    stats, pvals, reject = univariate_tests(X=eggs[['number_eggs_1hr']], 
                                            y=eggs['gene_name'], 
                                            test='ANOVA',
                                            control=CONTROL_STRAIN,
                                            comparison_type='multiclass',
                                            multitest_correction='fdr_by',
                                            alpha=0.05,
                                            n_permutation_test=None) # 'all'

    # get effect sizes
    effect_sizes = get_effect_sizes(X=eggs[['number_eggs_1hr']], 
                                    y=eggs['gene_name'],
                                    control=CONTROL_STRAIN,
                                    effect_type=None,
                                    linked_test='ANOVA')
    
    # compile
    test_results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
    test_results.columns = ['stats','effect_size','pvals','reject']     
    test_results = test_results.sort_values(by=['pvals'], ascending=True) # rank pvals
    
    # save results
    anova_path = Path(SAVE_DIR) / 'ANOVA_egg_laying_variation_on_food.csv'
    test_results.to_csv(anova_path, header=True, index=True)
          
    # TODO: Chi-square tests should be performed here, not t-tests!
    # t-tests: is egg count different on any food vs control?
    print("Performing t-tests comparing each antioxidant treatment to None (pooled window data)")
    stats_t, pvals_t, reject_t = univariate_tests(X=eggs[['number_eggs_1hr']],
                                                  y=eggs['gene_name'],
                                                  test='t-test',
                                                  control=CONTROL_STRAIN,
                                                  comparison_type='binary_each_group',
                                                  multitest_correction='fdr_by',
                                                  alpha=0.05)
    effect_sizes_t =  get_effect_sizes(X=eggs[['number_eggs_1hr']], 
                                       y=eggs['gene_name'], 
                                       control=CONTROL_STRAIN,
                                       effect_type=None,
                                       linked_test='t-test')
        
    # compile + save t-test results
    stats_t.columns = ['stats_' + str(c) for c in stats_t.columns]
    pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
    reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
    effect_sizes_t.columns = ['effect_size_' + str(c) for c in effect_sizes_t.columns]
    ttest_results = pd.concat([stats_t, effect_sizes_t, pvals_t, reject_t], axis=1)
    ttest_save_path = Path(SAVE_DIR) / 't-test_egg_laying_on_food.csv'
    ttest_save_path.parent.mkdir(exist_ok=True, parents=True)
    ttest_results.to_csv(ttest_save_path, header=True, index=True)
      
    # Plot bar chart
    plt.close()
    fig, ax = plt.subplots(figsize=(14, 8))
    ax = sns.barplot(x="gene_name", y="number_eggs_1hr", order=strain_list, data=eggs, 
                     estimator=np.mean, dodge=False, ci=95, capsize=.1, palette='plasma')
    ax = sns.stripplot(x="gene_name", y="number_eggs_1hr", order=strain_list, data=eggs, 
                       color='k', alpha=0.7, size=4, dodge=False, ax=ax)
    n_labs = len(eggs['gene_name'].unique())
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:n_labs], labels[:n_labs], fontsize=15, frameon=False, loc='upper right')
        
    # load t-test results + annotate p-values on plot
    for ii, strain in enumerate(strain_list):
        if strain == CONTROL_STRAIN:
            continue
        p = pvals_t.loc["number_eggs_1hr", ('pvals_' + strain)]
        text = ax.get_xticklabels()[ii]
        assert text.get_text() == strain
        p_text = 'P<0.001' if p < 0.001 else 'P=%.3f' % p
        #y = (y_bar[antiox] + 2 * IQR[antiox]) if scale_outliers_box else plot_df[feature].max()
        #h = (max(IQR) / 10) if scale_outliers_box else (y - plot_df[feature].min()) / 50
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        plt.plot([ii-.3, ii-.3, ii+.3, ii+.3], [0.99, 0.99, 0.99, 0.99], lw=1.5, c='k', transform=trans)
        ax.text(ii, 1.01, p_text, fontsize=9, ha='center', va='bottom', transform=trans, rotation=0)
        
    ax.set_xticks(range(len(strain_list)+1))
    ax.set_xlabel('Strain', fontsize=15, labelpad=10)
    ax.set_ylabel('Number of eggs after 1hr on food', fontsize=15, labelpad=10)
    
    # save plot
    fig_savepath = Path(SAVE_DIR) / 'egg_laying_boxplot.png'
    fig_savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_savepath, dpi=300)

    # TODO: Use univariate_tests function with chi_sq test to compare n_eggs between fepD vs BW
     