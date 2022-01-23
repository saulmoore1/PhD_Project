#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Egg Laying - Keio Acute Antioxidant Screen

Plots of number of eggs recorded on plates +24hrs after picking 10 worms onto 60mm plates seeded 
with either BW background or fepD knockout mutant bacteria, in combination with exogenous delivery
of antioxidants (with bluelight stimulus delivered every 5 minutes)

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

from statistical_testing.stats_helper import pairwise_ttest

from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
#from tierpsytools.drug_screenings.filter_compounds import compounds_with_low_effect_univariate

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Keio_Acute_Rescue"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Acute_Rescue/egg_counting"
CONTROL_STRAIN = "BW"
CONTROL_ANTIOXIDANT = "None"

#%% Main

if __name__ == "__main__":
    
    # Load metadata
    metadata_path = Path(PROJECT_DIR) / "AuxiliaryFiles" / "metadata.csv"
    metadata = pd.read_csv(metadata_path, dtype={"comments":str})
    
    # Extract egg count + compute average for each strain/antioxidant treatment combination
    eggs = metadata[['gene_name','antioxidant','number_eggs_24hrs']]
    # mean_eggs = eggs.groupby(['gene_name','antioxidant']).mean()
    # std_eggs = eggs.groupby(['gene_name','antioxidant']).std()
    
    # drop NaN entries
    eggs = eggs.dropna(subset=['gene_name','antioxidant','number_eggs_24hrs'])
    
    strain_list = [CONTROL_STRAIN] + [s for s in eggs['gene_name'].unique() if s != CONTROL_STRAIN]
    antioxidant_list = [CONTROL_ANTIOXIDANT] + [a for a in eggs['antioxidant'].unique() if
                                                a != CONTROL_ANTIOXIDANT]
    
    # 1. perform chi-sq tests to see if number of eggs laid is significantly different from control
    
    # perform ANOVA - is there variation in egg laying across antioxidants? (pooled strain data)
    stats, pvals, reject = univariate_tests(X=eggs[['number_eggs_24hrs']], 
                                            y=eggs['antioxidant'], 
                                            test='ANOVA',
                                            control=CONTROL_ANTIOXIDANT,
                                            comparison_type='multiclass',
                                            multitest_correction='fdr_by',
                                            alpha=0.05,
                                            n_permutation_test=None) # 'all'

    # get effect sizes
    effect_sizes = get_effect_sizes(X=eggs[['number_eggs_24hrs']], 
                                    y=eggs['antioxidant'],
                                    control=CONTROL_ANTIOXIDANT,
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
    # Is there any difference in egg laying between strains, after accounting for antioxidant treatment?
    # Do not have enough data for LMM!
    # print("Performing t-tests comparing each antioxidant treatment to None (pooled window data)")
    # signif_effect_strains, low_effect_strains, sig_dict = compounds_with_low_effect_univariate(
    #     feat=eggs[['number_eggs_24hrs']],
    #     drug_name=eggs['gene_name'],
    #     random_effect=eggs['antioxidant'],
    #     control=CONTROL_STRAIN,
    #     test='ANOVA',
    #     comparison_type='multiclass',
    #     multitest_method='fdr_by',
    #     fdr=0.05)
    
    # Is there a difference in egg laying between strain vs control for any antioxidant
    print("\nPairwise tests comparing number of eggs laid under the different antioxidant treatments")

    control_eggs = eggs[eggs['gene_name']==CONTROL_STRAIN]
    strain_eggs = eggs[eggs['gene_name']!=CONTROL_STRAIN]

    stats, pvals, reject = pairwise_ttest(control_eggs, 
                                          strain_eggs, 
                                          feature_list=['number_eggs_24hrs'], 
                                          group_by='antioxidant', 
                                          fdr_method='fdr_by',
                                          fdr=0.05)
 
    # compile table of results
    stats.columns = ['stats_' + str(c) for c in stats.columns]
    pvals.columns = ['pvals_' + str(c) for c in pvals.columns]
    reject.columns = ['reject_' + str(c) for c in reject.columns]
    test_results = pd.concat([stats, pvals, reject], axis=1)
    
    # save results
    ttest_strain_path = Path(SAVE_DIR) / 'pairwise_ttests' / 'antioxidant_results.csv'
    ttest_strain_path.parent.mkdir(parents=True, exist_ok=True)
    test_results.to_csv(ttest_strain_path, header=True, index=True)
        
    # Plot bar chart + save
    plt.close()
    fig, ax = plt.subplots(figsize=(14, 8))
    ax = sns.barplot(x="gene_name", y="number_eggs_24hrs", hue="antioxidant", 
                     order=strain_list, hue_order=antioxidant_list, data=eggs, 
                     estimator=np.mean, dodge=True, ci=95, capsize=.1, palette='plasma')
    ax = sns.swarmplot(x="gene_name", y="number_eggs_24hrs", hue='antioxidant', 
                       order=strain_list, hue_order=antioxidant_list, data=eggs,
                       color='k', alpha=0.7, size=4, dodge=True)
    n_labs = len(eggs['antioxidant'].unique())
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[n_labs:], labels[n_labs:], fontsize=15, frameon=False, loc=(1.01,0.8))
    ax.set_xlabel("")
    ax.set_ylabel("Number of eggs after 24hrs on food", fontsize=15, labelpad=10)
    
    # annotate p-values
    for ii, antiox in enumerate(antioxidant_list):
        p = pvals.loc['number_eggs_24hrs', ('pvals_' + antiox)]
        p_text = 'P < 0.001' if p < 0.001 else 'P = %.3f' % p
        offset = .32
        scalar = .16
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        plt.plot([1-offset+(scalar*ii)-.05, 1-offset+(scalar*ii)-.05, 
                  1-offset+(scalar*ii)+.05, 1-offset+(scalar*ii)+.05], 
                 [0.99, 0.99, 0.99, 0.99], #[y+h, y+2*h, y+2*h, y+h], 
                 lw=1.5, c='k', transform=trans)
        ax.text(1-offset+(scalar*ii), 1.01, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
    
    plt.subplots_adjust(right=0.85)
    # save plot
    save_path = Path(SAVE_DIR) / "eggs_after_24hrs_on_food.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    
    

    # TODO: Use univariate_tests function with chi_sq test to compare n_eggs between fepD vs BW