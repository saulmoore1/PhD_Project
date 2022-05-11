#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keio E. coli fepD cell supernatant and lysate experiments:
    
Adding fepD culture to BW lawns
- supernatant vs lysate
- solid vs liquid culture
- live vs dead bacteria

@author: sm5911
@date: 13/04/2022

"""

#%% Imports

import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import transforms
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patches as mpatches
from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from visualisation.plotting_helper import sig_asterix

#%% Globals

PROJECT_DIR = '/Volumes/hermes$/Keio_Supernatants_6WP'
SAVE_DIR = '/Users/sm5911/Documents/Keio_Supernatants'
IMAGING_DATES = ['20220412']
N_WELLS = 6

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
MIN_NSKEL_PER_VIDEO = None
MIN_NSKEL_SUM = 500

FEATURE = 'motion_mode_forward_fraction'

WINDOW_DICT_SECONDS = {0:(1790,1800), 1:(1805,1815), 2:(1815,1825),
                       3:(1850,1860), 4:(1865,1875), 5:(1875,1885),
                       6:(1910,1920), 7:(1925,1935), 8:(1935,1945)}

WINDOW_DICT_STIM_TYPE = {0:'prestim\n(30min)',1:'bluelight\n(30min)',2:'poststim\n(30min)',
                         3:'prestim\n(31min)',4:'bluelight\n(31min)',5:'poststim\n(31min)',
                         6:'prestim\n(32min)',7:'bluelight\n(32min)',8:'poststim\n(32min)'}

WINDOW_NUMBER = 2

drug_type_list = ['none','fepD']
extract_type_list = ['none','supernatant','lysate']
culture_type_list = ['none','liquid','solid']
is_dead_list = ['N','Y']
solvent_list = ['none','PBS','DMSO','PBS/DMSO','NGM/DMSO']
killing_method_list = ['none', 'ultraviolet', 'sonication', 'ultraviolet/sonication',
                       'methanol/sonication', 'ultraviolet/methanol/sonication']
treatment_list = ['none-none-none','fepD-lysate-solid','fepD-lysate-liquid',  
                  'fepD-supernatant-solid', 'fepD-supernatant-liquid']

# treatment_order = ['none-N-none-none', 'none-Y-none-none', 
#                    'fepD-N-solid-lysate', 'fepD-Y-solid-lysate',
#                    'fepD-N-liquid-lysate', 'fepD-Y-liquid-lysate',
#                    'fepD-N-solid-supernatant', 'fepD-Y-solid-supernatant',
#                    'fepD-N-liquid-supernatant', 'fepD-Y-liquid-supernatant']
    
#%% Functions

def do_stats(metadata, 
             features, 
             group_by,
             control,
             save_dir,
             feat=FEATURE, 
             pvalue_threshold=0.05, 
             fdr_method='fdr_by'):
    
    """ ANOVA / t-tests comparing a single feature of worm behaviour across treatments vs control 
        
        Parameters
        ----------
        metadata : pandas.DataFrame
        
        features : pandas.DataFrame
            Dataframe of compiled window summaries
            
        group_by : str
            Column name of variable containing control and other groups to compare, eg. 'gene_name'
            
        control : str
            Name of control group in 'group_by' column in metadata
            
        save_dir : str
            Path to directory to save results files
            
        feat : str
            Feature (or list of features) to test
        
        pvalue_threshold : float
            P-value significance threshold
            
        fdr_method : str
            Multiple testing correction method to use
    """
    
    from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes    
    from statistical_testing.perform_keio_stats import df_summary_stats
    from write_data.write import write_list_to_file

    # categorical variables to investigate: 'gene_name' and 'window'
    print("\nInvestigating variation in fraction of worms paused with respect to '%s'" % group_by)    

    groups = metadata[group_by].unique()
    test_groups = list([s for s in groups if s != control])    
    
    # check there will be no errors due to case-sensitivity
    assert len(groups) == len(metadata[group_by].str.upper().unique())
    
    # print mean sample size
    sample_size = df_summary_stats(metadata, columns=[group_by, 'window'])
    print("Mean sample size of %s/window: %d" % (group_by, int(sample_size['n_samples'].mean())))


    ### perform ANOVA - is there variation in worm motion mode among solvents used?
        
    fset = []
    if len(groups) > 2:        
        anova_path = Path(save_dir) / '{}_ANOVA_results.csv'.format(group_by)
        anova_path.parent.mkdir(parents=True, exist_ok=True)

        stats, pvals, reject = univariate_tests(X=features[feat], 
                                                y=metadata[group_by], 
                                                control=control, 
                                                test='ANOVA',
                                                comparison_type='multiclass',
                                                multitest_correction=fdr_method,
                                                alpha=pvalue_threshold,
                                                n_permutation_test=None)

        # get effect sizes
        effect_sizes = get_effect_sizes(X=features[feat], 
                                        y=metadata[group_by],
                                        control=control,
                                        effect_type=None,
                                        linked_test='ANOVA')

        # compile + save results
        test_results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
        test_results.columns = ['stats','effect_size','pvals','reject']     
        test_results['significance'] = sig_asterix(test_results['pvals'])
        test_results = test_results.sort_values(by=['pvals'], ascending=True) # rank by p-value
        test_results.to_csv(anova_path, header=True, index=True)

        # use reject mask to find significant feature set
        fset = pvals.loc[reject['ANOVA']].sort_values(by='ANOVA', ascending=True).index.to_list()

        if len(fset) > 0:
            print("%d significant features found by ANOVA for '%s' (P<%.2f, %s)" %\
                  (len(fset), group_by, pvalue_threshold, fdr_method))
            anova_sigfeats_path = anova_path.parent / 'ANOVA_sigfeats.txt'
            write_list_to_file(fset, anova_sigfeats_path)


    ### perform t-tests - compare each group to control            
                    
    if len(groups) == 2 or len(fset) > 0:
        
        ttest_strain_path = Path(save_dir) / '{}_ttest_results.csv'.format(group_by)
        ttest_strain_path.parent.mkdir(parents=True, exist_ok=True)

        stats, pvals, reject = univariate_tests(X=features[feat], 
                                                y=metadata[group_by], 
                                                control=control, 
                                                test='t-test',
                                                comparison_type='binary_each_group',
                                                multitest_correction=fdr_method,
                                                alpha=pvalue_threshold,
                                                n_permutation_test=None)

        # get effect sizes
        effect_sizes = get_effect_sizes(X=features[feat], 
                                        y=metadata[group_by],
                                        control=control,
                                        effect_type=None,
                                        linked_test='t-test')
     
        # compile table of results
        stats.columns = ['stats_' + str(c) for c in stats.columns]
        pvals.columns = ['pvals_' + str(c) for c in pvals.columns]
        reject.columns = ['reject_' + str(c) for c in reject.columns]
        effect_sizes.columns = ['effect_size_' + str(c) for c in effect_sizes.columns]
        test_results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
        
        # save results
        test_results.to_csv(ttest_strain_path, header=True, index=True)
        
        for group in test_groups:
            print("%s difference in '%s' between %s vs %s (t-test, P=%.3f, %s)" %\
                  (("SIGNIFICANT" if reject.loc[feat, 'reject_{}'.format(group)] else "No"), 
                  feat, group, control, pvals.loc[feat, 'pvals_{}'.format(group)], fdr_method))
        
    return

def supernatants_stats(metadata, 
                       features, 
                       save_dir,
                       window=2,
                       feature=FEATURE,
                       pvalue_threshold=0.05,
                       fdr_method='fdr_by'):
    """ T-tests comparing each of the following for BW vs BW+fepD:
        - fepD cell lysate vs supernatant
        - extracted from solid vs liquid media
        - added to live vs UV-killed BW25113 control bacteria (BW)
    """
    
    assert metadata.shape[0] == features.shape[0]
    
    window_meta = metadata.query("window==@window")
    
    control_meta = window_meta.query("drug_type=='none' and is_dead=='N' and solvent=='none'")

    ### Compare each treatment to BW control
    
    # TODO: Correct p-values for multiple t-test comparisons
    
    # live BW + fepD solid lysate
    fepD_live_solid_lysate_meta = window_meta.query("culture_type=='solid' and " +
                                                    "cell_extract_type=='lysate' and " +
                                                    "killing_method=='sonication' and " +
                                                    "is_dead=='N'")
    test_meta = pd.concat([control_meta, fepD_live_solid_lysate_meta], axis=0)
    do_stats(metadata=test_meta, 
             features=features.reindex(test_meta.index), 
             group_by='drug_type',
             control='none',
             save_dir=save_dir / 'fepD_lysate_solid_on_live_BW',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)
    
    # live BW + fepD solid supernatant
    fepD_live_solid_supernatant_meta = window_meta.query("culture_type=='solid' and " +
                                                         "cell_extract_type=='supernatant' and " +
                                                         "is_dead=='N'")
    
    test_meta = pd.concat([control_meta, fepD_live_solid_supernatant_meta], axis=0)
    do_stats(metadata=test_meta, 
             features=features.reindex(test_meta.index), 
             group_by='drug_type',
             control='none',
             save_dir=save_dir / 'fepD_supernatant_solid_on_live_BW',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)
    
    # live BW + fepD liquid lysate 
    fepD_live_liquid_lysate_meta = window_meta.query("culture_type=='liquid' and " +
                                                     "cell_extract_type=='lysate' and " +
                                                     "killing_method=='sonication' and " +
                                                     "is_dead=='N'")
    test_meta = pd.concat([control_meta, fepD_live_liquid_lysate_meta], axis=0)
    do_stats(metadata=test_meta, 
             features=features.reindex(test_meta.index), 
             group_by='drug_type',
             control='none',
             save_dir=save_dir / 'fepD_lysate_liquid_on_live_BW',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)
    
    # live BW + fepD liquid supernatant
    fepD_live_liquid_supernatant_meta = window_meta.query("culture_type=='liquid' and " +
                                                          "cell_extract_type=='supernatant' and " +
                                                          "is_dead=='N'")
    
    test_meta = pd.concat([control_meta, fepD_live_liquid_supernatant_meta], axis=0)
    do_stats(metadata=test_meta, 
             features=features.reindex(test_meta.index), 
             group_by='drug_type',
             control='none',
             save_dir=save_dir / 'fepD_supernatant_liquid_on_live_BW',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)   
    
    # dead BW + fepD solid lysate
    fepD_dead_solid_lysate_meta = window_meta.query("culture_type=='solid' and " +
                                                    "cell_extract_type=='lysate' and " +
                                                    "killing_method=='ultraviolet/sonication' and " +
                                                    "is_dead=='Y'")
    test_meta = pd.concat([control_meta, fepD_dead_solid_lysate_meta], axis=0)
    do_stats(metadata=test_meta, 
             features=features.reindex(test_meta.index), 
             group_by='drug_type',
             control='none',
             save_dir=save_dir / 'fepD_lysate_solid_on_dead_BW',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)
    
    # dead BW + fepD solid supernatant
    fepD_dead_solid_supernatant_meta = window_meta.query("culture_type=='solid' and " +
                                                         "cell_extract_type=='supernatant' and " +
                                                         "is_dead=='Y'")
    
    test_meta = pd.concat([control_meta, fepD_dead_solid_supernatant_meta], axis=0)
    do_stats(metadata=test_meta, 
             features=features.reindex(test_meta.index), 
             group_by='drug_type',
             control='none',
             save_dir=save_dir / 'fepD_supernatant_solid_on_dead_BW',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)
    
    # dead BW + fepD liquid lysate 
    fepD_dead_liquid_lysate_meta = window_meta.query("culture_type=='liquid' and " +
                                                     "cell_extract_type=='lysate' and " +
                                                     "killing_method=='ultraviolet/sonication' and " +
                                                     "is_dead=='Y'")
    test_meta = pd.concat([control_meta, fepD_dead_liquid_lysate_meta], axis=0)
    do_stats(metadata=test_meta, 
             features=features.reindex(test_meta.index), 
             group_by='drug_type',
             control='none',
             save_dir=save_dir / 'fepD_lysate_liquid_on_dead_BW',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)
    
    # dead BW + fepD liquid supernatant
    fepD_dead_liquid_supernatant_meta = window_meta.query("culture_type=='liquid' and " +
                                                          "cell_extract_type=='supernatant' and " +
                                                          "is_dead=='Y'")
    
    test_meta = pd.concat([control_meta, fepD_dead_liquid_supernatant_meta], axis=0)
    do_stats(metadata=test_meta, 
             features=features.reindex(test_meta.index), 
             group_by='drug_type',
             control='none',
             save_dir=save_dir / 'fepD_supernatant_liquid_on_dead_BW',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)   
 
    
    ### SOLVENT: is there a difference between solvents used on BW control?
    
    BW_solvent_meta = window_meta.query("drug_type=='none' and is_dead=='N'")
    do_stats(metadata=BW_solvent_meta,
             features=features.reindex(BW_solvent_meta.index),
             group_by='solvent',
             control='none',
             save_dir=save_dir / 'BW_solvents',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)
    
    # Were worms on PBS/DMSO significantly different from worms on just PBS? 
    BW_PBS_meta = BW_solvent_meta[['PBS' in s for s in BW_solvent_meta['solvent']]]
    do_stats(metadata=BW_PBS_meta,
             features=features.reindex(BW_PBS_meta.index),
             group_by='solvent',
             control='PBS',
             save_dir=save_dir / 'BW_solvents' / 'PBS vs PBS-DMSO',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)
    
    
    ### UV KILLING: BW control - dead vs alive
    
    BW_UV_meta = window_meta.query("drug_type=='none' and solvent=='none'")
    do_stats(metadata=BW_UV_meta,
             features=features.reindex(BW_UV_meta.index),
             group_by='is_dead',
             control='N',
             save_dir=save_dir / 'BW_UV_killed',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)
    

    ### UV KILLING: fepD added to live vs dead BW

    # fepD - supernatant vs lysate, from solid vs liquid culture, on dead vs live BW
    fepD_meta = metadata.query("drug_type=='fepD'")
    
    # fepD solid lysate on live vs dead BW
    fepD_solid_lysate_meta = fepD_meta.query("culture_type=='solid' and " +
                                             "cell_extract_type=='lysate'")
    do_stats(metadata=fepD_solid_lysate_meta,
             features=features.reindex(fepD_solid_lysate_meta.index),
             group_by='is_dead',
             control='N',
             save_dir=save_dir / 'live_vs_dead_BW' / 'fepD_solid_lysate',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)
    
    # fepD liquid lysate on live vs dead BW
    fepD_liquid_lysate_meta = fepD_meta.query("culture_type=='liquid' and " +
                                              "cell_extract_type=='lysate'")
    do_stats(metadata=fepD_liquid_lysate_meta,
             features=features.reindex(fepD_liquid_lysate_meta.index),
             group_by='is_dead',
             control='N',
             save_dir=save_dir / 'live_vs_dead_BW' / 'fepD_liquid_lysate',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)
    
    # fepD solid supernatant on live vs dead BW
    fepD_solid_supernatant_meta = fepD_meta.query("culture_type=='solid' and " +
                                                  "cell_extract_type=='supernatant'")
    do_stats(metadata=fepD_solid_supernatant_meta,
             features=features.reindex(fepD_solid_supernatant_meta.index),
             group_by='is_dead',
             control='N',
             save_dir=save_dir / 'live_vs_dead_BW' / 'fepD_solid_supernatant',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)
    
    # fepD liquid supernatant on live vs dead BW
    fepD_liquid_supernatant_meta = fepD_meta.query("culture_type=='liquid' and " +
                                                   "cell_extract_type=='supernatant'")
    do_stats(metadata=fepD_liquid_supernatant_meta,
             features=features.reindex(fepD_liquid_supernatant_meta.index),
             group_by='is_dead',
             control='N',
             save_dir=save_dir / 'live_vs_dead_BW' / 'fepD_liquid_supernatant',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method) 
    
    
    ### SUPERNATANT vs LYSATE - fepD added to BW lawns
    
    # fepD supernatant vs lysate (from solid culture added to live BW)
    fepD_live_solid_meta = fepD_meta.query("is_dead=='N' and culture_type=='solid'")
    do_stats(metadata=fepD_live_solid_meta,
             features=features.reindex(fepD_live_solid_meta.index),
             group_by='cell_extract_type',
             control='lysate',
             save_dir=save_dir / 'fepD_supernatant_vs_lysate' / 'solid_culture_on_live_BW',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)
        
    # fepD supernatant vs lysate (from liquid culture added to live BW)
    fepD_live_liquid_meta = fepD_meta.query("is_dead=='N' and culture_type=='liquid'")
    do_stats(metadata=fepD_live_liquid_meta,
             features=features.reindex(fepD_live_liquid_meta.index),
             group_by='cell_extract_type',
             control='lysate',
             save_dir=save_dir / 'fepD_supernatant_vs_lysate' / 'liquid_culture_on_live_BW',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)
    
    # fepD supernatant vs lysate (from solid culture added to dead BW)
    fepD_dead_solid_meta = fepD_meta.query("is_dead=='Y' and culture_type=='solid'")
    do_stats(metadata=fepD_dead_solid_meta,
             features=features.reindex(fepD_dead_solid_meta.index),
             group_by='cell_extract_type',
             control='lysate',
             save_dir=save_dir / 'fepD_supernatant_vs_lysate' / 'solid_culture_on_dead_BW',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)
    
    # fepD supernatant vs lysate (from liquid culture added to dead BW)
    fepD_dead_liquid_meta = fepD_meta.query("is_dead=='Y' and culture_type=='liquid'")
    do_stats(metadata=fepD_dead_liquid_meta,
             features=features.reindex(fepD_dead_liquid_meta.index),
             group_by='cell_extract_type',
             control='lysate',
             save_dir=save_dir / 'fepD_supernatant_vs_lysate' / 'liquid_culture_on_dead_BW',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)
    
    
    ### SOLID vs LIQUID CULTURE - fepD extracted from either O/N liquid culture or seeded lawns
    
    # fepD solid vs liquid culture (lysate on live BW)
    fepD_live_lysate_meta = fepD_meta.query("is_dead=='N' and cell_extract_type=='lysate'")
    do_stats(metadata=fepD_live_lysate_meta,
             features=features.reindex(fepD_live_lysate_meta.index),
             group_by='culture_type',
             control='solid',
             save_dir=save_dir / 'fepD_solid_vs_liquid_culture' / 'lysate_on_live_BW',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)
    
    # fepD solid vs liquid culture (supernatant on live BW)
    fepD_live_supernatant_meta = fepD_meta.query("is_dead=='N' and " +
                                                  "cell_extract_type=='supernatant'")
    do_stats(metadata=fepD_live_supernatant_meta,
             features=features.reindex(fepD_live_supernatant_meta.index),
             group_by='culture_type',
             control='solid',
             save_dir=save_dir / 'fepD_solid_vs_liquid_culture' / 'supernatant_on_live_BW',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)
    
    # fepD solid vs liquid culture (lysate on dead BW)
    fepD_dead_lysate_meta = fepD_meta.query("is_dead=='Y' and cell_extract_type=='lysate'")
    do_stats(metadata=fepD_dead_lysate_meta,
             features=features.reindex(fepD_dead_lysate_meta.index),
             group_by='culture_type',
             control='solid',
             save_dir=save_dir / 'fepD_solid_vs_liquid_culture' / 'lysate_on_dead_BW',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)

    # fepD solid vs liquid culture (supernatant on dead BW)
    fepD_dead_supernatant_meta = fepD_meta.query("is_dead=='Y' and " +
                                                 "cell_extract_type=='supernatant'")
    do_stats(metadata=fepD_dead_supernatant_meta,
             features=features.reindex(fepD_dead_supernatant_meta.index),
             group_by='culture_type',
             control='solid',
             save_dir=save_dir / 'fepD_solid_vs_liquid_culture' / 'supernatant_on_dead_BW',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)

    
    ### KILLING METHOD: Effect of use of methanol in addition to UV/sonication for killing fepD lysate

    # fepD lysate added to live BW - sonication vs sonication/methanol
    do_stats(metadata=fepD_live_lysate_meta,
             features=features.reindex(fepD_live_lysate_meta.index),
             group_by='killing_method',
             control='sonication',
             save_dir=save_dir / 'fepD_killing_method' / 'liquid_lysate_on_live_BW',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)
        
    # fepD lysate added to dead BW - UV/sonication vs methanol/UV/sonication
    do_stats(metadata=fepD_dead_lysate_meta,
             features=features.reindex(fepD_dead_lysate_meta.index),
             group_by='killing_method',
             control='ultraviolet/sonication',
             save_dir=save_dir / 'fepD_killing_method' / 'lysate_on_dead_BW',
             feat=feature,
             pvalue_threshold=pvalue_threshold,
             fdr_method=fdr_method)
   
    return

def supernatants_plots(metadata, 
                       features,
                       plot_dir,
                       stats_dir,
                       window=2,
                       feature=FEATURE,
                       pvalue_threshold=0.05):
    
    assert metadata.shape[0] == features.shape[0]
        
    window_meta = metadata.query("window==@window")
    window_meta['treatment'] = window_meta[['drug_type', 'cell_extract_type', 'culture_type']
                                           ].agg('-'.join, axis=1)
    
    # boxplots for live BW vs fepD supernatant/lysate from solid/liquid culture added to live BW
    live_meta = window_meta.query("is_dead=='N'")
    plot_df = live_meta.join(features.reindex(live_meta.index))
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12,8))
    sns.boxplot(x='treatment', y=feature, order=treatment_list, ax=ax, data=plot_df,
                palette='plasma', showfliers=False) 
                #hue='is_dead', hue_order=is_dead_list, dodge=True
    sns.stripplot(x='treatment', y=feature, order=treatment_list, ax=ax, data=plot_df,
                  s=5, marker='D', color='k')
    ax.set_xlabel('')
    ax.set_ylabel(feature.replace('_',' '), fontsize=12, labelpad=10)
    ax.set_title('Addition of fepD lysate or supernatant (from solid or liquid culture) to live BW', 
                 pad=30, fontsize=18)
    # annotate p-values - load t-test results for each treatment vs BW control
    for ii, treatment in enumerate(treatment_list):
        if treatment == 'none-none-none':
            continue
        ttest_path = stats_dir / (treatment.replace('-','_') + '_on_live_BW') /\
            'drug_type_ttest_results.csv'
        ttest_df = pd.read_csv(ttest_path, index_col=0)
        p = ttest_df.loc[feature, 'pvals_fepD']
        text = ax.get_xticklabels()[ii]
        assert text.get_text() == treatment
        p_text = '***\nP < 0.001' if p < 0.001 else sig_asterix([p])[0] + '\nP = %.3f' % p
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        # plt.plot([ii-.2,ii-.2,ii+.2,ii+.2],[0.98,0.99,0.99,0.98],lw=1.5,c='k',transform=trans)
        ax.text(ii, 0.97, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
    save_path = Path(plot_dir) / 'fepD_added_to_live_BW.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    

    # boxplots for dead BW vs fepD supernatant/lysate from solid/liquid culture added to dead BW
    dead_meta = window_meta.query("is_dead=='Y'")
    plot_df = dead_meta.join(features.reindex(dead_meta.index))
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12,8))
    sns.boxplot(x='treatment', y=feature, order=treatment_list, ax=ax, data=plot_df,
                palette='plasma', showfliers=False) 
                #hue='is_dead', hue_order=is_dead_list, dodge=True
    sns.stripplot(x='treatment', y=feature, order=treatment_list, ax=ax, data=plot_df,
                  s=5, marker='D', color='k')
    ax.set_xlabel('')
    ax.set_ylabel(feature.replace('_',' '), fontsize=12, labelpad=10)
    ax.set_title('Addition of fepD lysate or supernatant (from solid or liquid culture) to dead BW', 
                 pad=30, fontsize=18)
    # annotate p-values - load t-test results for each treatment vs BW control
    for ii, treatment in enumerate(treatment_list):
        if treatment == 'none-none-none':
            continue
        ttest_path = stats_dir / (treatment.replace('-','_') + '_on_dead_BW') /\
            'drug_type_ttest_results.csv'
        ttest_df = pd.read_csv(ttest_path, index_col=0)
        p = ttest_df.loc[feature, 'pvals_fepD']
        text = ax.get_xticklabels()[ii]
        assert text.get_text() == treatment
        p_text = '***\nP < 0.001' if p < 0.001 else sig_asterix([p])[0] + '\nP = %.3f' % p
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        # plt.plot([ii-.2,ii-.2,ii+.2,ii+.2],[0.98,0.99,0.99,0.98],lw=1.5,c='k',transform=trans)
        ax.text(ii, 0.97, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
    save_path = Path(plot_dir) / 'fepD_added_to_dead_BW.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)


    # SIGNIFICANT difference between lysate vs supernatant
    # No difference whether from solid vs liquid culture
    
    
    ### SOLVENTS
    
    # Boxplots of differences in effect of solvent used on BW control (no fepD added)
    BW_live_solvent_meta = window_meta.query("drug_type=='none' and is_dead=='N'")
    plot_df = BW_live_solvent_meta.join(features.reindex(BW_live_solvent_meta.index))
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10,8))
    sns.boxplot(x='solvent', y=feature, order=solvent_list, ax=ax, data=plot_df, 
                palette='plasma', showfliers=False)
    sns.stripplot(x='solvent', y=feature, order=solvent_list, ax=ax, data=plot_df, 
                  s=5, marker='D', color='k')
    ax.set_xlabel('')
    ax.set_ylabel(feature.replace('_',' '), fontsize=12, labelpad=10)
    ax.set_title('Effect of solvent added to BW control', pad=30, fontsize=18)
    # load t-test results for each solvent vs 'none'
    ttest_path = Path(stats_dir) / 'BW_solvents' / 'solvent_ttest_results.csv'
    ttest_df = pd.read_csv(ttest_path, index_col=0)
    pvals = ttest_df[[c for c in ttest_df.columns if 'pvals_' in c]]
    # annotate p-values
    for ii, solvent in enumerate(solvent_list):
        if solvent == 'none':
            continue
        p = pvals.loc[feature, 'pvals_' + solvent]
        text = ax.get_xticklabels()[ii]
        assert text.get_text() == solvent
        p_text = '***\nP < 0.001' if p < 0.001 else sig_asterix([p])[0] + '\nP = %.3f' % p
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        # plt.plot([ii-.2,ii-.2,ii+.2,ii+.2],[0.98,0.99,0.99,0.98],lw=1.5,c='k',transform=trans)
        ax.text(ii, 0.97, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
    save_path = Path(plot_dir) / 'BW_control' / 'effect_of_solvent_used.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    
    
    ### UV KILLING - No difference whether alive vs dead for BW control (no fepD added)
    
    # Boxplots of differences in worm motion mode on live vs dead BW control bacteria
    BW_UV_meta = window_meta.query("drug_type=='none' and solvent=='none'")
    plot_df = BW_UV_meta.join(features.reindex(BW_UV_meta.index))
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10,8))
    sns.boxplot(x='is_dead', y=feature, order=is_dead_list, ax=ax, data=plot_df,
                palette='Paired', showfliers=False)
    sns.stripplot(x='is_dead', y=feature, order=is_dead_list, ax=ax, data=plot_df,
                  color='k', s=5, marker='D')
    ax.set_xlabel('UV-killed?', fontsize=12, labelpad=10)
    ax.set_ylabel(feature.replace('_',' '), fontsize=12, labelpad=10)
    ax.set_title('No effect of UV-killing control BW bacteria on worm motion mode', 
                 pad=30, fontsize=18)
    # load t-test results for is dead 'Y' (Yes, dead) vs 'N' (No, alive)
    ttest_path = Path(stats_dir) / 'BW_UV_killed' / 'is_dead_ttest_results.csv'
    ttest_df = pd.read_csv(ttest_path, index_col=0)
    pvals = ttest_df[[c for c in ttest_df.columns if 'pvals_' in c]]
    # annotate p-values
    for ii, is_dead in enumerate(is_dead_list):
        if is_dead == 'N':
            continue
        p = pvals.loc[feature, 'pvals_' + is_dead]
        text = ax.get_xticklabels()[ii]
        assert text.get_text() == is_dead
        p_text = '***\nP < 0.001' if p < 0.001 else sig_asterix([p])[0] + '\nP = %.3f' % p
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(ii, 1.01, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
    ax.set_xticklabels(['No','Yes'])
    save_path = Path(plot_dir) / 'BW_control' / 'effect_of_UV_killing.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    
    
    ### UV KILLING - fepD lysate
        
    # Boxplots of differences in worm motion mode when fepD killed in different ways is added to 
    # BW that is either dead or alive
    fepD_lysate_meta = window_meta.query("drug_type=='fepD' and cell_extract_type=='lysate'")
    plot_df = fepD_lysate_meta.join(features.reindex(fepD_lysate_meta.index))
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10,8))
    col_dict1 = dict(zip(culture_type_list[1:], sns.color_palette('Paired', len(culture_type_list[1:]))))
    sns.boxplot(x='is_dead', y=feature, order=is_dead_list, 
                hue='culture_type', hue_order=culture_type_list[1:], dodge=True,
                ax=ax, data=plot_df, palette=col_dict1, showfliers=False)
    kill_labs = [i for i in killing_method_list if i in plot_df['killing_method'].unique()]
    col_dict2 = dict(zip(kill_labs, sns.color_palette('plasma', len(kill_labs))))
    for k, kill in enumerate(kill_labs):
        sns.stripplot(x='is_dead', y=feature, order=is_dead_list, 
                      hue='culture_type', hue_order=culture_type_list[1:], dodge=True,
                      ax=ax, data=plot_df.query("killing_method==@kill"),
                      s=10, marker='D', color=sns.set_palette(palette=[col_dict2[kill]], 
                                                              n_colors=len(culture_type_list[1:])))
    handles, labels = [], []
    for ct in culture_type_list[1:]:
        handles.append(mpatches.Patch(color=col_dict1[ct]))
        labels.append(ct)
    for kt in kill_labs:
        handles.append(Line2D([0], [0], marker='D', color='w', markersize=10,
                              markerfacecolor=col_dict2[kt]))
        labels.append(kt)
    ax.legend(handles, labels, loc='best', frameon=False)
    # annotate p-values
    for ii, is_dead in enumerate(is_dead_list):
        ttest_path = stats_dir / 'fepD_solid_vs_liquid_culture' /\
            'lysate_on_{}_BW'.format('live' if is_dead=='N' else 'dead') /\
            'culture_type_ttest_results.csv'
        ttest_df = pd.read_csv(ttest_path, index_col=0)
        p = ttest_df.loc[feature, 'pvals_liquid']
        text = ax.get_xticklabels()[ii]
        assert text.get_text() == is_dead
        p_text = '***\nP < 0.001' if p < 0.001 else sig_asterix([p])[0] + '\nP = %.3f' % p
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(ii, 1.01, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
    ax.set_xlabel('UV-killed?', fontsize=12, labelpad=10)
    ax.set_ylabel(feature.replace('_',' '), fontsize=12, labelpad=10)
    ax.set_title('Effect of culture type and killing method', pad=30, fontsize=18)
    ax.set_xticklabels(['No','Yes'])
    # save figure
    save_path = Path(plot_dir) / 'fepD_lysate' / 'solid_vs_liquid_on_dead_vs_alive_BW.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)

    
    ### SUPERNATANT vs LYSATE - fepD on live BW
    
    fepD_live_meta = window_meta.query("drug_type=='fepD' and is_dead=='N'")
    plot_df = fepD_live_meta.join(features.reindex(fepD_live_meta.index))
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10,8))
    col_dict = dict(zip(culture_type_list[1:], sns.color_palette('Paired_r', len(culture_type_list[1:]))))
    sns.boxplot(x='cell_extract_type', y=feature, order=extract_type_list[1:], 
                hue='culture_type', hue_order=culture_type_list[1:], 
                ax=ax, data=plot_df, palette=col_dict, dodge=True, showfliers=False)
    sns.stripplot(x='cell_extract_type', y=feature, order=extract_type_list[1:], 
                  hue='culture_type', hue_order=culture_type_list[1:], 
                  ax=ax, data=plot_df, color='k', dodge=True, marker='D', s=8)
    handles = []
    for label in col_dict.keys():
        handles.append(mpatches.Patch(color=col_dict[label]))
    ax.legend(handles, col_dict.keys(), loc='best', frameon=False)
    # annotate p-values
    for ii, extract_type in enumerate(extract_type_list[1:]):
        ttest_path = stats_dir / 'fepD_solid_vs_liquid_culture' /\
            '{}_on_live_BW'.format(extract_type) / 'culture_type_ttest_results.csv'
        ttest_df = pd.read_csv(ttest_path, index_col=0)
        p = ttest_df.loc[feature, 'pvals_liquid']
        text = ax.get_xticklabels()[ii]
        assert text.get_text() == extract_type
        p_text = '***\nP < 0.001' if p < 0.001 else sig_asterix([p])[0] + '\nP = %.3f' % p
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(ii, 1.01, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
    ax.set_xlabel('cell extract type', fontsize=12, labelpad=10)
    ax.set_ylabel(feature.replace('_',' '), fontsize=12, labelpad=10)
    ax.set_title('fepD supernatant and lysate (solid vs liquid culture)\nadded to live BW', 
                 pad=30, fontsize=18)
    # save figure
    save_path = Path(plot_dir) / 'fepD_culture_type' / 'fepD_solid_vs_liquid_culture_on_live_BW.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)

    ### SUPERNATANT vs LYSATE - fepD on dead BW
    
    fepD_dead_meta = window_meta.query("drug_type=='fepD' and is_dead=='Y'")
    plot_df = fepD_dead_meta.join(features.reindex(fepD_dead_meta.index))
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10,8))
    col_dict = dict(zip(culture_type_list[1:], sns.color_palette('Paired_r', len(culture_type_list[1:]))))
    sns.boxplot(x='cell_extract_type', y=feature, order=extract_type_list[1:], 
                hue='culture_type', hue_order=culture_type_list[1:], 
                ax=ax, data=plot_df, palette=col_dict, dodge=True, showfliers=False)
    sns.stripplot(x='cell_extract_type', y=feature, order=extract_type_list[1:], 
                  hue='culture_type', hue_order=culture_type_list[1:], 
                  ax=ax, data=plot_df, color='k', dodge=True, marker='D', s=8)
    handles = []
    for label in col_dict.keys():
        handles.append(mpatches.Patch(color=col_dict[label]))
    ax.legend(handles, col_dict.keys(), loc='best', frameon=False)
    # annotate p-values
    for ii, extract_type in enumerate(extract_type_list[1:]):
        ttest_path = stats_dir / 'fepD_solid_vs_liquid_culture' /\
            '{}_on_dead_BW'.format(extract_type) / 'culture_type_ttest_results.csv'
        ttest_df = pd.read_csv(ttest_path, index_col=0)
        p = ttest_df.loc[feature, 'pvals_liquid']
        text = ax.get_xticklabels()[ii]
        assert text.get_text() == extract_type
        p_text = '***\nP < 0.001' if p < 0.001 else sig_asterix([p])[0] + '\nP = %.3f' % p
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(ii, 1.01, p_text, fontsize=9, ha='center', va='bottom', transform=trans)
    ax.set_xlabel('cell extract type', fontsize=12, labelpad=10)
    ax.set_ylabel(feature.replace('_',' '), fontsize=12, labelpad=10)
    ax.set_title('fepD supernatant and lysate (solid vs liquid culture)\nadded to UV-killed BW', 
                 pad=30, fontsize=18)
    # save figure
    save_path = Path(plot_dir) / 'fepD_culture_type' / 'fepD_solid_vs_liquid_culture_on_dead_BW.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    
    return


#%% Main

if __name__ == '__main__':
    
    AUX_DIR = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    RES_DIR = Path(PROJECT_DIR) / 'Results'

    META_PATH = Path(SAVE_DIR) / 'metadata.csv'
    FEAT_PATH = Path(SAVE_DIR) / 'features.csv'
    
    if not META_PATH.exists() and not FEAT_PATH.exists():
    
        # compile metadata
        metadata, metadata_path = compile_metadata(aux_dir=AUX_DIR, 
                                                   imaging_dates=IMAGING_DATES, 
                                                   n_wells=N_WELLS,
                                                   add_well_annotations=N_WELLS==96,
                                                   from_source_plate=True)
        
        # compile window summaries
        features, metadata = process_feature_summaries(metadata_path=metadata_path, 
                                                       results_dir=RES_DIR, 
                                                       compile_day_summaries=True,
                                                       imaging_dates=IMAGING_DATES, 
                                                       align_bluelight=False,
                                                       window_summaries=True,
                                                       n_wells=N_WELLS)
        
        # clean results
        features, metadata = clean_summary_results(features, 
                                                   metadata,
                                                   feature_columns=None,
                                                   nan_threshold_row=NAN_THRESHOLD_ROW,
                                                   nan_threshold_col=NAN_THRESHOLD_COL,
                                                   max_value_cap=1e15,
                                                   imputeNaN=True,
                                                   min_nskel_per_video=MIN_NSKEL_PER_VIDEO,
                                                   min_nskel_sum=MIN_NSKEL_SUM,
                                                   drop_size_related_feats=False,
                                                   norm_feats_only=False,
                                                   percentile_to_use=None)
    
        assert not features.isna().sum(axis=1).any()
        assert not (features.std(axis=1) == 0).any()
    
        # save features
        metadata.to_csv(META_PATH, index=False)
        features.to_csv(FEAT_PATH, index=False) 
    
    else:
        # load clean metadata and features
        metadata = pd.read_csv(META_PATH, dtype={'comments':str, 'source_plate_id':str})
        features = pd.read_csv(FEAT_PATH, index_col=None)
        
    supernatants_stats(metadata, 
                       features, 
                       save_dir=Path(SAVE_DIR) / "Stats",
                       window=WINDOW_NUMBER,
                       feature=FEATURE)
    
    supernatants_plots(metadata,
                       features,
                       plot_dir=Path(SAVE_DIR) / "Plots",
                       stats_dir=Path(SAVE_DIR) / "Stats",
                       window=WINDOW_NUMBER,
                       feature=FEATURE)

