#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined plots of mutants created based on proteomics results in BW wild-type and fepD E. coli 
background strains - analysis conducted separately for BW background mutants, BW-iptg mutants, 
fepD background mutants and fepD-iptg mutants. Total n=79 mutants tested.

@author: sm5911
@date: 06/02/2023 (updated: 17/11/2024)

"""

#%% Imports

import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import transforms

from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from visualisation.plotting_helper import sig_asterix
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes

#%% Globals

PROJECT_DIR_LIST = ['/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Proteomics_Mutants',
                    '/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_FepD_Oxidative_Stress_Mutants',
                    '/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_FepD_Mutants',
                    '/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_FepD_Mutants_2',
                    '/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Bacteria_Mutants',
                    '/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_FepD_Ent_Mutants',
                    '/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_FepD_Ent_Mutants_2']

SAVE_DIR = '/Users/sm5911/Documents/PhD_DLBG/Keio_FepD_Mutants_Combined'

N_WELLS = 6
NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
FEATURE = 'speed_50th'
P_VALUE_THRESHOLD = 0.05
FDR_METHOD = 'fdr_bh'
DPI = 900
FPS = 25
BLUELIGHT_TIMEPOINTS_SECONDS = [(60, 70),(160, 170),(260, 270)]

OMIT_DRUGS_LIST = ['deferoxamine']
OMIT_STRAINS_LIST = ['fepD; empty plasmid'] # remove samples for empty plasmid without IPTG

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
        anova_results = anova_results.sort_values(by=['pvals'], ascending=True) # rank by p-value

        if save_dir is not None:
            anova_path = Path(save_dir) / 'ANOVA_results.csv'
            anova_path.parent.mkdir(parents=True, exist_ok=True)
            anova_results.to_csv(anova_path, header=True, index=True)

        # # use reject mask to find significant feature set
        # fset = pvals.loc[reject['ANOVA']].sort_values(by='ANOVA', ascending=True).index.to_list()            
        # if len(fset) > 0:
        #     print("%d significant features found by ANOVA by '%s' (P<%.2f, %s)" %\
        #           (len(fset), group_by, pvalue_threshold, fdr_method))
        #     if save_dir is not None:
        #         anova_sigfeats_path = anova_path.parent / 'ANOVA_sigfeats.txt'
        #         write_list_to_file(fset, anova_sigfeats_path)
             
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
        ttest_path = Path(save_dir) / 't-test' / 't-test_results.csv'
        ttest_path.parent.mkdir(exist_ok=True, parents=True)
        ttest_results.to_csv(ttest_path, header=True, index=True)
    
    # nsig = sum(reject_t.sum(axis=1) > 0)
    # print("%d significant features between any %s vs %s (t-test, P<%.2f, %s)" %\
    #       (nsig, group_by, control, pvalue_threshold, fdr_method))

    return anova_results, ttest_results


def main():
    
    metadata_combined_path = Path(SAVE_DIR) / 'metadata.csv'
    features_combined_path = Path(SAVE_DIR) / 'features.csv'
    
    # compile combined metadata & feature summaries, clean data and save
    if not (metadata_combined_path.exists() or features_combined_path.exists()):

        metadata_list = []
        features_list = []
        for project_dir in tqdm(PROJECT_DIR_LIST):
            
            aux_dir = Path(project_dir) / 'AuxiliaryFiles'
            res_dir = Path(project_dir) / 'Results'

            _metadata, _metadata_path = compile_metadata(aux_dir, 
                                                         n_wells=N_WELLS, 
                                                         add_well_annotations=False,
                                                         from_source_plate=True)
            
            _features, _metadata = process_feature_summaries(_metadata_path, 
                                                             results_dir=res_dir, 
                                                             compile_day_summaries=True, 
                                                             imaging_dates=None, 
                                                             align_bluelight=False, 
                                                             window_summaries=True,
                                                             n_wells=N_WELLS)
            
            if 'food_type' in _metadata.columns:
                _metadata = _metadata.rename({'food_type':'bacteria_strain'}, axis=1)
                
            metadata_list.append(_metadata)
            features_list.append(_features)
            
        # subset for window (290:300 seconds)
        meta_list = []
        feat_list = []
        for i, (meta, feat) in enumerate(zip(metadata_list, features_list)):
            if meta['window'].nunique() == 6:
                meta = meta[meta['window']==5] # corresponds to 20-30 seconds after final BL pulse
                feat = feat.reindex(meta.index)
            elif meta['window'].nunique() == 1: # if summaries only generated for final BL window
                assert meta['window'].unique()[0] == 0
            meta_list.append(meta)
            feat_list.append(feat)
            
        metadata = pd.concat(meta_list, axis=0).reset_index(drop=True)
        features = pd.concat(feat_list, axis=0).reset_index(drop=True)
        
        # clean combined results - remove bad wells + features with many NaNs/zero std + impute NaNs
        features, metadata = clean_summary_results(features, 
                                                   metadata,
                                                   feature_columns=None,
                                                   nan_threshold_row=NAN_THRESHOLD_ROW,
                                                   nan_threshold_col=NAN_THRESHOLD_COL,
                                                   max_value_cap=1e15,
                                                   imputeNaN=True,
                                                   min_nskel_per_video=None,
                                                   min_nskel_sum=None,
                                                   drop_size_related_feats=False,
                                                   norm_feats_only=False)
        
        # save clean metadata and features
        metadata.to_csv(metadata_combined_path, index=False)
        features.to_csv(features_combined_path, index=False)          
        
    else:
        # load metadata and features summaries
        metadata = pd.read_csv(metadata_combined_path, header=0, index_col=None, 
                               dtype={'comments':str})
        features = pd.read_csv(features_combined_path, header=0, index_col=None)

    # subset metadata results for bluelight videos only 
    bluelight_videos = [i for i in metadata['imgstore_name'] if 'bluelight' in i]
    metadata = metadata[metadata['imgstore_name'].isin(bluelight_videos)]

    assert not features.isna().sum(axis=1).any()
    assert not (features.std(axis=1) == 0).any()    
    assert not metadata['bacteria_strain'].isna().any()
    
    #omit deferoxamine supplementation from metadata (failed) - "keio_fepD_oxidative_stress_mutants"
    n_samples =  metadata.shape[0]
    metadata = metadata[~metadata['drug_type'].isin(OMIT_DRUGS_LIST)]

    # rename 'control_BW' to 'BW' in bacteria_strain column - "keio_fepD_ent_mutants"
    metadata['bacteria_strain'] = ['BW' if i == 'control_BW' else i for i in 
                                   metadata['bacteria_strain'].copy()]           
    metadata['treatment'] = metadata[['bacteria_strain','drug_type','drug_solvent']
                                     ].astype(str).agg('-'.join, axis=1)
    metadata['treatment'] = [i.replace('OE_','OE ').replace('_','; ').replace('-nan','').replace(
        'iptg','IPTG') for i in metadata['treatment']]
    
    metadata = metadata[~metadata['treatment'].isin(OMIT_STRAINS_LIST)]
    print("%d samples dropped" % (n_samples - metadata.shape[0]))
    
    # reindex features after subsetting metadata
    features = features.reindex(metadata.index)

    cs = ['BW', 'fepD']
    ts = [i for i in sorted(metadata['treatment'].unique()) if i not in cs]
    fs = [i for i in ts if i.startswith('fepD')]
    bs = [i for i in ts if not i.startswith('fepD')]
    strain_list_fepD = cs + fs  # n=48
    strain_list_BW = cs + bs    # n=33
    
    BW_strains = [i for i in strain_list_BW if not 'IPTG' in i]                     # n=24
    BW_strains_iptg = [i for i in strain_list_BW if 'IPTG' in i]
    BW_strains_iptg.insert(1, 'fepD-IPTG')                                          # n=10
    fepD_strains = [i for i in strain_list_fepD if not 'IPTG' in i]                 # n=37
    fepD_strains_iptg = ['BW-IPTG'] + [i for i in strain_list_fepD if 'IPTG' in i]  # n=12
    
    BACTERIA_GROUP_DICT = {'BW mutants':BW_strains,
                           'BW-IPTG mutants':BW_strains_iptg,
                           'fepD mutants':fepD_strains,
                           'fepD-IPTG mutants':fepD_strains_iptg}
    FIGSIZE_DICT = dict(zip(BACTERIA_GROUP_DICT.keys(),[[20,10],[12,10],[25,10],[12,10]]))
    
    for group in tqdm(BACTERIA_GROUP_DICT.keys()):
        group_strains = BACTERIA_GROUP_DICT[group]
        
        meta = metadata[metadata['treatment'].isin(group_strains)]
        feat = features.reindex(meta.index)
        
        stats_dir = Path(SAVE_DIR) / group / 'Stats'
        plots_dir = Path(SAVE_DIR) / group / 'Plots'

        # do stats - compare each mutant-treatment combination to N2-BW without paraquat
        control = group_strains[0] if 'BW' in group else group_strains[1]
        anova_results, ttest_results = stats(meta,
                                             feat,
                                             group_by='treatment',
                                             control=control,
                                             feat='speed_50th',
                                             save_dir=Path(stats_dir),
                                             pvalue_threshold=P_VALUE_THRESHOLD,
                                             fdr_method=FDR_METHOD)

        # extract t-test pvals
        pvals = ttest_results[[c for c in ttest_results.columns if 'pval' in c]]
        pvals.columns = [c.replace('pvals_','') for c in pvals.columns]

        # save data for plot
        plot_df = meta[['treatment','date_yyyymmdd']].join(feat[['speed_50th']])   
        plot_df.to_csv(plots_dir / 'figure_data.csv', header=True, index=False)
        
        # boxplot        
        lut = dict(zip(['BW','fepD'], sns.color_palette(palette='tab10', n_colors=2)))
        colour_dict = {i:(lut['fepD'] if 'fepD' in i else lut['BW']) for i in group_strains}        

        plt.close('all')
        sns.set_style('ticks')
        fig = plt.figure(figsize=FIGSIZE_DICT[group])
        ax = fig.add_subplot(1,1,1)
        sns.boxplot(x='treatment',
                    y='speed_50th',
                    data=plot_df, 
                    order=group_strains,
                    palette=colour_dict,
                    hue='treatment',
                    hue_order=group_strains,
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
                          order=group_strains,
                          palette=[date_lut[date]] * len(group_strains),
                          hue='treatment',
                          hue_order=group_strains,
                          color='dimgray',
                          marker=".",
                          edgecolor='k',
                          linewidth=0.3,
                          s=12) #facecolors="none"
        
        ax.axes.get_xaxis().get_label().set_visible(False) # remove y axis label
        ax.axes.set_xticklabels([l.get_text() for l in ax.axes.get_xticklabels()], fontsize=20, rotation=90)
        ax.tick_params(axis='x', which='major', pad=15)
        ax.axes.set_ylabel('Speed (µm s$^{-1}$)', fontsize=25, labelpad=25)                             
        plt.yticks(fontsize=20)
        plt.ylim(-20, 300)
                
        # scale y axis for annotations    
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled
        
        # add pvalues to plot
        for i, text in enumerate(ax.axes.get_xticklabels()):
            treatment = text.get_text()
            if treatment == control:
                continue
            else:
                p = pvals.loc['speed_50th', treatment]
                p_text = sig_asterix([p])[0]
                ax.text(i, 1.03, p_text, fontsize=20, ha='center', va='center', transform=trans)

        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles[:2], labels[:2], loc='best', frameon=False, fontsize=15)
        boxplot_path = plots_dir / '{}.svg'.format(group)
        boxplot_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(boxplot_path, bbox_inches='tight', transparent=True)

    return  

#%% Main

if __name__ == '__main__':
    main()
    