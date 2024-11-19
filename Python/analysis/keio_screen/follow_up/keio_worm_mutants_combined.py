#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combinbed results for worm mutant experiments so far on fepD / BW both with/without paraquat 
(boxplots and timeseries)

@author: sm5911
@date: 10/06/2023

"""

#%% Imports 

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from itertools import chain
from matplotlib import pyplot as plt
from matplotlib import transforms

from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from visualisation.plotting_helper import sig_asterix
from time_series.plot_timeseries import plot_timeseries, get_strain_timeseries
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes

#%% Globals 

PROJECT_DIR_LIST = ['/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Worm_Stress_Mutants',
                    '/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Worm_Stress_Mutants_2',
                    '/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Worm_Stress_Mutants_3',
                    '/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Worm_Stress_Mutants_4']

SAVE_DIR = '/Users/sm5911/Documents/PhD_DLBG/31_Keio_Worm_Stress_Mutants_Combined'

FEATURE = 'speed_50th'
FDR_METHOD = 'fdr_bh'
P_VALUE_THRESHOLD = 0.05
NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
N_WELLS = 6
DPI = 900
FPS = 25
BLUELIGHT_TIMEPOINTS_SECONDS = [(60, 70),(160, 170),(260, 270)]

OMIT_STRAINS_LIST = ["RID(RNAi)::unc-31", "PY7505", "MQ1333", "nuo-6"]

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
                    #"nuo-6":"nuo-6(qm200)", # MQ1333, same as above (alternative name in metadata)
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
                                  "prdx-2(gk169)",
                                  "sod-1",
                                  "sod-1; sod-5",
                                  "sod-2",
                                  "sod-1; sod-2",
                                  "sod-3",
                                  "sod-2; sod-3",
                                  "sod-4",
                                  "sod-4; sod-5",
                                  "sod-5",
                                  "sod-3; sod-5",
                                  "OE sod-1",
                                  "OE sod-2",
                                  "OE ctl-1+ctl-2+ctl-3",
                                  "clk-1(qm30)",
                                  "gas-1(fc21)",
                                  "msra-1(tm1421)"],
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
            metadata_list.append(_metadata)
            features_list.append(_features)
            
        # subset for window (290:300 seconds)
        meta_list = []
        feat_list = []
        for i, (meta, feat) in enumerate(zip(metadata_list, features_list)):
            if i == 0: # only for '/Volumes/hermes$/Keio_Worm_Stress_Mutants' metadata
                assert meta['window'].nunique() == 6
                meta = meta[meta['window']==5]
                feat = feat.reindex(meta.index)
            else:
                assert meta['window'].nunique() == 1
            meta_list.append(meta)
            feat_list.append(feat)
            
        metadata = pd.concat(meta_list, axis=0).reset_index(drop=True)
        features = pd.concat(feat_list, axis=0).reset_index(drop=True)
        
        # clean combined results - remove bad wells + features with too many NaNs/zero std + impute NaNs
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
        metadata = pd.read_csv(metadata_combined_path, header=0, index_col=None, dtype={'comments':str})
        features = pd.read_csv(features_combined_path, header=0, index_col=None)

    # subset metadata results for bluelight videos only 
    bluelight_videos = [i for i in metadata['imgstore_name'] if 'bluelight' in i]
    metadata = metadata[metadata['imgstore_name'].isin(bluelight_videos)]
    
    # omit strains from OMIT_STRAINS_LIST for worm_strain
    n_samples =  metadata.shape[0]
    metadata = metadata[~metadata['worm_strain'].isin(OMIT_STRAINS_LIST)]
    print("%d samples dropped (omitted strains)" % (n_samples - metadata.shape[0]))
    
    # strain name mapping
    metadata['worm_strain'] = metadata['worm_strain'].map(WORM_STRAIN_DICT)
    metadata['treatment'] = metadata[['worm_strain','bacteria_strain','drug_type']
                                     ].astype(str).agg('-'.join, axis=1)
    metadata['treatment'] = [i.replace('-nan','') for i in metadata['treatment']]

    features = features.reindex(metadata.index)
        
    assert set(list(chain.from_iterable(WORM_GROUP_DICT.values()))) == set(metadata.worm_strain.unique())
    
    worm_group_list = list(WORM_GROUP_DICT.keys())
    for group in tqdm(worm_group_list):
        group_strains = WORM_GROUP_DICT[group]
        
        meta = metadata[metadata['worm_strain'].isin(group_strains)]
        feat = features.reindex(meta.index)
        
        stats_dir = Path(SAVE_DIR) / group / 'Stats'
        plots_dir = Path(SAVE_DIR) / group / 'Plots'

        # do stats - compare each mutant-treatment combination to N2-BW without paraquat
        control = 'N2-BW'
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
        
        # boxplot without paraquat
        meta_nopara = meta[meta['drug_type']!='Paraquat']
        feat_nopara = feat.reindex(meta_nopara.index)
        plot_df = meta_nopara.join(feat_nopara)
        
        order = ['N2'] + [w for w in sorted(plot_df['worm_strain'].unique()) if w != 'N2']
        colour_dict = dict(zip(['BW','fepD'], sns.color_palette(palette='tab10', n_colors=2)))

        plt.close('all')
        sns.set_style('ticks')
        fig = plt.figure(figsize=FIGSIZE_DICT[group])
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
        date_lut = dict(zip(dates, sns.color_palette(palette="Greys", n_colors=len(dates))))
        for date in dates:
            date_df = plot_df[plot_df['date_yyyymmdd']==date]
            sns.stripplot(x='worm_strain',
                          y='speed_50th',
                          data=date_df,
                          order=order,
                          hue='bacteria_strain',
                          hue_order=['BW','fepD'],
                          dodge=True,
                          palette=[date_lut[date]] * len(order),
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
        plt.ylim(-40, 270)
                
        # scale y axis for annotations    
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled
        
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

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], loc='best', frameon=False, fontsize=15)
        boxplot_path = plots_dir / 'worm_{}_mutants.svg'.format(group)
        boxplot_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(boxplot_path, bbox_inches='tight', transparent=True)
        
        
        # boxplot with paraquat
        meta_para = meta[meta['drug_type']=='Paraquat']
        feat_para = feat.reindex(meta_para.index)
        plot_df = meta_para.join(feat_para)

        order = ['N2'] + [w for w in sorted(plot_df['worm_strain'].unique()) if w != 'N2']
        colour_dict = dict(zip(['BW','fepD'], sns.color_palette(palette='tab10', n_colors=2)))

        plt.close('all')
        sns.set_style('ticks')
        fig = plt.figure(figsize=FIGSIZE_DICT[group])
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
        date_lut = dict(zip(dates, sns.color_palette(palette="Greys", n_colors=len(dates))))
        for date in dates:
            date_df = plot_df[plot_df['date_yyyymmdd']==date]
            sns.stripplot(x='worm_strain',
                          y='speed_50th',
                          data=date_df,
                          order=order,
                          hue='bacteria_strain',
                          hue_order=['BW','fepD'],
                          dodge=True,
                          palette=[date_lut[date]] * len(order),
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
        plt.ylim(-40, 270)
                
        # scale y axis for annotations    
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled
        
        # add pvalues to plot
        for i, text in enumerate(ax.axes.get_xticklabels()):
            worm = text.get_text()
            if worm == 'N2':
                p = pvals.loc['speed_50th','N2-fepD-Paraquat']
                p_text = sig_asterix([p])[0]
                ax.text(i+0.2, 1.03, p_text, fontsize=20, ha='center', va='center', transform=trans)
            else:
                p1 = pvals.loc['speed_50th',worm+'-BW-Paraquat']
                p2 = pvals.loc['speed_50th',worm+'-fepD-Paraquat']
                p1_text = sig_asterix([p1])[0]
                p2_text = sig_asterix([p2])[0]
                ax.text(i-0.2, 1.03, p1_text, fontsize=20, ha='center', va='center', transform=trans)
                ax.text(i+0.2, 1.03, p2_text, fontsize=20, ha='center', va='center', transform=trans)                

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], loc='best', frameon=False, fontsize=15)
        boxplot_path = plots_dir / 'worm_{}_mutants-Paraquat.svg'.format(group)
        boxplot_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(boxplot_path, bbox_inches='tight', transparent=True)
        
        
    # timeseries 
    
    worm_strain_list = sorted([i for i in metadata['worm_strain'].unique() if i != 'N2'])       
    bluelight_frames = [(i*FPS, j*FPS) for (i, j) in BLUELIGHT_TIMEPOINTS_SECONDS]
    feature = 'speed'
    save_dir = Path(SAVE_DIR) / 'timeseries-speed'
    
    # without paraquat
    
    print("Plotting timeseries speed (no paraquat)")   
    for worm in tqdm(worm_strain_list):
        
        save_dir.mkdir(exist_ok=True, parents=True)
        save_path = save_dir / '{}_speed_bluelight.pdf'.format(worm)
        
        if not save_path.exists():
        
            groups = ['N2-BW', 'N2-fepD', worm+'-BW', worm+'-fepD']
            colour_dict = dict(zip(groups, sns.color_palette(palette='tab10', n_colors=4)))
        
            plt.close('all')
            fig, ax = plt.subplots(figsize=(15,6), dpi=300)
            
            for group in groups:
                
                # subset metadata for group
                group_meta = metadata.groupby('treatment').get_group(group)
                group_meta['dirpath'] = [str(s).split('/Results/')[0] for s in group_meta['featuresN_filename']]
                dirnames = sorted(group_meta['dirpath'].unique())
                
                # if combining timeseries results across multiple project directories
                if len(dirnames) > 1:
                    group_ts_list = []
                    for project_dir in dirnames:        
                        
                        # get timeseries from each project directory
                        dirmeta = group_meta.loc[group_meta['dirpath']==project_dir]                
                        
                        group_ts_dir = get_strain_timeseries(dirmeta,
                                                             project_dir=Path(project_dir),
                                                             strain=group,
                                                             group_by='treatment',
                                                             feature_list=[feature],
                                                             save_dir=save_dir / Path(project_dir).name,
                                                             n_wells=N_WELLS,
                                                             verbose=True)
                        
                        group_ts_list.append(group_ts_dir)
                        
                    group_ts = pd.concat(group_ts_list, axis=0).reset_index(drop=True)
                else:
                    group_ts = get_strain_timeseries(group_meta,
                                                     project_dir=Path(dirnames[0]),
                                                     strain=group,
                                                     group_by='treatment',
                                                     feature_list=[feature],
                                                     save_dir=save_dir / Path(dirnames[0]).name,
                                                     n_wells=N_WELLS,
                                                     verbose=True)
            
                # plot timeseries            
                ax = plot_timeseries(df=group_ts,
                                     feature=feature,
                                     error=True,
                                     max_n_frames=360*FPS, 
                                     smoothing=10*FPS, 
                                     ax=ax,
                                     bluelight_frames=bluelight_frames,
                                     colour=colour_dict[group])
        
            plt.ylim(-20, 320)
            xticks = np.linspace(0, 360*FPS, int(360/60)+1)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
            ax.set_xlabel('Time (minutes)', fontsize=20, labelpad=10)
            ylab = feature.replace('_50th'," (µm s$^{-1}$)")
            ax.set_ylabel(ylab, fontsize=20, labelpad=10)
            ax.legend(groups, fontsize=12, frameon=False, loc='best', handletextpad=1)
            plt.subplots_adjust(left=0.1, top=0.98, bottom=0.15, right=0.98)
        
            # save plot
            print("Saving to: %s" % save_path)
            plt.savefig(save_path)
      
    # with paraquat
    
    print("Plotting timeseries speed (+ 1mM paraquat)")   
    for worm in tqdm(worm_strain_list):
        
        save_dir.mkdir(exist_ok=True, parents=True)
        save_path = save_dir / '{}_paraquat_speed_bluelight.pdf'.format(worm)
        
        if not save_path.exists():
        
            groups = ['N2-BW-Paraquat', 'N2-fepD-Paraquat', worm+'-BW-Paraquat', worm+'-fepD-Paraquat']
            colour_dict = dict(zip(groups, sns.color_palette(palette='tab10', n_colors=4)))
    
            plt.close('all')
            fig, ax = plt.subplots(figsize=(15,6), dpi=300)
            
            for group in groups:
                
                # subset metadata for group
                group_meta = metadata.groupby('treatment').get_group(group)
                group_meta['dirpath'] = [str(s).split('/Results/')[0] for s in group_meta['featuresN_filename']]
                dirnames = sorted(group_meta['dirpath'].unique())
                
                # if combining timeseries results across multiple project directories
                if len(dirnames) > 1:
                    group_ts_list = []
                    for project_dir in dirnames:        
                        
                        # get timeseries from each project directory
                        dirmeta = group_meta.loc[group_meta['dirpath']==project_dir]                
                        
                        group_ts_dir = get_strain_timeseries(dirmeta,
                                                             project_dir=Path(project_dir),
                                                             strain=group,
                                                             group_by='treatment',
                                                             feature_list=[feature],
                                                             save_dir=save_dir / Path(project_dir).name,
                                                             n_wells=N_WELLS,
                                                             verbose=True)
                        
                        group_ts_list.append(group_ts_dir)
                        
                    group_ts = pd.concat(group_ts_list, axis=0).reset_index(drop=True)
                else:
                    group_ts = get_strain_timeseries(group_meta,
                                                     project_dir=Path(dirnames[0]),
                                                     strain=group,
                                                     group_by='treatment',
                                                     feature_list=[feature],
                                                     save_dir=save_dir / Path(dirnames[0]).name,
                                                     n_wells=N_WELLS,
                                                     verbose=True)
            
                # plot timeseries            
                ax = plot_timeseries(df=group_ts,
                                     feature=feature,
                                     error=True,
                                     max_n_frames=360*FPS, 
                                     smoothing=10*FPS, 
                                     ax=ax,
                                     bluelight_frames=bluelight_frames,
                                     colour=colour_dict[group])
        
            plt.ylim(-20, 320)
            xticks = np.linspace(0, 360*FPS, int(360/60)+1)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(x/FPS/60)) for x in xticks])   
            ax.set_xlabel('Time (minutes)', fontsize=20, labelpad=10)
            ylab = feature.replace('_50th'," (µm s$^{-1}$)")
            ax.set_ylabel(ylab, fontsize=20, labelpad=10)
            ax.legend(groups, fontsize=12, frameon=False, loc='best', handletextpad=1)
            plt.subplots_adjust(left=0.1, top=0.98, bottom=0.15, right=0.98)
        
            # save plot
            print("Saving to: %s" % save_path)
            plt.savefig(save_path)
    
# =============================================================================
#     # without paraquat
#     
#     meta_nopara = metadata[metadata['drug_type']!='Paraquat']
#     feat_nopara = features.reindex(meta_nopara.index)    
# 
#     # do stats
#     control = 'N2-BW'
#     anova_results, ttest_results = stats(meta_nopara,
#                                          feat_nopara,
#                                          group_by='treatment',
#                                          control=control,
#                                          feat='speed_50th',
#                                          pvalue_threshold=P_VALUE_THRESHOLD,
#                                          fdr_method=FDR_METHOD)
#     
#     # t-test pvals
#     pvals = ttest_results[[c for c in ttest_results.columns if 'pval' in c]]
#     pvals.columns = [c.replace('pvals_','') for c in pvals.columns]
# 
#     # boxplots - for each group of related worm mutants
#     worm_group_list = list(WORM_GROUP_DICT.keys())
#     
#     for group in tqdm(worm_group_list):
#         group_strains = WORM_GROUP_DICT[group]
#         
#         meta = meta_nopara[meta_nopara['worm_strain'].isin(group_strains)]
#         feat = feat_nopara.reindex(meta.index)
#     
#         plot_df = meta.join(feat)
#         ctl = ['N2-BW','N2-fepD']
#         order = ctl + [i for i in sorted(meta['treatment'].unique()) if i not in ctl]
#         cols = sns.color_palette(palette='tab10', n_colors=2)
#         colour_dict = {i:(cols[0] if '-BW' in i else cols[1]) for i in order}
# 
#         plt.close('all')
#         sns.set_style('ticks')
#         fig = plt.figure(figsize=[14,18])
#         ax = fig.add_subplot(1,1,1)
#         sns.boxplot(x='speed_50th',
#                     y='treatment',
#                     data=plot_df, 
#                     order=order,
#                     palette=colour_dict,
#                     showfliers=False, 
#                     showmeans=False,
#                     meanprops={"marker":"x", 
#                                "markersize":5,
#                                "markeredgecolor":"k"},
#                     flierprops={"marker":"x", 
#                                 "markersize":15, 
#                                 "markeredgecolor":"r"})
#         dates = meta['date_yyyymmdd'].unique()
#         sns.stripplot(x='speed_50th',
#                       y='treatment',
#                       data=plot_df,
#                       s=12,
#                       order=order,
#                       hue=None,
#                       palette=None,
#                       color='dimgray',
#                       marker=".",
#                       edgecolor='k',
#                       linewidth=0.3) #facecolors="none"
#         
#         ax.axes.get_yaxis().get_label().set_visible(False) # remove y axis label
#         ax.axes.set_yticklabels([l.get_text() for l in ax.axes.get_yticklabels()], fontsize=25)
#         ax.tick_params(axis='y', which='major', pad=15)
#         ax.axes.set_xlabel('Speed (µm s$^{-1}$)', fontsize=30, labelpad=25)                             
#         plt.xticks(fontsize=20)
#         plt.xlim(-20, 250)
#                 
#         # scale x axis for annotations    
#         trans = transforms.blended_transform_factory(ax.transAxes, ax.transData) #x=scaled
#         
#         # add pvalues to plot
#         for i, treatment in enumerate(order, start=0):
#             if treatment == control:
#                 continue
#             else:
#                 p = pvals.loc['speed_50th', treatment]
#                 text = ax.get_yticklabels()[i]
#                 assert text.get_text() == treatment
#                 p_text = sig_asterix([p])[0]
#                 ax.text(1.03, i, p_text, fontsize=30, ha='left', va='center', transform=trans)
#                 
#         plt.subplots_adjust(left=0.4, right=0.9, bottom=0.15, top=0.95)
#         plt.savefig(Path(SAVE_DIR) / 'worm_{}_mutants.png'.format(group), dpi=DPI)  
# 
# 
#     # with paraquat
#     
#     meta_para = metadata[metadata['drug_type']=='Paraquat']
#     feat_para = features.reindex(meta_para.index)
#  
#     # do stats
#     control = 'N2-BW-Paraquat'
#     anova_results, ttest_results = stats(meta_para,
#                                          feat_para,
#                                          group_by='treatment',
#                                          control=control,
#                                          feat='speed_50th',
#                                          pvalue_threshold=P_VALUE_THRESHOLD,
#                                          fdr_method=FDR_METHOD)
#     
#     # t-test pvals
#     pvals = ttest_results[[c for c in ttest_results.columns if 'pval' in c]]
#     pvals.columns = [c.replace('pvals_','') for c in pvals.columns]
# 
#     # boxplots - for each group of related worm mutants
#     worm_group_list = list(WORM_GROUP_DICT.keys())
# 
#     for group in tqdm(worm_group_list):
#         group_strains = WORM_GROUP_DICT[group]
#         
#         meta = meta_para[meta_para['worm_strain'].isin(group_strains)]
#         feat = feat_para.reindex(meta.index)
#     
#         plot_df = meta.join(feat)
#         ctl = ['N2-BW-Paraquat','N2-fepD-Paraquat']
#         order = ctl + [i for i in sorted(meta['treatment'].unique()) if i not in ctl]
#         cols = sns.color_palette(palette='tab10', n_colors=2)
#         colour_dict = {i:(cols[0] if '-BW' in i else cols[1]) for i in order}
# 
#         plt.close('all')
#         sns.set_style('ticks')
#         fig = plt.figure(figsize=[14,18])
#         ax = fig.add_subplot(1,1,1)
#         sns.boxplot(x='speed_50th',
#                     y='treatment',
#                     data=plot_df, 
#                     order=order,
#                     palette=colour_dict,
#                     showfliers=False, 
#                     showmeans=False,
#                     meanprops={"marker":"x", 
#                                "markersize":5,
#                                "markeredgecolor":"k"},
#                     flierprops={"marker":"x", 
#                                 "markersize":15, 
#                                 "markeredgecolor":"r"})
#         dates = meta['date_yyyymmdd'].unique()
#         sns.stripplot(x='speed_50th',
#                       y='treatment',
#                       data=plot_df,
#                       s=12,
#                       order=order,
#                       hue=None,
#                       palette=None,
#                       color='dimgray',
#                       marker=".",
#                       edgecolor='k',
#                       linewidth=0.3) #facecolors="none"
#         
#         ax.axes.get_yaxis().get_label().set_visible(False) # remove y axis label
#         ax.axes.set_yticklabels([l.get_text() for l in ax.axes.get_yticklabels()], fontsize=25)
#         ax.tick_params(axis='y', which='major', pad=15)
#         ax.axes.set_xlabel('Speed (µm s$^{-1}$)', fontsize=30, labelpad=25)                             
#         plt.xticks(fontsize=20)
#         plt.xlim(-20, 250)
#                 
#         # scale x axis for annotations    
#         trans = transforms.blended_transform_factory(ax.transAxes, ax.transData) #x=scaled
#         
#         # add pvalues to plot
#         for i, treatment in enumerate(order, start=0):
#             if treatment == control:
#                 continue
#             else:
#                 p = pvals.loc['speed_50th', treatment]
#                 text = ax.get_yticklabels()[i]
#                 assert text.get_text() == treatment
#                 p_text = sig_asterix([p])[0]
#                 ax.text(1.03, i, p_text, fontsize=30, ha='left', va='center', transform=trans)
#                 
#         plt.subplots_adjust(left=0.4, right=0.9, bottom=0.15, top=0.95)
#         plt.savefig(Path(SAVE_DIR) / 'worm_{}_mutants_Paraquat.png'.format(group), dpi=DPI) 
# =============================================================================

#%% Main

if __name__ == '__main__':
    main()
    