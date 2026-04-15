#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C. elegans pdfr-1 RNAi mutant speed during blue light on atfs-1 and L4440 
E. coli in BW25113 wildtype and ΔfepD background

C. elegans strains:
FGC126: 
    pdfr-1(o);him-5(o);[rab-3::nCre+unc-122::rfp], [pdfr-1::LoxP::inv pdfr-1cDNAisofD::sl2::GFP::LoxP+unc122::GFP] pdfr-1(ok3425)III;him-5(e1490)V;olels5[rab-3::nCre(3ng/µl)+unc-122::RFP(30ng/µl)]; olels3[pdfr-1::LoxP::inv pdfr-1cDNAisofD::sl2::GFP::LoxP(10ng/µl)+unc122::GFP(10ng/µl)]
FGC127: 
    pdfr-1(o);him-5(o);[pdfr-1::LoxP::inv pdfr-1cDNAisofD::sl2::GFP::LoxP+unc122::GFP];oleEx72[tdc-1::nCre+glr-3::nCre+ttx-3::nCre+unc122::RFP] pdfr-1(ok3425)III;him-5(e1490)V;olels3[pdfr-1::LoxP::inv pdfr-1cDNAisofD::sl2::GFP::LoxP(10ng/µl)+unc122::GFP(10ng/µl)];oleEx72[tdc-1::nCre(20ng/µl pSF178)+glr-3::nCre(50ng/µl pSF179)+ttx-3::nCre(5ng/µl pLAU 11)+unc122::RFP(25ng/µl)]
FGC131-1: 
    pdfr-1(o);him-5(o);[rab-3::nCre+unc-122::rfp] pdfr-1(ok3425)III;him-5(e1490)V;olels5[rab-3::nCre(3ng/µl)+unc-122::RFP(30ng/µl)]

E. coli strains:
BW25113(DE3) L4440
BW25113(DE3) ΔfepD L4440
BW25113(DE3) atfs-1 RNAi
BW25113(DE3) ΔfepD atfs-1 L4440

@author: sm5911
@date: 11/04/2026

"""

#%% Imports

from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import transforms
import seaborn as sns

from preprocessing.compile_hydra_data import compile_metadata
from preprocessing.compile_hydra_data import process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results
from tierpsytools.analysis.statistical_tests import univariate_tests
from tierpsytools.analysis.statistical_tests import get_effect_sizes
from tierpsytools.analysis.statistical_tests import _multitest_correct
from visualisation.plotting_helper import sig_asterix

#%% Globals

PROJECT_DIR = "/Users/sm5911/Documents/PhD_DLBG/Keio_worm_pdfr-1_RNAi_mutants_on_atfs-1_and_L4440_bacteria"

#%% Functions

def stats(metadata,
          features,
          group_by,
          control,
          feat='speed_50th',
          pvalue_threshold=0.05,
          fdr_method=None):
        
    assert all(metadata.index == features.index)
    features = features[[feat]]
             
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

    return ttest_results


def main():
    
    metadata_path_clean = Path(AUX_DIR) / "metadata_clean.csv"
    features_path_clean = Path(AUX_DIR) / "features_clean.csv"
    
    # compile combined metadata & feature summaries, clean data and save
    if not metadata_path_clean.exists() or not features_path_clean.exists():

        # compile metadata and feature summaries        
        metadata, metadata_path = compile_metadata(AUX_DIR, 
                                                   n_wells=6, 
                                                   add_well_annotations=False,
                                                   from_source_plate=False)
                
        features, metadata = process_feature_summaries(metadata_path, 
                                                       results_dir=RES_DIR, 
                                                       compile_day_summaries=False, 
                                                       imaging_dates=None, 
                                                       align_bluelight=False, 
                                                       window_summaries=True,
                                                       n_wells=6)

        # Clean results - Remove bad well data + features with too many NaNs/zero std + impute NaNs
        features, metadata = clean_summary_results(features, 
                                                   metadata,
                                                   feature_columns=None,
                                                   nan_threshold_row=0.8,
                                                   nan_threshold_col=0.05,
                                                   max_value_cap=1e15,
                                                   imputeNaN=True,
                                                   min_nskel_per_video=None,
                                                   min_nskel_sum=None,
                                                   drop_size_related_feats=False,
                                                   norm_feats_only=False)
        
        # save clean metadata and features
        metadata.to_csv(metadata_path_clean, index=False)
        features.to_csv(features_path_clean, index=False)
        
    else:
        metadata = pd.read_csv(metadata_path_clean, header=0, index_col=None, 
                               dtype={'comments':str, 'date_yyyymmdd':int})
        features = pd.read_csv(features_path_clean, header=0, index_col=None)
        
    # rename columns in metadata
    metadata['worm_strain'] = metadata['worm_strain_y']
    metadata['bacteria_strain'] = metadata['RNA_strain']
    
    # subset metadata results for bluelight videos only 
    bluelight_videos = [i for i in metadata['imgstore_name'] if 'bluelight' in i]
    metadata = metadata[metadata['imgstore_name'].isin(bluelight_videos)]
    
    features = features.reindex(metadata.index)
    
    worm_strain_list = sorted(metadata['worm_strain'].unique())
    
    # STATS: t-tests of L4440 vs atfs-1 in BW and fepD background for each 
    # worm strain (corrected for multiple comparisons afterwards)  
    control = 'BW L4440'
    test_results_dict = {}
    for worm in worm_strain_list:
        meta_worm = metadata[metadata['worm_strain']==worm]
        feat_worm = features.reindex(meta_worm.index)
        
        # All bacterial strains vs control strain BW L4440
        ttest_results_worm = stats(meta_worm,
                                   feat_worm,
                                   group_by='bacteria_strain',
                                   control=control,
                                   feat='speed_50th',
                                   pvalue_threshold=0.05,
                                   fdr_method=None)
        test_results_dict[worm] = ttest_results_worm
            
    # extract results into dataframe and correct for multiple comparisons
    test_results_df = pd.DataFrame()
    for key, values in test_results_dict.items():
        values.index = [key]
        test_results_df = pd.concat([test_results_df, values])
            
    pval_cols = [c for c in test_results_df.columns if 'pvals_' in c]
    c_reject, c_pvals = _multitest_correct(test_results_df[pval_cols], 
                                           multitest_method='fdr_by', 
                                           fdr=0.05)
    c_reject.columns = [c.replace('pvals_','reject_') for c in c_reject.columns]
    test_results_df[c_reject.columns] = c_reject
    test_results_df[c_pvals.columns] = c_pvals

    # save results
    test_results_path = RES_DIR / 't-test_results.csv'
    test_results_df.to_csv(test_results_path, header=True, index=True)

    # PLOTS: boxplots of L4440 vs atfs-1 in BW and fepD background for each 
    # worm strain
    metadata['treatment'] = metadata[['worm_strain','bacteria_strain']
                                      ].astype(str).agg('_'.join, axis=1)
    
    for worm in worm_strain_list:
        meta_worm = metadata[metadata['worm_strain']==worm]
        feat_worm = features.reindex(meta_worm.index)
        
        plot_df = meta_worm[['worm_strain','bacteria_strain','date_yyyymmdd']
                            ].join(feat_worm[['speed_50th']]).sort_values(
                                by=['worm_strain','bacteria_strain','date_yyyymmdd'])
                                
        # save plot data to file
        plot_data_path = RES_DIR / '{}_data.csv'.format(worm)
        plot_df.to_csv(plot_data_path, header=True, index=False)
        
        order = ['BW L4440', 'fepD L4440', 'BW atfs-1', 'fepD atfs-1']
        colour_dict = dict(zip(order, 
                               sns.color_palette(palette='tab10', n_colors=2) +\
                               sns.color_palette(palette='tab10', n_colors=2)))
            
        plt.close('all')
        sns.set_style('ticks')
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(1,1,1)
        sns.boxplot(x='bacteria_strain',
                    y='speed_50th',
                    data=plot_df, 
                    order=order,
                    hue='bacteria_strain',
                    hue_order=order,
                    dodge=False,
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
            sns.stripplot(x='bacteria_strain',
                          y='speed_50th',
                          data=date_df,
                          order=order,
                          hue='bacteria_strain',
                          hue_order=order,
                          dodge=False,
                          palette=[date_lut[date]] * len(order),
                          color='dimgray',
                          marker=".",
                          edgecolor='k',
                          linewidth=0.3,
                          s=12) #facecolors="none"
            
        ax.axes.get_xaxis().get_label().set_visible(False) # remove y axis label
        ax.axes.set_xticklabels([l.get_text() for l in ax.axes.get_xticklabels()], 
                                fontsize=20) #rotation=90
        ax.tick_params(axis='x', which='major', pad=15)
        ax.axes.set_ylabel('Speed (µm s$^{-1}$)', fontsize=25, labelpad=15)                             
        plt.yticks(fontsize=20)
        plt.ylim(-40, 220)
                
        # scale y axis for annotations    
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled
        
        # add p-values to plot
        for i, text in enumerate(ax.axes.get_xticklabels()):
            strain = text.get_text()
            if strain != control:
                p = test_results_df.loc[worm, 'pvals_' + strain]
                p_text = sig_asterix([p])[0]
                ax.text(i, 1.02, p_text, fontsize=20, 
                        ha='center', va='center', transform=trans)
                # # Plot the bar: [x1,x1,x2,x2],[bar_tips,bar_height,bar_height,bar_tips]
                # plt.plot([i-1, i-1, i, i],[0.92, 0.94, 0.94, 0.92], 
                #          lw=1, c='k', transform=trans)

        boxplot_path = RES_DIR / '{}.svg'.format(worm)
        plt.savefig(boxplot_path, bbox_inches='tight', transparent=True)

    return

#%% Main

if __name__ == "__main__":
    
    AUX_DIR = Path(PROJECT_DIR) / "AuxiliaryFiles"
    RES_DIR = Path(PROJECT_DIR) / "Results"
    
    main()