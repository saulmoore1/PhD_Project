#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse Initial Keio Screen 

- compile metadata and feature summaries
- clean metadata and feature summaries
- run analysis

@author: sm5911
@date: 28/11/2024

"""

#%% Imports

import numpy as np
import pandas as pd
from pathlib import Path

from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from preprocessing.append_supplementary_info import load_supplementary_7, append_supplementary_7
from filter_data.clean_feature_summaries import clean_summary_results
from tierpsytools.preprocessing.filter_data import select_feat_set
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes, _multitest_correct
from visualisation.plotting_helper import sig_asterix

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Screen_Initial"
SAVE_DIR = "/Users/sm5911/Documents/PhD_DLBG/3_Keio_Screen_Initial"

PATH_SUP_INFO = Path(PROJECT_DIR) / "AuxiliaryFiles/Baba_et_al_2006/Supporting_Information/Supplementary_Table_7.xls"

EXPERIMENT_DATES = ["20210406", "20210413", "20210420", "20210427", "20210504", "20210511"]
N_TIERPSY_FEATS = 256
MIN_NSKEL_SUM = 6000

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
P_VALUE_THRESHOLD = 0.05
FDR_METHOD = 'fdr_bh' # fdr_by
N_WELLS = 96
N_SIG_FEATS = 50

RENAME_DICT = {"FECE" : "fecE",
               "AroP" : "aroP",
               "TnaB" : "tnaB"}

#BLUELIGHT_TIMEPOINTS_SECONDS = [(60, 70),(160, 170),(260, 270)]

#%% Functions

def stats(metadata,
          features,
          group_by='gene_name',
          control='wild_type',
          test='ANOVA',
          feature_list=None,
          save_dir=None,
          p_value_threshold=0.05,
          fdr_method='fdr_bh'):
    
    """ Perform ANOVA tests to compare worms on each bacterial food vs BW25113 control for each 
        Tierpsy feature in feature_list """

    assert all(metadata.index == features.index)
    
    if feature_list is None:
        feature_list = list(features.columns)
    else:
        assert type(feature_list) == list
        assert all(f in features.columns for f in feature_list)
        features = features[feature_list]
        
    n_strains = metadata[group_by].nunique()
    print("Performing %s tests for %d %ss (across %d features)" % (test, n_strains, group_by, len(feature_list)))    
    
    # perform ANOVA + record results before & after correcting for multiple comparisons               
    stats, pvals, reject = univariate_tests(X=features, 
                                            y=metadata[group_by], 
                                            control=control, 
                                            test=test,
                                            comparison_type='multiclass',
                                            multitest_correction=None,
                                            alpha=p_value_threshold,
                                            n_permutation_test=None)

    # get effect sizes
    effect_sizes = get_effect_sizes(X=features, 
                                    y=metadata[group_by],
                                    control=control,
                                    effect_type=None,
                                    linked_test=test)

    # compile + save results (uncorrected)
    results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
    results.columns = ['stats','effect_size','pvals','reject']     
    results['significance'] = sig_asterix(results['pvals'])
    results = results.sort_values(by=['pvals'], ascending=True) # rank pvals
    
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        results_path = Path(save_dir) / '{}_results_uncorrected.csv'.format(test)
        results.to_csv(results_path, header=True, index=True)

    # correct for multiple comparisons
    if fdr_method is not None:
        reject_corrected, pvals_corrected = _multitest_correct(pvals, 
                                                               multitest_method=fdr_method,
                                                               fdr=p_value_threshold)
                                                
        # compile + save results (corrected)
        results = pd.concat([stats, effect_sizes, pvals_corrected, reject_corrected], axis=1)
        results.columns = ['stats','effect_size','pvals','reject']     
        results['significance'] = sig_asterix(results['pvals'])
        results = results.sort_values(by=['pvals'], ascending=True) # rank pvals
        
        if save_dir is not None:
            results_corrected_path = Path(save_dir) / '{}_results_corrected.csv'.format(test)
            results.to_csv(results_corrected_path, header=True, index=True)
        
    # use reject mask to find significant feature set
    fset = pvals.loc[reject[test]].sort_values(by=test, ascending=True).index.to_list()
    assert set(fset) == set(results['pvals'].index[np.where(results['pvals'] < p_value_threshold)[0]])

    if len(fset) > 0:
        print("%d significant features found by %s for '%s' (P<%.2f, %s)" %\
              (len(fset), test, group_by, p_value_threshold, fdr_method))
        if save_dir is not None:
            sigfeats_path = save_dir / '{}_sigfeats.txt'.format(test)
            
            # write significant feature list to file
            with open(str(sigfeats_path), 'w') as fid:
                for line in fset:
                    fid.write("%s\n" % line)
    
    return results, fset

def main():
    
    metadata_path_local = Path(SAVE_DIR) / 'metadata.csv'
    features_path_local = Path(SAVE_DIR) / 'features.csv'
    
    aux_dir = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    res_dir = Path(PROJECT_DIR) / 'Results'
    
    if not metadata_path_local.exists() or not features_path_local.exists():

        # compile metadata and feature summaries        
        metadata, metadata_path = compile_metadata(aux_dir, 
                                                   imaging_dates=EXPERIMENT_DATES,
                                                   n_wells=N_WELLS, 
                                                   add_well_annotations=True,
                                                   from_source_plate=False)
        
        features, metadata = process_feature_summaries(metadata_path, 
                                                       results_dir=res_dir, 
                                                       compile_day_summaries=True, 
                                                       imaging_dates=EXPERIMENT_DATES, 
                                                       align_bluelight=True, 
                                                       window_summaries=False,
                                                       n_wells=N_WELLS)

        # fill in 'gene_name' for control as 'wild_type' (for control plates where gene name is missing)
        metadata.loc[metadata['source_plate_id'] == "BW", 'gene_name'] = "wild_type"

        # rename gene names in metadata
        for k, v in RENAME_DICT.items():
            metadata.loc[metadata['gene_name'] == k, 'gene_name'] = v

        # clean results - Remove bad well data + features with too many NaNs/zero std + impute NaNs
        features, metadata = clean_summary_results(features,
                                                   metadata,
                                                   feature_columns=None,
                                                   nan_threshold_row=NAN_THRESHOLD_ROW,
                                                   nan_threshold_col=NAN_THRESHOLD_COL,
                                                   max_value_cap=1e15,
                                                   imputeNaN=True,
                                                   min_nskel_per_video=None,
                                                   min_nskel_sum=MIN_NSKEL_SUM,
                                                   drop_size_related_feats=False,
                                                   norm_feats_only=False,
                                                   percentile_to_use=None,
                                                   no_nan_cols=['worm_strain','gene_name'])
        
        assert not any(metadata['worm_strain'].isna()) and not any(metadata['gene_name'].isna())
        
        # assert no duplicate genes due to case-sensitivity
        assert len(metadata['gene_name'].unique()) == len(metadata['gene_name'].str.upper().unique())
        
        # add COG category info from Baba et al. (2006) supplementary info to metadata
        if not 'COG_category' in metadata.columns:
            supplementary_7 = load_supplementary_7(PATH_SUP_INFO)
            metadata = append_supplementary_7(metadata, supplementary_7, column_name='gene_name')
            assert set(metadata.index) == set(features.index)    
        COG_families = {'Information storage and processing' : ['J', 'K', 'L', 'D', 'O'], 
                        'Cellular processes' : ['M', 'N', 'P', 'T', 'C', 'G', 'E'], 
                        'Metabolism' : ['F', 'H', 'I', 'Q', 'R'], 
                        'Poorly characterised' : ['S', 'U', 'V']}
        COG_mapping_dict = {i : k for (k, v) in COG_families.items() for i in v}        
        COG_info = []
        for i in metadata['COG_category']:
            try:
                COG_info.append(COG_mapping_dict[i])
            except:
                COG_info.append('Unknown') # np.nan
        metadata['COG_info'] = COG_info
        
        # save clean metadata and features
        metadata.to_csv(metadata_path_local, index=False)
        features.to_csv(features_path_local, index=False)
        
    else:
        metadata = pd.read_csv(metadata_path_local, header=0, index_col=None, dtype={'comments':str})
        features = pd.read_csv(features_path_local, header=0, index_col=None)
        
    # subset for Tierpsy 256 feature set
    features = select_feat_set(features,
                               tierpsy_set_name='tierpsy_{}'.format(N_TIERPSY_FEATS), 
                               append_bluelight=True)
    
    stats_dir = Path(SAVE_DIR) / 'Stats'
    plots_dir = Path(SAVE_DIR) / 'Plots'
    
    strain_list = sorted(metadata['gene_name'].unique())
    print("%d bacterial strains in total will be analysed" % len(strain_list))
    
# =============================================================================
#     # F-test for equal variances - since sample sizes are not equal, first check for homogeneity of
#     # variance before performing t-tests or ANOVA (if not, then use Mann-Whitney or Kruskal-Wallis)
#
#     from statistical_testing.stats_helper import levene_f_test
#     f_test_stats_path = stats_dir / 'f-test_results.csv'
#     stats_dir.mkdir(parents=True, exist_ok=True)
#     if not f_test_stats_path.exists():
#         group_by = 'gene_name'
#         levene_stats = levene_f_test(features, 
#                                      metadata, 
#                                      group_by,
#                                      p_value_threshold=P_VALUE_THRESHOLD, 
#                                      multitest_method=FDR_METHOD,
#                                      saveto=f_test_stats_path,
#                                      del_if_exists=False)
#         # if p<0.05 then variances are not equal and sample size matters
#         prop_eqvar = (levene_stats['pval'] > P_VALUE_THRESHOLD).sum() / len(levene_stats['pval'])
#         print("Percentage equal variance %.1f%%" % (prop_eqvar * 100))
# =============================================================================

    # Perform ANOVA for each feature and correct for multiple comparisons
    anova_results, fset = stats(metadata,
                                features,
                                group_by='gene_name',
                                control='wild_type',
                                test='ANOVA', # 'Kruskal-Wallis'
                                feature_list=None,
                                save_dir=stats_dir,
                                p_value_threshold=P_VALUE_THRESHOLD,
                                fdr_method=FDR_METHOD)
    

    return

#%% Main

if __name__ == '__main__':
    main()
