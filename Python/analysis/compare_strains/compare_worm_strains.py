#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse between strain variation

@author: saul.moore11@lms.mrc.ac.uk
@date: 13/01/2021
"""

#%% Imports

import sys
import argparse
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import (ttest_ind,
                         #f_oneway,
                         #kruskal,
                         zscore)

# Path to Github helper functions (USER-DEFINED path to local copy of Github repo)
PATH_LIST = ['/Users/sm5911/Tierpsy_Versions/tierpsy-tools-python/',
             '/Users/sm5911/Documents/GitHub/PhD_Project/Python']
for sysPath in PATH_LIST:
    if sysPath not in sys.path:
        sys.path.insert(0, sysPath)
        
# Custom functions
from preprocessing.compile_hydra_data import process_metadata, process_feature_summaries
from preprocessing.clean_data import clean_features_summaries
from analysis.statistical_testing import (shapiro_normality_test,
                                          ranksumtest,
                                          ttest_by_feature,
                                          anova_by_feature)

from analysis.compare_strains.helper import (load_top256,
                                             plot_day_variation,
                                             barplot_sigfeats,
                                             boxplots_sigfeats,
                                             boxplots_grouped,
                                             plot_clustermap,
                                             plot_barcode_heatmap,
                                             plot_pca,
                                             remove_outliers_pca,
                                             plot_tSNE,
                                             plot_umap)

from tierpsytools.analysis.significant_features import k_significant_feat
from tierpsytools.drug_screenings.filter_compounds import compounds_with_low_effect_univariate #lmm

#%% Functions

def sig_asterix(pvalues_array):
    """ A function for converting p-values to asterisks 
        for showing significance on plots """
    asterix = []
    for p in pvalues_array:
        if p < 0.001:
            asterix.append('***')
        elif p < 0.01:
            asterix.append('**')
        elif p < 0.05:
            asterix.append('*')
        else:
            asterix.append('')
    return asterix
                
#%% Main

#TODO: Make so that you can provide a manual set of selected features for fset and it will plot 
#       for those features only

if __name__ == "__main__":
    
    # Accept command-line inputs # TODO: Read from JSON instead?
    parser = argparse.ArgumentParser(description='Analyse Tierpsy results (96-well)')
    parser.add_argument('--project_dir', help="Project root directory,\
                        containing 'AuxiliaryFiles', 'RawVideos',\
                        'MaskedVideos' and 'Results' folders",
                        default='/Volumes/hermes$/KeioScreen_96WP', type=str)
    parser.add_argument('--grouping_variable', help="Categorical variable that you wish to \
                        investigate (eg. 'worm_strain')", nargs='+', default='food_type')
    # Keio = ['food_type','instrument_name','lawn_growth_duration_hours','lawn_storage_type']
    parser.add_argument('--compile_day_summaries', help="Compile full feature summaries from \
                        day feature summary results", default=True, action='store_false')
    parser.add_argument('--add_well_annotations', help="Add 'is_bad_well' labels \
                        from WellAnnotator GUI", default=True,
                        action='store_false')
    parser.add_argument('--omit_strains', help="List of strains in 'analyse_variables' \
                        to omit from the analysis", nargs='+', default=['NA'])
    parser.add_argument('--dates', help="List of imaging dates to use for analysis.\
                        If None, all imaging dates will be investigated", nargs='+',
                        default=['20210126','20210127']) # ['20201208', '20201209']
    parser.add_argument('--runs', help="List of imaging run numbers to use for \
                        analysis. If None, all imaging runs will be investigated",
                        nargs='+', default=None)
    parser.add_argument('--test', help="Choose between 'LMM' (if >1 day replicate), 'ANOVA' or \
                        'Kruskal' (Kruskal tests will be performed instead of ANOVA if check_normal \
                        and data is not normally distributed) If significant features are found, \
                        pairwise t-tests are performed.", default='Kruskal')
    parser.add_argument('--use_top256', help="Use Tierpsy Top256 features only",
                        default=False, action='store_true')
    parser.add_argument('--percentile_to_use', help="Use only feature summaries for given \
                        percentile in analysis (eg. '50th')", default=None, type=str)
                        # TODO: Change this to percentile to use
    parser.add_argument('--drop_size_features', help="Remove size-related Tierpsy \
                        features from analysis", default=False, action='store_true')
                        # TODO: Size-related features only (drop behaviour)
    parser.add_argument('--norm_features_only', help="Use only normalised \
                        size-invariant features ('_norm') for analysis",
                        default=False, action='store_true')
    parser.add_argument('--check_normal', help="Perform Shapiro-Wilks test for \
                        normality to decide between parametric/non-parametric \
                        statistics", default=False, action='store_true')
    parser.add_argument('--remove_outliers', help="Use Mahalanobis distance to \
                        exclude outliers from analysis", default=False, 
                        action='store_true')  
    parser.add_argument('--nan_threshold', help="Threshold proportion of NaN values \
                        to drop feature from analysis", default=0.2, type=float)
    parser.add_argument('--pval_threshold', help="Threshold p-value for statistical \
                        significance", default=0.05, type=float)
    parser.add_argument('--k_sig_features', help="Number of most significantly \
                        different features to plot", default=100, type=int)  
    parser.add_argument('--selected_features_path', help="Path to manually selected intuitive \
                        features for publication", default=None, type=str)
                       #'/Users/sm5911/Documents/tmp_analysis/Filipe/manually_selected_features.csv'
    args = parser.parse_args()
    
    print("\nInputs:\n")
    for arg in list(args._get_kwargs()):
        print('%s -- %s' % (arg[0], str(arg[1])))
    print('\n')
        
    PROJECT_DIR = Path(args.project_dir)                        # str
    TEST_NAME = args.test                                       # str
    IMAGING_DATES = args.dates                                  # list
    IMAGING_RUNS = args.runs                                    # list
    GROUPING_VAR = args.grouping_variable                       # list
    OMIT_STRAINS = args.omit_strains                            # list
    COMPILE_DAY_SUMMARIES = args.compile_day_summaries          # bool
    ADD_WELL_ANNOTATIONS = args.add_well_annotations            # bool
    CHECK_NORMAL = args.check_normal                            # bool
    USE_TOP256 = args.use_top256                                # bool
    FILTER_SIZE_FEATS = args.drop_size_features                 # bool
    NORM_FEATS_ONLY = args.norm_features_only                   # bool
    PERCENTILE = args.percentile_to_use                         # bool
    REMOVE_OUTLIERS = args.remove_outliers                      # bool
    NAN_THRESHOLD = args.nan_threshold                          # float
    P_VALUE_THRESHOLD = args.pval_threshold                     # float
    K_SIG_FEATS = args.k_sig_features                           # int
    
    # IO paths
    aux_dir = PROJECT_DIR / "AuxiliaryFiles"
    results_dir = PROJECT_DIR / "Results"
    save_dir = Path('/Users/sm5911/Documents/tmp_analysis/Keio') # PROJECT_DIR / "Analysis"

    fn = 'Top256' if USE_TOP256 else 'All_features'
    fn = fn + '_noSize' if FILTER_SIZE_FEATS else fn
    fn = fn + '_norm' if NORM_FEATS_ONLY else fn
    fn = fn + '_' + PERCENTILE if PERCENTILE is not None else fn
    fn = fn + '_noOutliers' if REMOVE_OUTLIERS else fn

    # Other Globals
    CONTROL_DICT = {'worm_strain': 'N2',
                    'drug_type': 'DMSO',
                    'food_type': 'icd', #'OP50',
                    'instrument_name': 'Hydra01',
                    'worm_life_stage': 'D1',
                    'lawn_growth_duration_hours': '8',
                    'lawn_storage_type': 'old'}
    
    CONTROL = CONTROL_DICT[GROUPING_VAR]
    FDR_METHOD = 'fdr_by' # Benjamini-Yekutieli correction for multiple testing
    RANDOM_EFFECT = 'date_yyyymmdd'
    USE_K_SIG_FEATS_OVERLAP = False # Restrict significant feature set to overlap with k sig feats?
    DO_STATS = False
    MAX_FEATURES_PLOT_CAP = None

    #%% Compile and clean results
    
    # Process metadata    
    metadata = process_metadata(aux_dir=aux_dir, 
                                imaging_dates=IMAGING_DATES, 
                                add_well_annotations=ADD_WELL_ANNOTATIONS)
    
    # # Calculate duration on food + L1 diapause duration
    # metadata = duration_on_food(metadata) 
    # metadata = duration_L1_diapause(metadata)
       
    # Process feature summary results
    features, metadata = process_feature_summaries(metadata, 
                                                   results_dir,
                                                   compile_day_summaries=COMPILE_DAY_SUMMARIES,
                                                   imaging_dates=IMAGING_DATES,
                                                   add_bluelight=True)
    
    # Clean: remove data with too many NaNs/zero std and impute remaining NaNs
    features, metadata = clean_features_summaries(features, 
                                                  metadata,
                                                  feature_columns=None,
                                                  imputeNaN=True,
                                                  nan_threshold=NAN_THRESHOLD,
                                                  max_value_cap=1e15,
                                                  drop_size_related_feats=FILTER_SIZE_FEATS,
                                                  norm_feats_only=NORM_FEATS_ONLY,
                                                  percentile_to_use=PERCENTILE)
        
    # Save full results to file
    full_results_path = save_dir / (fn + '_full_results.csv')
    if not full_results_path.exists():
        full_results_path.parent.mkdir(exist_ok=True, parents=True)
        fullresults = metadata.join(features) # join metadata + results
        print("Saving full results (features/metadata) to:\n '%s'" % full_results_path)
        fullresults.to_csv(full_results_path, index=False)

    #%% IO assertions + subset results
    
    # Load Tierpsy Top256 feature set
    if USE_TOP256:     
        top256_path = aux_dir / 'top256_tierpsy_no_blob_no_eigen_only_abs_no_norm.csv'
        top256 = load_top256(top256_path, remove_path_curvature=True, add_bluelight=True)
        
        # Ensure results exist for features in featurelist
        top256_feat_list = [feat for feat in top256 if feat in features.columns]
        print("Dropped %d features in Top256 that are missing from results" %\
              (len(top256)-len(top256_feat_list)))
        
        # Select features for analysis
        features = features[top256_feat_list]
    
    # Subset results (rows) for imaging dates of interest
    if IMAGING_DATES:
        assert all([i in list(metadata['date_yyyymmdd'].unique().astype(str)) for 
                    i in IMAGING_DATES])
        metadata = metadata[metadata['date_yyyymmdd'].isin(IMAGING_DATES)]
        features = features.reindex(metadata.index)
            
    # Subset results (rows) to remove NA groups
    metadata = metadata[~metadata[GROUPING_VAR].isna()]
    features = features.reindex(metadata.index)
    
    # Subset results (rows) for strains of interest (check case-sensitive)
    strain_list = list(metadata[GROUPING_VAR].unique())
    assert len(strain_list)==len(metadata[GROUPING_VAR].str.upper().unique())
    
    if OMIT_STRAINS:
        strain_list = [s for s in strain_list if s.upper() not in [o.upper() for o in OMIT_STRAINS]]
        metadata = metadata[metadata[GROUPING_VAR].isin(strain_list)]
        features = features.reindex(metadata.index)
       
    # Record imaging runs to analyse
    if IMAGING_RUNS:
        assert type(IMAGING_RUNS) == list
        imaging_run_list = IMAGING_RUNS if type(IMAGING_RUNS) == list else [IMAGING_RUNS]
    else:
        imaging_run_list = list(metadata['imaging_run_number'].unique())
        print("Found %d imaging runs to analyse: %s" % (len(imaging_run_list), imaging_run_list))
    
    # If LMM is chosen, ensure that there are multiple day replicates to compare at each timepoint
    assert TEST_NAME in ['ANOVA','Kruskal','LMM']
    if TEST_NAME == 'LMM':
        assert GROUPING_VAR in ['worm_strain','food_type','drug_type']
        assert all(len(metadata.loc[metadata['imaging_run_number']==t, RANDOM_EFFECT].unique()) > 1 
                   for t in imaging_run_list)
        
    if CHECK_NORMAL:
        # Sample data from a random run to see if normal. If not, use Kruskal-Wallis test instead.
        _r = np.random.choice(imaging_run_list, size=1)[0]
        _rMeta = metadata[metadata['imaging_run_number']==_r]
        _rFeat = features.reindex(_rMeta.index)
        
        normtest_savepath = save_dir / (fn + "_run{}_shapiro_results.csv".format(_r))
        normtest_savepath.parent.mkdir(exist_ok=True, parents=True)
        (prop_features_normal, is_normal) = shapiro_normality_test(features_df=_rFeat,
                                                                metadata_df=_rMeta,
                                                                group_by=GROUPING_VAR,
                                                                p_value_threshold=P_VALUE_THRESHOLD,
                                                                verbose=True)  
        if not is_normal:
            TEST_NAME = 'Kruskal' if TEST_NAME == 'ANOVA' else TEST_NAME
            print("WARNING: Data is not normal! Kruskal-Wallis tests will be used instead of ANOVA")
            
        # Save normailty test results to file
        prop_features_normal.to_csv(normtest_savepath, 
                                    index=True, 
                                    index_label=GROUPING_VAR, 
                                    header='prop_normal')
    else:
        TEST_NAME = 'Kruskal' if TEST_NAME == 'ANOVA' else TEST_NAME
        is_normal = False # Default non-parametric
        
    #%% Analyse variables

    print("\nInvestigating '%s' variation" % GROUPING_VAR)    
    for run in imaging_run_list:
        print("\nAnalysing imaging run %d" % run)
        
        # Subset results to investigate single imaging run
        meta_df = metadata[metadata['imaging_run_number']==run]
        feat_df = features.reindex(meta_df.index)
        
        print("n = %d wells for run %d" % (meta_df.shape[0], run))
        mean_sample_size = int(np.round(meta_df.join(feat_df).groupby([GROUPING_VAR, 
                               'date_yyyymmdd'], as_index=False).size().mean()))
        print("Mean sample size: %d" % mean_sample_size)
        
        # Clean subsetted data: drop NaNs, zero std, etc
        feat_df, meta_df = clean_features_summaries(feat_df, 
                                                    meta_df, 
                                                    max_value_cap=False,
                                                    imputeNaN=False)
        # Save paths
        run_save_dir = save_dir / fn / "Run_{}".format(run) / (GROUPING_VAR + '_variation')
        stats_dir =  run_save_dir / "Stats"
        plot_dir = run_save_dir / "Plots"

        #%% STATISTICS
        #   One-way ANOVA/Kruskal-Wallis tests for significantly different 
        #   features across groups (e.g. strains)
    
        # When comparing more than 2 groups, perform ANOVA and proceed only to 
        # pairwise two-sample t-tests if there is significant variability among 
        # all groups for any feature
        run_strain_list = list(meta_df[GROUPING_VAR].unique())
        
        # Record name of t-test
        T_TEST = ttest_ind if is_normal else ranksumtest
        T_TEST_NAME = str(T_TEST).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]
        
        # TODO: Make this instead 'if not exists stats results file...'
        if DO_STATS:
            # Create table to store statistics results
            grouped = feat_df.join(meta_df[GROUPING_VAR]).groupby(by=GROUPING_VAR)
            stats_table = grouped.mean().T
            mean_cols = ['mean ' + v for v in stats_table.columns.to_list()]
            stats_table.columns = mean_cols
            for group in grouped.size().index:
                stats_table['sample size {}'.format(group)] = grouped.size().loc[group]

            # if not (TEST_NAME == 'ANOVA' and len(strain_list) == 2):
            #     print("WARNING: Not enough groups for ANOVA. Performing t-tests instead")
            # (signif_effect, low_effect, error, mask, 
            #   pvalues)=compounds_with_low_effect_univariate(feat=feat_df, 
            #                                           drug_name=meta_df[GROUPING_VAR], 
            #                                           drug_dose=None, 
            #                                           random_effect=(meta_df[RANDOM_EFFECT] if 
            #                                                          TEST_NAME == 'LMM' else None), 
            #                                           control=CONTROL, 
            #                                           test=TEST_NAME, 
            #                                           comparison_type='multiclass',
            #                                           multitest_method=FDR_METHOD,
            #                                           ignore_names=None, 
            #                                           return_pvals=True)
            
            if (TEST_NAME == 'ANOVA' or TEST_NAME == 'Kruskal'):
                if len(run_strain_list) > 2:
                    pvalues, anova_sigfeats_list = anova_by_feature(feat_df=feat_df, 
                                                                    meta_df=meta_df, 
                                                                    group_by=GROUPING_VAR, 
                                                                    strain_list=run_strain_list, 
                                                                    p_value_threshold=P_VALUE_THRESHOLD, 
                                                                    is_normal=is_normal, 
                                                                    fdr_method=FDR_METHOD)
                    # TODO: Use TT function: compounds_with_low_effect_univariate
                                
                    # Record name of statistical test used (kruskal/f_oneway)
                    col = '{} p-value'.format(TEST_NAME)
                    stats_table[col] = pvalues.loc['pval', stats_table.index]
        
                    # TODO: Save statistics at the end
                    # Save test statistics to file
                    stats_outpath = stats_dir / '{}_results.csv'.format(TEST_NAME)
                    stats_outpath.parent.mkdir(exist_ok=True, parents=True)
                    pvalues.to_csv(stats_outpath)
                    
                    sigfeats_outpath = Path(str(stats_outpath).replace('_results.csv',
                                                                        '_significant_features.csv'))
                    anova_sigfeats_list.to_csv(sigfeats_outpath, index=False)
                    
                    # Record number of signficant features by ANOVA
                    fset = list(pvalues.columns[pvalues.loc['pval'] < P_VALUE_THRESHOLD])  
                    
                    if len(fset) > 0:
                        print("%d significant features found for %s (run %d, %s, P<%.2f, %s)" %\
                              (len(fset), GROUPING_VAR, run, TEST_NAME, P_VALUE_THRESHOLD, FDR_METHOD))
                else:
                    fset = []
                    print("WARNING: Not enough groups for ANOVA (n=%d groups, run %d)" %\
                          (len(run_strain_list), run))
        
            ###   Linear Mixed Models (LMMs) to account for day-to-day variation when comparing 
            #     between worm/food/drug type
            elif TEST_NAME == 'LMM':
                with warnings.catch_warnings():
                    # Filter warnings as parameter is often on the boundary
                    warnings.filterwarnings("ignore")
                    #warnings.simplefilter("ignore", ConvergenceWarning)
                    (signif_effect, low_effect, error, mask, 
                      pvalues)=compounds_with_low_effect_univariate(feat=feat_df, 
                                                              drug_name=meta_df[GROUPING_VAR], 
                                                              drug_dose=None, 
                                                              random_effect=meta_df[RANDOM_EFFECT], 
                                                              control=CONTROL, 
                                                              test=TEST_NAME, 
                                                              comparison_type='multiclass',
                                                              multitest_method=FDR_METHOD,
                                                              ignore_names=None, 
                                                              return_pvals=True)
                assert len(error) == 0
                
                if len(run_strain_list) == 2:
                    col = '{} p-value'.format(TEST_NAME)
                    stats_table[col] = pvalues.loc[pvalues.index[0], stats_table.index]  
                
                # Save LMM significant features
                lmm_path = stats_dir / '{}_results.csv'.format(TEST_NAME)
                lmm_path.parent.mkdir(exist_ok=True, parents=True)
                pvalues.to_csv(lmm_path, header=True, index=True)
                # Ideally report as: parameter | beta | lower-95 | upper-95 | random effect (SD)
                
                # Significant feature set = select feature if significant for any food vs control
                fset = list(pvalues.columns[(pvalues < P_VALUE_THRESHOLD).any() > 0])
                
                if len(signif_effect) > 0:
                    print(("%d significant features found (%d significant %ss vs %s, "\
                          % (len(fset), len(signif_effect), GROUPING_VAR.replace('_',' '), 
                              CONTROL) if len(signif_effect) > 0 else\
                          "No significant differences found between %s "\
                          % GROUPING_VAR.replace('_',' '))
                          + "after accounting for %s variation (run %d, %s, P<%.2f, %s)"\
                          % (RANDOM_EFFECT.split('_yyyymmdd')[0], run, TEST_NAME, P_VALUE_THRESHOLD,
                              FDR_METHOD))
            else:
                fset = []
                
            ###   T-TESTS: If significance is found by ANOVA/LMM, or only 2 groups, perform 
            #     t-tests/rank-sum tests for significant features between each group vs control        
            if len(fset) > 0 or len(run_strain_list) == 2:
                (pvalues_ttest, 
                  sigfeats_table, 
                  sigfeats_df) = ttest_by_feature(feat_df, 
                                                  meta_df, 
                                                  group_by=GROUPING_VAR, 
                                                  control_strain=CONTROL, 
                                                  is_normal=is_normal, 
                                                  p_value_threshold=P_VALUE_THRESHOLD,
                                                  fdr_method=FDR_METHOD,
                                                  verbose=False)
    
                # Record significant feature set
                fset_ttest = list(pvalues_ttest.columns[(pvalues_ttest < 
                                                          P_VALUE_THRESHOLD).sum(axis=0) > 0])
                if len(fset_ttest) > 0:
                    print("%d signficant features found by %s (run %d, %s, P<%.2f)" %\
                          (len(fset_ttest), T_TEST_NAME, run, GROUPING_VAR, P_VALUE_THRESHOLD))
                elif len(fset_ttest) == 0:
                    print("No significant features found for any %s (run %d, %s, P<%.2f)" %\
                          (GROUPING_VAR, run, T_TEST_NAME, P_VALUE_THRESHOLD))
                                     
                # Save test statistics to file
                ttest_outpath = stats_dir / '{}_results.csv'.format(T_TEST_NAME)
                ttest_outpath.parent.mkdir(exist_ok=True, parents=True)
                ttest_sigfeats_outpath = Path(str(ttest_outpath).replace('_results.csv',
                                                                          '_significant_features.csv'))
                sigfeats_df.to_csv(ttest_sigfeats_outpath, index=False) # Save feature list to file
                pvalues_ttest.to_csv(ttest_outpath) # Save test results to CSV
                
                # Investigate t-test significant features if comparing just 2 strains
                if len(run_strain_list) == 2:
                    print("Only 2 groups. Preferring t-test over ANOVA")
                    pvalues = pvalues_ttest
                    fset = fset_ttest
                    
                    # Add pvalues to stats table
                    col = '{} p-value'.format(T_TEST_NAME)
                    stats_table[col] = pvalues.loc[pvalues.index[0], stats_table.index]
    
            # Add stats results to stats table
            if len(run_strain_list) == 2:
                stats_table['significance'] = sig_asterix(pvalues.values[0])
            else:
                stats_table['significance'] = sig_asterix(pvalues.loc['pval'].values)
    
            # Barplot of number of significantly different features for each strain   
            prop_sigfeats = barplot_sigfeats(test_pvalues_df=(pvalues_ttest if (len(fset) > 0 or 
                                                              len(run_strain_list) == 2) else None), 
                                              saveDir=plot_dir,
                                              p_value_threshold=P_VALUE_THRESHOLD,
                                              test_name=T_TEST_NAME)
            
            ### K significant features
            # k_sigfeat_dir = plot_dir / 'k_sig_feats'
            # k_sigfeat_dir.mkdir(exist_ok=True, parents=True)      
            fset_ksig, (scores, pvalues_ksig), support = k_significant_feat(feat=feat_df, 
                                                            y_class=meta_df[GROUPING_VAR], 
                                                            k=(len(fset) if len(fset) > K_SIG_FEATS 
                                                                else K_SIG_FEATS), 
                                                            score_func='f_classif', 
                                                            scale=None, 
                                                            feat_names=None, 
                                                            plot=False, 
                                                            k_to_plot=None, 
                                                            close_after_plotting=True,
                                                            saveto=None, #k_sigfeat_dir
                                                            figsize=None, 
                                                            title=None, 
                                                            xlabel=None)
            
            pvalues_ksig = pd.DataFrame(pd.Series(data=pvalues_ksig, 
                                                  index=fset_ksig, 
                                                  name='k_significant_features')).T
            # Save k most significant features
            pvalues_ksig.to_csv(stats_dir / 'k_significant_features.csv', header=True, index=False)   
            
            # col = 'Top{} k-significant p-value'.format(K_SIG_FEATS)
            # stats_table[col] = np.nan
            # stats_table.loc[fset_ksig, col] = pvalues_ksig.loc['k_significant_features', fset_ksig]
            
            if len(fset) > 0:
                fset_overlap = set(fset).intersection(set(fset_ksig))
                prop_overlap = len(fset_overlap) / len(fset)
                if prop_overlap < 0.5 and len(fset) > 100:
                    raise Warning("Inconsistency in statistics for feature set agreement between "
                                  + "%s and k significant features!" % (T_TEST_NAME if 
                                  len(run_strain_list) == 2 else TEST_NAME)) 
                elif USE_K_SIG_FEATS_OVERLAP:
                    fset = pvalues_ksig.loc['k_significant_features', 
                                            fset_overlap].sort_values(axis=0, ascending=True).index
            else:
                print("NO SIGNIFICANT FEATURES FOUND! "
                      + "Falling back on 'k_significant_feat' feature set for plotting.")
                fset = fset_ksig
            
                # TODO: Actually fall back on k_sig_feats for plotting :P
            ## Save feature set to file
            # save_test_name = T_TEST_NAME if len(run_strain_list) == 2 else TEST_NAME
            # fset_out = run_save_dir / '{}_significant_feature_set.csv'.format(save_test_name)
            # pd.Series(fset).to_csv(fset_out)

        #%% Load statistics results
        
        # Read stats results from file and infer fset
        print("Loading feature set")
        path_pvalues_anova = stats_dir / '{}_results.csv'.format(TEST_NAME)
        path_pvalues_ttest = stats_dir / '{}_results.csv'.format(T_TEST_NAME)
                    
        if path_pvalues_anova.exists() and len(run_strain_list) > 2:
            # Read ANOVA results and record significant features
            pvalues_anova = pd.read_csv(path_pvalues_anova, index_col=0)
            fset = pvalues_anova.columns[pvalues_anova.loc['pval'] < P_VALUE_THRESHOLD].to_list()
            print("%d significant features found by %s (run %d, P<%.2f)" %\
                  (len(fset), TEST_NAME, run, P_VALUE_THRESHOLD))
        elif path_pvalues_ttest.exists() and len(run_strain_list) == 2:
            # Read t-test results for feature summaries
            pvalues_ttest = pd.read_csv(path_pvalues_ttest, index_col=0)
            assert all(f in pvalues_ttest.columns for f in feat_df.columns)
                
            # Record significant features by t-test
            fset = list(pvalues_ttest.columns[(pvalues_ttest < P_VALUE_THRESHOLD).sum(axis=0) > 0])
        else:
            raise Warning("Stats results not found! Please perform stats first.")
        
        # # Read feature set from file
        # fset_in = run_save_dir / '{}_significant_feature_set.csv'.format(load_test_name)
        # fset = pd.read_csv()
        
        # Read pairwise t-test results from file for p-value annotations on plots
        if len(fset) > 0 or len(run_strain_list) == 2:
            print("Loading %s p-values for plotting" % T_TEST_NAME)
            
            if path_pvalues_ttest.exists():
                # Read t-test results for feature summaries
                pvalues_ttest = pd.read_csv(path_pvalues_ttest, index_col=0)
                assert all(f in pvalues_ttest.columns for f in feat_df.columns)
                
                # Record significant features by t-test
                fset_ttest = list(pvalues_ttest.columns[(pvalues_ttest < 
                                                         P_VALUE_THRESHOLD).sum(axis=0) > 0])
                if len(fset_ttest) == 0:
                    pvalues_ttest = None
            else:
                raise Warning("T-test results not found! Please perform stats first.")
                pvalues_ttest =  None
        else:
            pvalues_ttest = None
                              
        #%% Plot day variation
        
        # TODO: Look into why these plots take so long?!
        swarmDir = plot_dir / '{}_variation'.format(RANDOM_EFFECT.split('_yyyymmdd')[0])
        plot_day_variation(feat_df=feat_df,
                           meta_df=meta_df,
                           group_by=GROUPING_VAR,
                           test_pvalues_df=pvalues_ttest,
                           control=CONTROL,
                           day_var='date_yyyymmdd',
                           feature_set=fset,
                           max_features_plot_cap=MAX_FEATURES_PLOT_CAP,
                           p_value_threshold=P_VALUE_THRESHOLD,
                           saveDir=swarmDir,
                           figsize=[(len(run_strain_list)/3 if len(run_strain_list)>10 else 6), 6],
                           sns_colour_palette="tab10",
                           dodge=False, 
                           ranked=True,
                           drop_insignificant=False)
                                                               
        #%% Boxplots of most significantly different features for each strain vs control
        # features ranked by test pvalue significance (lowest first)
        
        # Boxplots of significant features by ANOVA/LMM (across all groups)
        # TODO: Add ANOVA/LMM pval to title of plot?
        boxplots_grouped(feat_meta_df=meta_df.join(feat_df), 
                         group_by=GROUPING_VAR,
                         control_group=CONTROL,
                         test_pvalues_df=pvalues_ttest,
                         feature_set=fset,
                         saveDir=(plot_dir / 'grouped_boxplots'),
                         max_features_plot_cap=None, 
                         max_groups_plot_cap=None,
                         p_value_threshold=P_VALUE_THRESHOLD,
                         drop_insignificant=False,
                         sns_colour_palette="tab10",
                         figsize=[6, (len(run_strain_list)/3 if len(run_strain_list)>10 else 12)],
                         saveFormat='png')
                
        # Boxplots of significant features by pairwise t-test (for each group vs control)
        if pvalues_ttest is not None:
            boxplots_sigfeats(feat_meta_df=meta_df.join(feat_df), 
                              test_pvalues_df=pvalues_ttest, 
                              group_by=GROUPING_VAR, 
                              control_strain=CONTROL, 
                              feature_set=fset, #['speed_norm_50th_bluelight'],
                              saveDir=plot_dir / 'paired_boxplots',
                              max_features_plot_cap=K_SIG_FEATS,
                              p_value_threshold=P_VALUE_THRESHOLD,
                              drop_insignificant=True,
                              verbose=False)
                
        # from tierpsytools.analysis.significant_features import plot_feature_boxplots
        # plot_feature_boxplots(feat_to_plot=fset,
        #                       y_class=GROUPING_VAR,
        #                       scores=pvalues.rank(axis=1),
        #                       feat_df=feat_df,
        #                       pvalues=np.asarray(pvalues).flatten(),
        #                       saveto=None,
        #                       close_after_plotting=False)
        
        #%% Hierarchical Clustering Analysis
        #   - Clustermap of features by strain, to see if data cluster into groups
        #   - Control data is clustered first, feature order is stored and ordering applied to 
        #     full data for comparison
        
        heatmap_saveFormat = 'pdf'
        
        # Extract data for control
        control_feat_df = feat_df[meta_df[GROUPING_VAR]==CONTROL]
        control_meta_df = meta_df.reindex(control_feat_df.index)
        
        # TODO: Investigate control variation module of helper functions for plotting PCA by day, 
        #       rig, well & temperature/humidity during/across runs
        
        control_feat_df, control_meta_df = clean_features_summaries(features=control_feat_df,
                                                                    metadata=control_meta_df)
        
        # Ensure no NaNs or features with zero standard deviation before normalisation
        assert not control_feat_df.isna().sum(axis=0).any()
        assert not (control_feat_df.std(axis=0) == 0).any()

        #zscores = (df-df.mean())/df.std() # minus mean, divide by std
        controlZ_feat_df = control_feat_df.apply(zscore, axis=0)

        # Drop features with NaN values after normalising
        n_cols = len(controlZ_feat_df.columns)
        controlZ_feat_df.dropna(axis=1, inplace=True)
        n_dropped = n_cols - len(controlZ_feat_df.columns)
        if n_dropped > 0:
            print("Dropped %d features after normalisation (NaN)" % n_dropped)

        # plot clustermap for control        
        if len(control_meta_df[RANDOM_EFFECT].unique()) > 1:
            control_clustermap_path = plot_dir / 'HCA' / ('{}_clustermap'.format(CONTROL) + 
                                                          '.{}'.format(heatmap_saveFormat))
            cg = plot_clustermap(featZ=controlZ_feat_df,
                                 meta=control_meta_df,
                                 group_by=[GROUPING_VAR,'date_yyyymmdd'],
                                 col_linkage=None,
                                 method='complete',#[linkage, complete, average, weighted, centroid]
                                 figsize=[18,6],
                                 saveto=control_clustermap_path)
    
            # Extract linkage + clustered features
            col_linkage = cg.dendrogram_col.calculated_linkage
            clustered_features = np.array(controlZ_feat_df.columns)[cg.dendrogram_col.reordered_ind]
        else:
            clustered_features = None
        
        assert not feat_df.isna().sum(axis=0).any()
        assert not (feat_df.std(axis=0) == 0).any()
        
        featZ_df = feat_df.apply(zscore, axis=0)
        
        # Drop features with NaN values after normalising
        # TODO: Do we need these checks?
        n_cols = len(featZ_df.columns)
        featZ_df.dropna(axis=1, inplace=True)
        n_dropped = n_cols - len(featZ_df.columns)
        if n_dropped > 0:
            print("Dropped %d features after normalisation (NaN)" % n_dropped)

        if DO_STATS:
            # Add z-normalised values to stats table
            z_stats = featZ_df.join(meta_df[GROUPING_VAR]).groupby(by=GROUPING_VAR).mean().T
            z_mean_cols = ['z-mean ' + v for v in z_stats.columns.to_list()]
            z_stats.columns = z_mean_cols
            stats_table = stats_table.join(z_stats)
            first_cols = [m for m in stats_table.columns if 'mean' in m]
            last_cols = [c for c in stats_table.columns if c not in first_cols]
            first_cols.extend(last_cols)
            stats_table = stats_table[first_cols].reset_index()
            first_cols.insert(0, 'feature')
            stats_table.columns = first_cols
            stats_table['feature'] = [' '.join(f.split('_')) for f in stats_table['feature']]
            stats_table = stats_table.sort_values(by='{} p-value'.format((T_TEST_NAME if 
                                         len(run_strain_list) == 2 else TEST_NAME)), ascending=True)
            
            # Save stats table to CSV
            stats_table_path = stats_dir / 'stats_summary_table.csv'
            stats_table.to_csv(stats_table_path, header=True, index=None)
        
        # Clustermap of full data       
        full_clustermap_path = plot_dir / 'HCA' / ('{}_full_clustermap'.format(GROUPING_VAR) + 
                                                   '.{}'.format(heatmap_saveFormat))
        fg = plot_clustermap(featZ=featZ_df, 
                             meta=meta_df, 
                             group_by=GROUPING_VAR,
                             col_linkage=None,
                             method='complete',
                             figsize=[20, (len(run_strain_list) / 4 if 
                                           len(run_strain_list) > 10 else 6)],
                             saveto=full_clustermap_path)
        if not clustered_features:
            # If no control clustering (due to no day variation) then use clustered features for 
            # all strains to order barcode heatmaps
            clustered_features = np.array(featZ_df.columns)[fg.dendrogram_col.reordered_ind]
        
        if len(run_strain_list) > 2:
            pvalues_heatmap = pvalues_anova.loc['pval', clustered_features]
        elif len(run_strain_list) == 2:
            pvalues_heatmap = pvalues_ttest.loc[pvalues_ttest.index[0], clustered_features]
        pvalues_heatmap.name = 'P < {}'.format(P_VALUE_THRESHOLD)

        assert all(f in featZ_df.columns for f in pvalues_heatmap.index)

        # Heatmap barcode with selected features, ordered by control clustered feature order
        #   - Read in selected features list  
        if args.selected_features_path is not None and run == 3 and GROUPING_VAR == 'worm_strain':
            fset = pd.read_csv(Path(args.selected_features_path), index_col=None)
            fset = [s for s in fset['feature'] if s in featZ_df.columns] # TODO: Assert this?
            
        # Plot barcode heatmap (grouping by date)
        if len(control_meta_df[RANDOM_EFFECT].unique()) > 1:
            heatmap_date_path = plot_dir / 'HCA' / ('{}_date_heatmap'.format(GROUPING_VAR) + 
                                                    '.{}'.format(heatmap_saveFormat))
            plot_barcode_heatmap(featZ=featZ_df[clustered_features], 
                                 meta=meta_df, 
                                 group_by=['date_yyyymmdd',GROUPING_VAR], 
                                 pvalues_series=pvalues_heatmap,
                                 p_value_threshold=P_VALUE_THRESHOLD,
                                 selected_feats=fset if len(fset) > 0 else None,
                                 saveto=heatmap_date_path,
                                 figsize=[20, (len(run_strain_list) / 4 if 
                                               len(run_strain_list) > 10 else 6)],
                                 sns_colour_palette="Pastel1")
        
        # Plot group-mean heatmap (averaged across days)
        heatmap_path = plot_dir / 'HCA' / ('{}_heatmap'.format(GROUPING_VAR) + 
                                           '.{}'.format(heatmap_saveFormat))
        plot_barcode_heatmap(featZ=featZ_df[clustered_features], 
                             meta=meta_df, 
                             group_by=[GROUPING_VAR], 
                             pvalues_series=pvalues_heatmap,
                             p_value_threshold=P_VALUE_THRESHOLD,
                             selected_feats=fset if len(fset) > 0 else None,
                             saveto=heatmap_path,
                             figsize=[20, (len(run_strain_list) / 4 if 
                                           len(run_strain_list) > 10 else 6)],
                             sns_colour_palette="Pastel1")        
                        
        # feature sets for each stimulus type
        featsets = {}
        for stim in ['_prestim','_bluelight','_poststim']:
            featsets[stim] = [f for f in features.columns if stim in f]

        # TODO: Timeseries analysis of feature across timepoints/stimulus windows/on-off food/etc
        # TODO: sns.relplot / sns.jointplot / sns.lineplot for visualising covariance/correlation
        # between selected features
    
        #%% Principal Components Analysis (PCA)

        if REMOVE_OUTLIERS:
            outlier_path = plot_dir / 'mahalanobis_outliers.pdf'
            feat_df, inds = remove_outliers_pca(df=feat_df, 
                                                features_to_analyse=None, 
                                                saveto=outlier_path)
            meta_df = meta_df.reindex(feat_df.index)
            featZ_df = feat_df.apply(zscore, axis=0)
  
        # plot PCA
        #from tierpsytools.analysis.decomposition import plot_pca
        pca_dir = plot_dir / 'PCA'
        projected_df = plot_pca(featZ=featZ_df, 
                                meta=meta_df, 
                                group_by=GROUPING_VAR, 
                                n_dims=2,
                                control=CONTROL,
                                var_subset=None, 
                                saveDir=pca_dir,
                                PCs_to_keep=10,
                                n_feats2print=10,
                                sns_colour_palette="tab10",
                                hypercolor=False)      
        # TODO: Ensure sns colour palette doees not plot white points
         
        #%%     t-distributed Stochastic Neighbour Embedding (tSNE)

        tsne_dir = plot_dir / 'tSNE'
        perplexities = [5,15,30]
        try:
            tSNE_df = plot_tSNE(featZ=featZ_df,
                                meta=meta_df,
                                group_by=GROUPING_VAR,
                                var_subset=None,
                                saveDir=tsne_dir,
                                perplexities=perplexities,
                                 # NB: perplexity parameter should be roughly equal to group size
                                sns_colour_palette="tab10")
        except Exception as e:
            print("WARNING: Could not plot tSNE\n", e)
       
        #%%     Uniform Manifold Projection (UMAP)

        umap_dir = plot_dir / 'UMAP'
        n_neighbours = [5,15,30]
        min_dist = 0.1 # Minimum distance parameter
        try:
            umap_df = plot_umap(featZ=featZ_df,
                                meta=meta_df,
                                group_by=GROUPING_VAR,
                                var_subset=None,
                                saveDir=umap_dir,
                                n_neighbours=n_neighbours,
                                # NB: n_neighbours parameter should be roughly equal to group size
                                min_dist=min_dist,
                                sns_colour_palette="tab10")
        except Exception as e:
            print("WARNING: Could not plot UMAP\n", e)
