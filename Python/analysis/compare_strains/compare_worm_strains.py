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
                         f_oneway,
                         kruskal,
                         zscore)

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
   
# Path to Github helper functions (USER-DEFINED path to local copy of Github repo)
PATH_LIST = ['/Users/sm5911/Tierpsy_Versions/tierpsy-tools-python/']
for sysPath in PATH_LIST:
    if sysPath not in sys.path:
        sys.path.insert(0, sysPath)

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

if __name__ == "__main__":
    
    # Accept command-line inputs # TODO: Read from JSON instead?
    parser = argparse.ArgumentParser(description='Analyse Tierpsy results (96-well)')
    parser.add_argument('--project_dir', help="Project root directory,\
                        containing 'AuxiliaryFiles', 'RawVideos',\
                        'MaskedVideos' and 'Results' folders",
                        default='/Volumes/hermes$/KeioScreen_96WP', type=str)
    parser.add_argument('--grouping_variable', help="Categorical variable that you wish to \
                        investigate", nargs='+', default='food_type') # 'worm_strain'
    parser.add_argument('--compile_day_summaries', help="Compile full feature summaries from \
                        day feature summary results", default=False, action='store_true')
    # Keio = ['food_type','instrument_name','lawn_growth_duration_hours','lawn_storage_type']
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
    parser.add_argument('--test', help="Choose between 'LMM' (if > 1 day replicate) or 'ANOVA' \
                        (Kruskal tests will be performed instead of ANOVA if check_normal and \
                        data is not normally distributed) If features are found to be significant, \
                        pairwise t-tests are performed.", default='ANOVA')
    parser.add_argument('--use_top256', help="Use Tierpsy Top256 features only",
                        default=False, action='store_true')
    parser.add_argument('--feature_means_only', help="Use only 50th percentile feature summaries \
                        for each feature", default=False, action='store_false')
                        # TODO: Change this to percentile to use
    parser.add_argument('--drop_size_features', help="Remove size-related Tierpsy \
                        features from analysis", default=False, action='store_true')
                        # TODO: Size-related features only (drop behaviour)
    parser.add_argument('--norm_features_only', help="Use only normalised \
                        size-invariant features ('_norm') for analysis",
                        default=False, action='store_true')
    parser.add_argument('--check_normal', help="Perform Shapiro-Wilks test for \
                        normality to decide between parametric/non-parametric \
                        statistics", default=True, action='store_false')
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
    FEAT_MEANS_ONLY = args.feature_means_only                   # bool
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
    fn = fn + '_50th' if FEAT_MEANS_ONLY else fn
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
                                                  mean_feats_only=FEAT_MEANS_ONLY)
        
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
        # Save normailty test results to file
        prop_features_normal.to_csv(normtest_savepath, 
                                    index=True, 
                                    index_label=GROUPING_VAR, 
                                    header='prop_normal')
    else:
        is_normal = False # Default non-parametric

    # If LMM is chosen, ensure that there are multiple day replicates to compare at each timepoint
    if TEST_NAME == 'LMM':
        assert all(len(metadata.loc[metadata['imaging_run_number']==t, RANDOM_EFFECT].unique()) > 1 
                   for t in imaging_run_list)
    elif TEST_NAME == 'ANOVA' and not is_normal:
        print("WARNING: Non-parametric tests will be preferred. Performing Kruskal-Wallis tests " +
              "instead of ANOVA")
        TEST_NAME = 'Kruskal'
        
    #%% Analyse variables

    print("\nInvestigating '%s' variation" % GROUPING_VAR)    
    for run in imaging_run_list:
        print("\nAnalysing imaging run %d" % run)
        
        # Subset results to investigate single imaging run
        meta_df = metadata[metadata['imaging_run_number']==run]
        feat_df = features.reindex(meta_df.index)
        
        print(meta_df.shape)
        print(meta_df.groupby[GROUPING_VAR].count().head())
        
        # Clean subsetted data: drop NaNs, zero std, etc
        feat_df, meta_df = clean_features_summaries(feat_df, 
                                                    meta_df, 
                                                    max_value_cap=False,
                                                    imputeNaN=False)
        # Save paths
        stats_dir = save_dir / fn / "Run_{}".format(run) / "Stats" / (GROUPING_VAR + '_variation')
        plot_dir = save_dir / fn / "Run_{}".format(run) / "Plots" / (GROUPING_VAR + '_variation')

        #%% STATISTICS
        #   One-way ANOVA/Kruskal-Wallis tests for significantly different 
        #   features across groups (e.g. strains)
    
        # When comparing more than 2 groups, perform ANOVA and proceed only to 
        # pairwise two-sample t-tests if there is significant variability among 
        # all groups for any feature
        var_list = list(meta_df[GROUPING_VAR].unique())
        
        grouped = feat_df.join(meta_df[GROUPING_VAR]).groupby(by=GROUPING_VAR)
        stats_table = grouped.mean().T
        mean_cols = ['mean ' + v for v in stats_table.columns.to_list()]
        stats_table.columns = mean_cols
        for group in grouped.size().index:
            stats_table['sample size {}'.format(group)] = grouped.size().loc[group]
            
        if len(var_list) > 2:
            (pvalues_anova, 
            anova_sigfeats_list) = anova_by_feature(feat_df=feat_df, 
                                                    meta_df=meta_df, 
                                                    group_by=GROUPING_VAR, 
                                                    strain_list=var_list, 
                                                    p_value_threshold=P_VALUE_THRESHOLD, 
                                                    is_normal=is_normal, 
                                                    fdr_method=FDR_METHOD)
                        
            # Record name of statistical test used (kruskal/f_oneway)
            TEST = f_oneway if is_normal else kruskal
            test_name = str(TEST).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]
            col = '{} p-value'.format(test_name)
            stats_table[col] = pvalues_anova.loc['pval', stats_table.index]

            # Save test statistics to file
            stats_outpath = stats_dir / '{}_results.csv'.format(test_name)
            sigfeats_outpath = Path(str(stats_outpath).replace('_results.csv',
                                                               '_significant_features.csv'))
            pvalues_anova.to_csv(stats_outpath)
            anova_sigfeats_list.to_csv(sigfeats_outpath, index=False)
            
            # Record number of signficant features by ANOVA
            fset_anova = list(pvalues_anova.columns[pvalues_anova.loc['pval'] < 
                                                    P_VALUE_THRESHOLD])
        else:
            fset_anova = []     
    
        ###   t-tests/rank-sum tests for significantly different features between 
        #     each group vs control
     
        if len(fset_anova) > 0 or len(var_list) == 2:
            
            (pvalues_ttest, 
             sigfeats_table, 
             sigfeats_list) = ttest_by_feature(feat_df, 
                                               meta_df, 
                                               group_by=GROUPING_VAR, 
                                               control_strain=CONTROL, 
                                               is_normal=is_normal, 
                                               p_value_threshold=P_VALUE_THRESHOLD,
                                               fdr_method=FDR_METHOD)

            # Record name of statistical test
            TEST = ttest_ind if is_normal else ranksumtest
            test_name = str(TEST).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]

            if len(var_list) == 2:
                col = '{} p-value'.format(test_name)
                stats_table[col] = pvalues_ttest.loc[pvalues_ttest.index[0], stats_table.index]
                                               
            # Print number of significant features
            print("%d significant features found for %s (run %d, %s, P<%.2f, %s)"\
                  % (len(sigfeats_list), GROUPING_VAR.replace('_',' '), run, 
                     test_name, P_VALUE_THRESHOLD, FDR_METHOD))
            # Save test statistics to file
            stats_outpath = stats_dir / '{}_results.csv'.format(test_name)
            stats_outpath.parent.mkdir(exist_ok=True, parents=True)
            sigfeats_outpath = Path(str(stats_outpath).replace('_results.csv',
                                                               '_significant_features.csv'))
            sigfeats_list.to_csv(sigfeats_outpath, index=False) # Save feature list to file
            pvalues_ttest.to_csv(stats_outpath) # Save test results to CSV
            
            fset_ttest = list(pvalues_ttest.columns[(pvalues_ttest < 
                                                     P_VALUE_THRESHOLD).sum(axis=0) > 0])
        else:
            fset_ttest = []
            
        ###   Linear Mixed Models (LMMs) to account for day-to-day variation when comparing 
        #     between worm/food/drug type
        if (GROUPING_VAR in ['worm_strain','food_type','drug_type'] and 
           len(meta_df[RANDOM_EFFECT].unique()) > 1):
            
            test_name = 'LMM'
            with warnings.catch_warnings():
                # Filter warnings as parameter is often on the boundary
                warnings.filterwarnings("ignore")
                #warnings.simplefilter("ignore", ConvergenceWarning)
                (signif_effect, 
                 low_effect, 
                 error, 
                 mask, 
                 pvalues_lmm)=compounds_with_low_effect_univariate(feat=feat_df, 
                                                          drug_name=meta_df[GROUPING_VAR], 
                                                          drug_dose=None, 
                                                          random_effect=meta_df[RANDOM_EFFECT], 
                                                          control=CONTROL, 
                                                          test=test_name, 
                                                          comparison_type='multiclass',
                                                          multitest_method=FDR_METHOD,
                                                          ignore_names=None, 
                                                          return_pvals=True)
            assert len(error) == 0
            
            if len(var_list) == 2:
                col = '{} p-value'.format(test_name)
                stats_table[col] = pvalues_lmm.loc[pvalues_lmm.index[0], stats_table.index]  
            
            # Save LMM significant features
            lmm_path = stats_dir / '{}_results.csv'.format(test_name)
            lmm_path.parent.mkdir(exist_ok=True, parents=True)
            pvalues_lmm.to_csv(lmm_path, header=True, index=True)
            
            # Ideally report as: parameter | beta | lower-95 | upper-95 | random effect (SD)
            fset_lmm = list(pvalues_lmm.columns[(pvalues_lmm < P_VALUE_THRESHOLD).any() > 0])
            
            if len(signif_effect) > 0:
                print(("%d significant features found (%d significant %ss vs %s, "\
                      % (len(fset_lmm), len(signif_effect), GROUPING_VAR.replace('_',' '), 
                         CONTROL) if len(signif_effect) > 0 else\
                      "No significant differences found between %s "\
                      % GROUPING_VAR.replace('_',' '))
                      + "after accounting for %s variation (run %d, %s, P<%.2f, %s)"\
                      % (RANDOM_EFFECT.split('_yyyymmdd')[0], run, test_name, P_VALUE_THRESHOLD,
                         FDR_METHOD))
        else:
            fset_lmm = []

        # Prefer LMM over ttest if applicable, and compare with k sig feats
        # If there is little overlap, investigate why...
        if ((len(var_list) > 2 and len(fset_anova) == 0) or len(fset_ttest)==0):
            print("NO SIGNIFICANT FEATURES (%s, run %d)" % (GROUPING_VAR.replace('_',' '), run))
            fset = []
        elif GROUPING_VAR in ['worm_strain','food_type','drug_type']:
            if len(fset_lmm) == 0:
                # No significant features after accounting for day variation (LMM)
                print("NO SIGNIFICANT FEATURES (%s, run %d)" % (GROUPING_VAR.replace('_',
                                                                                     ' '), run))
                fset = []
            else:
                fset = fset_lmm
        elif len(fset_anova) > 0 or len(var_list) == 2:
            fset = fset_ttest
        else:
            assert len(fset_anova) > 0
            fset = fset_anova
            
        # Add stats results to stats table
        if len(var_list) == 2:
            stats_table['significance'] = sig_asterix(pvalues_lmm.values[0])
        else:
            stats_table['significance'] = sig_asterix(pvalues_anova.loc['pval'].values)
            
        ### K significant features
        #   Compare feature set overlap with k significant features

        # k_sigfeat_dir = plot_dir / 'k_sig_feats'
        # k_sigfeat_dir.mkdir(exist_ok=True, parents=True)
            
        # Infer feature set
        
        #K_SIG_FEATS = len(fset) if (fset != None and len(fset) > K_SIG_FEATS) else K_SIG_FEATS
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
            NO_SIGNIFICANCE = False
            fset_overlap = set(fset).intersection(set(fset_ksig))
            prop_overlap = len(fset_overlap) / len(fset)
            if prop_overlap < 0.5 and len(fset) > 100:
                raise Warning("Insufficient consistency in statistics for feature set agreement!") 
            else:
                fset = pvalues_ksig.loc['k_significant_features', 
                                        fset_overlap].sort_values(axis=0, ascending=True).index
        else:
            print("NO SIGNIFICANT FEATURES FOUND! Using 'k_significant_feat' features instead")
            NO_SIGNIFICANCE = True
            fset = fset_ksig
            
        #%% Plot day variation
                
        print("Loading %s model p-values for plotting" % test_name)
        pvalues = pd.read_csv((stats_dir / '{}_results.csv'.format(test_name)), index_col=0)
        assert all(f in pvalues.columns for f in feat_df.columns)             
        
        swarmDir = plot_dir / '{}_variation'.format(RANDOM_EFFECT.split('_yyyymmdd')[0])
        plot_day_variation(feat_df=feat_df,
                           meta_df=meta_df,
                           group_by=GROUPING_VAR,
                           test_pvalues_df=None if NO_SIGNIFICANCE else pvalues,
                           control=CONTROL,
                           day_var='date_yyyymmdd',
                           fset=fset,
                           p_value_threshold=P_VALUE_THRESHOLD,
                           saveDir=swarmDir,
                           sns_colour_palette="tab10",
                           dodge=False, 
                           ranked=True)
                                                               
        #%% Boxplots of most significantly different features for each strain vs control
        #   features ranked by test pvalue significance (lowest first)

        # Barplot of number of significantly different features for each strain   
        prop_sigfeats = barplot_sigfeats(test_pvalues_df=None if NO_SIGNIFICANCE else pvalues, 
                                         saveDir=plot_dir,
                                         p_value_threshold=P_VALUE_THRESHOLD)
        
        # TODO: Plot k_sig_feats as boxplots anyway even if no sigfeats are found by LMM tests
        # TODO: boxplots for fset provided without pvalues
        # Boxplots of significant features (for each group vs control)
        boxplots_sigfeats(feat_meta_df=meta_df.join(feat_df), 
                          test_pvalues_df=pvalues, 
                          group_by=GROUPING_VAR, 
                          control_strain=CONTROL, 
                          selected_features=fset, #['speed_norm_50th_bluelight'],
                          saveDir=plot_dir / 'paired_boxplots',
                          n_sig_feats_to_plot=K_SIG_FEATS,
                          p_value_threshold=P_VALUE_THRESHOLD)
                
        # from tierpsytools.analysis.significant_features import plot_feature_boxplots
        # plot_feature_boxplots(feat_to_plot=fset,
        #                       y_class=GROUPING_VAR,
        #                       scores=pvalues.rank(axis=1),
        #                       feat_df=feat_df,
        #                       pvalues=np.asarray(pvalues).flatten(),
        #                       saveto=None,
        #                       close_after_plotting=False)
        
        boxplots_grouped(feat_meta_df=meta_df.join(feat_df), 
                         group_by=GROUPING_VAR,
                         control_group=CONTROL,
                         test_pvalues_df=pvalues if len(fset) > 1 else None,
                         fset=fset,
                         saveDir=(plot_dir / 'grouped_boxplots'),
                         max_features_plot_cap=K_SIG_FEATS, 
                         max_groups_plot_cap=48,
                         p_value_threshold=0.05,
                         drop_insignificant=False,
                         sns_colour_palette="tab10",
                         figsize=[8,12],
                         saveFormat='png')
        
        #%% Hierarchical Clustering Analysis
        #   - Clustermap of features by strain, to see if data cluster into groups
        #   - Control data is clustered first, feature order is stored and ordering applied to 
        #     full data for comparison
        
        # Extract data for control
        control_feat_df = feat_df[meta_df[GROUPING_VAR]==CONTROL]
        control_meta_df = meta_df.reindex(control_feat_df.index)
        
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
        control_clustermap_path = plot_dir / 'HCA' / '{}_clustermap.pdf'.format(CONTROL)
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
                    
        # Clustermap of full data using control clustered features order
        assert not feat_df.isna().sum(axis=0).any()
        assert not (feat_df.std(axis=0) == 0).any()
        
        featZ_df = feat_df.apply(zscore, axis=0)
        
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
        stats_table = stats_table.sort_values(by='{} p-value'.format(test_name), ascending=True)
        
        # Save stats table to CSV
        stats_table_path = stats_dir / 'stats_summary_table.csv'
        stats_table.to_csv(stats_table_path, header=True, index=None)

        # Drop features with NaN values after normalising
        n_cols = len(featZ_df.columns)
        featZ_df.dropna(axis=1, inplace=True)
        n_dropped = n_cols - len(featZ_df.columns)
        if n_dropped > 0:
            print("Dropped %d features after normalisation (NaN)" % n_dropped)
        
        full_clustermap_path = plot_dir / 'HCA' / 'full_{}_clustermap.pdf'.format(GROUPING_VAR)
        fg = plot_clustermap(featZ=featZ_df, 
                             meta=meta_df, 
                             group_by=GROUPING_VAR,
                             col_linkage=None,
                             method='complete',
                             figsize=[20,5],
                             saveto=full_clustermap_path)
        
        if len(var_list) > 2:
            pvalues_heatmap = pvalues_anova.loc['pval', clustered_features]
        elif len(var_list) == 2:
            pvalues_heatmap = pvalues.loc[pvalues.index[0], clustered_features]
        pvalues_heatmap.name = 'P < {}'.format(P_VALUE_THRESHOLD)

        assert all(f in featZ_df.columns for f in pvalues_heatmap.index)

        # Heatmap barcode with selected features
        #   - Read in selected features list
        
        if args.selected_features_path is not None and run == 3 and GROUPING_VAR == 'worm_strain':
            fset = pd.read_csv(Path(args.selected_features_path), index_col=None)
            fset = [s for s in fset['feature'] if s in featZ_df.columns] # TODO: Assert this?
            
        # Plot barcode hewatmap, grouping also by date
        heatmap_date_path = plot_dir / 'HCA' /\
            '{}_date_heatmap.pdf'.format(GROUPING_VAR)
        plot_barcode_heatmap(featZ=featZ_df[clustered_features], 
                             meta=meta_df, 
                             group_by=['date_yyyymmdd',GROUPING_VAR], 
                             pvalues_series=pvalues_heatmap,
                             p_value_threshold=P_VALUE_THRESHOLD,
                             selected_feats=fset if len(fset) > 0 else None,
                             saveto=heatmap_date_path,
                             figsize=[18,6],
                             sns_colour_palette="Pastel1")
        
        # Plot group-mean heatmap (averaged across days)
        heatmap_path = plot_dir / 'HCA' / '{}_heatmap.pdf'.format(GROUPING_VAR)
        plot_barcode_heatmap(featZ=featZ_df[clustered_features], 
                             meta=meta_df, 
                             group_by=[GROUPING_VAR], 
                             pvalues_series=pvalues_heatmap,
                             p_value_threshold=P_VALUE_THRESHOLD,
                             selected_feats=fset if len(fset) > 0 else None,
                             saveto=heatmap_path,
                             figsize=[18,6],
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
                                var_subset=None, 
                                saveDir=pca_dir,
                                PCs_to_keep=10,
                                n_feats2print=10)      
         
        #%%     t-distributed Stochastic Neighbour Embedding (tSNE)

        perplexities = [5,15,30,50] # tSNE: Perplexity parameter for tSNE mapping

        tsne_dir = plot_dir / 'tSNE'
        tSNE_df = plot_tSNE(featZ=featZ_df,
                            meta=meta_df,
                            group_by=GROUPING_VAR,
                            var_subset=None,
                            saveDir=tsne_dir,
                            perplexities=perplexities)
        
        #%%     Uniform Manifold Projection (UMAP)

        n_neighbours = [5,15,30,50] # N-neighbours parameter
        min_dist = 0.1 # Minimum distance parameter
        
        umap_dir = plot_dir / 'UMAP'
        umap_df = plot_umap(featZ=featZ_df,
                            meta=meta_df,
                            group_by=GROUPING_VAR,
                            var_subset=None,
                            saveDir=umap_dir,
                            n_neighbours=n_neighbours,
                            min_dist=min_dist)
