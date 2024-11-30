#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistics helper functions (t-test/ANOVA) for comparing between 2 or more groups

@author: sm5911
@date: 9/2/21

"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from functools import partial
from statsmodels.stats import multitest as smm # AnovaRM
from scipy.stats import ttest_ind, chi2, f_oneway, kruskal, shapiro, ranksums, levene

from write_data.write import write_list_to_file
from visualisation.plotting_helper import sig_asterix

from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes, _multitest_correct
from tierpsytools.preprocessing.filter_data import feat_filter_std
from tierpsytools.analysis.statistical_tests_helper import stats_test



def multiple_test_correction(pvalues, fdr_method='fdr_by', fdr=0.05):
    """
    Multiple comparisons correction of pvalues from univariate tests
    
    Parameters
    ----------
    pvalues : pandas.Series shape=(n_features,) OR
              pandas.DataFrame shape=(n_features, n_groups)
    fdr_method : str
        The method to use in statsmodels.stats.multitest.multipletests function
    fdr : float
        False discovery rate threshold
        
    Returns
    -------
    pvalues : pandas.DataFrame shape=(n_features, n_groups)
        Dataframe of corrected pvalues for each feature
    """
 
    assert type(pvalues) in [pd.DataFrame, pd.Series]
    if type(pvalues) == pd.Series:
        pvalues = pd.DataFrame(pvalues).T
        
    for idx in pvalues.index:
        # Locate pvalue results for strain (row)
        _pvals = pvalues.loc[idx]
 
        # Perform correction for multiple features comparisons
        _corrArray = smm.multipletests(_pvals.values, 
                                       alpha=fdr, 
                                       method=fdr_method,
                                       is_sorted=False, 
                                       returnsorted=False)
        
        # Get pvalues for features that passed the Benjamini/Yekutieli (negative) correlation test
        pvalues.loc[idx,:] = _corrArray[1]
        
        # Record significant features (after correction)
        sigfeats = pvalues.loc[idx, pvalues.columns[_corrArray[1] < fdr]].sort_values(ascending=True)
        print("%d significant features found for %s (method='%s', fdr=%s)" % (len(sigfeats), idx, 
                                                                              fdr_method, fdr))
    
    return pvalues

def shapiro_normality_test(features_df, 
                           metadata_df, 
                           group_by, 
                           p_value_threshold=0.05,
                           verbose=True):
    """ Perform a Shapiro-Wilks test for normality among feature summary results separately for 
        each test group in 'group_by' column provided, e.g. group_by='worm_strain', and return 
        whether or not theee feature data can be considered normally distributed for parameetric 
        statistics 
    """

    if verbose:
        print("Checking for feature normality..")
        
    is_normal_threshold = 1 - p_value_threshold
    strain_list = list(metadata_df[group_by].unique())
    prop_features_normal = pd.Series(data=None, index=strain_list, name='prop_normal')
    for strain in strain_list:
        strain_meta = metadata_df[metadata_df[group_by]==strain]
        strain_feats = features_df.reindex(strain_meta.index)
        if verbose and not strain_feats.shape[0] > 2:
            print("Not enough data for normality test for %s" % strain)
        else:
            strain_feats = strain_feats.dropna(axis=1, how='all')
            fset = strain_feats.columns
            normality_results = pd.DataFrame(data=None, index=['stat','pval'], columns=fset)
            for f, feature in enumerate(fset):
                try:
                    stat, pval = shapiro(strain_feats[feature])
                    # NB: UserWarning: Input data for shapiro has range zero 
                    # Some features for that strain contain all zeros - shapiro(np.zeros(5))
                    normality_results.loc['stat',feature] = stat
                    normality_results.loc['pval',feature] = pval
                    
                    ## Q-Q plots to visualise whether data fit Gaussian distribution
                    #from statsmodels.graphics.gofplots import qqplot
                    #qqplot(data[feature], line='s')
                    
                except Exception as EE:
                    print("WARNING: %s" % EE)
                    
            prop_normal = (normality_results.loc['pval'] < p_value_threshold).sum()/len(fset)
            prop_features_normal.loc[strain] = prop_normal
            if verbose:
                print("%.1f%% of features are normal for %s (n=%d)" % (prop_normal*100, strain, 
                                                                       strain_feats.shape[0]))

    # Determine whether to perform parametric or non-parametric statistics
    # NB: Null hypothesis - feature summary results for individual strains are normally distributed
    total_prop_normal = np.mean(prop_features_normal)
    if total_prop_normal > is_normal_threshold:
        is_normal = True
        if verbose:
            print('More than %d%% of features (%.1f%%) were found to obey a normal distribution '\
                  % (is_normal_threshold*100, total_prop_normal*100)
                  + 'so parametric analyses will be preferred.')
    else:
        is_normal = False
        if verbose:
            print('Less than %d%% of features (%.1f%%) were found to obey a normal distribution '\
                  % (is_normal_threshold*100, total_prop_normal*100)
                  + 'so non-parametric analyses will be preferred.')
    
    return prop_features_normal, is_normal

def levene_f_test(features, 
                  metadata, 
                  grouping_var, 
                  p_value_threshold=0.05, 
                  multitest_method='fdr_by', 
                  saveto=None, 
                  del_if_exists=False):
    """ Apply Levene's F-test for equal variances between strains for each feature and return a 
        dataframe of test results containing 'stat' and 'pval' columns
    """
        
    levene_stats = None
    
    if saveto is not None:
        if Path(saveto).exists() and not del_if_exists:
            print("Reading Levene stats from file")
            levene_stats = pd.read_csv(saveto, index_col=0)

    if levene_stats is None:
        # inputs with partial
        func = partial(stats_test, test=levene, vectorized=False)
        stats, pvals = func(X=features, 
                            y=metadata[grouping_var], 
                            n_jobs=-1)
    
        levene_stats = pd.DataFrame(data={'stat' : stats, 
                                          'pval' : pvals}, index=features.columns)
        # for feat in tqdm(features.columns):
        #     stat, pval = levene(*[features[metadata[grouping_var]==strain][feat] for 
        #                           strain in metadata[grouping_var].unique()], center='median')
        #     levene_stats.loc[feat, :] = stat, pval
     
        # Perform correction for multiple comparisons
        _corrArray = smm.multipletests(levene_stats['pval'], 
                                       alpha=p_value_threshold, 
                                       method=multitest_method,
                                       is_sorted=False, 
                                       returnsorted=False)
        
        # Update pvalues with multiple test correction
        levene_stats['pval'] = _corrArray[1]

                
    if saveto is not None:
        if Path(saveto).exists():
            if del_if_exists:
                Path(saveto).unlink()
        else:        
            print("Saving Levene stats to %s" %  saveto)
            saveto.parent.mkdir(exist_ok=True, parents=True)       
            levene_stats.to_csv(saveto, header=True, index=True)
        
    return levene_stats

def ranksumtest(test_data, control_data):
    """ Wilcoxon rank sum test (column-wise between 2 dataframes of equal dimensions)
        Returns 2 lists: a list of test statistics, and a list of associated p-values
    """
    
    colnames = list(test_data.columns)
    J = len(colnames)
    statistics = np.zeros(J)
    pvalues = np.zeros(J)
    
    for j in range(J):
        test_feat_data = test_data[colnames[j]]
        control_feat_data = control_data[colnames[j]]
        statistic, pval = ranksums(test_feat_data, control_feat_data)
        pvalues[j] = pval
        statistics[j] = statistic
        
    return statistics, pvalues

def pairwise_ttest(control_df, strain_df, feature_list, test_name='t-test', group_by='antioxidant', 
                   fdr_method='fdr_by', fdr=0.05):
    """ Perform pairwise t-tests between each strain and control, for each treatment group in 
        'group_by' column
        
        Inputs
        ------
        control_df : pd.Dataframe
            Combined metadata + features dataframe for control strain only
        strain_df : pd.Dataframe
            Combined metadata + features dataframe for strains to be tested against control
        feature_list : list
            List of features (column names) to test
        group_by : str
            Column name of variable to group by for t-tests (eg. 'antioxidant' or 'window')
        fdr_method : str
            Specify method for multiple tests correction
        fdr : float
            Significance threshold for controlling the false discovery rate
            
        Returns
        -------
        strain_stats, strain_pvals, strain_reject
    """

    groups = control_df[group_by].unique()
    
    if test_name == 't-test' or test_name.startswith('ttest'):
        test = ttest_ind
    elif test_name == 'chi2' or test_name.startswith('chi'):
        test = chi2
    else:
        raise IOError("Please choose from: ['ttest', 'chi2']")

    strain_pvals_list = []
    strain_stats_list = []
    for group in groups:
        test_control = control_df[control_df[group_by]==group]
        test_strain = strain_df[strain_df[group_by]==group]
        
        pvals = []
        stats = []
        for feature in feature_list:
            _stat, _pval = test(test_control[feature], test_strain[feature], axis=0)
            pvals.append(_pval)
            stats.append(_stat)
        
        pvals = pd.DataFrame(np.array(pvals).T, index=feature_list, columns=[group])
        stats = pd.DataFrame(np.array(stats).T, index=feature_list, columns=[group])
            
        strain_pvals_list.append(pvals)
        strain_stats_list.append(stats)
        
    strain_pvals = pd.concat(strain_pvals_list, axis=1)
    strain_stats = pd.concat(strain_stats_list, axis=1)
    strain_reject = strain_pvals < fdr

    # correct for multiple feature/antioxidant comparisons
    if fdr_method is not None:
        strain_reject, strain_pvals = _multitest_correct(strain_pvals, fdr_method, fdr)

    return strain_stats, strain_pvals, strain_reject

def ttest_by_feature(feat_df, 
                     meta_df, 
                     group_by, 
                     control_strain, 
                     is_normal=True,
                     p_value_threshold=0.05,
                     fdr_method='fdr_by',
                     verbose=False):
    """ Perform t-tests for significant differences between each strain and the
        control, for each feature. If is_normal=False, rank-sum tests will be 
        performed instead 
    """   

    # Record name of statistical test used (ttest/ranksumtest)
    TEST = ttest_ind if is_normal else ranksumtest
    test_name = str(TEST).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]
    print("Computing %s tests for each %s vs %s" % (test_name, group_by, control_strain))

    # Extract control results
    control_meta = meta_df[meta_df[group_by] == control_strain]
    control_feats = feat_df.reindex(control_meta.index)
    
    # Record test strains
    test_strains = [strain for strain in meta_df[group_by].unique() if strain != control_strain]

    # Pre-allocate dataframes for storing test statistics and p-values
    test_stats_df = pd.DataFrame(index=list(test_strains), columns=feat_df.columns)
    test_pvalues_df = pd.DataFrame(index=list(test_strains), columns=feat_df.columns)
    sigfeats_table = pd.DataFrame(index=test_pvalues_df.index, 
                                  columns=['sigfeats','sigfeats_corrected'])
    
    # Compute test statistics for each strain, comparing to control for each feature
    for t, strain in enumerate(tqdm(test_strains, position=0)):
            
        # Grab feature summary results for that strain
        strain_meta = meta_df[meta_df[group_by] == strain]
        strain_feats = feat_df.reindex(strain_meta.index)
        
        # Drop columns that contain only zeros
        n_cols = len(strain_feats.columns)
        strain_feats = feat_filter_std(strain_feats, threshold=0.0)
        control_feats = feat_filter_std(control_feats, threshold=0.0)
        zero_std_cols = n_cols - len(strain_feats.columns)
        if zero_std_cols > 0 and verbose:
            print("Dropped %d feature summaries for %s (zero std)" % (zero_std_cols, strain))
            
        # Use only shared feature summaries between control data and test data
        shared_colnames = control_feats.columns.intersection(strain_feats.columns)
        strain_feats = strain_feats[shared_colnames]
        control_feats = control_feats[shared_colnames]
    
        # Perform rank-sum tests comparing between strains for each feature
        test_stats, test_pvalues = TEST(strain_feats, control_feats)
    
        # Add test results to out-dataframe
        test_stats_df.loc[strain][shared_colnames] = test_stats
        test_pvalues_df.loc[strain][shared_colnames] = test_pvalues
        
        # Record the names and number of significant features 
        sigfeats = pd.Series(test_pvalues_df.columns[np.where(test_pvalues < p_value_threshold)])
        sigfeats.name = strain
        sigfeats_table.loc[strain,'sigfeats'] = len(sigfeats)
                
    # Benjamini-Yekutieli corrections for multiple comparisons    
    sigfeats_list = []
    for strain in test_pvalues_df.index:
        # Locate pvalue results for strain (row)
        strain_pvals = test_pvalues_df.loc[strain]
        
        # Perform correction for multiple features comparisons
        _corrArray = smm.multipletests(strain_pvals.values, 
                                       alpha=p_value_threshold, 
                                       method=fdr_method,
                                       is_sorted=False, 
                                       returnsorted=False)
        
        # Get pvalues for features that passed the Benjamini/Yekutieli (negative) correlation test
        test_pvalues_df.loc[strain,:] = _corrArray[1]
        
        # Record the names and number of significant features (after BY correction)
        sigfeats = pd.Series(test_pvalues_df.columns[np.where(_corrArray[1] < p_value_threshold)])
        sigfeats.name = strain
        sigfeats_list.append(sigfeats)
        sigfeats_table.loc[strain,'sigfeats_corrected'] = len(sigfeats)

    # Concatentate into dataframe of sigfeats for each strain 
    sigfeats_list = pd.concat(sigfeats_list, axis=1, ignore_index=True, sort=False)
    sigfeats_list.columns = test_pvalues_df.index

    return test_pvalues_df, sigfeats_table, sigfeats_list

def anova_by_feature(feat_df, 
                     meta_df, 
                     group_by, 
                     strain_list=None, 
                     p_value_threshold=0.05, 
                     is_normal=True,
                     fdr_method='fdr_by'):
    """ One-way ANOVA/Kruskal-Wallis tests for pairwise differences across 
        strains for each feature 
    """

    # Drop features with zero std
    n_cols = len(feat_df.columns)
    feat_df = feat_filter_std(feat_df)
    zero_std = n_cols - len(feat_df.columns)
    if zero_std > 0:
        print("Dropped %d features with zero standard deviation" % zero_std)
  
    # Record name of statistical test used (kruskal/f_oneway)
    TEST = f_oneway if is_normal else kruskal
    test_name = str(TEST).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]
    print("\nComputing %s tests between strains for each feature..." % test_name)

    # Perform 1-way ANOVAs for each feature between test strains
    test_pvalues_df = pd.DataFrame(index=['stat','pval'], columns=feat_df.columns)
    for f, feature in enumerate(tqdm(feat_df.columns, position=0)): # leave=True
            
        # Perform test and capture outputs: test statistic + p value
        test_stat, test_pvalue = TEST(*[feat_df[meta_df[group_by]==strain][feature] \
                                      for strain in meta_df[group_by].unique()])
        test_pvalues_df.loc['stat',feature] = test_stat
        test_pvalues_df.loc['pval',feature] = test_pvalue

    # Perform correction for multiple comparisons on one-way ANOVA pvalues
    _corrArray = smm.multipletests(test_pvalues_df.loc['pval'], 
                                   alpha=p_value_threshold, 
                                   method=fdr_method,
                                   is_sorted=False, 
                                   returnsorted=False)
    
    # Update pvalues with Benjamini-Yekutieli correction
    test_pvalues_df.loc['pval',:] = _corrArray[1]
    
    # Store names of features that show significant differences across the test bacteria
    sigfeats = test_pvalues_df.columns[np.where(test_pvalues_df.loc['pval'] < p_value_threshold)]
    print("Complete!\n%d/%d (%.1f%%) features exhibit significant differences between strains "\
          % (len(sigfeats), len(test_pvalues_df.columns), 
             len(sigfeats) / len(test_pvalues_df.columns)*100) + 
          '(%s, P<%.2f, %s)' % (test_name, p_value_threshold, fdr_method))
    
    # Compile list to store names of significant features
    sigfeats_list = pd.Series(test_pvalues_df.columns[np.where(test_pvalues_df.loc['pval'] <
                                                               p_value_threshold)])
    sigfeats_list.name = 'significant_features_' + test_name
    sigfeats_list = pd.DataFrame(sigfeats_list)
    
    # Rank pvalues by ascending order
    topfeats = test_pvalues_df.loc['pval'].sort_values(ascending=True)
    test_pvalues_df = test_pvalues_df[topfeats.index]
    
    print("Top 10 significant features by %s test:\n" % test_name)
    for feat in topfeats.index[:10]:
        print(feat)

    return test_pvalues_df, sigfeats_list

# def linear_mixed_model(feat_df, 
#                        meta_df,
#                        fixed_effect,
#                        control,
#                        random_effect='date_yyyymmdd', 
#                        fdr=0.05, 
#                        fdr_method='fdr_by', 
#                        comparison_type='infer',
#                        n_jobs=-1):
#     """ Test whether a given group differs siginificantly from the control, taking into account one 
#         random effect, eg. date of experiment. Each feature is tested independently using a Linear 
#         Mixed Model with fixed slope and variable intercept to account for the random effect.
#         The pvalues from the different features are corrected for multiple comparisons using the
#         multitest methods of statsmodels.
        
#         Parameters
#         ----------
#         feat_df :         TYPE - pd.DataFrame
#                           DESCRIPTION - Dataframe of feature summary results
#         fixed_effect :        TYPE - str
#                           DESCRIPTION - fixed effect variable (grouping variable)
#         random_effect :   TYPE - str
#                           DESCRIPTION - random effect variable
#         control :         TYPE float, optional. Default is 0.
#                           DESCRIPTION - The dose of the control points in drug_dose.
#         fdr :             TYPE - float
#                           DESCRIPTION. False discovery rate threshold [0-1]. Default is 0.05.
#         fdr_method :      TYPE - str
#                           DESCRIPTION - Method for multitest correction. Default is 'fdr_by'.
#         comparison_type : TYPE - str
#                           DESCRIPTION - ['continuous', 'categorical', 'infer']. Default is 'infer'.
#         n_jobs :          TYPE - int
#                           DESCRIPTION - Number of jobs for parallelisation of model fit.
#                                         The default is -1.
                 
#         Returns
#         -------
#         pvals :           TYPE - pd.Series or pd.DataFrame
#                           DESCRIPTION - P-values for each feature. If categorical, a dataframe is 
#                                         returned with each group compared separately
#     """

#     assert type(fixed_effect) == str and fixed_effect in meta_df.columns
#     assert type(random_effect) == str and random_effect in meta_df.columns
    
#     feat_names = feat_df.columns.to_list()
#     df = feat_df.assign(fixed_effect=meta_df[fixed_effect]).assign(random_effect=meta_df[random_effect])

#     # select only the control points that belong to groups that have non-control members
#     groups = np.unique(df['random_effect'][df['fixed_effect']!=control])
#     df = df[np.isin(df['random_effect'], groups)]

#     # Convert fixed effect variable to categorical if you want to compare by group
#     fixed_effect_type = type(df['fixed_effect'].iloc[0])
#     if comparison_type == 'infer':
#         if fixed_effect_type in [float, int, np.int64]:
#             comparison_type = 'continuous'
#             df['fixed_effect'] = df['fixed_effect'].astype(float)
#         elif fixed_effect_type == str:
#             comparison_type = 'categorical'
#         else:
#             raise TypeError('Cannot infer fixed effect dtype!')
#     elif comparison_type == 'continuous':
#         if not fixed_effect_type in [float, int, np.int64]:
#             raise TypeError('Cannot cast fixed effect dtype to float!')
#         else:
#             df['fixed_effect'] = df['fixed_effect'].astype(float)
#     elif comparison_type == 'categorical':
#         if not fixed_effect_type in [str, int, np.int64]:
#             raise TypeError('Cannot cast fixed effect type to str!')
#         else:
#             df['fixed_effect'] = df['fixed_effect'].astype(str)
#     else:
#         raise ValueError('Comparison type not recognised!')

#     # Intitialize pvals as series or dataframe (based on the number of comparisons per feature)
#     if comparison_type == 'continuous':
#         pvals = pd.Series(index=feat_names)
#     elif comparison_type == 'categorical':
#         groups = np.unique(df['fixed_effect'][df['fixed_effect']!=control])
#         pvals = pd.DataFrame(index=feat_names, columns=groups)

#     # Local function to perform LMM test for a single feature
#     def lmm_fit(feature, df):
#         # remove groups with fewer than 3 members
#         data = pd.concat([data for _, data in df.groupby(by=['fixed_effect', 'random_effect'])
#                           if data.shape[0] > 2])

#         # Define LMM
#         md = smf.mixedlm("{} ~ fixed_effect".format(feature), 
#                          data,
#                          groups=data['random_effect'].astype(str),
#                          re_formula="")
#         # Fit LMM
#         try:
#             mdf = md.fit()
#             pval = mdf.pvalues[[k for k in mdf.pvalues.keys() if k.startswith('fixed_effect')]]
#             pval = pval.min()
#         except:
#             pval = np.nan

#         return feature, pval

#     ## Fit LMMs for each feature
#     if n_jobs==1:
#         # Using a for loop is faster than launching a single job with joblib
#         for feature in tqdm (feat_names, desc="Testing features…", ascii=False):
#             _, pvals.loc[feature] = lmm_fit(feature, 
#                                             df[[feature,
#                                                 'fixed_effect',
#                                                 'random_effect']].dropna(axis=0))
#     else: 
#         # Parallelize jobs with joblib
#         parallel = Parallel(n_jobs=n_jobs, verbose=True)
#         func = delayed(lmm_fit)

#         res = parallel(func(feature, df[[feature,'fixed_effect','random_effect']].dropna(axis=0))
#                        for feature in feat_names)
#         for feature, pval in res:
#             pvals.loc[feature] = pval
    
#     # Benjamini-Yekutieli corrections for multiple comparisons
#     pvals_corrected = multiple_test_correction(pvals.T, fdr_method=fdr_method, fdr=fdr)
    
#     return pvals_corrected

def do_stats(metadata, 
             features, 
             group_by,
             control,
             save_dir,
             feat=['motion_mode_forward_fraction'], 
             pvalue_threshold=0.05, 
             fdr_method='fdr_by',
             ttest_if_nonsig=False,
             verbose=True):
    
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

    # categorical variables to investigate: 'gene_name' and 'window'
    if verbose:
        print("\nTesting for feature variation with respect to '%s'" % group_by)    

    groups = metadata[group_by].unique()
    test_groups = list([s for s in groups if s != control])    
    
    # check there will be no errors due to case-sensitivity
    assert len(groups) == len(metadata[group_by].str.upper().unique())
    
    # feature list input handling
    if isinstance(feat, str):
        feat = [feat]    
    assert isinstance(feat, list) and all(f in features.columns for f in feat)

    ### perform ANOVA - is there variation in worm motion mode among solvents used?
        
    fset = []
    if len(groups) > 2:        
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

        # compile results
        test_results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
        test_results.columns = ['stats','effect_size','pvals','reject']     
        test_results['significance'] = sig_asterix(test_results['pvals'])
        test_results = test_results.sort_values(by=['pvals'], ascending=True) # rank by p-value
        
        # save results
        if save_dir is not None:
            anova_path = Path(save_dir) / '{}_ANOVA_results.csv'.format(group_by)
            anova_path.parent.mkdir(parents=True, exist_ok=True)
            test_results.to_csv(anova_path, header=True, index=True)
        
        if verbose:
            print(test_results)

        # use reject mask to find significant feature set
        fset = pvals.loc[reject['ANOVA']].sort_values(by='ANOVA', ascending=True).index.to_list()

        # print results
        if verbose:
            print("%d significant features found by ANOVA for '%s' (P<%.2f, %s)" %\
                  (len(fset), group_by, pvalue_threshold, fdr_method))
        # save ANOVA significant features
        if len(fset) > 0 and save_dir is not None:
            anova_sigfeats_path = anova_path.parent / 'ANOVA_sigfeats.txt'
            write_list_to_file(fset, anova_sigfeats_path)


    ### perform t-tests - compare each group to control            
                    
    if len(groups) == 2 or len(fset) > 0 or ttest_if_nonsig:
        
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
        if save_dir is not None:
            ttest_strain_path = Path(save_dir) / '{}_ttest_results.csv'.format(group_by)
            ttest_strain_path.parent.mkdir(parents=True, exist_ok=True)        
            test_results.to_csv(ttest_strain_path, header=True, index=True)
            
        # print results
        if verbose:
            for group in test_groups:
                nsig_feats = sum(test_results['reject_' + group])
                print("%d significant features differing between %s vs %s (t-test, P<%.2f, %s)" %\
                      (nsig_feats, group, control, pvalue_threshold, fdr_method))
                
        if save_dir is None:
            return test_results
        
    return

# TODO: deprecated: single_feature_window_stats
def window_stats(metadata, 
                 features, 
                 group_by,
                 control,
                 save_dir,
                 windows=None,
                 feature_list=['motion_mode_forward_fraction'], 
                 pvalue_threshold=0.05, 
                 fdr_method='fdr_by'):
    
    """ ANOVA and t-tests for each window comparing Tierpsy features of worm behaviour on 
        different treatment conditions vs control 
        
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
            
        windows : list
            List of window numbers at which to compare strains (not corrected for multiple testing)
            
        feat : str
            Feature to test
        
        pvalue_threshold : float
            P-value significance threshold
            
        fdr_method : str
            Multiple testing correction method to use
    """
    

    # categorical variables to investigate: 'gene_name' and 'window'
    print("\nInvestigating variation in fraction of worms paused between hit strains and control " +
          "(for each window)")    

    # check there will be no errors due to case-sensitivity
    assert len(metadata[group_by].unique()) == len(metadata[group_by].str.upper().unique())
        
    # subset for list of windows
    if windows is None:
        windows = sorted(metadata['window'].unique())
    else:
        assert all(w in sorted(metadata['window'].unique()) for w in windows)
        metadata = metadata[metadata['window'].isin(windows)]
    
    if feature_list is not None:
        if isinstance(feature_list, str):
            feature_list = [feature_list]
        assert isinstance(feature_list, list) and all(f in features.columns for f in feature_list)
        
    features = features[feature_list].reindex(metadata.index)

    # print mean sample size
    sample_size = metadata.groupby([group_by, 'window']).count()
    print("Mean sample size of %s/window: %d" % (group_by, 
                                                 int(sample_size[sample_size.columns[-1]].mean())))
        
    for window in tqdm(windows):
        window_meta = metadata.query("window==@window")
        window_feat = features.reindex(window_meta.index)

        fset = []
        n = len(window_meta[group_by].unique())
        
        # Perform ANOVA - is there variation among strains at each window?
        if n > 2:
            anova_path = Path(save_dir) / 'ANOVA' / 'ANOVA_results_window_{}.csv'.format(window)
            anova_path.parent.mkdir(parents=True, exist_ok=True)
    
            stats, pvals, reject = univariate_tests(X=window_feat, 
                                                    y=window_meta[group_by], 
                                                    control=control, 
                                                    test='ANOVA',
                                                    comparison_type='multiclass',
                                                    multitest_correction=fdr_method,
                                                    alpha=pvalue_threshold,
                                                    n_permutation_test=None)
    
            # get effect sizes
            effect_sizes = get_effect_sizes(X=window_feat,
                                            y=window_meta[group_by],
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
    
        # perform t-tests comparing each strain to control
        if n == 2 or len(fset) > 0:
            ttest_path = Path(save_dir) / 't-tests' / 't-test_results_window_{}.csv'.format(window)
            ttest_path.parent.mkdir(parents=True, exist_ok=True)
            
            stats_t, pvals_t, reject_t = univariate_tests(X=window_feat,
                                                          y=window_meta[group_by],
                                                          control=control,
                                                          test='t-test',
                                                          comparison_type='binary_each_group',
                                                          multitest_correction=fdr_method,
                                                          alpha=pvalue_threshold,
                                                          n_permutation_test=None)
            
            effect_sizes_t = get_effect_sizes(X=window_feat, 
                                              y=window_meta[group_by],
                                              control=control,
                                              effect_type=None,
                                              linked_test='t-test')
            
            # compile results
            stats_t.columns = ['stats_' + str(c) for c in stats_t.columns]
            pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
            reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
            effect_sizes_t.columns = ['effect_size_' + str(c) for c in effect_sizes_t.columns]
            test_results = pd.concat([stats_t, pvals_t, reject_t, effect_sizes_t], axis=1)            
                            
            # save results
            test_results.to_csv(ttest_path, header=True, index=True)
                
            n_sig = sum(reject_t.sum(axis=1) > 0)
            print("%d significant features between any %s vs %s in window %d (t-test, P=%.2f, %s)" %\
                  (n_sig, group_by, control, window, pvalue_threshold, fdr_method))
        
    return
