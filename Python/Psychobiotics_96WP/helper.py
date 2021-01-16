#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper Functions

@author: sm5911
@date: 21/11/2020

"""

#%% Imports
import os
import re
import sys
import umap
import datetime
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path, PosixPath
from scipy.stats import (ttest_ind, 
                         ranksums, 
                         f_oneway, 
                         kruskal, 
                         zscore, 
                         shapiro)
from statsmodels.stats import multitest as smm # AnovaRM
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.covariance import MinCovDet
from matplotlib import pyplot as plt
#from matplotlib import patches, transforms
from matplotlib.gridspec import GridSpec
from matplotlib.axes._axes import _log as mpl_axes_logger
from mpl_toolkits.mplot3d import Axes3D

# Path to Github helper functions (USER-DEFINED path to local copy of Github repo)
PATH_LIST = ['/Users/sm5911/Tierpsy_Versions/tierpsy-tools-python/']
for sysPath in PATH_LIST:
    if sysPath not in sys.path:
        sys.path.insert(0, sysPath)

from tierpsytools.hydra.compile_metadata import add_imgstore_name # concatenate_days_metadata, get_camera_serial
                                                 
from tierpsytools.read_data.compile_features_summaries import compile_tierpsy_summaries
from tierpsytools.read_data.hydra_metadata import (read_hydra_metadata, 
                                                   align_bluelight_conditions)
from tierpsytools.hydra.match_wells_annotations import (import_wells_annoations_in_folder,
                                                        match_rawvids_annotations,
                                                        update_metadata)
from tierpsytools.feature_processing.filter_features import (filter_nan_inf, 
                                                             feat_filter_std, 
                                                             drop_bad_wells,
                                                             drop_ventrally_signed)

CUSTOM_STYLE = '/Users/sm5911/Documents/GitHub/PhD_Project/Python/Psychobiotics_96WP/analysis_heatmap.mplstyle'

#%% Functions
def duration_L1_diapause(df):
    """ Calculate L1 diapause duration (if possible) and append to results """
    
    diapause_required_columns = ['date_bleaching_yyyymmdd','time_bleaching',\
                                 'date_L1_refed_yyyymmdd','time_L1_refed_OP50']
    
    if all(x in df.columns for x in diapause_required_columns) and \
       all(df[x].any() for x in diapause_required_columns):
        # Extract bleaching dates and times
        bleaching_datetime = [datetime.datetime.strptime(date_str + ' ' +\
                              time_str, '%Y%m%d %H:%M:%S') for date_str, time_str\
                              in zip(df['date_bleaching_yyyymmdd'].astype(str),\
                              df['time_bleaching'])]
        # Extract dispensing dates and times
        dispense_L1_datetime = [datetime.datetime.strptime(date_str + ' ' +\
                                time_str, '%Y%m%d %H:%M:%S') for date_str, time_str\
                                in zip(df['date_L1_refed_yyyymmdd'].astype(str),\
                                df['time_L1_refed_OP50'])]
        # Estimate duration of L1 diapause
        L1_diapause_duration = [dispense - bleach for bleach, dispense in \
                                zip(bleaching_datetime, dispense_L1_datetime)]
        
        # Add duration of L1 diapause to df
        df['L1_diapause_seconds'] = [int(timedelta.total_seconds()) for \
                                     timedelta in L1_diapause_duration]
    else:
        missingInfo = [x for x in diapause_required_columns if x in df.columns\
                       and not df[x].any()]
        print("""WARNING: Could not calculate L1 diapause duration.\n\t\
         Required column info: %s""" % missingInfo)

    return df

def duration_on_food(df):
    """ Calculate time worms since worms dispensed on food for each video 
        entry in metadata """
    
    duration_required_columns = ['date_yyyymmdd','time_recording',
                                 'date_worms_on_test_food_yyyymmdd',
                                 'time_worms_on_test_food_yyyymmdd']
    
    if all(x in df.columns for x in duration_required_columns) and \
        all(df[x].any() for x in duration_required_columns):
        # Extract worm dispensing dates and times
        dispense_datetime = [datetime.datetime.strptime(date_str + ' ' +\
                             time_str, '%Y%m%d %H:%M:%S') for date_str, time_str\
                             in zip(df['date_worms_on_test_food_yyyymmdd'].astype(int).astype(str),\
                             df['time_worms_on_test_food_yyyymmdd'])]
        # Extract imaging dates and times
        imaging_datetime = [datetime.datetime.strptime(date_str + ' ' +\
                            time_str, '%Y%m%d %H:%M:%S') for date_str, time_str\
                            in zip(df['date_yyyymmdd'].astype(int).astype(str),df['time_recording'])]
        # Estimate duration worms have spent on food at time of imaging
        on_food_duration = [image - dispense for dispense, image in \
                            zip(dispense_datetime, imaging_datetime)]
            
        # Add duration on food to df
        df['duration_on_food_seconds'] = [int(timedelta.total_seconds()) for \
                                          timedelta in on_food_duration]
   
    return df

def process_metadata(aux_dir, 
                     imaging_dates=None, 
                     align_bluelight=True, 
                     add_well_annotations=True):
    """ Compile metadata from individual day metadata 
        - Add 'imgstore_name'
        - Add well annotations from WellAnnotator GUI
        - Add camera serial number 
        - Add duration on food
    """
    
    compiled_metadata_path = Path(aux_dir) / "metadata.csv"
    
    if not compiled_metadata_path.exists():
        print("Compiling from day-metadata in '%s'" % aux_dir)
        
        AuxFileList = os.listdir(aux_dir)
        ExperimentDates = sorted([expdate for expdate in AuxFileList if re.match(r'\d{8}', expdate)])
        
        if imaging_dates:
            ExperimentDates = [expdate for expdate in ExperimentDates if expdate in imaging_dates]
        else:
            imaging_dates = ExperimentDates
    
        day_meta_list = []
        for expdate in imaging_dates:
            day_meta_path = Path(aux_dir) / expdate / '{}_day_metadata.csv'.format(expdate)
                    
            day_meta = pd.read_csv(day_meta_path, dtype={"comments":str})
            
            # Rename metadata columns for compatibility with TierpsyTools functions 
            day_meta = day_meta.rename(columns={'date_recording_yyyymmdd': 'date_yyyymmdd',
                                                'well_number': 'well_name',
                                                'plate_number': 'imaging_plate_id',
                                                'run_number': 'imaging_run_number',
                                                'camera_number': 'camera_serial'})
             
            # Get path to RawVideo directory for day metadata
            rawDir = Path(str(day_meta_path).replace("AuxiliaryFiles","RawVideos")).parent
            
            # Get imgstore name + camera serial
            if 'imgstore_name' not in day_meta.columns:
                if 'camera_serial' in day_meta.columns:
                     day_meta = day_meta.drop(columns='camera_serial')
                day_meta = add_imgstore_name(day_meta, rawDir)
            
            # Get filename
            day_meta['filename'] = [rawDir.parent / day_meta.loc[i,'imgstore_name']\
                                    for i in range(len(day_meta['filename']))]
            
            # Overwrite day metadata
            print("Updating day metadata for: %s" % expdate)
            day_meta.to_csv(day_meta_path, index=False)
            
            day_meta_list.append(day_meta)
        
        # Concatenate list of day metadata into full metadata
        meta_df = pd.concat(day_meta_list, axis=0, ignore_index=True, sort=False)

        # Ensure no missing filenames
        assert not any(list(~np.array([isinstance(path, PosixPath) or\
                       isinstance(path, str) for path in meta_df['filename']])))
                
        # Calculate duration on food
        meta_df = duration_on_food(meta_df) 
        
        # Calculate L1 diapause duration
        meta_df = duration_L1_diapause(meta_df)
        
        meta_df.to_csv(compiled_metadata_path, index=None) 
        
        if add_well_annotations:
            annotations_df = import_wells_annoations_in_folder(aux_dir=aux_dir)
            
            rawDir = aux_dir.parent / "RawVideos"
            matched_long = match_rawvids_annotations(rawvid_dir=rawDir, 
                                                     annotations_df=annotations_df)
            # overwrite metadata with annotations added
            meta_df = update_metadata(aux_dir=aux_dir, 
                                      matched_long=matched_long, 
                                      saveto=compiled_metadata_path,
                                      del_if_exists=True)
            prop_bad = meta_df.is_bad_well.sum()/len(meta_df.is_bad_well)
            print("%.1f%% of data are labelled as 'bad well' data" % (prop_bad*100))
            
        print("Metadata saved to: %s" % compiled_metadata_path)
    else:
        # load metadata
        meta_df = pd.read_csv(compiled_metadata_path, dtype={"comments":str}, header=0)
        print("Metadata loaded.")
        
        if imaging_dates:
            meta_df = meta_df.loc[meta_df['date_yyyymmdd'].isin(imaging_dates),:]
            print("Extracted metadata for imaging dates provided.")
        if add_well_annotations:
            if not 'is_bad_well' in meta_df.columns:
                raise Warning("Bad well annotations not found in metadata!\n\
                         Please delete + re-compile with annotations")
            else:
                prop_bad = meta_df.is_bad_well.sum()/len(meta_df.is_bad_well)
                print("%.1f%% of data are labelled as 'bad well' data" % (prop_bad*100))
                
    return meta_df

def process_feature_summaries(metadata, 
                              results_dir, 
                              imaging_dates=None, 
                              add_bluelight=True):
    """ Compile feature summary results and join with metadata to produce
        combined full results. Clean full results by imputing NaN values 
        according to 'nan_threshold' given, where feature column containing 
        more than threshold number of NaN values are dropped from the analysis 
    """    
    combined_feats_path = results_dir / "full_features.csv"
    combined_fnames_path = results_dir / "full_filenames.csv"
 
    if not np.logical_and(combined_feats_path.is_file(), 
                          combined_fnames_path.is_file()):
        print("Processing feature summary results..")
        
        if imaging_dates:
            feat_files = []
            fname_files = []
            for date in imaging_dates:
                date_dir = Path(results_dir) / date
                feat_files.extend([file for file in Path(date_dir).rglob('features_summary*.csv')])
                fname_files.extend([Path(str(file).replace("/features_","/filenames_")) for file in feat_files])
        else:
            feat_files = [file for file in Path(results_dir).rglob('features_summary*.csv')]
            fname_files = [Path(str(file).replace("/features_", "/filenames_")) for file in feat_files]
            
        # Keep only features files for which matching filenames_summaries exist
        feat_files = [feat_fl for feat_fl,fname_fl in zip(np.unique(feat_files),\
                      np.unique(fname_files)) if fname_fl is not None]
        fname_files = [fname_fl for fname_fl in np.unique(fname_files) if\
                       fname_fl is not None]
        
        # Compile feature summaries for mathed features/filename summaries
        compile_tierpsy_summaries(feat_files=feat_files, 
                                  compiled_feat_file=combined_feats_path,
                                  compiled_fname_file=combined_fnames_path,
                                  fname_files=fname_files)
    
    # Read features/filename summaries
    feature_summaries = pd.read_csv(combined_feats_path, comment='#')
    filename_summaries = pd.read_csv(combined_fnames_path, comment='#')
    print("Feature summary results loaded.")

    features, metadata = read_hydra_metadata(feature_summaries, 
                                             filename_summaries,
                                             metadata,
                                             add_bluelight=add_bluelight)
    if add_bluelight:
        features, metadata = align_bluelight_conditions(feat=features, 
                                                        meta=metadata, 
                                                        how='outer',
                                                        merge_on_cols=['date_yyyymmdd',
                                                                       'imaging_run_number',
                                                                       'imaging_plate_id',
                                                                       'well_name'])
    assert set(features.index) == set(metadata.index)
    
    return features, metadata

def clean_features_summaries(features, 
                             metadata, 
                             feature_columns=None, 
                             imputeNaN=True,
                             nan_threshold=0.2, 
                             drop_size_related_feats=False,
                             norm_feats_only=False):
    """ Clean features summary results:
        - Drop features with too many NaN/Inf values (> nan_threshold)
        - Impute remaining NaN values with global mean value for each feature
        - Drop features with zero standard deviation
        - Drop features that are ventrally signed
    """

    assert set(features.index) == set(metadata.index)

    if feature_columns is not None:
        assert all([feat in features.columns for feat in feature_columns])
        features = features[feature_columns]
    else:
        feature_columns = features.columns
               
    # Drop bad well data
    features, metadata = drop_bad_wells(features, metadata, bad_well_cols=['is_bad_well'], verbose=False)
    assert not any(features.sum(axis=1) == 0) # ensure no missing row data

    # Drop feature columns with too many NaN values
    features = filter_nan_inf(features, threshold=nan_threshold, axis=0)
    nan_cols = [col for col in feature_columns if col not in features.columns]
    if len(nan_cols) > 0:
        print("Dropped %d features with >%.1f%% NaNs" % (len(nan_cols), nan_threshold*100))
    # Dropped feats are 'food_edge'-related (n = 93 *3=279) which is undefined so NaNs are expected
    
    # if any((features == 0).all()): # check for columns containing only zeros
    #     print("%d features found with all zero values" % len(features.columns[(features == 0).all()]))
 
    # Drop feature columns with zero standard deviation
    feature_columns = features.columns
    features = feat_filter_std(features, threshold=0.0)
    zero_std_feats = [col for col in feature_columns if col not in features.columns]
    if len(zero_std_feats) > 0:
        print("Dropped %d features with zero standard deviation" % len(zero_std_feats))
    
    # Impute remaining NaN values (using global mean feature value for each strain)
    if imputeNaN:
        n_nans = features.isna().sum(axis=0).sum()
        if n_nans > 0:
            print("Imputing %d missing values (%.2f%% data), using global mean value for each feature.."\
                  % (n_nans, n_nans/features.count().sum()*100)) 
            features = features.fillna(features.mean(axis=0))
        else:
            print("No need to impute! No remaining NaN values found in feature summary results.")

    # Drop ventrally-signed features
    # In general, for the curvature and angular velocity features we should only 
    # use the 'abs' versions, because the sign is assigned based on whether the worm 
    # is on its left or right side and this is not known for the multiworm tracking data
    feature_columns = features.columns
    features = drop_ventrally_signed(features)
    ventrally_signed_feats = [col for col in feature_columns if col not in features.columns]
    if len(ventrally_signed_feats) > 0:
        print("Dropped %d features that are ventrally signed" % len(ventrally_signed_feats))
    
    # Drop size-related features
    if drop_size_related_feats:
        size_feat_keys = ['blob','box','width','length','area']
        size_features = []
        for feature in list(features.columns):
            for key in size_feat_keys:
                if key in feature:
                    size_features.append(feature)
        feature_columns = [feat for feat in features.columns if feat not in size_features]
        features = features[feature_columns]
        print("Dropped %d features that are size-related" % len(size_features))
        
    # Use '_norm' features only
    if norm_feats_only:
        feature_columns = features.columns
        norm_features = [feat for feat in feature_columns if '_norm' in feat]
        features = features[norm_features]
        print("Dropped %d features that are not '_norm' features" % (len(feature_columns)-len(features.columns)))
        
    return features, metadata

def load_top256(top256_path, remove_path_curvature=True, add_bluelight=True):
    """ Load Tierpsy Top256 set of features describing N2 behaviour on E. coli 
        OP50 bacteria 
    """   
    top256_df = pd.read_csv(top256_path, header=0)
    top256 = list(top256_df[top256_df.columns[0]])
    n = len(top256)
    print("Feature list loaded (n=%d features)" % n)

    # Remove features from Top256 that are path curvature related
    top256 = [feat for feat in top256 if "path_curvature" not in feat]
    n_feats_after = len(top256)
    print("Dropped %d features from Top%d that are related to path curvature"\
          % ((n - n_feats_after), n))
        
    if add_bluelight:
        bluelight_suffix = ['_prestim','_bluelight','_poststim']
        top256 = [col + suffix for suffix in bluelight_suffix for col in top256]

    return top256

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
                    # NB: UserWarning: Input data for shapiro has range zero # Some features for that strain contain all zeros - shapiro(np.zeros(5))
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
    # NB: Null hypothesis - feature summary results for individual strains are normally distributed (Gaussian)
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
            print("""Less than %d%% of features (%.1f%%) were found to obey a normal distribution, 
                  so non-parametric analyses will be preferred.""" % (is_normal_threshold*100, 
                                                                      total_prop_normal*100))
    
    return prop_features_normal, is_normal

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
    # Drop columns that contain only zeros
    n_cols = len(feat_df.columns)
    feat_df = feat_df.drop(columns=feat_df.columns[(feat_df == 0).all()])
    zero_cols = n_cols - len(feat_df.columns)
    if zero_cols > 0:
        print("Dropped %d feature summaries (all zeros)" % zero_cols)
  
    # Record name of statistical test used (kruskal/f_oneway)
    TEST = f_oneway if is_normal else kruskal
    test_name = str(TEST).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]
    print("\nComputing %s tests between strains for each feature..." % test_name)

    # Perform 1-way ANOVAs for each feature between test strains
    test_pvalues_df = pd.DataFrame(index=['stat','pval'], columns=feat_df.columns)
    for f, feature in enumerate(tqdm(feat_df.columns)):
            
        # Perform test and capture outputs: test statistic + p value
        test_stat, test_pvalue = TEST(*[feat_df[meta_df[group_by]==strain][feature]\
                                           for strain in meta_df[group_by].unique()])
        test_pvalues_df.loc['stat',feature] = test_stat
        test_pvalues_df.loc['pval',feature] = test_pvalue

    # Perform Bonferroni correction for multiple comparisons on one-way ANOVA pvalues
    _corrArray = smm.multipletests(test_pvalues_df.loc['pval'], 
                                   alpha=p_value_threshold, 
                                   method=fdr_method,
                                   is_sorted=False, 
                                   returnsorted=False)
    
    # Update pvalues with Benjamini-Yekutieli correction
    test_pvalues_df.loc['pval',:] = _corrArray[1]
    
    # Store names of features that show significant differences across the test bacteria
    sigfeats = test_pvalues_df.columns[np.where(test_pvalues_df.loc['pval'] < p_value_threshold)]
    print("Complete!\n%d/%d (%.1f%%) features exhibit significant differences between strains (%s test, %s)"\
          % (len(sigfeats), len(test_pvalues_df.columns), 
             len(sigfeats)/len(test_pvalues_df.columns)*100, test_name, fdr_method))
    
    # Compile list to store names of significant features
    sigfeats_list = pd.Series(test_pvalues_df.columns[np.where(test_pvalues_df.loc['pval'] < p_value_threshold)])
    sigfeats_list.name = 'significant_features_' + test_name
    sigfeats_list = pd.DataFrame(sigfeats_list)
      
    topfeats = test_pvalues_df.loc['pval'].sort_values(ascending=True)[:10]
    print("Top 10 significant features by %s test:\n" % test_name)
    for feat in topfeats.index:
        print(feat)

    return test_pvalues_df, sigfeats_list

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

def ttest_by_feature(feat_df, 
                     meta_df, 
                     group_by, 
                     control_strain, 
                     is_normal=True,
                     p_value_threshold=0.05,
                     fdr_method='fdr_by'):
    """ Perform t-tests for significant differences between each strain and the
        control, for each feature. If is_normal=False, rank-sum tests will be 
        performed instead 
    """
    # Record name of statistical test used (ttest/ranksumtest)
    TEST = ttest_ind if is_normal else ranksumtest
    test_name = str(TEST).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]
    
    # Extract control results
    control_meta = meta_df[meta_df[group_by] == control_strain]
    control_feats = feat_df.reindex(control_meta.index)
    
    # Record test strains
    test_strains = [strain for strain in meta_df[group_by].unique() if strain != control_strain]

    # Pre-allocate dataframes for storing test statistics and p-values
    test_stats_df = pd.DataFrame(index=list(test_strains), columns=feat_df.columns)
    test_pvalues_df = pd.DataFrame(index=list(test_strains), columns=feat_df.columns)
    sigfeats_table = pd.DataFrame(index=test_pvalues_df.index, columns=['sigfeats','sigfeats_corrected'])
    
    # Compute test statistics for each strain, comparing to control for each feature
    for t, strain in enumerate(test_strains):
        print("Computing %s tests for %s vs %s..." % (test_name, control_strain, strain))
            
        # Grab feature summary results for that strain
        strain_meta = meta_df[meta_df[group_by] == strain]
        strain_feats = feat_df.reindex(strain_meta.index)
        
        # Drop columns that contain only zeros
        n_cols = len(strain_feats.columns)
        strain_feats = feat_filter_std(strain_feats, threshold=0.0)
        control_feats = feat_filter_std(control_feats, threshold=0.0)
        # strain_feats = strain_feats.drop(columns=strain_feats.columns[(strain_feats == 0).all()])
        # control_feats = control_feats.drop(columns=control_feats.columns[(control_feats == 0).all()])
        zero_std_cols = n_cols - len(strain_feats.columns)
        if zero_std_cols > 0:
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

def barplot_sigfeats_ttest(test_pvalues_df, 
                           saveDir=None,
                           p_value_threshold=0.05):
    """ """
    # Proportion of features significantly different from control
    prop_sigfeats = ((test_pvalues_df < p_value_threshold).sum(axis=1)/len(test_pvalues_df.columns))*100
    prop_sigfeats = prop_sigfeats.sort_values(ascending=False)
    
    # Plot proportion significant features for each strain
    plt.ioff() if saveDir else plt.ion()
    plt.close('all')
    fig = plt.figure(figsize=[7,10])
    ax = fig.add_subplot(1,1,1)
    prop_sigfeats.plot.barh(ec='black') # fc
    ax.set_xlabel('% significantly different features', fontsize=16, labelpad=10)
    ax.set_ylabel('Strain', fontsize=17, labelpad=10)
    plt.xlim(0,100)
    plt.tight_layout(rect=[0.02, 0.02, 0.96, 1])
    
    if saveDir:
        Path(saveDir).mkdir(exist_ok=True, parents=True)
        savePath = Path(saveDir) / 'percentage_sigfeats.eps'
        print("Saving figure: %s" % savePath.name)
        plt.savefig(savePath, format='eps', dpi=600)
        plt.close()
    else:
        plt.show()
    
    return prop_sigfeats
    
def boxplots_top_feats(feat_meta_df, 
                       test_pvalues_df, 
                       group_by, 
                       control_strain, 
                       saveDir=None, 
                       p_value_threshold=0.05,
                       n_top_features=None,
                       sns_colour_palette="tab10"):
    """ Box plots of most significantly different features between strains """    
        
    plt.ioff() if saveDir else plt.ion()
    plt.close('all')
       
    for strain in tqdm(test_pvalues_df.index):
        pvals = test_pvalues_df.loc[strain]
        
        n_sigfeats = sum(pvals < p_value_threshold)
                    
        if pvals.isna().all():
            print("No signficant features found for %s" % strain)
        elif n_sigfeats > 0:       
            ranked_pvals = pvals.sort_values(ascending=True) # rank p-values in ascending order
            ranked_pvals = ranked_pvals.dropna(axis=0) # drop NaNs
            topfeats = ranked_pvals[ranked_pvals < p_value_threshold] # drop non-sig feats  
            
            # select top ranked p-values
            if not n_top_features:
                n_top_features = n_sigfeats
            if len(topfeats) < n_top_features:
                print("Only %d significant features found for %s" % (n_sigfeats, str(strain)))
                n_top_features = len(topfeats)
            else:
                topfeats = topfeats[:n_top_features]
            #topfeats = pd.Series(index=['curvature_midbody_abs_90th'])
            
            # print("\nTop %d features for %s:\n" % (len(topfeats), strain))
            # print(*[feat + '\n' for feat in list(topfeats.index)])
    
            # Subset feature summary results for test-strain + control only
            plot_df = feat_meta_df[np.logical_or(feat_meta_df[group_by]==control_strain,
                                                 feat_meta_df[group_by]==str(strain))]
        
            # Colour/legend dictionary
            # Create colour palette for plot loop
            colour_labels = sns.color_palette(sns_colour_palette, 2)
            colour_dict = {control_strain:colour_labels[0], str(strain):colour_labels[1]} # '#0C9518', '#C2FDBE'
                                                  
            # Boxplots of control vs test-strain for each top-ranked significant feature
            for f, feature in enumerate(topfeats.index):
                plt.close('all')
                sns.set_style('darkgrid')
                fig = plt.figure(figsize=[10,8])
                ax = fig.add_subplot(1,1,1)
                sns.boxplot(x=group_by, 
                            y=feature, 
                            data=plot_df, 
                            palette=colour_dict,
                            showfliers=False, 
                            showmeans=True,
                            meanprops={"marker":"x", 
                                       "markersize":5,
                                       "markeredgecolor":"k"},
                            flierprops={"marker":"x", 
                                        "markersize":15, 
                                        "markeredgecolor":"r"})
                sns.stripplot(x=group_by, 
                              y=feature, 
                              data=plot_df, 
                              s=6, 
                              marker=".", 
                              color='k')
                ax.set_xlabel('Strain', fontsize=15, labelpad=12)
                ax.set_ylabel(feature, fontsize=15, labelpad=12)
                ax.set_title(feature, fontsize=20, pad=40)
    
                # # Add plot legend
                # patch_list = []
                # for l, key in enumerate(colour_dict.keys()):
                #     patch = patches.Patch(color=colour_dict[key], label=key)
                #     patch_list.append(patch)
                #     if key == strain:
                #         ylim = plot_df[plot_df[group_by]==key][feature].max()
                #         pval = test_pvalues_df.loc[key, feature]
                #         if isinstance(pval, float) and pval < p_value_threshold:
                #             trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                #             ax.text(l - 0.1, 1, 'p={:g}'.format(float('{:.2g}'.format(pval))),\
                #             fontsize=13, color='k', verticalalignment='bottom', transform=trans)
                # plt.tight_layout(rect=[0.04, 0, 0.84, 0.96])
                # plt.legend(handles=patch_list, labels=colour_dict.keys(), loc=(1.02, 0.8),\
                #           borderaxespad=0.4, frameon=False, fontsize=15)
    
                # Save figure
                if saveDir:
                    plot_path = saveDir / str(strain) / ('{0}_'.format(f + 1) + feature + '.eps')
                    plot_path.parent.mkdir(exist_ok=True, parents=True)
                    plt.savefig(plot_path, format='eps', dpi=300)
                else:
                    plt.show()
                    
                plt.close(fig) # Close plot
    
def boxplots_by_strain(df,
                       group_by,
                       test_pvalues_df,
                       control_strain,
                       features2plot=None,
                       saveDir=None,
                       p_value_threshold=0.05,
                       max_features_plot_cap=None, 
                       max_groups_plot_cap=48,
                       figsize=[8,12],
                       sns_colour_palette="tab10"):
    """ Boxplots comparing all strains to control for a given feature """
        
    if features2plot is not None:
        assert all(feat in df.columns for feat in features2plot)
        # Drop insignificant features
        features2plot = [feature for feature in features2plot if \
                         (test_pvalues_df[feature] < p_value_threshold).any()]
        
        if max_features_plot_cap and len(features2plot) > max_features_plot_cap:
            print("WARNING: Too many features to plot! Capping at %d plots"\
                  % max_features_plot_cap)
            features2plot = features2plot[:max_features_plot_cap]
    else:
        # Plot all sig feats between any strain and control
        features2plot = [feature for feature in test_pvalues_df.columns if \
                         (test_pvalues_df[feature] < p_value_threshold).any()]
    
    # OPTIONAL: Plot cherry-picked features
    #features2plot = ['speed_50th','curvature_neck_abs_50th','major_axis_50th','angular_velocity_neck_abs_50th']
    
    # Seaborn boxplots with swarmplot overlay for each feature - saved to file
    plt.ioff() if saveDir else plt.ion()
    sns.set(color_codes=True); sns.set_style('darkgrid')
    for f, feature in enumerate(tqdm(features2plot)):
        sortedPvals = test_pvalues_df[feature].sort_values(ascending=True)
        strains2plt = list(sortedPvals.index)
        if len(strains2plt) > max_groups_plot_cap:
            strains2plt = list(sortedPvals[:max_groups_plot_cap].index)
            
        strains2plt.insert(0, control_strain)
        plot_df = df[df[group_by].isin(strains2plt)]
        
        # Rank by median
        rankMedian = plot_df.groupby(group_by)[feature].median().sort_values(ascending=True)
        #plot_df = plot_df.set_index(group_by).loc[strains2plt].reset_index()
        plot_df = plot_df.set_index(group_by).loc[rankMedian.index].reset_index()
        
        if len(strains2plt) > 10:
            colour_dict = {strain: "r" if strain == control_strain else \
                           "darkgray" for strain in plot_df[group_by].unique()}
            colour_dict2 = {strain: "b" for strain in list(sortedPvals[sortedPvals < p_value_threshold].index)}
            colour_dict.update(colour_dict2)
        else:
            colour_labels = sns.color_palette(sns_colour_palette, len(strains2plt))
            colour_dict = {key:col for (key,col) in zip(plot_df[group_by].unique(), colour_labels)}
        
        # Seaborn boxplot for each feature (only top strains)
        plt.close('all')
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)
        sns.boxplot(x=feature, y=group_by, data=plot_df, showfliers=False,\
                    showmeans=True,\
                    meanprops={"marker":"x", "markersize":5, "markeredgecolor":"k"},\
                    flierprops={"marker":"x", "markersize":15, "markeredgecolor":"r"},\
                    palette=colour_dict)
        ax.set_xlabel(feature, fontsize=18, labelpad=10)
        ax.set_ylabel(group_by, fontsize=18, labelpad=10)
        locs, labels = plt.yticks() # Get y-axis tick positions and labels
        labs = [lab.get_text() for lab in labels]
        #trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        
        for l, strain in enumerate(labs):
            if strain == control_strain:
                plt.axvline(x=rankMedian[control_strain], c='dimgray', ls='--')
                continue
            pval = test_pvalues_df.loc[strain, feature]
            if isinstance(pval, float) and pval < p_value_threshold:
                xmin, xmax = ax.get_xlim()
                xtext = xmin + 1*(xmax - xmin)
                ax.text(xtext, locs[l], 'p={:g}'.format(float('{:.2g}'.format(pval))),\
                        fontsize=10, color='k', 
                        horizontalalignment='left', verticalalignment='center')
        plt.subplots_adjust(top=0.9,bottom=0.1,left=0.2,right=0.85)
         
        # Save boxplot
        if saveDir:
            saveDir.mkdir(exist_ok=True, parents=True)
            plot_path = Path(saveDir) / (str(f + 1) + '_' + feature + '.eps')
            plt.savefig(plot_path, format='eps', dpi=300)
        else:
            plt.show()

def plot_clustermap(featZ, 
                    meta, 
                    group_by, 
                    test_pvalues_df=None,
                    selected_feats=None,
                    saveto=None,
                    sns_colour_palette="tab10"):
    """ Seaborn clustermap (hierarchical clustering heatmap) of normalised """                
    
    assert (featZ.index == meta.index).all()
    
    # Store feature names
    fset = featZ.columns
        
    # Compute average value for strain for each feature (not each well)
    featZ_grouped = featZ.join(meta).groupby([group_by,'date_yyyymmdd']).mean().reset_index()
    
    # Map colors for strains
    var_list = list(featZ_grouped[group_by].unique())
    var_colour_dict = dict(zip(var_list, sns.color_palette(sns_colour_palette, len(var_list))))
    date_list = list(featZ_grouped['date_yyyymmdd'].unique())
    date_colour_dict = dict(zip(date_list, sns.color_palette("Set2", len(date_list))))
    #date_colour_dict = dict(zip(set(date_list), sns.hls_palette(len(set(date_list)), l=0.5, s=0.8)))
    bluelight_colour_dict = dict(zip(['prestim','bluelight','poststim'], sns.color_palette("Set2", 3)))
    feat_colour_dict = {f:bluelight_colour_dict[f.split('_')[-1]] for f in fset}
    
    #Create additional row_colors here
    row_cols_var = featZ_grouped[group_by].map(var_colour_dict)
    row_cols_date = featZ_grouped['date_yyyymmdd'].map(date_colour_dict)

    # TODO: import fastcluster ??
    # Plot clustermap
    plt.ioff() if saveto else plt.ion()
    plt.close('all')
    sns.set(font_scale=0.6)
    cg = sns.clustermap(data=featZ_grouped[fset], 
                        row_colors=[row_cols_var, row_cols_date],
                        col_colors=fset.map(feat_colour_dict),
                        #standard_scale=1, 
                        #z_score=1,
                        metric='euclidean', 
                        method='complete',
                        vmin=-2, vmax=2,
                        figsize=[15,10],
                        xticklabels=fset if len(fset) < 256 else False,
                        yticklabels=featZ_grouped[group_by])
    #cg.ax_heatmap.axes.set_xticklabels([])
    #cg.ax_heatmap.axes.set_yticklabels([])
    if len(fset) <= 256:
        plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), fontsize=10)
    
    # patch_list = []
    # for l, key in enumerate(strain_colour_dict.keys()):
    #     patch = patches.Patch(color=strain_colour_dict[key], label=key)
    #     patch_list.append(patch)
    # plt.legend(handles=patch_list, labels=strain_colour_dict.keys(),\
    #            borderaxespad=0.4, frameon=False, loc=(-3, -13), fontsize=8)
     
    plt.subplots_adjust(top=0.95,bottom=0.05,left=0.02,right=0.92,hspace=0.2,wspace=0.2)
    plt.tight_layout(rect=[0, 0, 0.89, 1], w_pad=0.5)
    
    # Extract clustered features
    clustered_features = np.array(fset)[cg.dendrogram_col.reordered_ind]
    
    # Save clustermap
    if saveto:
        saveto.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(saveto, dpi=300)
    else:
        plt.show()
    
    return clustered_features

def plot_barcode_clustermap(featZ, 
                            meta, 
                            group_by, 
                            control=None,
                            test_pvalues_df=None,
                            p_value_threshold=0.05,
                            selected_feats=None,
                            saveto=None,
                            sns_colour_palette="tab10"):
    
    assert set(featZ.index) == set(meta.index)
    
    # Store feature names
    fset = featZ.columns
        
    # Compute average value for strain for each feature (not each well)
    featZ_grouped = featZ.join(meta).groupby(group_by).mean()#.reset_index()
       
    # Plot barcode clustermap
    plt.ioff() if saveto else plt.ion()
    plt.close('all')  
    # Make dataframe for heatmap plot
    heatmap_df_list = [featZ_grouped[fset]]
    
    if test_pvalues_df is not None:
        assert set(test_pvalues_df.columns) == set(fset)
        assert len(test_pvalues_df.index) == 1
        
        # rename index for plot label
        test_pvalues_df.index = ['p<{}'.format(p_value_threshold)]
        heatmap_df_list.append(-np.log10(test_pvalues_df))
    
    # Map colors for stimulus type
    _stim = pd.DataFrame(data=[f.split('_')[-1] for f in fset], columns=['stim_type'])
    _stim['stim_type'] = _stim['stim_type'].map({'prestim':1,'bluelight':2,'poststim':3})
    _stim = _stim.transpose().rename(columns={c:v for c,v in enumerate(fset)})
    heatmap_df_list.append(_stim)
    heatmap_df = pd.concat(heatmap_df_list)
    
    # Add barcode - asterisk (*) to highlight selected features
    cm=['inferno', 'inferno', 'Greys', 'Pastel1']
    vmin_max = [(-2,2), (-2,2), (0,20), (1,3)]
    sns.set_style('ticks')
    plt.style.use(CUSTOM_STYLE)  
    
    f = plt.figure(figsize= (20,3))
    gs = GridSpec(4, 1,wspace=0, hspace=0, height_ratios=[3,3,1,1])
    cbar_ax = f.add_axes([.91, .3, .03, .4])
    
    for n, ((ix, r), c, v) in enumerate(zip(heatmap_df.iterrows(), cm, vmin_max)):
        axis = f.add_subplot(gs[n])
        sns.heatmap(r.to_frame().transpose().astype(float),
                    yticklabels=[ix],
                    xticklabels=[],
                    ax=axis,
                    cmap=c,
                    cbar=n==0, #only plots colorbar for first plot
                    cbar_ax=None if n else cbar_ax,
                    vmin=v[0],
                    vmax=v[1])
        axis.set_yticklabels(labels=[ix], rotation=0, fontsize=20)
        
        if n>2:
            c = sns.color_palette('Pastel1',3)
            sns.heatmap(r.to_frame().transpose(),
                    yticklabels=[ix],
                    xticklabels=[],
                    ax=axis,
                    cmap=c,
                    cbar=n==0, 
                    cbar_ax=None if n else cbar_ax,
                    vmin=v[0],
                    vmax=v[1])
            axis.set_yticklabels(labels=[ix], rotation=0, fontsize=20)
        cbar_ax.set_yticklabels(labels = cbar_ax.get_yticklabels())#, fontdict=font_settings)
        #f.tight_layout(rect=[0, 0, 0.89, 1], w_pad=0.5)
    
    if selected_feats is not None:
        for feat in selected_feats:
            try:
                axis.text(heatmap_df.columns.get_loc(feat), 1, '*')
            except KeyError:
                print('{} not in featureset'.format(feat))
    
    if saveto:
        saveto.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(saveto, dpi=600)
    else:
        plt.show()
                
    return 

def pcainfo(pca, zscores, PC=0, n_feats2print=10):
    """ A function to plot PCA explained variance, and print the most 
        important features in the given principal component (P.C.)
    """
        
    cum_expl_var_frac = np.cumsum(pca.explained_variance_ratio_)

    # Plot explained variance
    fig, ax = plt.subplots()
    plt.plot(range(1,len(cum_expl_var_frac)+1),
             cum_expl_var_frac,
             marker='o')
    ax.set_xlabel('Number of Principal Components', fontsize=15)
    ax.set_ylabel('explained $\sigma^2$', fontsize=15)
    ax.set_ylim((0,1.05))
    fig.tight_layout()
    
    # Print important features
    # important_feats_list = []
    # for pc in range(PCs_to_keep):
    important_feats = pd.DataFrame(pd.Series(zscores.columns[np.argsort(pca.components_[PC]**2)\
                                      [-n_feats2print:][::-1]], name='PC_{}'.format(str(PC))))
    # important_feats_list.append(pd.Series(important_feats, 
    #                                       name='PC_{}'.format(str(pc+1))))
    # important_feats = pd.DataFrame(important_feats_list).T
    
    print("\nTop %d features in Principal Component %d:\n" % (n_feats2print, PC))
    for feat in important_feats['PC_{}'.format(PC)]:
        print(feat)

    return important_feats, fig

def plot_pca(featZ, 
             meta, 
             group_by, 
             n_dims=2,
             var_subset=None, 
             saveDir=None,
             PCs_to_keep=10,
             n_feats2print=10,
             sns_colour_palette="tab10"):
    """ Perform principal components analysis 
        - group_by : column in metadata to group by for plotting (colours) 
        - n_dims : number of principal component dimensions to plot (2 or 3)
        - var_subset : subset list of categorical names in featZ[group_by]
        - saveDir : directory to save PCA results
        - PCs_to_keep : number of PCs to project
        - n_feats2print : number of top features influencing PCs to store """
    
    assert (featZ.index == meta.index).all()
    if var_subset is not None:
        assert all([strain in meta[group_by].unique() for strain in var_subset])
    else:
        var_subset = list(meta[group_by].unique())
    
    # Perform PCA on extracted features
    print("\nPerforming Principal Components Analysis (PCA)...")

    # Fit the PCA model with the normalised data
    pca = PCA()
    pca.fit(featZ)

    # Plot summary data from PCA: explained variance (most important features)
    plt.ioff() if saveDir else plt.ion()
    important_feats, fig = pcainfo(pca=pca, 
                                   zscores=featZ, 
                                   PC=0, 
                                   n_feats2print=n_feats2print)
           
    # Save plot of PCA explained variance
    if saveDir:
        pca_path = Path(saveDir) / 'PCA_explained.eps'
        pca_path.parent.mkdir(exist_ok=True, parents=True)
        plt.tight_layout()
        plt.savefig(pca_path, format='eps', dpi=300)

        # Save PCA important features list
        pca_feat_path = Path(saveDir) / 'PC_top{}_features.csv'.format(str(n_feats2print))
        important_feats.to_csv(pca_feat_path, index=False)        
    else:
        plt.show(); plt.pause(2)

    # Project data (zscores) onto PCs
    projected = pca.transform(featZ) # A matrix is produced
    # NB: Could also have used pca.fit_transform() OR decomposition.TruncatedSVD().fit_transform()

    # Store the results for first few PCs in dataframe
    projected_df = pd.DataFrame(data=projected[:,:PCs_to_keep],
                                columns=['PC' + str(n+1) for n in range(PCs_to_keep)],
                                index=featZ.index)
    
    # TODO: Use sklearn.kdeplot Cluster boundaries (convex polygon?) and quantify overlap?

    plt.close('all')
    if n_dims == 2:
        # OPTION 1: Plot PCA - 2 principal components
        plt.rc('xtick',labelsize=15)
        plt.rc('ytick',labelsize=15)
        fig, ax = plt.subplots(figsize=[9,8])
                
        # Create colour palette for plot loop
        colour_labels = sns.color_palette(sns_colour_palette, len(var_subset))
        palette = dict(zip(var_subset, colour_labels))
        
        grouped = meta.join(projected_df).groupby(group_by)
        for key, group in grouped:
            group.plot(ax=ax, 
                       kind='scatter', 
                       x='PC1', 
                       y='PC2', 
                       label=key, 
                       color=palette[key])
        sns.kdeplot(x='PC1', 
                    y='PC2', 
                    data=meta.join(projected_df), 
                    hue=group_by, 
                    palette=palette,
                    fill=False,
                    thresh=0.05,
                    levels=1)        
        ax.set_xlabel('Principal Component 1', fontsize=20, labelpad=12)
        ax.set_ylabel('Principal Component 2', fontsize=20, labelpad=12)
        ax.set_title("PCA by '{}'".format(group_by), fontsize=20)
        sns.set_style("whitegrid")
        if len(var_subset) <= 15:
            plt.tight_layout(rect=[0.04, 0, 0.84, 0.96])
            ax.legend(var_subset, frameon=False, loc='upper right', fontsize=15)
        ax.grid()
        plt.tight_layout()
        plt.show()

        
    elif n_dims == 3:
        # OPTION 2: Plot PCA - 3 principal components  
        plt.rc('xtick',labelsize=12)
        plt.rc('ytick',labelsize=12)
        fig = plt.figure(figsize=[10,10])
        mpl_axes_logger.setLevel('ERROR') # Work-around for 3D plot colour warnings
        ax = Axes3D(fig) # ax = fig.add_subplot(111, projection='3d')
        
        # Create colour palette for plot loop
        palette = itertools.cycle(sns.color_palette(sns_colour_palette, len(var_subset)))
        
        for g_var in var_subset:
            g_var_projected_df = projected_df[meta[group_by]==g_var]
            ax.scatter(xs=g_var_projected_df['PC1'], 
                       ys=g_var_projected_df['PC2'], 
                       zs=g_var_projected_df['PC3'],
                       zdir='z', s=30, c=next(palette), depthshade=False)
        ax.set_xlabel('Principal Component 1', fontsize=15, labelpad=12)
        ax.set_ylabel('Principal Component 2', fontsize=15, labelpad=12)
        ax.set_zlabel('Principal Component 3', fontsize=15, labelpad=12)
        ax.set_title("PCA by '{}'".format(group_by), fontsize=20)
        if len(var_subset) <= 15:
            ax.legend(var_subset, frameon=False, fontsize=12)
            #ax.set_rasterized(True)
        ax.grid()
    else:
        raise ValueError("Value for 'n_dims' must be either 2 or 3")

    # Save PCA plot
    if saveDir:
        pca_path = Path(saveDir) / ('pca_by_{}'.format(group_by) 
                                    + ('.png' if n_dims == 3 else '.eps'))
        plt.savefig(pca_path, format='png' if n_dims == 3 else 'eps', 
                    dpi=600 if n_dims == 3 else 300) # rasterized=True
    else:
        # Rotate the axes and update plot        
        if n_dims == 3:
            for angle in range(0, 360):
                ax.view_init(270, angle)
                plt.draw(); plt.pause(0.0001)
        else:
            plt.show()
    
    return projected_df

def find_outliers_mahalanobis(featMatProjected, extremeness=2., saveto=None):
    """ A function to determine to return a list of outlier indices using the
        Mahalanobis distance. 
        Outlier threshold = std(Mahalanobis distance) * extremeness degree 
        [extreme_values=2, very_extreme_values=3 --> according to 68-95-99.7 rule]
    """
    # NB: Euclidean distance puts more weight than it should on correlated variables
    # Chicken and egg situation, we can’t know they are outliers until we calculate 
    # the stats of the distribution, but the stats of the distribution are skewed by outliers!
    # Mahalanobis gets around this by weighting by robust estimation of covariance matrix
    
    # Fit a Minimum Covariance Determinant (MCD) robust estimator to data 
    robust_cov = MinCovDet().fit(featMatProjected[:,:10]) # Use the first 10 principal components
    
    # Get the Mahalanobis distance
    MahalanobisDist = robust_cov.mahalanobis(featMatProjected[:,:10])
    
    projectedTable = pd.DataFrame(featMatProjected[:,:10],\
                      columns=['PC' + str(n+1) for n in range(10)])

    plt.ioff() if saveto else plt.ion()
    plt.close('all')
    plt.rc('xtick',labelsize=15)
    plt.rc('ytick',labelsize=15)
    fig, ax = plt.subplots(figsize=[10,10])
    ax.set_facecolor('#F7FFFF')
    plt.scatter(np.array(projectedTable['PC1']), 
                np.array(projectedTable['PC2']), 
                c=MahalanobisDist) # colour PCA by Mahalanobis distance
    plt.title('Mahalanobis Distance for Outlier Detection', fontsize=20)
    plt.colorbar()
    
    if saveto:
        saveto.parent.mkdir(exist_ok=True, parents=True)
        suffix = Path(saveto).suffix.strip('.')
        plt.savefig(saveto, format=suffix, dpi=300)
    else:
        plt.show()
        
    k = np.std(MahalanobisDist) * extremeness
    upper_t = np.mean(MahalanobisDist) + k
    outliers = []
    for i in range(len(MahalanobisDist)):
        if (MahalanobisDist[i] >= upper_t):
            outliers.append(i)
    print("Outliers found: %d" % len(outliers))
            
    return np.array(outliers)

def remove_outliers_pca(df, features_to_analyse=None, saveto=None):
    """ Remove outliers from dataset based on Mahalanobis distance metric 
        between points in PCA space. """

    if features_to_analyse:
        data = df[features_to_analyse]
    else:
        data = df
            
    # Normalise the data before PCA
    zscores = data.apply(zscore, axis=0)
    
    # Drop features with NaN values after normalising
    colnames_before = list(zscores.columns)
    zscores.dropna(axis=1, inplace=True)
    colnames_after = list(zscores.columns)
    nan_cols = [col for col in colnames_before if col not in colnames_after]
    if len(nan_cols) > 0:
        print("Dropped %d features with NaN values after normalization:\n%s" %\
              (len(nan_cols), nan_cols))

    print("\nPerforming PCA for outlier removal...")
    
    # Fit the PCA model with the normalised data
    pca = PCA()
    pca.fit(zscores)
    
    # Project data (zscores) onto PCs
    projected = pca.transform(zscores) # A matrix is produced
    # NB: Could also have used pca.fit_transform()

    # Plot summary data from PCA: explained variance (most important features)
    important_feats, fig = pcainfo(pca, zscores, PC=0, n_feats2print=10)        
    
    # Remove outliers: Use Mahalanobis distance to exclude outliers from PCA
    indsOutliers = find_outliers_mahalanobis(projected, saveto=saveto)
    
    # Get outlier indices in original dataframe
    indsOutliers = np.array(data.index[indsOutliers])
    plt.pause(5); plt.close()
    
    # Drop outlier(s)
    print("Dropping %d outliers from analysis" % len(indsOutliers))
    df = df.drop(index=indsOutliers)
        
    return df, indsOutliers

def plot_tSNE(featZ,
              meta,
              group_by,
              var_subset=None,
              saveDir=None,
              perplexities=[10],
              n_components=2,
              sns_colour_palette="tab10"):
    """ t-distributed stochastic neighbour embedding """
    
    assert (meta.index == featZ.index).all()
    assert type(perplexities) == list
    
    if var_subset is not None:
        assert all([strain in meta[group_by].unique() for strain in var_subset])
    else:
        var_subset = list(meta[group_by].unique())    
        
    print("Performing t-distributed stochastic neighbour embedding (t-SNE)")
    for perplex in perplexities:
        # 2-COMPONENT t-SNE
        tSNE_embedded = TSNE(n_components=n_components, 
                             init='random', 
                             random_state=42,\
                             perplexity=perplex, 
                             n_iter=3000).fit_transform(featZ)
        tSNE_df = pd.DataFrame(tSNE_embedded, columns=['tSNE_1', 'tSNE_2']).set_index(featZ.index)
        
        # Plot 2-D tSNE
        plt.ioff() if saveDir else plt.ion()
        plt.close('all')
        plt.rc('xtick',labelsize=12)
        plt.rc('ytick',labelsize=12)
        fig = plt.figure(figsize=[10,10])
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('tSNE Component 1', fontsize=15, labelpad=12)
        ax.set_ylabel('tSNE Component 2', fontsize=15, labelpad=12)
        ax.set_title('2-component tSNE (perplexity={0})'.format(perplex), fontsize=20)
        
        # Create colour palette for plot loop
        palette = itertools.cycle(sns.color_palette(sns_colour_palette, len(var_subset))) # 'gist_rainbow'
        
        for var in var_subset:
            tSNE_var = tSNE_df[meta[group_by]==var]
            sns.scatterplot(x='tSNE_1', y='tSNE_2', data=tSNE_var, color=next(palette), s=100)
        plt.tight_layout(rect=[0.04, 0, 0.84, 0.96])
        ax.legend(var_subset, frameon=False, loc=(1, 0.1), fontsize=15)
        ax.grid()
        
        if saveDir:
            saveDir.mkdir(exist_ok=True, parents=True)
            savePath = Path(saveDir) / 'tSNE_perplex={0}.eps'.format(perplex)
            plt.savefig(savePath, tight_layout=True, dpi=300)
        else:
            plt.show(); plt.pause(2)
        
    return tSNE_df
    
def plot_umap(featZ,
              meta,
              group_by,
              var_subset=None,
              saveDir=None,
              n_neighbours=[10],
              min_dist=0.3,
              sns_colour_palette="tab10"):
    """ Uniform manifold projection """
    
    assert (meta.index == featZ.index).all()
    assert type(n_neighbours) == list
    
    if var_subset is not None:
        assert all([strain in meta[group_by].unique() for strain in var_subset])
    else:
        var_subset = list(meta[group_by].unique())    

    print("Performing uniform manifold projection (UMAP)")
    for n in n_neighbours:
        UMAP_projection = umap.UMAP(n_neighbors=n,
                                    min_dist=min_dist,
                                    metric='correlation').fit_transform(featZ)
        
        UMAP_projection_df = pd.DataFrame(UMAP_projection, 
                                          columns=['UMAP_1', 'UMAP_2']).set_index(featZ.index)
        UMAP_projection_df.name = 'n={}'.format(str(n))
        
        # Plot 2-D UMAP
        plt.close('all')
        sns.set_style('whitegrid')
        plt.rc('xtick',labelsize=12)
        plt.rc('ytick',labelsize=12)
        fig = plt.figure(figsize=[11,10])
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('UMAP Component 1', fontsize=15, labelpad=12)
        ax.set_ylabel('UMAP Component 2', fontsize=15, labelpad=12)
        ax.set_title('2-component UMAP (n_neighbours={0})'.format(n), fontsize=20)
                
        # Create colour palette for plot loop
        palette = itertools.cycle(sns.color_palette(sns_colour_palette, len(var_subset)))
        
        for var in var_subset:
            UMAP_var = UMAP_projection_df[meta[group_by]==var]
            sns.scatterplot(x='UMAP_1', y='UMAP_2', data=UMAP_var, color=next(palette), s=100)
        plt.tight_layout(rect=[0.04, 0, 0.84, 0.96])
        ax.legend(var_subset, frameon=False, loc=(1, 0.1), fontsize=15)
        ax.grid()
        
        if saveDir:
            saveDir.mkdir(exist_ok=True, parents=True)
            savePath = Path(saveDir) / 'UMAP_n_neighbours={0}.eps'.format(n)
            plt.savefig(savePath, tight_layout=True, dpi=300)
        else:
            plt.show(); plt.pause(2)
        
    return UMAP_projection_df

# =============================================================================
# def control_variation(control_feats, control_meta, saveDir, 
#                       features_to_analyse=None, 
#                       variables_to_analyse=["date_yyyymmdd"], 
#                       remove_outliers=False, 
#                       p_value_threshold=0.05, 
#                       PCs_to_keep=10):
#     """ A function written to analyse control variation over time across with respect 
#         to a defined grouping variable (factor), eg. day of experiment, run number, 
#         duration of L1 diapause, camera/rig ID, etc. """
#            
#     # Record non-data columns before dropping feature columns   
#     other_colnames = [col for col in df.columns if col not in features_to_analyse]
#         
#     # Drop columns that contain only zeros
#     colnames_before = list(df.columns)
#     AllZeroFeats = df[features_to_analyse].columns[(df[features_to_analyse] == 0).all()]
#     df = df.drop(columns=AllZeroFeats)
#     colnames_after = list(df.columns)
#     zero_cols = [col for col in colnames_before if col not in colnames_after]
#     if len(zero_cols) > 0:
#         print("Dropped %d features with all-zero summaries:\n%s" % (len(zero_cols), zero_cols))
#     
#     # Record feature column names after dropping zero data
#     features_to_analyse = [feat for feat in df.columns if feat not in other_colnames]
#     
#     # # Remove outliers from the dataset 
#     # if remove_outliers:
#     #     df, indsOutliers = removeOutliersMahalanobis(df, features_to_analyse)
#     #     remove_outliers = False 
#     #     # NB: Ensure Mahalanobis operation to remove outliers is performed only once!
# 
#     # Check for normality in features to analyse in order decide which 
#     # statistical test to use: one-way ANOVA (parametric) or Kruskal-Wallis 
#     # (non-parametric) test
#     TEST = check_normality(df, features_to_analyse, p_value_threshold)
# 
#     # Record name of statistical test used (kruskal/f_oneway)
#     test_name = str(TEST).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]
# 
#     # CONTROL VARIATION: STATS (ANOVAs)
#     # - Does N2 worm behaviour on control vary across experiment days? 
#     #       (worms are larger? Shorter L1 diapuase? Camera focus/FOV adjusted? Skewed by non-worm tracked objects?
#     #       Did not record time when worms were refed! Could be this. If so, worms will be bigger across all foods on that day) 
#     # - Perform ANOVA to see if features vary across imaging days for control
#     # - Perform Tukey HSD post-hoc analyses for pairwise differences between imaging days
#     # - Highlight outlier imaging days and investigate reasons why
#     # - Save list of top significant features for outlier days - are they size-related features?
#     for grouping_variable in variables_to_analyse:
#         print("\nTESTING: %s\n" % grouping_variable)
#         
#         if not len(df[grouping_variable].unique()) > 1:
#             print("Need at least two groups for stats to investigate %s" % grouping_variable)
#         else:
#             print("Performing %s tests for '%s'" % (test_name, grouping_variable))            
#     
#             test_results_df, sigfeats_out = \
#                 topfeats_ANOVA_by_group(df, 
#                                         grouping_variable, 
#                                         features_to_analyse,
#                                         TEST,
#                                         p_value_threshold)
#             
#             # Ensure directory exists to save results
#             Path(outDir).mkdir(exist_ok=True, parents=True)
#             
#             # Define outpaths
#             froot = 'control_variation_in_' + grouping_variable + '_' + test_name
#             stats_outpath = outDir / (froot + "_results.csv")
#             sigfeats_outpath = outDir / (froot + "_significant_features.csv")
#                                    
#             # Save test statistics + significant features list to file
#             test_results_df.to_csv(stats_outpath)
#             sigfeats_out.to_csv(sigfeats_outpath, header=False)
# 
#             # Box plots
#             plotDir = outDir / "Plots"
#             topfeats_boxplots_by_group(df, 
#                                        test_results_df, 
#                                        grouping_variable,
#                                        plot_save_dir=plotDir, #save to plotDir
#                                        p_value_threshold=p_value_threshold)
#                         
#             # PCA (coloured by grouping variable, eg. experiment date)
#             df = doPCA(df, 
#                        grouping_variable, 
#                        features_to_analyse,
#                        plot_save_dir = plotDir,
#                        PCs_to_keep = PCs_to_keep)
# =============================================================================
            
#%% Plot PCA - All bacterial strains (food)

# topNstrains = 5

# if is_normal:
#     test_name = 'ttest_ind'
# else:
#     test_name = 'ranksumtest'
    
# stats_inpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Stats', test_name + '_results.csv')
# test_pvalues_df = pd.read_csv(stats_inpath, index_col=0)
# print("Loaded %s results." % test_name)

# propfeatssigdiff = ((test_pvalues_corrected_df < p_value_threshold).sum(axis=1)/len(featurelist))*100
# propfeatssigdiff = propfeatssigdiff.sort_values(ascending=False)

# topStrains = list(propfeatssigdiff[:topNstrains].index)
# topStrains_projected_df = projected_df[projected_df['food_type'].str.upper().isin(topStrains)]
# topStrains.insert(0, CONTROL_STRAIN)

# otherStrains = [strain for strain in BACTERIAL_STRAINS if strain not in topStrains]
# otherStrains_projected_df = projected_df[projected_df['food_type'].str.upper().isin(otherStrains)]

# # Create colour palette for plot loop
# #colour_dict_other = {strain: "r" if strain == "OP50" else "darkgray" for strain in otherStrains}
# colour_dict_other = {strain: "darkgray" for strain in otherStrains}
# topcols = sns.color_palette("Paired", len(topStrains))
# colour_dict_top = {strain: topcols[i] for i, strain in enumerate(topStrains)}
# #colour_dict.update(colour_dict2)
# #palette = itertools.cycle(sns.color_palette("gist_rainbow", len(BACTERIAL_STRAINS)))

# plt.close()
# plotpath_2d = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'All', 'PCA', 'PCA_2PCs_byStrain.eps')
# title = None #"""2-Component PCA (Top256 features)"""
# plt.rc('xtick',labelsize=15)
# plt.rc('ytick',labelsize=15)
# sns.set_style("whitegrid")
# fig, ax = plt.subplots(figsize=[10,10])

# palette_other = itertools.cycle(list(colour_dict_other.values()))
# for strain in otherStrains:
#     strain_projected_df = projected_df[projected_df['food_type'].str.upper()==strain]
#     sns.scatterplot(strain_projected_df['PC1'], strain_projected_df['PC2'],\
#                     color=next(palette_other), s=50, alpha=0.65, linewidth=0)

# palette_top = itertools.cycle(list(colour_dict_top.values()))
# for strain in topStrains:
#     strain_projected_df = projected_df[projected_df['food_type'].str.upper()==strain]
#     sns.scatterplot(strain_projected_df['PC1'], strain_projected_df['PC2'],\
#                     color=next(palette_top), s=70, edgecolor='k') # marker="^"
    
# ax.set_xlabel('Principal Component 1', fontsize=20, labelpad=12)
# ax.set_ylabel('Principal Component 2', fontsize=20, labelpad=12)
# if title:
#     ax.set_title(title, fontsize=20)

# # Add plot legend
# patches = []
# for l, key in enumerate(colour_dict_top.keys()):
#     patch = mpatches.Patch(color=colour_dict_top[key], label=key)
#     patches.append(patch)
# plt.legend(handles=patches, labels=colour_dict_top.keys(), frameon=False, fontsize=12)
# ax.grid()

# # Save PCA scatterplot of first 2 PCs
# savefig(plotpath_2d, tight_layout=False, tellme=True, saveFormat='png') # rasterized=True
# plt.show(); plt.pause(2)

# =============================================================================
# def remove_outliers_pca(projected_df, feat_df):
#     """ Remove outliers in dataset for PCA using Mahalanobis distance metric """
#     
#     # Remove outliers: Use Mahalanobis distance to exclude outliers from PCA
#     
#     indsOutliers = find_outliers_mahalanobis(projected, showplot=True)
#     plt.pause(5); plt.close()
#     
#     # Drop outlier observation(s)
#     print("Dropping %d outliers from analysis" % len(indsOutliers))
#     indsOutliers = results_feats.index[indsOutliers]
#     results_feats = results_feats.drop(index=indsOutliers)
#     fullresults = fullresults.drop(index=indsOutliers)
#     
#     # Re-normalise data
#     zscores = results_feats.apply(zscore, axis=0)
#     
#     # Drop features with NaN values after normalising
#     zscores.dropna(axis=1, inplace=True)
#     print("Dropped %d features after normalisation (NaN)" % (len(results_feats.columns)-len(zscores.columns)))
#     
#     # Use Top256 features
#     print("Using Top256 feature list for dimensionality reduction...")
#     top256featcols = [feat for feat in zscores.columns if feat in featurelist]
#     zscores = zscores[top256featcols]
#     
#     # Project data on PCA axes again
#     pca = PCA()
#     pca.fit(zscores)
#     projected = pca.transform(zscores) # project data (zscores) onto PCs
#     important_feats, fig = pcainfo(pca=pca, zscores=zscores, PC=1, n_feats2print=10)
#     plt.pause(5); plt.close()
#     
#     # Store the results for first few PCs
#     projected_df = pd.DataFrame(projected[:,:10],\
#                                   columns=['PC' + str(n+1) for n in range(10)])
#     projected_df.set_index(fullresults.index, inplace=True) # Do not lose index position
#     projected_df = pd.concat([fullresults[metadata_colnames], projected_df], axis=1)
#     
# 
#     return projected_df, feat_df
# =============================================================================
