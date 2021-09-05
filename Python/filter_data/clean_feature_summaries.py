#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean feature summary and associated metadata to remove:
    Row entries:
    - For 
@author: sm5911
@date: 9/2/21

"""

def clean_summary_results(features, 
                          metadata, 
                          feature_columns=None, 
                          nan_threshold_row=0.8, 
                          nan_threshold_col=0.05,
                          max_value_cap=1e15,
                          imputeNaN=True,
                          min_nskel_per_video=None,
                          min_nskel_sum=None,
                          drop_size_related_feats=False,
                          norm_feats_only=False,
                          percentile_to_use=None):
    """ Clean features summary results
        - Drop bad wells from WellAnnotator annotations file
        - Drop samples with >nan_threshold_row proportion of NaN features
        - Drop features with >nan_threshold_col proportion of NaNs
        - Drop features with zero standard deviation
        - Drop features that are ventrally signed
        - Drop features that are path curvature related
    
        Parameters
        ----------
        feature_columns : list, None
            List of feature column names to clean
        nan_threshold_row : float
            Drop samples with too many NaN/Inf values across features (> nan_threshold_row)
        nan_threshold_col : float
            Drop features with too many NaN/Inf values across samples (> nan_threshold_col)
        max_value_cap : int, float
            Maximum value for feature summary results (features will be capped at this value)
        imputeNaN : bool
            Impute remaining NaN values with global mean value for each feature
        filter_based_on_skeletons : bool
            Drop samples where Tierpsy did not find many worm skeletons throughout the video
        drop_size_related  : bool
            Drop features that are size-related
        norm_feats_only : bool
            Drop faetures that are not length-normalised (size-invariant)
        percentile_to_use : str, None
            Use only given percentile of feature summary distribution
        
        Returns
        -------
        features, metadata
        
    """
    from tierpsytools.preprocessing.filter_data import (drop_bad_wells,
                                                        filter_nan_inf, 
                                                        feat_filter_std, 
                                                        drop_ventrally_signed,
                                                        cap_feat_values,
                                                        filter_n_skeletons)

    assert set(features.index) == set(metadata.index)

    if feature_columns is not None:
        assert all([feat in features.columns for feat in feature_columns])
        features = features[feature_columns]
    else:
        feature_columns = features.columns
               
    # Drop bad well data
    features, metadata = drop_bad_wells(features, 
                                        metadata, 
                                        bad_well_cols=['is_bad_well'], 
                                        verbose=False)
    assert not any(features.sum(axis=1) == 0) # ensure no missing row data

    # Drop rows based on percentage of NaN values across features for each row
    # NB: axis=1 will sum the NaNs across all the columns for each row
    features = filter_nan_inf(features, threshold=nan_threshold_row, axis=1, verbose=True)
    metadata = metadata.reindex(features.index)
    
    # Drop feature columns with too many NaN values
    # NB: to remove features with NaNs across all results, eg. food_edge related features which are not calculated
    features = filter_nan_inf(features, threshold=nan_threshold_col, axis=0, verbose=False)
    nan_cols = [col for col in feature_columns if col not in features.columns]
    if len(nan_cols) > 0:
        print("Dropped %d features with >%.1f%% NaNs" % (len(nan_cols), nan_threshold_col*100))
    
    # Drop feature columns with zero standard deviation
    feature_columns = features.columns
    features = feat_filter_std(features, threshold=0.0)
    zero_std_feats = [col for col in feature_columns if col not in features.columns]
    if len(zero_std_feats) > 0:
        print("Dropped %d features with zero standard deviation" % len(zero_std_feats))
    
    # Drop ventrally-signed features
    # In general, for the curvature and angular velocity features we should only 
    # use the 'abs' versions, because the sign is assigned based on whether the worm 
    # is on its left or right side and this is not known for the multiworm tracking data
    feature_columns = features.columns
    features = drop_ventrally_signed(features)
    ventrally_signed_feats = [f for f in feature_columns if f not in features.columns]
    if len(ventrally_signed_feats) > 0:
        print("Dropped %d features that are ventrally signed" % len(ventrally_signed_feats))
    
    # Cap feature values to max value for given feature
    if max_value_cap:
        features = cap_feat_values(features, cutoff=max_value_cap)
    
    # Remove 'path_curvature' features
    path_curvature_feats = [f for f in features.columns if 'path_curvature' in f]
    if len(path_curvature_feats) > 0:
        features = features.drop(columns=path_curvature_feats)
        print("Dropped %d features that are derived from path curvature"\
              % len(path_curvature_feats))
    
    # Drop rows from feature summaries where any videos has less than min_nskel_per_video
    if min_nskel_per_video is not None:
        features, metadata = filter_n_skeletons(features, metadata, 
                                                min_nskel_per_video=min_nskel_per_video)
        
    # Drop rows from feature summaries where the sum number of skeletons across 
    # prestim/bluelight/poststim videos is less than min_nskel_sum
    if min_nskel_sum is not None:
        features, metadata = filter_n_skeletons(features, metadata, 
                                                min_nskel_sum=min_nskel_sum)

    # Impute remaining NaN values (using global mean feature value for each strain)
    if imputeNaN:
        n_nans = features.isna().sum(axis=0).sum()
        if n_nans > 0:
            print("Imputing %d missing values (%.2f%% data) " % (n_nans, 
                                                                 n_nans/features.count().sum()*100)
                  + "using global mean value for each feature..") 
            features = features.fillna(features.mean(axis=0))
    
    # Drop size-related features
    if drop_size_related_feats:
        size_feat_keys = ['blob','box','width','length','area']
        size_features = []
        for feature in list(features.columns):
            for key in size_feat_keys:
                if key in feature:
                    size_features.append(feature)
        feature_columns = [f for f in features.columns if f not in size_features]
        features = features[feature_columns]
        print("Dropped %d features that are size-related" % len(size_features))
        
    # Use '_norm' features only
    if norm_feats_only:
        not_norm = [f for f in features.columns if not '_norm' in f]
        if len(not_norm) > 0:
            features = features.drop(columns=not_norm)
            print("Dropped %d features that are not '_norm' features" % len(not_norm))
            
    # Use '_50th' percentile data only
    if percentile_to_use is not None:
        assert type(percentile_to_use) == str
        not_perc = [f for f in features.columns if not percentile_to_use in f]
        if len(not_perc) > 0:
            features = features.drop(columns=not_perc)
            print("Dropped %d features that are not %s features" % (len(not_perc), 
                                                                    percentile_to_use))

    return features, metadata

def subset_results(features, metadata, column, groups, omit=False, verbose=True):
    """ Subset features and metadata for groups in a given column 
    
        Parameters
        ----------
        features, metadata : pd.DataFrame
            Separate dataframes for data and metadata information
        column : str
            A column name belonging to a column in metadata
        groups : list
            List of groups that you would like to subset
        omit_groups : bool
            If True, groups are omitted from dataframe, instead of extracted
    """
    
    assert set(features.index) == set(metadata.index)
    assert column in metadata.columns
    assert isinstance(groups, list)
    
    if len(groups) > 0:
        assert all([i in metadata[column].unique() for i in groups])
    
        if omit:
            if verbose:
                print("Omitting %d '%s'" % (len(groups), column))
            subset_metadata = metadata[~metadata[column].isin(groups)]
        else:
            if verbose:
                print("Subsetting for %d '%s'" % (len(groups), column))
            subset_metadata = metadata[metadata[column].isin(groups)]
            
        subset_features = features.reindex(subset_metadata.index)
        
        return subset_features, subset_metadata
    else:
        print("No groups to subset")
        return features, metadata
    