#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean feature summary and associated metadata to remove:
    Row entries:
    - For 
@author: sm5911
@date: 9/2/21

"""

def clean_features_summaries(features, 
                             metadata, 
                             feature_columns=None, 
                             imputeNaN=True,
                             nan_threshold=0.2, 
                             max_value_cap=1e15,
                             mean_feats_only=True,
                             drop_size_related_feats=False,
                             norm_feats_only=False):
    """ Clean features summary results:
        - Drop features with too many NaN/Inf values (> nan_threshold)
        - Impute remaining NaN values with global mean value for each feature
        - Drop features with zero standard deviation
        - Drop features that are ventrally signed
        - Drop features that are path curvature related
        - Drop features that are size-related (OPTIONAL)
        - Drop faetures that are not '_norm' (OPTIONAL)
        - Drop features that are not '_50th' percentiles (OPTIONAL)
    """

    from tierpsytools.preprocessing.filter_data import (filter_nan_inf, 
                                                        feat_filter_std, 
                                                        drop_ventrally_signed,
                                                        cap_feat_values,
                                                        drop_bad_wells)

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

    # Drop feature columns with too many NaN values
    features = filter_nan_inf(features, threshold=nan_threshold, axis=0)
    nan_cols = [col for col in feature_columns if col not in features.columns]
    if len(nan_cols) > 0:
        print("Dropped %d features with >%.1f%% NaNs" % (len(nan_cols), nan_threshold*100))

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
            print("Imputing %d missing values (%.2f%% data) " % (n_nans, 
                                                                 n_nans/features.count().sum()*100)
                  + "using global mean value for each feature..") 
            features = features.fillna(features.mean(axis=0))

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
        feature_columns = features.columns
        not_norm = [f for f in feature_columns if not '_norm' in f]
        if len(not_norm) > 0:
            features = features.drop(columns=not_norm)
            print("Dropped %d features that are not '_norm' features" % len(not_norm))
            
    # Use '_50th' perrcentile data only
    if mean_feats_only:
        not_50th = [f for f in features.columns if not '_50th' in f]
        if len(not_50th) > 0:
            features = features.drop(columns=not_50th)
            print("Dropped %d features that are not '_50th' features" % len(not_50th))
        
    return features, metadata
