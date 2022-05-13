#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test multiple test correction method across features/samples

@author: sm5911
@date: 11/05/2022

"""

import pandas as pd
from pathlib import Path
from statistical_testing.stats_helper import do_stats
from tierpsytools.analysis.statistical_tests import _multitest_correct


path_example_meta = '/Users/sm5911/Documents/Keio_ubiC/metadata.csv'
path_example_feat = '/Users/sm5911/Documents/Keio_ubiC/features.csv'

metadata = pd.read_csv(path_example_meta, dtype={'comments':str, 'source_plate_id':str})
features = pd.read_csv(path_example_feat, index_col=None)

save_dir = Path(path_example_meta).parent / 'multiple_testing_checks'
save_dir.mkdir(exist_ok=True, parents=True)

strain_list = [s for s in list(metadata['food_type'].unique()) if s != 'BW']


### multitest correct within loop (for features) and afterwards (for strains)
print("Testing multiple test correction within loop...")
pvals1_list = []
for strain in strain_list:
    # subset for each strain + control in turn
    test_meta = metadata.query("food_type=='BW' or food_type==@strain")
    
    test_results = do_stats(metadata=test_meta, 
                            features=features.reindex(test_meta.index),
                            group_by='food_type',
                            control='BW',
                            feat=features.columns.tolist(),
                            save_dir=None,
                            pvalue_threshold=0.05,
                            # correct for the fact that we conducted separate tests for each feature
                            fdr_method='fdr_by',
                            ttest_if_nonsig=True,
                            verbose=False)
    
    pvals1 = test_results[[c for c in test_results.columns if 'pvals' in c]]
    pvals1_list.append(pvals1)
    
pvals1 = pd.concat(pvals1_list, axis=1)

# save p-values (corrected only for features only, within loop for each strain)
save_path = save_dir / 'pvals_corrected_features_only.csv'
pvals1.to_csv(save_path, header=True, index=True)
    
# now correct for the fact that we conducted separate tests for each strain
for feat in pvals1.index.tolist():
    # correct for the multiple strains compared for each feature in turn
    feat_pvals = pvals1.loc[feat, :]
    feat_reject, feat_pvals = _multitest_correct(pvals=feat_pvals, 
                                                 multitest_method='fdr_by', 
                                                 fdr=0.05)
    # update pvals table with corrected p-value
    pvals1.loc[feat, :] = feat_pvals

# save p-values (corrected for each feature within loop, and for each strain after the loop)
save_path = save_dir / 'pvals_corrected_features_in_loop_strains_after.csv'
pvals1.to_csv(save_path, header=True, index=True)

### multitest correct after loop (in one go)
print("\nTesting multiple test correction after loop...")
pvals2_list = []
for strain in strain_list:
    # subset for each strain + control in turn
    test_meta = metadata.query("food_type=='BW' or food_type==@strain")
    
    test_results = do_stats(metadata=test_meta, 
                            features=features.reindex(test_meta.index),
                            group_by='food_type',
                            control='BW',
                            feat=features.columns.tolist(),
                            save_dir=None,
                            pvalue_threshold=0.05,
                            # correct for the fact that we conducted separate tests for each feature
                            fdr_method=None,
                            ttest_if_nonsig=True,
                            verbose=False)
    
    pvals2 = test_results[[c for c in test_results.columns if 'pvals' in c]]
    pvals2_list.append(pvals2)
    
pvals2 = pd.concat(pvals2_list, axis=1)

# save p-values (uncorrected - no multiple test correction)
save_path = save_dir / 'pvals_uncorrected.csv'
pvals2.to_csv(save_path, header=True, index=True)

reject, pvals2 = _multitest_correct(pvals=pvals2, 
                                    multitest_method='fdr_by', 
                                    fdr=0.05)

# save p-values (corrected after loop for both features and strains tested)
save_path = save_dir / 'pvals_corrected_after_loop.csv'
pvals2.to_csv(save_path, header=True, index=True)
    
print("Done!")

# CONCLUSION: They are different! I owe Andre a beer :/

