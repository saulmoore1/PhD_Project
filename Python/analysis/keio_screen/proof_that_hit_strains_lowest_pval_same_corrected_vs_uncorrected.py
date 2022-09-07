#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 19:04:47 2022

@author: sm5911
"""

import pandas as pd

df = pd.read_csv("/Users/sm5911/Documents/Keio_Screen/Top16/gene_name/Stats/fdr_bh/t-test_results_uncorrected.csv", index_col=0)
df = df[[c for c in df.columns if 'pval' in c]]
ranked_lowest_pval = df.min(axis=0).sort_values(ascending=True)
hit_strain_list = ranked_lowest_pval.index[:100]
hit_strain_list = [s.split('pvals_')[-1] for s in hit_strain_list]
unc = set(hit_strain_list)


df_by = pd.read_csv("/Users/sm5911/Documents/Keio_Screen/Top16/gene_name/Stats/fdr_bh/t-test_results.csv", index_col=0)
df_by = df_by[[c for c in df_by.columns if 'pval' in c]]
ranked_lowest_pval_by = df_by.min(axis=0).sort_values(ascending=True)
hit_strain_list_by = ranked_lowest_pval_by.index[:100]
hit_strain_list_by = [s.split('pvals_')[-1] for s in hit_strain_list_by]
by = set(hit_strain_list_by) 

by - unc
unc - by
len(by.union(unc))
len(by.intersection(unc))

curated_list_path = '/Users/sm5911/Documents/Keio_Screen/Top16/59_selected_strains_from_initial_keio_top100_lowest_pval_tierpsy16_fdr_bh.txt'

chosen_strains = []
with open(curated_list_path, 'r') as fid:
    for line in fid:
        chosen_strains.append(line.strip('\n'))
        
chosen = set(chosen_strains)

len(chosen)
len(unc.intersection(chosen))
len(by.intersection(chosen))

selected_hits_from_by = by.intersection(chosen)
added_strains_by = chosen - by

selected_hits_from_unc = unc.intersection(chosen)
added_strains_unc = chosen - unc

added_strains_by.difference(added_strains_unc)    

