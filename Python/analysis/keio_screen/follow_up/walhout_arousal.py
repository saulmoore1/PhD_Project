#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-referencing Walhout 2019 list of 244 hits with my initial Keio screen results to look for 
arousal phenotypes in any of these strains, which have been shown to slow worm development 
(through low iron). This developmental delay can be rescued by iron or antioxidant supplementation.
 
@author: sm5911
@date: 27/02/2022

"""

#%% Imports

import pandas as pd
from tqdm import tqdm
from pathlib import Path

from visualisation.plotting_helper import errorbar_sigfeats
from write_data.write import write_list_to_file
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from visualisation.plotting_helper import sig_asterix

#%% Globals

PROJECT_DIR = "/Users/sm5911/Documents/Keio_Screen"

WALHOUT_SI_PATH = "/Users/sm5911/Documents/Keio_Screen/Walhout_2019_arousal_crossref/Walhout_2019_SI_Table1.xlsx"

FEATURE_LIST = ['speed_50th']

STIMULUS_LIST = ['bluelight']

#%% Functions

def load_walhout_244(supplementary_path=WALHOUT_SI_PATH):
    """ Function to load Walhout 2019 SI Table 1 and extract the full list of 244 strains that slow
        C. elegans development 
    """
    xl = pd.ExcelFile(supplementary_path)
    SI_Table1 = xl.parse(sheet_name=xl.sheet_names[0], header=0, index_col=None)
    walhout_244_gene_list = list(sorted(SI_Table1['Keio Gene Name'].unique()))
    
    return walhout_244_gene_list

def stats(metadata,
          features,
          group_by='gene_name',
          control='wild_type',
          feature_set=['speed_50th_bluelight'],
          pvalue_threshold=0.05,
          fdr_method='fdr_bh'):
    
    # check case-sensitivity
    assert len(metadata[group_by].unique()) == len(metadata[group_by].str.upper().unique())
    
    if feature_set is not None:
        feature_set = [feature_set] if isinstance(feature_set, str) else feature_set
        assert isinstance(feature_set, list)
        assert(all(f in features.columns for f in feature_set))
    else:
        feature_set = features.columns.tolist()
        
    features = features[feature_set].reindex(metadata.index)

    # print mean sample size
    sample_size = metadata.groupby(group_by).count()
    print("Mean sample size of %s: %d" % (group_by, int(sample_size[sample_size.columns[-1]].mean())))

    n = len(metadata[group_by].unique())
        
    fset = []
    if n > 2:
   
        # Perform ANOVA - is there variation among strains at each window?
        stats, pvals, reject = univariate_tests(X=features, 
                                                y=metadata[group_by], 
                                                control=control, 
                                                test='ANOVA',
                                                comparison_type='multiclass',
                                                multitest_correction=fdr_method,
                                                alpha=pvalue_threshold,
                                                n_permutation_test=None)

        # get effect sizes
        effect_sizes = get_effect_sizes(X=features,
                                        y=metadata[group_by],
                                        control=control,
                                        effect_type=None,
                                        linked_test='ANOVA')

        # compile results
        test_results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
        test_results.columns = ['stats','effect_size','pvals','reject']     
        test_results['significance'] = sig_asterix(test_results['pvals'])
        test_results = test_results.sort_values(by=['pvals'], ascending=True) # rank by p-value

        # use reject mask to find significant feature set
        fset = pvals.loc[reject['ANOVA']].sort_values(by='ANOVA', ascending=True).index.to_list()

        if len(fset) > 0:
            print("%d significant features found by ANOVA by '%s' (P<%.2f, %s)" %\
                  (len(fset), group_by, pvalue_threshold, fdr_method))
             
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
        
    nsig = sum(reject_t.sum(axis=1) > 0)
    print("%d significant features between any %s vs %s (t-test, P<%.2f, %s)" %\
          (nsig, group_by, control, pvalue_threshold, fdr_method))

    return ttest_results

def main():
    
    # load Walhout 2019 gene list
    walhout_244 = load_walhout_244()
    walhout_244.insert(0, 'wild_type')
    
    save_dir = Path(WALHOUT_SI_PATH).parent

    # load my Keio screen results
    metadata = pd.read_csv(Path(PROJECT_DIR) / "metadata.csv", dtype={"comments":str})
    features = pd.read_csv(Path(PROJECT_DIR) / "features.csv")
        
    fset = [f + '_' + s for f in FEATURE_LIST for s in STIMULUS_LIST]
    assert all(f in features for f in fset)
    
    # filter my results for Walhout low-iron (slow development) genes
    meta_walhout = metadata[metadata['gene_name'].isin(walhout_244)]
    feat_walhout = features.reindex(meta_walhout.index)

    # errorbar plots for Walhout genes only
    print("\nMaking errorbar plots")
    errorbar_sigfeats(feat_walhout, meta_walhout, 
                      group_by='gene_name', 
                      fset=fset, 
                      control='wild_type', 
                      rank_by='mean',
                      max_feats2plt=None, 
                      figsize=(30,6), 
                      fontsize=5,
                      ms=8,
                      elinewidth=1.5,
                      fmt='.',
                      tight_layout=[0.01,0.01,0.99,0.99],
                      saveDir=save_dir / 'errorbar' / 'walhout_only')
    
    # errorbar plots for all genes (highlighting Walhout genes)
    errorbar_sigfeats(features, metadata, 
                      group_by='gene_name',
                      fset=fset,
                      control='wild_type',
                      highlight_subset=[s for s in walhout_244 if s != 'wild_type'],
                      rank_by='mean',
                      figsize=(30,20),
                      fontsize=3,
                      ms=2,
                      elinewidth=1.5,
                      fmt='.',
                      tight_layout=[0.01,0.01,0.99,0.99],
                      saveDir=save_dir / 'errorbar' / 'walhout_labelled')
    
    # save Walhout ranking
    grouped = metadata[['gene_name']].join(features).groupby('gene_name')
    
    median_strain = grouped[fset].median()
                
    # save ranking
    for f, feat in enumerate(tqdm(fset)):
        order = median_strain[feat].sort_values(ascending=True).reset_index(drop=False)
        walhout_ranking = order[order['gene_name'].isin(walhout_244)].reset_index(drop=False)
        walhout_ranking = walhout_ranking.rename({'index':'rank'}, axis='columns')
        save_path = save_dir / 'ranking' / '{}.csv'.format(feat)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        walhout_ranking.to_csv(save_path, header=True, index=False)
        
        # save full Keio ranking
        order = order.reset_index(drop=False).rename({'index':'rank'}, axis='columns')
        order.to_csv(str(save_path).replace('.csv','_full.csv'), header=True, index=False)
 
    # errorbar plots for all genes (highlighting fep genes)
    fep_list = ['fepB','fepC','fepD','fepG']
    errorbar_sigfeats(features, metadata, 
                      group_by='gene_name',
                      fset=fset,
                      control='wild_type',
                      highlight_subset=[s for s in list(metadata.gene_name.unique()) if s in fep_list],
                      rank_by='mean',
                      figsize=(150,6),
                      fontsize=3,
                      ms=6,
                      elinewidth=1.2,
                      fmt='.',
                      tight_layout=[0.01,0.01,0.99,0.99],
                      saveDir=save_dir / 'errorbar' / 'fep_labelled')
    
    # read initial screen t-test results
    ttest_results = stats(metadata, 
                          features, 
                          group_by='gene_name',
                          control='wild_type',
                          feature_set=['speed_50th_bluelight'],
                          pvalue_threshold=0.05,
                          fdr_method='fdr_bh')
    pvals = ttest_results[[c for c in ttest_results.columns if 'pvals_' in c]]
    pvals.columns = [c.replace('pvals_','') for c in pvals.columns]
    speed_pvals = pvals[pvals < 0.05]
        
#%% Main

if __name__ == "__main__":
    main()
    