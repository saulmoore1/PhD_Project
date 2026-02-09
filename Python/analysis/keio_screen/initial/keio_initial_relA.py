#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RelA
"""

#%% IMPORTS

import pandas as pd
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import zscore
from tierpsytools.preprocessing.filter_data import select_feat_set
from clustering.hierarchical_clustering import plot_clustermap
from matplotlib import transforms
from visualisation.plotting_helper import sig_asterix
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes

#%% GLOBALS

PROJ_DIR = "/Users/sm5911/Documents/PhD_DLBG/Other/Keio_Initial_Screen/heatmap"
SAVE_DIR = "/Users/sm5911/Documents/PhD_DLBG/Other/Keio_Initial_Screen/RelA"

# analysis parameters
FEATURE_SET = "tierpsy_256"
COLLAPSE_CONTROL = True
FDR_METHOD = "fdr_bh"
DATES = ["20210406", "20210413", "20210427", "20210504", "20210511"] # "20210420"

# data cleaning parameters used
ALIGN_BLUELIGHT = True     # append stimulus type to feature names
IMPUTE_NANS = True         # fill missing feature data with global mean for that feature
MAX_VALUE_CAP = 1e15       # maximum cap on feature value (drop extreme erroneus values)
NAN_THRESHOLD_ROW = 0.8    # drop samples with missing values for >80% of features
NAN_THRESHOLD_COL = 0.05   # drop features where >5% samples have missing values for that feature
MIN_NSKEL_PER_VIDEO = None # do not drop samples with less than n skeletons per video
MIN_NSKEL_SUM = 6000       # drop samples with <6000 skeletons in total across video frames

#%% FUNCTIONS
        
def average_control_data(metadata, features, 
                         control='wild_type', 
                         grouping_var='gene_name', 
                         average_by='date_yyyymmdd'):
    
    """ Average data for control on each experiment day/plate/etc to yield a single 
        mean datapoint for the control. This reduces the control sample size to 
        equal the test strain sample size, for t-test comparison. Information 
        for the first well in the control sample for each 'average_by' group
        is used as the accompanying metadata for mean feature results. 
        
        Input
        -----
        features, metadata : pd.DataFrame
            Feature summary results and metadata dataframe with multiple 
            entries per 'average_by' e.g. day/imaging plate/etc
            
        Returns
        -------
        features, metadata : pd.DataFrame
            Feature summary results and metadata with control data averaged 
            (single sample per 'average_by' e.g. day/imaging plate/etc)
    """
        
    # Subset results for control data
    control_metadata = metadata[metadata[grouping_var]==control]
    control_features = features.reindex(control_metadata.index)

    # calculate mean of control for each plate (collapses data to a single 
    # datapoint for strain comparison)
    mean_control = control_metadata[[grouping_var, average_by]].join(
        control_features).groupby(by=[grouping_var, average_by]).mean().reset_index()
    
    # Append remaining control metadata column info (with first well data for each date)
    remaining_cols = [c for c in control_metadata.columns.to_list() if 
                      c not in [grouping_var, average_by]]
    
    mean_control_row_data = []
    for i in mean_control.index:
        # for each 'average_by' group:
        group = mean_control.loc[i, average_by]
        # get the metadata for the first well of the group
        control_group_meta = control_metadata.loc[control_metadata[average_by] == group]
        first_well_meta = control_group_meta.loc[control_group_meta.index[0], remaining_cols]
        # get the mean feature values for the group
        group_mean = mean_control.loc[mean_control[average_by] == group].squeeze(axis=0)
        # concatenate metadata and mean feature values for group + append to list
        group_mean_and_meta = pd.concat([group_mean,first_well_meta])
        mean_control_row_data.append(group_mean_and_meta)
    
    # create dataframe of all control plate mean data + split into metadata and features
    control_mean = pd.DataFrame.from_records(mean_control_row_data)
    control_metadata = control_mean[control_metadata.columns.to_list()]
    control_features = control_mean[control_features.columns.to_list()]
    
    # replace control data with control plate means
    features = pd.concat([features.loc[metadata[grouping_var] != control, :], 
                          control_features], axis=0).reset_index(drop=True)        
    metadata = pd.concat([metadata.loc[metadata[grouping_var] != control, :], 
                          control_metadata.loc[:, metadata.columns.to_list()]], 
                          axis=0).reset_index(drop=True)
    
    assert all(metadata.index == features.index)

    return metadata, features

def stats(metadata,
          features,
          group_by='treatment',
          control='BW',
          feat='speed_50th',
          save_dir=None,
          pvalue_threshold=0.05,
          fdr_method='fdr_bh'):
        
    assert all(metadata.index == features.index)
    if feat is not None:
        features = features[[feat]]
    
    # check case-sensitivity
    assert len(metadata[group_by].unique()) == len(metadata[group_by].str.upper().unique())

    # print mean sample size
    sample_size = metadata.groupby(group_by).count()
    print("Mean sample size per %s: %d" % (group_by, int(sample_size.max(axis=1).mean())))

    n = len(metadata[group_by].unique())
    if n > 2:
   
        # Perform ANOVA - is there variation among strains?
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

        # compile ANOVA results
        anova_results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
        anova_results.columns = ['stats','effect_size','pvals','reject']     
        anova_results['significance'] = sig_asterix(anova_results['pvals'])
        anova_results = anova_results.sort_values(by=['pvals'], ascending=True)

        if save_dir is not None:
            anova_path = Path(save_dir) / 'ANOVA_results.csv'
            anova_path.parent.mkdir(parents=True, exist_ok=True)
            anova_results.to_csv(anova_path, header=True, index=True)

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

    # save results
    if save_dir is not None:
        ttest_path = Path(save_dir) / 't-test_results.csv'
        ttest_path.parent.mkdir(exist_ok=True, parents=True)
        ttest_results.to_csv(ttest_path, header=True, index=True)
    
    return ttest_results

def main():
        
    metadata_path_local = Path(PROJ_DIR) / 'metadata.csv'
    features_path_local = Path(PROJ_DIR) / 'features.csv'

    metadata = pd.read_csv(metadata_path_local, header=0, index_col=None, 
                           dtype={'comments':str, 'date_yyyymmdd':str})
    features = pd.read_csv(features_path_local, header=0, index_col=None)

    assert not features.isna().sum(axis=1).any()
    assert not (features.std(axis=1) == 0).any()
    assert set(features.index) == set(metadata.index)    
    assert (len(metadata['gene_name'].unique()) ==
            len(metadata['gene_name'].str.upper().unique()))
    
    # subset for selected dates
    metadata = metadata[metadata["date_yyyymmdd"].isin(DATES)]
    features = features.reindex(metadata.index)    
    
    if COLLAPSE_CONTROL:
        print("\nCollapsing control data (mean of each day)")
        metadata, features = average_control_data(metadata,
                                                  features,
                                                  control='wild_type', 
                                                  grouping_var='gene_name', 
                                                  average_by='date_yyyymmdd')
    
    strain_list = ["wild_type","relA"]
    metadata = metadata[metadata["gene_name"].isin(strain_list)]
    features = features.reindex(metadata.index)
    
    # save raw feature data for control and relA to file
    raw_data = metadata[['gene_name','date_yyyymmdd']].join(features)
    save_path = Path(SAVE_DIR) / "raw_data.csv"
    Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)
    raw_data.to_csv(save_path, header=True, index=False)

    features_subset = select_feat_set(features, 
                                      tierpsy_set_name=FEATURE_SET, 
                                      append_bluelight=True)
    
    # heatmap (averaged control per day, tierpsy256)
    featZ = features_subset.apply(zscore, axis=0)
    figsize = [30,15] if FEATURE_SET == "tierpsy_16" else [60,15]

    clustermap_path = Path(SAVE_DIR) / 'RelA_heatmap_{0}.pdf'.format(FEATURE_SET)
    clustermap_data_path = Path(SAVE_DIR) / 'RelA_heatmap_{0}_data.csv'.format(FEATURE_SET)
    data = plot_clustermap(featZ, metadata[['gene_name']], 
                           group_by='gene_name',
                           row_colours=None,
                           method='complete', 
                           metric='euclidean',
                           figsize=figsize,
                           sub_adj={'bottom':0.40 if FEATURE_SET == "tierpsy_16" else 0.05,
                                    'left':0,'top':1,'right':0.92},
                           saveto=clustermap_path,
                           label_size=(10,20),
                           show_xlabels=True if FEATURE_SET == "tierpsy_16" else False)
    data.to_csv(clustermap_data_path, header=True, index=True)
    

    metadata['gene_name_date'] = metadata[['gene_name','date_yyyymmdd']
                                     ].astype(str).agg('-'.join, axis=1)
    clustermap_path = Path(SAVE_DIR) / 'RelA_date_heatmap_{0}.pdf'.format(FEATURE_SET)
    clustermap_data_path = Path(SAVE_DIR) / 'RelA_date_heatmap_{0}_data.csv'.format(FEATURE_SET)
    data = plot_clustermap(featZ, metadata[['gene_name_date']], 
                           group_by=['gene_name_date'],
                           row_colours=None,
                           method='complete', 
                           metric='euclidean',
                           figsize=figsize,
                           sub_adj={'bottom':0.40 if FEATURE_SET == "tierpsy_16" else 0.05,
                                    'left':0,'top':1,'right':0.90},
                           saveto=clustermap_path,
                           label_size=(10,20),
                           show_xlabels=True if FEATURE_SET == "tierpsy_16" else False)
    data.to_csv(clustermap_data_path, header=True, index=True)

    # stats + boxplots of sigfeats

    plot_df = metadata[['gene_name','date_yyyymmdd']].join(features)

    ttest_results = stats(metadata,
                          features,
                          group_by='gene_name',
                          control='wild_type',
                          feat=None,
                          save_dir=Path(SAVE_DIR),
                          pvalue_threshold=0.05,
                          fdr_method='fdr_bh')
    
    # extract t-test pvals
    pvals = ttest_results[[c for c in ttest_results.columns if 'pval' in c]]
    pvals.columns = [c.replace('pvals_','') for c in pvals.columns]
    sigfeats = pvals[pvals['relA'] < 0.05].index.tolist()
        
    colour_dict = dict(zip(strain_list, sns.color_palette(palette='tab10', n_colors=2)))
    for feat in sigfeats:
        plt.close('all')
        sns.set_style('ticks')
        fig = plt.figure(figsize=[8,5])
        ax = fig.add_subplot(1,1,1)
        sns.boxplot(x='gene_name',
                    y=feat,
                    data=plot_df, 
                    order=strain_list,
                    hue='gene_name',
                    hue_order=strain_list,
                    dodge=False,
                    palette=colour_dict,
                    showfliers=True, 
                    showmeans=True,
                    meanprops={"marker":"x", 
                               "markersize":5,
                               "markeredgecolor":"k"},
                    flierprops={"marker":"x", 
                                "markersize":15, 
                                "markeredgecolor":"r"})
        dates = sorted(plot_df['date_yyyymmdd'].unique())
        date_lut = dict(zip(dates, sns.color_palette(palette="Greys", 
                                                     n_colors=len(dates))))
        for date in dates:
            date_df = plot_df[plot_df['date_yyyymmdd']==date]
            sns.stripplot(x='gene_name',
                          y=feat,
                          data=date_df,
                          order=strain_list,
                          hue='gene_name',
                          hue_order=strain_list,
                          dodge=False,
                          palette=[date_lut[date]] * plot_df['gene_name'].nunique(),
                          color='dimgray',
                          marker=".",
                          edgecolor='k',
                          linewidth=0.3,
                          s=12) #facecolors="none"
        
        ax.axes.get_xaxis().get_label().set_visible(False) # remove y axis label
        ax.axes.set_xticklabels([l.get_text() for l in ax.axes.get_xticklabels()], 
                                fontsize=20, rotation=0)
        ax.tick_params(axis='x', which='major', pad=15)
        ax.axes.set_ylabel(feat, fontsize=10, labelpad=25)                             
        plt.yticks(fontsize=20)
        
        # add pvalues to plot
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled
        for i, text in enumerate(ax.axes.get_xticklabels()):
            strain = text.get_text()
            if strain != 'wild_type':
                p = pvals.loc[feat, strain]
                p_text = sig_asterix([p])[0]
                ax.text(i, 1.03, p_text, fontsize=20, ha='center', va='center', 
                        transform=trans)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], loc='best', frameon=False, fontsize=15)
        # plt.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.95, 
        #                     hspace=0.01, wspace=0.01)
        
        boxplot_path = Path(SAVE_DIR) / 'Plots' / 'RelA_{0}.svg'.format(feat)
        boxplot_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(boxplot_path, bbox_inches='tight', transparent=True)


    return
    
#%% MAIN
if __name__ == "__main__":
    main()

