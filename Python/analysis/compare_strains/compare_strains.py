#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic script to compile metadata and features summary results, load and filter the data, 
perform statistics and plot boxplots and heatmaps comparing between strains. 

Please note that filepath and parameter default are hard coded for the example dataset 
in the 'Globals' section. Please provide all arguments at command-line via argparse 
if using this script to analyse a different dataset.

@author: sm5911
@date: 15/12/2021

"""

#%% Imports

import re
import argparse
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.stats import zscore

from tierpsytools.hydra.compile_metadata import populate_96WPs, get_day_metadata
from tierpsytools.hydra.match_wells_annotations import update_metadata_with_wells_annotations
from tierpsytools.read_data.hydra_metadata import read_hydra_metadata, align_bluelight_conditions
from tierpsytools.preprocessing.filter_data import select_feat_set
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from tierpsytools.preprocessing.filter_data import filter_nan_inf, feat_filter_std
# other filtering options: drop_bad_wells, drop_ventrally_signed, filter_n_skeletons

#%% Globals

AUX_DIR = '/Volumes/behavgenom$/Tanara/Screen1_20211210/AuxiliaryFiles'
IMAGING_DATES = ['20211210']

STRAIN_COLNAME = 'worm_gene' # 'worm_code'
CONTROL_STRAIN = 'elegans'   # 'N2'

N_TIERPSY_FEATURES = 16 # 256, None
COMPILED_FEATURES_PATH = '/Volumes/behavgenom$/Tanara/Screen1_20211210/Results/features_summary_tierpsy_plate_20211213_083702.csv'
COMPILED_FILENAMES_PATH = '/Volumes/behavgenom$/Tanara/Screen1_20211210/Results/filenames_summary_tierpsy_plate_20211213_083702.csv'

#%% Functions

def compile_day_metadata(aux_dir, day):
    """ Compile experiment day metadata from wormsorter and hydra rig metadata for a given day in 
        'AuxiliaryFiles' directory 
        
        Parameters
        ----------
        aux_dir : str
            Path to "AuxiliaryFiles" containing metadata  
        day : str, None
            Experiment day folder in format 'YYYYMMDD'
            
        Returns
        -------
        compiled_day_metadata : pandas.DataFrame
            Compiled metadata dataframe for given imaging date
    """
        
    day_dir = Path(aux_dir) / str(day)
    wormsorter_meta = day_dir / (str(day) + '_wormsorter.csv')
    hydra_meta = day_dir / (str(day) + '_manual_metadata.csv')
      
    # expand wormsorter metadata to have separate row for each well
    plate_metadata = populate_96WPs(wormsorter_meta)
    
    day_metadata = get_day_metadata(complete_plate_metadata=plate_metadata, 
                                    hydra_metadata_file=hydra_meta,
                                    merge_on=['imaging_plate_id'],
                                    n_wells=96,
                                    run_number_regex='run\\d+_',
                                    saveto=None,
                                    del_if_exists=False,
                                    include_imgstore_name=True,
                                    raw_day_dir=None)
    
    return day_metadata

def compile_metadata(aux_dir, imaging_dates=None):
    """ Compile experiment metadata from day metadata files in imaging date sub-directories within 
        'AuxiliaryFiles' directory

        Parameters
        ----------
        aux_dir : str
            Path to "AuxiliaryFiles" containing metadata.
        imaging_dates : list, optional
            List of experiment dates to compile metadata for. If None, use all dates found in directory.
    
        Returns
        -------
        compiled_metadata : pandas.DataFrame
            Compiled metadata dataframe (for all imaging dates if None given)

    """
    
    metadata_path = Path(aux_dir) / 'metadata.csv'
    
    dates_found = re.findall(r'20\d{6}', aux_dir)
    if imaging_dates is None:
        imaging_dates = dates_found
    else: 
        assert type(imaging_dates) == list and all(d in dates_found for d in imaging_dates)
        
    day_meta_list = []
    for day in imaging_dates:
        day_metadata_path = Path(aux_dir) / day / '{}_day_metadata.csv'.format(day)
        if not day_metadata_path.exists():
            day_meta = compile_day_metadata(aux_dir, day)
        else:
            day_meta = pd.read_csv(day_metadata_path, index_col=False)
        day_meta_list.append(day_meta)
    
    metadata_df = pd.concat(day_meta_list, axis=0).reset_index()
    
    # save day metadata as metadata.csv
    metadata_df.to_csv(metadata_path, index=False)
    
    return metadata_df
    
#%% Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare between strains for given dataset")
    parser.add_argument('--aux_dir', type=str, default=AUX_DIR,
                        help="Path to 'AuxiliaryFiles' directory containing experiment metadata")
    parser.add_argument('--imaging_dates', type=list, nargs='+', default=IMAGING_DATES, 
                        help="List of imaging dates to process")
    parser.add_argument('--strain_colname', type=str, default=STRAIN_COLNAME, 
                        help="Column name in metadata for strain info")
    parser.add_argument('--control', type=str, default=CONTROL_STRAIN, 
                        help="Control strain name")
    parser.add_argument('--features_file', type=str, default=COMPILED_FEATURES_PATH,
                        help="Path to compiled features summary file")
    parser.add_argument('--filenames_file', type=str, default=COMPILED_FILENAMES_PATH,
                        help="Path to compiled filenames summary file")
    parser.add_argument('--n_features', type=int, default=N_TIERPSY_FEATURES, 
                        help="Tierpsy feature set to use for analysis, choose from: [8,16,256] \
                        (default=16 for 'tierpsy_16')")
    args = parser.parse_args()
    
    save_dir = Path(args.aux_dir).parent / 'Analysis' / args.strain_colname
    metadata_path = Path(args.aux_dir) / 'metadata.csv'
    
    ### compile metadata
    
    if not metadata_path.exists():
        metadata_df = compile_metadata(args.aux_dir, imaging_dates=args.imaging_dates)
    else:
        metadata_df = pd.read_csv(metadata_path, index_col=False)
    
    # add well annotations to metadata
    annotated_metadata_path = Path(str(metadata_path).replace('.csv','_annotated.csv'))
    if not annotated_metadata_path.exists():
        metadata_df = update_metadata_with_wells_annotations(Path(args.aux_dir), 
                                                             saveto=annotated_metadata_path)        
    
    # read metadata + features summaries
    features_df, metadata_df = read_hydra_metadata(feat_file=args.features_file, 
                                                   fname_file=args.filenames_file, 
                                                   meta_file=annotated_metadata_path)

    # align bluelight conditions (as separate feature columns)
    features_df, metadata_df = align_bluelight_conditions(features_df, metadata_df, 
                                                          merge_on_cols=['date_yyyymmdd',
                                                                         'imaging_plate_id',
                                                                         'well_name'])

    ### clean data
    
    # remove rows with missing strain information (n=10)
    metadata_df = metadata_df[~metadata_df[args.strain_colname].isna()]
    features_df = features_df.reindex(metadata_df.index)  
      
    # subset for Tierpsy features only
    if args.n_features is not None:
        features_df = select_feat_set(features_df, 
                                      tierpsy_set_name='tierpsy_{}'.format(args.n_features),
                                      append_bluelight=True)

    # subset for given imaging dates
    metadata_df = metadata_df[metadata_df['date_yyyymmdd'].astype(str).isin(args.imaging_dates)]
    features_df = features_df.reindex(features_df.index)
    
    # use tierpsytools functions to features clean data and remove NaNs
    
    # Drop rows based on percentage of NaN values across features for each row
    # NB: axis=1 will sum the NaNs across all the columns for each row
    features_df = filter_nan_inf(features_df, threshold=0.8, axis=1, verbose=True)
    metadata_df = metadata_df.reindex(features_df.index)
 
    # Drop feature columns with too many NaN values
    # NB: to remove features with NaNs across all results, eg. food_edge related features
    features_df = filter_nan_inf(features_df, threshold=0.2, axis=0, verbose=False)
        
    # Drop feature columns with zero standard deviation
    features_df = feat_filter_std(features_df, threshold=0.0)

    # Fill in NaNs with global mean
    features_df = features_df.fillna(features_df.mean(axis=0))
        
    feature_list = features_df.columns.to_list()
    strain_list = list(metadata_df[args.strain_colname].unique())

    ### statistics 
    # ANOVA to test to variation among strains 
    if len(metadata_df[args.strain_colname].unique()) > 2:
        stats, pvals, reject = univariate_tests(X=features_df,
                                                y=metadata_df[args.strain_colname],
                                                control=args.control,
                                                test='ANOVA',
                                                comparison_type='multiclass',
                                                multitest_correction='fdr_by',
                                                alpha=0.05)
        # get effect sizes
        effect_sizes = get_effect_sizes(X=features_df, 
                                        y=metadata_df[args.strain_colname],
                                        control=args.control,
                                        effect_type=None,
                                        linked_test='ANOVA')
        # compile + save results
        test_results = pd.concat([stats, effect_sizes, pvals, reject], axis=1)
        test_results.columns = ['stats','effect_size','pvals','reject']     
        test_results = test_results.sort_values(by=['pvals'], ascending=True) # rank pvals
        anova_save_path = save_dir / 'stats' / 'ANOVA_results.csv'
        anova_save_path.parent.mkdir(exist_ok=True, parents=True)
        test_results.to_csv(anova_save_path, header=True, index=True)

    # t-tests between each strain vs control
    stats_t, pvals_t, reject_t = univariate_tests(X=features_df, 
                                                  y=metadata_df[args.strain_colname],
                                                  control=args.control,
                                                  test='t-test',
                                                  comparison_type='binary_each_group',
                                                  multitest_correction='fdr_by',
                                                  alpha=0.05)
    
    effect_sizes_t =  get_effect_sizes(X=features_df, 
                                       y=metadata_df[args.strain_colname], 
                                       control=args.control,
                                       effect_type=None,
                                       linked_test='t-test')
            
    # compile + save t-test results
    stats_t.columns = ['stats_' + str(c) for c in stats_t.columns]
    pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
    reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
    effect_sizes_t.columns = ['effect_size_' + str(c) for c in effect_sizes_t.columns]
    ttest_results = pd.concat([stats_t, effect_sizes_t, pvals_t, reject_t], axis=1)
    ttest_save_path = save_dir / 'stats' / 't-test_results.csv'
    ttest_save_path.parent.mkdir(exist_ok=True, parents=True)
    ttest_results.to_csv(ttest_save_path, header=True, index=True)

    ### boxplots
    
    plot_df = metadata_df.join(features_df)
    
    print("\nPlotting feature boxplots..")
    for feature in tqdm(feature_list):
        plt.close('all')
        fig, ax = plt.subplots(figsize=(15,6) if len(strain_list) > 15 else (8,6)) #
        sns.boxplot(x=args.strain_colname, y=feature, data=plot_df, ax=ax, showfliers=False)
        sns.stripplot(x=args.strain_colname, y=feature, data=plot_df, ax=ax, 
                      color='k', alpha=0.5, size=3)
        if len(strain_list) > 15:
            plt.xticks(rotation=90)
            plt.tight_layout()

        # save boxplot
        box_save_path = save_dir / 'boxplots' / (feature + '.png')
        box_save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(box_save_path, dpi=300)
    
    ### heatmap 
    
    # Z-normalise control data
    featZ = features_df.apply(zscore, axis=0)
    
    featZ_grouped = featZ.join(metadata_df).groupby(args.strain_colname, 
                                                    dropna=False).mean().reset_index()
    
    # colour columns (features) by prestim/bluelight/poststim
    bluelight_colour_dict = dict(zip(['prestim','bluelight','poststim'], 
                                      sns.color_palette('Pastel1', 3)))
    feat_colour_dict = {f:bluelight_colour_dict[f.split('_')[-1]] for f in feature_list}

    print("\nPlotting heatmap..")
    plt.close('all')
    cg = sns.clustermap(data=featZ_grouped[feature_list], 
                        #row_colors=row_colours,
                        col_colors=featZ_grouped[feature_list].columns.map(feat_colour_dict),
                        #standard_scale=1, z_score=1,
                        #col_linkage=col_linkage,
                        metric='euclidean',
                        method='complete', #'average'
                        vmin=-2, vmax=2,
                        #figsize=(10,10),
                        xticklabels=feature_list,
                        yticklabels=featZ_grouped[args.strain_colname],
                        linewidths=0)
    # save heatmap
    heatmap_save_path = save_dir / 'heatmap' / '{}_heatmap.pdf'.format(args.strain_colname)
    heatmap_save_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(heatmap_save_path, dpi=600)

    print("Done!")
    
