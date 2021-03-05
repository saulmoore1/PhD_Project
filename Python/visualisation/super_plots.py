#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUPER PLOTS

Plotting helper module for producing super-plots for comparing variation among experimental variables

@author: sm5911
@date: 05/03/2021

"""

#%% Imports

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

# TODO: Update timeseries plots for plotting windowed summaries
# Making more summaries and handling windows is a pain, no tierpsytools functions exist yet
# Luigi wrote something for Ida but is not ready for tierpsytools
# "Unless Andre is specifically telling you to look at windowed summaries, you should not 
#  go down the rabbit hole" - Luigi

from filter_data.clean_feature_summaries import clean_summary_results

from tierpsytools.read_data.hydra_metadata import read_hydra_metadata
from tierpsytools.hydra.platechecker import fix_dtypes

#%% Globals

EXAMPLE_METADATA_PATH = "/Volumes/hermes$/Keio_Tests_96WP/AuxiliaryFiles/metadata_annotated.csv"
EXAMPLE_RESULTS_DIR = "/Volumes/hermes$/Keio_Tests_96WP/Results"
EXAMPLE_FEATURE_LIST = ['speed_50th']

IMAGING_RUN = 3

# Mapping stimulus order for plotting
STIMULUS_DICT = {'prestim' : 0, 
                 'bluelight' : 1, 
                 'poststim' : 2}

CUSTOM_STYLE = 'visualisation/style_sheet_20210126.mplstyle'
DODGE = True
    
#%% Functions    
def superplot_1(features, metadata, feat, 
                x1="food_type", x2="date_yyyymmdd", # 'imaging_run_number', 'instrument_name'
                plot_type='violin', max_points=30,
                sns_colour_palettes=["plasma","viridis"], dodge=False, show=True): #**kwargs
    """ Plot timeseries for strains by Hydra day replicates """

    import seaborn as sns
    from matplotlib import patches
    from matplotlib import pyplot as plt
    
    assert set(metadata.index) == set(features.index)
    assert feat in features.columns

    if x2 is not None:
        df = metadata[[x1, x2]].join(features[feat])
        av_df = df.groupby([x1, x2], as_index=False).agg({feat: "mean"})
    else:
        df = metadata[[x1]].join(features[feat])
        av_df = df.groupby([x1], as_index=False).agg({feat: "mean"})

    # Create colour palette for bluelight colours
    x1_list = list(metadata[x1].unique())
    x1_labels = sns.color_palette(sns_colour_palettes[0], len(x1_list))
    x1_col_dict = dict(zip(x1_list, x1_labels))

    if x1 == 'bluelight':
        x1_order = list(dict(sorted((value, key) for (key, value) in STIMULUS_DICT.items())).values())
    else:
        x1_order = list(sorted(metadata[x1].unique()))

    if x2 is not None:
        x2_list = list(metadata[x2].unique())
        x2_labels = sns.color_palette(sns_colour_palettes[1], len(x2_list))
        x2_col_dict = dict(zip(x2_list, x2_labels))
        x2_order = list(sorted(metadata[x2].unique()))
        
    # Plot time-series
    plt.close('all')
    plt.style.use(CUSTOM_STYLE)
    fig, ax = plt.subplots(figsize=[10,8])
    mean_sample_size = df.groupby([x1, x2], as_index=False).size().mean()
    
    # Plot violin plot if lots of data points, else stripplot
    if 'box' in plot_type.lower():
        sns.boxplot(x=x1, 
                    y=feat,
                    order=x1_order,
                    hue=x2 if x2 is not None else None,
                    hue_order=x2_order if x2 is not None else None,
                    palette=x1_col_dict if x2 is None else x2_col_dict,
                    showfliers=False,
                    showmeans=True if x2 is not None else None,
                    # meanline=True, 
                    # meanprops={"marker":"x", "markersize":5, "markeredgecolor":"k"},
                    # flierprops={"marker":"x", "markersize":15, "markeredgecolor":"r"},
                    dodge=dodge, ax=ax, data=df)
    elif 'violin' in plot_type.lower():       
        if mean_sample_size > max_points:
            sns.violinplot(x=x1, 
                           y=feat, 
                           order=x1_order, 
                           hue=x2 if x2 is not None else None, 
                           hue_order=x2_order if x2 is not None else None,
                           palette=x1_col_dict if x2 is None else x2_col_dict, #size=5,
                           dodge=dodge, ax=ax, data=df)
            for violin, alpha in zip(ax.collections, np.repeat(0.5, len(ax.collections))):
                violin.set_alpha(alpha)
        else:
            sns.stripplot(x=x1, 
                          y=feat, 
                          order=x1_order, 
                          hue=x2 if x2 is not None else None, 
                          hue_order=x2_order if x2 is not None else None,
                          palette=x2_col_dict if x2 is not None else None, 
                          color=None if x2 is not None else 'gray',
                          s=10, marker=".", edgecolor='k', linewidth=.3, # facecolors="none"
                          dodge=dodge, ax=ax, data=df)
    # Plot group means
    sns.swarmplot(x=x1, 
                  y=feat, 
                  order=x1_order, 
                  hue=x2 if x2 is not None else None, 
                  hue_order=x2_order if x2 is not None else None,
                  palette=x2_col_dict if x2 is not None else None, 
                  size=13, edgecolor='k', linewidth=2, 
                  dodge=dodge, ax=ax, data=av_df)

    # from matplotlib import transforms
    # trans = transforms.blended_transform_factory(ax.transData, # y=none
    #                                              ax.transAxes) # x=scaled
    
    # Add custom legend
    patch_labels = []
    patch_handles = []
    if x2 is not None:
        patch_labels.extend(x2_order)
        for key in x2_order:
            patch = patches.Patch(color=x2_col_dict[key], label=key)
            patch_handles.append(patch) 
    else:
        patch_labels.extend(x1_order)
        for key in x1_order:
            patch = patches.Patch(color=x1_col_dict[key], label=key)
            patch_labels.append(key)
            patch_handles.append(patch)

    # handles, labels = ax.get_legend_handles_labels()        
    plt.legend(labels=patch_labels, 
               handles=patch_handles,
               loc=(1.05, 0.8), #'upper right'
               borderaxespad=0.4, 
               frameon=True, 
               framealpha=1, 
               fontsize=15,
               handletextpad=0.5)
    
    plt.xlim(right=len(x1_order)-0.4)
    plt.ylabel(''); plt.xlabel('')
    plt.title(feat.replace('_',' '), fontsize=20, pad=20)
    plt.subplots_adjust(right=0.85) # bottom, right, top, wspace, hspace
    #plt.tight_layout() #rect=[0.04, 0, 0.84, 0.96]
    
    if show:
        plt.show()

def superplot_2(features, metadata, feat, 
                x1="bluelight", x2="imaging_run_number",
                sns_colour_palettes=["plasma","viridis"], dodge=False, show=True):
    """ Plot timeseries before/during/after stimulus by Hydra imaging runs """

    import seaborn as sns
    from matplotlib import patches
    from matplotlib import pyplot as plt
    
    assert set(metadata.index) == set(features.index)
    assert feat in features.columns
    
    if x2 is not None:
        df = metadata[[x1, x2]].join(features[feat])
    else:
        df = metadata[[x1]].join(features[feat])

    # Create colour palette for bluelight colours
    x1_list = list(metadata[x1].unique())
    x1_labels = sns.color_palette(sns_colour_palettes[0], len(x1_list))
    x1_col_dict = dict(zip(x1_list, x1_labels))
    
    if x1 == 'bluelight':
        x1_order = list(dict(sorted((value, key) for (key, value) in STIMULUS_DICT.items())).values())
    else:
        x1_order = list(sorted(metadata[x1].unique()))

    if x2 is not None:
        x2_list = list(metadata[x2].unique())
        x2_labels = sns.color_palette(sns_colour_palettes[1], len(x2_list))
        x2_col_dict = dict(zip(x2_list, x2_labels))
        x2_order = list(sorted(metadata[x2].unique()))

    plt.close('all')
    plt.style.use(CUSTOM_STYLE)
    fig, ax = plt.subplots(figsize=[10,8])
    sns.boxplot(x=x1, 
                y=feat,
                order=x1_order,
                hue=x2 if x2 is not None else None,
                hue_order=x2_order if x2 is not None else None,
                palette=x1_col_dict if x2 is None else x2_col_dict,
                showfliers=False,
                showmeans=True if x2 is not None else None,
                # meanline=True, 
                # meanprops={"marker":"x", "markersize":5, "markeredgecolor":"k"},
                # flierprops={"marker":"x", "markersize":15, "markeredgecolor":"r"},
                dodge=dodge, ax=ax, data=df)
    sns.stripplot(x=x1, 
                  y=feat, 
                  order=x1_order,
                  hue=x2 if x2 is not None else None,
                  hue_order=x2_order if x2 is not None else None,
                  palette=x2_col_dict if x2 is not None else None,
                  color=None if x2 is not None else 'gray',
                  s=10, marker=".", edgecolor='k', linewidth=.3, # facecolors="none",
                  dodge=dodge, ax=ax, data=df)    
    
    # from matplotlib import transforms
    # trans = transforms.blended_transform_factory(ax.transData, # y=none
    #                                              ax.transAxes) # x=scaled
    
    # Add custom legend  
    patch_labels = x1_order + x2_order
    patch_handles = []
    for key in x1_order:
        patch = patches.Patch(color=x1_col_dict[key], label=key)
        patch_handles.append(patch)
    for key in x2_order:
        patch = patches.Patch(color=x2_col_dict[key], label=key)
        patch_handles.append(patch) 

    # handles, labels = ax.get_legend_handles_labels()        
    plt.legend(labels=patch_labels, 
               handles=patch_handles,
               loc='upper right', # (1.02, 0.8)
               borderaxespad=0.4, 
               frameon=True, 
               framealpha=1, 
               fontsize=15,
               handletextpad=0.5)
    
    plt.xlim(right=len(x1_order)-0.4)
    plt.ylabel(''); plt.xlabel('')
    plt.title(feat.replace('_',' '), fontsize=20, pad=20)
    #plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
    #plt.tight_layout() #rect=[0.04, 0, 0.84, 0.96]
    
    if show:
        plt.show()
                                                 
#%% Main

if __name__ == '__main__':
    # Accept feature list from command line
    parser = argparse.ArgumentParser(description='Time-series analysis of selected features')
    parser.add_argument("--compiled_metadata_path", help="Path to compiled metadata file",
                        default=EXAMPLE_METADATA_PATH)
    parser.add_argument("--results_dir", help="Path to 'Results' directory containing full features\
                        and filenames summary files", default=EXAMPLE_RESULTS_DIR)
    parser.add_argument('--feature_list', help="List of selected features for time-series analysis", 
                        nargs='+', default=EXAMPLE_FEATURE_LIST)
    args = parser.parse_args()

    assert Path(args.compiled_metadata_path).exists()
    assert Path(args.results_dir).is_dir()
    assert type(args.feature_list) == list

    combined_feats_path = Path(args.results_dir) / "full_features.csv"
    combined_fnames_path = Path(args.results_dir) / "full_filenames.csv"
    
    # Ensure align bluelight is False
    # NB: leaves the df in a "long format" that seaborn likes    
    features, metadata = read_hydra_metadata(feat_file=combined_feats_path,
                                             fname_file=combined_fnames_path,
                                             meta_file=args.compiled_metadata_path,
                                             add_bluelight=True)

    # Convert metadata column dtypes, ie. stringsAsFactors, no floats, Δ, etc
    metadata = fix_dtypes(metadata)
    metadata['food_type'] = [f.replace("Δ","_") for f in metadata['food_type']]
    
    features, metadata = clean_summary_results(features, metadata)
    
    # Find masked HDF5 video files
    print("%d selected features loaded." % len(args.feature_list))

    # # Subset data for given imaging run
    # from filter_data.clean_feature_summaries import subset_results
    # run_feats, run_meta = subset_results(features, metadata, 'imaging_run_number', [IMAGING_RUN])
    
    # Time-series plots of day/run variation for selected features
    for feat in args.feature_list:
        superplot_1(features, 
                    metadata, 
                    feat, 
                    x1='food_type', 
                    x2='date_yyyymmdd', 
                    dodge=DODGE)
        plt.pause(10)
        superplot_2(features, 
                    metadata, 
                    feat, 
                    x1='bluelight', 
                    x2='imaging_run_number', 
                    dodge=DODGE)
        plt.pause(10)