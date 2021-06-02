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
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Custom imports
from filter_data.clean_feature_summaries import clean_summary_results
from read_data.read import load_topfeats

from tierpsytools.read_data.hydra_metadata import read_hydra_metadata
from tierpsytools.hydra.platechecker import fix_dtypes

#%% Globals

EXAMPLE_METADATA_PATH = "/Volumes/hermes$/Keio_Tests_96WP/AuxiliaryFiles/metadata_annotated.csv"
EXAMPLE_RESULTS_DIR = "/Volumes/hermes$/Keio_Tests_96WP/Results"

IMAGING_RUN = 3

# Mapping stimulus order for plotting
STIMULUS_DICT = {'prestim' : 0, 
                 'bluelight' : 1, 
                 'poststim' : 2}

CUSTOM_STYLE = "visualisation/style_sheet_20210126.mplstyle"
DODGE = True
    
#%% Functions    
def superplot(features, metadata, feat, 
              x1="food_type", x2="date_yyyymmdd", # 'imaging_run_number', 'instrument_name'
              plot_type='box', show_points=None, plot_means=True, max_points=30, 
              sns_colour_palettes=["plasma","viridis"], 
              dodge=False, saveDir=None, **kwargs):
    """ Plot timeseries for strains by Hydra day replicates """
    
    import numpy as np
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
        try:
            x1_order = list(sorted(metadata[x1].unique()))
        except:
            x1_order = list(metadata[x1].unique())

    x2_list = []
    if x2 is not None:
        x2_list = list(metadata[x2].unique())
        x2_labels = sns.color_palette(sns_colour_palettes[1], len(x2_list))
        x2_col_dict = dict(zip(x2_list, x2_labels))
        try:
            x2_order = list(sorted(metadata[x2].unique()))
        except:
            x2_order = list(metadata[x2].unique())
        
    # Plot time-series
    plt.close('all')
    plt.style.use(CUSTOM_STYLE)
    plt.ioff() if saveDir is not None else plt.ion()
    fig, ax = plt.subplots(figsize=[20,7])
    
    mean_sample_size = df.groupby(([x1, x2] if x2 is not None else [x1]), as_index=False).size().mean()
    
    if x2 is not None:
        if len(x2_list) > max_points:
            palette = None
        else:
            palette = x2_col_dict
    else:
        palette = x1_col_dict
        
    # Plot violin plot if lots of data points, else stripplot
    if 'box' in plot_type.lower():
        sns.boxplot(x=x1, 
                    y=feat,
                    order=x1_order,
                    hue=x2 if x2 is not None else None,
                    hue_order=x2_order if x2 is not None else None,
                    palette=palette,
                    showfliers=False,
                    showmeans=True if x2 is not None else None,
                    # meanline=True, 
                    # meanprops={"marker":"x", "markersize":5, "markeredgecolor":"k"},
                    # flierprops={"marker":"x", "markersize":15, "markeredgecolor":"r"},
                    dodge=dodge, ax=ax, data=df)
    elif 'violin' in plot_type.lower():       
        sns.violinplot(x=x1, 
                       y=feat, 
                       order=x1_order, 
                       hue=x2 if x2 is not None else None, 
                       hue_order=x2_order if x2 is not None else None,
                       palette=palette, #size=5,
                       dodge=dodge, ax=ax, data=df)
        for violin, alpha in zip(ax.collections, np.repeat(0.5, len(ax.collections))):
            violin.set_alpha(alpha)

    show_points = False if (show_points is None and mean_sample_size > max_points) else True
    if show_points:
        sns.stripplot(x=x1, 
                      y=feat, 
                      order=x1_order, 
                      hue=x2 if x2 is not None else None, 
                      hue_order=x2_order if x2 is not None else None,
                      palette=palette if x2 is not None else None, 
                      color=None if x2 is not None else 'gray',
                      s=10, marker=".", edgecolor='k', linewidth=.3, # facecolors="none"
                      dodge=dodge, ax=ax, data=df)
        
    # Plot group means
    if plot_means:
        sns.swarmplot(x=x1, 
                      y=feat, 
                      order=x1_order, 
                      hue=x2 if x2 is not None else None, 
                      hue_order=x2_order if x2 is not None else None,
                      palette=palette if x2 is not None else None, 
                      size=13, edgecolor='k', linewidth=2, 
                      dodge=dodge, ax=ax, data=av_df)

    # from matplotlib import transforms
    # trans = transforms.blended_transform_factory(ax.transData, # y=none
    #                                              ax.transAxes) # x=scaled

# =============================================================================
#     # Add p-value to plot  
#     if test_pvalues_df is not None:
#         for ii, group in enumerate(groups[1:]):
#             pval = test_pvalues_df.loc[group, feature]
#             text = ax.get_xticklabels()[ii+1]
#             assert text.get_text() == group
#             if isinstance(pval, float) and pval < p_value_threshold:
#                 y = df[feature].max() 
#                 h = (y - df[feature].min()) / 50
#                 plt.plot([0, 0, ii+1, ii+1], [y+h, y+2*h, y+2*h, y+h], lw=1.5, c='k')
#                 pval_text = 'P < 0.001' if pval < 0.001 else 'P = %.3f' % pval
#                 ax.text((ii+1)/2, y+2*h, pval_text, fontsize=12, ha='center', va='bottom')
#     plt.subplots_adjust(left=0.15) #top=0.9,bottom=0.1,left=0.2
# =============================================================================

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=5)
    plt.subplots_adjust(bottom=0.15)
        
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
   
    if saveDir:
        savePath = Path(saveDir) / (x1 + ('_wrt_' + x2 if x2 is not None else '')) / (feat + '.png')
        savePath.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(savePath, dpi=300)
    else:
        plt.show(); plt.pause(2)
             
#%% Main

if __name__ == '__main__':
    # Accept feature list from command line
    parser = argparse.ArgumentParser(description='Time-series analysis of selected features')
    parser.add_argument("--compiled_metadata_path", help="Path to compiled metadata file",
                        default=EXAMPLE_METADATA_PATH)
    parser.add_argument("--results_dir", help="Path to 'Results' directory containing full features\
                        and filenames summary files", default=EXAMPLE_RESULTS_DIR)
    parser.add_argument("--save_dir", help="Directory to save super-plots", default=None)
    parser.add_argument('--feature_list_from_csv', help="Path to saved list of selected features\
                        to plot (CSV)", nargs='+', default=None)
    args = parser.parse_args()

    assert Path(args.compiled_metadata_path).exists()
    assert Path(args.results_dir).is_dir()
    args.save_dir = (args.save_dir if args.save_dir is not None else 
                     Path(args.results_dir).parent / "Analysis" / "Superplots")

    combined_feats_path = Path(args.results_dir) / "full_features.csv"
    combined_fnames_path = Path(args.results_dir) / "full_filenames.csv"
    
    # NB: leaves the df in a "long format" that seaborn 'likes'   
    features, metadata = read_hydra_metadata(feat_file=combined_feats_path,
                                             fname_file=combined_fnames_path,
                                             meta_file=args.compiled_metadata_path,
                                             add_bluelight=True)

    # Convert metadata column dtypes, ie. stringsAsFactors, no floats, Δ, etc
    metadata = fix_dtypes(metadata)
    metadata['food_type'] = [f.replace("Δ","_") for f in metadata['food_type']]
    
    features, metadata = clean_summary_results(features, metadata)
        
    # Load feature list from file
    if args.feature_list_from_csv is not None:
        assert Path(args.feature_list_from_csv).exists()
        
        feature_list = pd.read_csv(args.feature_list_from_csv)
        feature_list = list(feature_list[feature_list.columns[0]].unique())
    elif args.n_top_feats is not None:
        top_feats_path = Path(args.tierpsy_top_feats_dir) / "tierpsy_{}.csv".format(str(args.n_top_feats))        
        topfeats = load_topfeats(top_feats_path, add_bluelight=True, 
                                 remove_path_curvature=True, header=None)

        # Drop features that are not in results
        feature_list = [feat for feat in list(topfeats) if feat in features.columns]
        features = features[feature_list]

    print("%d features loaded." % len(feature_list))

    # # Subset data for given imaging run
    # from filter_data.clean_feature_summaries import subset_results
    # run_feats, run_meta = subset_results(features, metadata, 'imaging_run_number', [IMAGING_RUN])
    
    # Time-series plots of day/run variation for selected features
    for feat in tqdm(feature_list):
        superplot(features, 
                  metadata, 
                  feat, 
                  x1='food_type', 
                  x2='date_yyyymmdd', 
                  dodge=DODGE,
                  saveDir=args.save_dir)
        
# TODO: Update timeseries plots for plotting windowed summaries
# Making more summaries and handling windows is a pain, no tierpsytools functions exist yet
# Luigi wrote something for Ida but is not ready for tierpsytools
# "Unless Andre is specifically telling you to look at windowed summaries, you should not 
#  go down the rabbit hole" - Luigi
