#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper Functions

@author: sm5911
@date: 21/11/2020

"""

#%% Imports
import sys
import umap
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from scipy.stats import zscore
# import scipy.spatial as sp
# import scipy.cluster.hierarchy as hc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.covariance import MinCovDet
from matplotlib import pyplot as plt
from matplotlib import patches, transforms
from matplotlib.gridspec import GridSpec
from matplotlib.axes._axes import _log as mpl_axes_logger
from mpl_toolkits.mplot3d import Axes3D

# Path to Github helper functions (USER-DEFINED path to local copy of Github repo)
PATH_LIST = ['/Users/sm5911/Tierpsy_Versions/tierpsy-tools-python/']
for sysPath in PATH_LIST:
    if sysPath not in sys.path:
        sys.path.insert(0, sysPath)

CUSTOM_STYLE = '/Users/sm5911/Documents/GitHub/PhD_Project/Python/analysis/analysis_20210126.mplstyle'

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
                                 'time_worms_on_test_food']
    
    if all(x in df.columns for x in duration_required_columns) and \
        all(df[x].any() for x in duration_required_columns):
        # Extract worm dispensing dates and times
        dispense_datetime = [datetime.datetime.strptime(date_str + ' ' +\
                             time_str, '%Y%m%d %H:%M:%S') for date_str, time_str in
                             zip(df['date_worms_on_test_food_yyyymmdd'].astype(int).astype(str),
                             df['time_worms_on_test_food'])]
        # Extract imaging dates and times
        imaging_datetime = [datetime.datetime.strptime(date_str + ' ' +\
                            time_str, '%Y%m%d %H:%M:%S') for date_str, time_str in 
                            zip(df['date_yyyymmdd'].astype(int).astype(str), df['time_recording'])]
        # Estimate duration worms have spent on food at time of imaging
        on_food_duration = [image - dispense for dispense, image in \
                            zip(dispense_datetime, imaging_datetime)]
            
        # Add duration on food to df
        df['duration_on_food_seconds'] = [int(timedelta.total_seconds()) for \
                                          timedelta in on_food_duration]
   
    return df

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

def plot_day_variation(feat_df,
                       meta_df,
                       group_by,
                       control,
                       test_pvalues_df=None,
                       day_var='date_yyyymmdd',
                       feature_set=None,
                       max_features_plot_cap=None,
                       p_value_threshold=0.05,
                       saveDir=None,
                       figsize=[6,6],
                       sns_colour_palette="tab10",
                       dodge=False,
                       ranked=True,
                       drop_insignificant=True):
    """
    Parameters
    ----------
    pvalues : pandas.Series
    """
    
    if feature_set is not None:
        assert all(f in feat_df.columns for f in feature_set)
    else:
        feature_set = [f for f in feat_df.columns]
    
    if max_features_plot_cap is not None:
        feature_set = feature_set[:max_features_plot_cap]
    
    groups = list(meta_df[group_by].unique())
    groups.remove(control)  
    
    if test_pvalues_df is not None:
        assert all(f in test_pvalues_df.columns for f in feature_set)
        assert all(g in list(test_pvalues_df.index) for g in groups)

        if drop_insignificant:    
            feature_set = [f for f in feature_set if (test_pvalues_df[f] < p_value_threshold).any()]         
    
    groups.insert(0, control)
    
    plt.ioff() if saveDir else plt.ion()
    for idx, feature in enumerate(tqdm(feature_set, position=0)):
                
        df = meta_df[[group_by, day_var]].join(feat_df[feature])
        RepAverage = df.groupby([group_by, day_var], as_index=False).agg({feature: "mean"})
        #RepAvPivot = RepAverage.pivot_table(columns=group_by, values=f, index=random_effect)
        #stat, pval = ttest_rel(RepAvPivot[control], RepAvPivot[g])
        date_list = list(df[day_var].unique())
        date_colours = sns.color_palette(sns_colour_palette, len(date_list))
        date_dict = dict(zip(date_list, date_colours))
            
        plt.close('all')
        plt.style.use(CUSTOM_STYLE)
        fig, ax = plt.subplots(figsize=figsize)
        mean_sample_size = df.groupby([group_by, day_var], as_index=False).size().mean()
        if mean_sample_size > 10:
            sns.violinplot(x=group_by, 
                           y=feature, 
                           order=groups,
                           hue=day_var, 
                           palette=date_dict,
                           #size=5,
                           data=df,
                           ax=ax,
                           dodge=dodge)
            for violin, alpha in zip(ax.collections, np.repeat(0.5, len(ax.collections))):
                violin.set_alpha(alpha)
        else:
            sns.stripplot(x=group_by, 
                          y=feature, 
                          data=df,
                          s=10,
                          order=groups,
                          hue=day_var,
                          palette=date_dict,
                          color=None,
                          marker=".",
                          edgecolor='k',
                          linewidth=.3) #facecolors="none"
        sns.swarmplot(x=group_by, 
                      y=feature, 
                      order=groups,
                      hue=day_var,
                      palette=date_dict,
                      size=13,
                      edgecolor='k',
                      linewidth=2,
                      data=RepAverage,
                      ax=ax,
                      dodge=dodge)
        handles, labels = ax.get_legend_handles_labels()
        n_labs = len(meta_df[day_var].unique())
        ax.legend(handles[:n_labs], labels[:n_labs], fontsize=10)
        plt.title(feature.replace('_',' '), fontsize=12, pad=20)
        plt.xlim(right=len(groups)-0.4)
        plt.ylabel(''); plt.xlabel('')
       
        # Add p-value to plot  
        if test_pvalues_df is not None:
            for ii, group in enumerate(groups[1:]):
                pval = test_pvalues_df.loc[group, feature]
                text = ax.get_xticklabels()[ii+1]
                assert text.get_text() == group
                if isinstance(pval, float) and pval < p_value_threshold:
                    y = df[feature].max() 
                    h = (y - df[feature].min()) / 50
                    plt.plot([0, 0, ii+1, ii+1], [y+h, y+2*h, y+2*h, y+h], lw=1.5, c='k')
                    pval_text = 'P < 0.001' if pval < 0.001 else 'P = %.3f' % pval
                    ax.text((ii+1)/2, y+2*h, pval_text, fontsize=12, ha='center', va='bottom')
        plt.subplots_adjust(left=0.15) #top=0.9,bottom=0.1,left=0.2

        if len(groups) > 10:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            plt.subplots_adjust(bottom=0.15)
            
        if saveDir is not None:
            Path(saveDir).mkdir(exist_ok=True, parents=True)
            savePath = Path(saveDir) / ((('{0}_'.format(idx + 1) + feature) if ranked 
                                         else feature) + '.png')
            plt.savefig(savePath, dpi=300) # dpi=600
        else:
            plt.show(); plt.pause(2)

def barplot_sigfeats(test_pvalues_df=None, saveDir=None, p_value_threshold=0.05,
                     test_name=None):
    """ Plot barplot of number of significant features from test p-values """
    
    if test_pvalues_df is not None:
        # Proportion of features significantly different from control
        prop_sigfeats = ((test_pvalues_df < p_value_threshold).sum(axis=1) /\
                         len(test_pvalues_df.columns))*100
        prop_sigfeats = prop_sigfeats.sort_values(ascending=False)
        
        n = len(prop_sigfeats.index)
        
        # Plot proportion significant features for each strain
        plt.ioff() if saveDir else plt.ion()
        plt.close('all')
        plt.style.use(CUSTOM_STYLE)  
        fig = plt.figure(figsize=[6, n/4 if n > 20 else 7]) # width, height
        ax = fig.add_subplot(1,1,1)
        ax.barh(prop_sigfeats,width=1)
        prop_sigfeats.plot.barh(x=prop_sigfeats.index, 
                                y=prop_sigfeats.values, 
                                color='gray',
                                ec='black') # fc
        ax.set_xlabel('% significant features') # fontsize=16, labelpad=10
        plt.xlim(0,100)
        for i, (l, v) in enumerate((test_pvalues_df < p_value_threshold).sum(axis=1).items()):
            ax.text(prop_sigfeats.loc[l] + 2, i, str(v), color='k', 
                    va='center', ha='left') #fontweight='bold'
        plt.text(0.85, 1, 'n = %d' % len(test_pvalues_df.columns), ha='center', va='bottom', 
                 transform=ax.transAxes)  
        plt.tight_layout(rect=[0.02, 0.02, 0.96, 1]) #
    
        if saveDir:
            Path(saveDir).mkdir(exist_ok=True, parents=True)
            savePath = Path(saveDir) / ('percentage_sigfeats_{}.png'.format(test_name) if \
                                        test_name is not None else 'percentage_sigfeats.png')
            print("Saving figure: %s" % savePath.name)
            plt.savefig(savePath, dpi=300)
        else:
            plt.show()
        
        return prop_sigfeats
    
def boxplots_sigfeats(feat_meta_df, 
                      test_pvalues_df, 
                      group_by, 
                      control_strain, 
                      saveDir=None, 
                      p_value_threshold=0.05,
                      max_features_plot_cap=None,
                      feature_set=None,
                      sns_colour_palette="tab10",
                      colour_by_date=False,
                      drop_insignificant=True,
                      verbose=True):
    """ Box plots of most significantly different features between strains """    
       
    if feature_set is not None:
        if not type(feature_set) == list:
            try:
                feature_set = list(feature_set)
            except:
                raise IOError("Please provide selected features as a list!")
        
    for strain in tqdm(test_pvalues_df.index, position=0):
        pvals = test_pvalues_df.loc[strain]
        
        n_sigfeats = sum(pvals < p_value_threshold)

        if (pvals.isna().all() or n_sigfeats == 0) and verbose:
            print("No signficant features found for %s" % strain)
        elif n_sigfeats > 0 or not drop_insignificant:   
            # ranked p-values
            ranked_pvals = pvals.sort_values(ascending=True) # rank p-values in ascending order
            ranked_pvals = ranked_pvals.dropna(axis=0) # drop NaNs
            topfeats = ranked_pvals[ranked_pvals < p_value_threshold] # drop non-sig feats  
            
            if feature_set is not None:
                select_feat_pvals = pvals[feature_set]
                topfeats = select_feat_pvals.append(topfeats)
                
            if max_features_plot_cap is not None and len(topfeats) > max_features_plot_cap:
                topfeats = topfeats[:max_features_plot_cap]
                if verbose:
                    print("\nCapping plots for %s at %d features..\n" % (strain, len(topfeats)))
            else:
                if verbose:
                    print("%d significant features found for %s" % (n_sigfeats, str(strain)))

            # Subset feature summary results for test-strain + control only
            plot_df = feat_meta_df[np.logical_or(feat_meta_df[group_by]==control_strain,
                                                 feat_meta_df[group_by]==str(strain))]
        
            # Colour/legend dictionary
            # Create colour palette for plot loop
            colour_labels = sns.color_palette(sns_colour_palette, 2)
            colour_dict = {control_strain:colour_labels[0], str(strain):colour_labels[1]}
            
            if colour_by_date:
                date_colours = sns.color_palette("Paired", len(plot_df['date_yyyymmdd'].unique()))
                date_dict = dict(zip(plot_df['date_yyyymmdd'].unique(), date_colours))
                
            order = list(plot_df[group_by].unique())
            order.remove(control_strain)
            order.insert(0, control_strain)
                                                  
            # Boxplots of control vs test-strain for each top-ranked significant feature
            plt.ioff() if saveDir else plt.ion()
            for f, feature in enumerate(topfeats.index):
                plt.close('all')
                plt.style.use(CUSTOM_STYLE) 
                sns.set_style('ticks')
                fig = plt.figure(figsize=[10,8])
                ax = fig.add_subplot(1,1,1)
                sns.boxplot(x=group_by, 
                            y=feature, 
                            data=plot_df, 
                            order=order,
                            palette=colour_dict,
                            showfliers=False, 
                            showmeans=True if colour_by_date else None,
                            #meanline=True,
                            meanprops={"marker":"x", 
                                       "markersize":5,
                                       "markeredgecolor":"k"},
                            flierprops={"marker":"x", 
                                        "markersize":15, 
                                        "markeredgecolor":"r"})
                sns.stripplot(x=group_by, 
                              y=feature, 
                              data=plot_df,
                              s=10,
                              order=order,
                              hue='date_yyyymmdd' if colour_by_date else None,
                              palette=date_dict if colour_by_date else None,
                              color=None if colour_by_date else 'gray',
                              marker=".",
                              edgecolor='k',
                              linewidth=.3) #facecolors="none"
                ax.axes.get_xaxis().get_label().set_visible(False) # remove x axis label

                ylab = ' '.join(feature.split('_')[:-2])
                if any(f in feature for f in ['length','width']):
                    ylab = r'{} ($\mu m$)'.format(ylab)
                elif 'speed' in feature:
                    ylab = r'{} ($\mu m/s$)'.format(ylab)
                elif 'area' in feature:
                    ylab = r'{} ($\mu m^2$)'.format(ylab)
                plt.ylabel(ylab, fontsize=18) #fontsize=15, labelpad=12

                if colour_by_date:
                    plt.xlim(right=len(order)-0.3)
                    plt.legend(loc='upper right', title='Date')
                #plt.title(feature.replace('_',' '), fontsize=18, pad=20)
                
                # Add p-value to plot
                for i, strain in enumerate(order[1:]):
                    pval = test_pvalues_df.loc[strain, feature]
                    text = ax.get_xticklabels()[i+1]
                    assert text.get_text() == strain
                    if ((isinstance(pval, float) or isinstance(pval, int)) and 
                        (pval < p_value_threshold or feature in feature_set)):
                        y = plot_df[feature].max() 
                        h = (y - plot_df[feature].min()) / 50
                        plt.plot([0, 0, i+1, i+1], [y+h, y+2*h, y+2*h, y+h], lw=1.5, c='k')
                        pval_text = 'P < 0.001' if pval < 0.001 else 'P = %.3f' % pval
                        ax.text((i+1)/2, y+2*h, pval_text, fontsize=12, ha='center', va='bottom')
                
                plt.subplots_adjust(left=0.15)
                # #Custom legend
                # patch_list = []
                # for l, key in enumerate(colour_dict.keys()):
                #     patch = patches.Patch(color=colour_dict[key], label=key)
                #     patch_list.append(patch)
                # plt.tight_layout(rect=[0.04, 0, 0.84, 0.96])
                # plt.legend(handles=patch_list, labels=colour_dict.keys(), loc=(1.02, 0.8),\
                #           borderaxespad=0.4, frameon=False, fontsize=15)
    
                # Save figure
                if saveDir:
                    plot_path = saveDir / str(strain) / ('{0}_'.format(f + 1) + feature + '.pdf')
                    plot_path.parent.mkdir(exist_ok=True, parents=True)
                    plt.savefig(plot_path, dpi=300)
                else:
                    plt.show(); plt.pause(2)
                     
def boxplots_grouped(feat_meta_df,
                     group_by,
                     control_group,
                     test_pvalues_df=None,
                     feature_set=None,
                     saveDir=None,
                     p_value_threshold=0.05,
                     drop_insignificant=False,
                     max_features_plot_cap=None, 
                     max_groups_plot_cap=None,
                     sns_colour_palette="tab10",
                     figsize=[8,12],
                     saveFormat=None,
                     **kwargs):
    """ Boxplots comparing all strains to control for each feature in feature set """
        
    if feature_set is not None:
        assert all(feat in feat_meta_df.columns for feat in feature_set)
        
        # Drop insignificant features
        if drop_insignificant and (test_pvalues_df is not None):
            feature_set = [feature for feature in feature_set if (test_pvalues_df[feature] < 
                                                                  p_value_threshold).any()]
        
        if max_features_plot_cap is not None and len(feature_set) > max_features_plot_cap:
            print("WARNING: Too many features to plot! Capping at %d plots"\
                  % max_features_plot_cap)
            feature_set = feature_set[:max_features_plot_cap]
    elif test_pvalues_df is not None:
        # Plot all sig feats between any strain and control
        feature_set = [feature for feature in test_pvalues_df.columns if
                       (test_pvalues_df[feature] < p_value_threshold).any()]
    else:
        raise IOError()
    
    # OPTIONAL: Plot cherry-picked features
    #feature_set = ['speed_50th','curvature_neck_abs_50th','angular_velocity_neck_abs_50th']
            
    # Seaborn boxplots with swarmplot overlay for each feature - saved to file
    plt.ioff() if saveDir else plt.ion()
    for f, feature in enumerate(tqdm(feature_set, position=0)):
        if test_pvalues_df is not None:
            sortedPvals = test_pvalues_df[feature].sort_values(ascending=True)
            strains2plt = list(sortedPvals.index)
        else:
            strains2plt = [s for s in list(feat_meta_df[group_by].unique()) if s != control_group]
        
        if max_groups_plot_cap is not None and len(strains2plt) > max_groups_plot_cap:
            print("Capping at %d strains" % max_groups_plot_cap)
            strains2plt = strains2plt[:max_groups_plot_cap]
            
        strains2plt.insert(0, control_group)
        plot_df = feat_meta_df[feat_meta_df[group_by].isin(strains2plt)]
        
        # Rank by median
        rankMedian = plot_df.groupby(group_by)[feature].median().sort_values(ascending=True)
        #plot_df = plot_df.set_index(group_by).loc[rankMedian.index].reset_index()
        
        if len(strains2plt) > 10:
            colour_dict = {strain: "r" if strain == control_group else \
                           "darkgray" for strain in plot_df[group_by].unique()}
            if drop_insignificant:
                colour_dict2 = {strain: "b" for strain in
                                list(sortedPvals[sortedPvals < p_value_threshold].index)}
                colour_dict.update(colour_dict2)
        else:
            colour_labels = sns.color_palette(sns_colour_palette, len(strains2plt))
            colour_dict = {key:col for (key,col) in zip(plot_df[group_by].unique(), colour_labels)}
        
        # Seaborn boxplot for each feature (only top strains)
        plt.close('all')
        plt.style.use(CUSTOM_STYLE) 
        sns.set_style('ticks')
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)
        sns.boxplot(x=feature, 
                    y=group_by,
                    data=plot_df, 
                    order=rankMedian.index,
                    showfliers=False,
                    meanline=False,
                    showmeans=True,
                    meanprops={"marker":"x","markersize":10,"markeredgecolor":"k"},
                    flierprops={"marker":"x","markersize":15,"markeredgecolor":"r"},
                    palette=colour_dict) # **kwargs
        ax.set_xlabel(feature.replace('_',' '), fontsize=18, labelpad=10)
        ax.axes.get_yaxis().get_label().set_visible(False) # remove y axis label
        #ax.set_ylabel(group_by, fontsize=18, labelpad=10)
        
        # Add p-value to plot
        #c_pos = np.where(rankMedian.index == control_group)[0][0]
        for i, strain in enumerate(rankMedian.index):
            if strain == control_group:
                plt.axvline(x=rankMedian[control_group], c='dimgray', ls='--')
                continue
            if test_pvalues_df is not None:
                pval = test_pvalues_df.loc[strain, feature]
                text = ax.get_yticklabels()[i]
                assert text.get_text() == strain
                if isinstance(pval, float) or isinstance(pval, int): # and pval < p_value_threshold
                    trans = transforms.blended_transform_factory(ax.transAxes, # x=scaled
                                                                 ax.transData) # y=none
                    text = 'P < 0.001' if pval < 0.001 else 'P = %.3f' % pval
                    ax.text(1.02, i, text, fontsize=12, ha='left', va='center', transform=trans)
        plt.subplots_adjust(right=0.85) #top=0.9,bottom=0.1,left=0.2

        # Save boxplot
        if saveDir:
            saveDir.mkdir(exist_ok=True, parents=True)
            saveFormat = saveFormat if saveFormat is not None else 'png'
            plot_path = Path(saveDir) / (str(f + 1) + '_' + feature + '.{}'.format(saveFormat))
            plt.savefig(plot_path, format=saveFormat, dpi=300)
        else:
            plt.show()

def plot_clustermap(featZ, 
                    meta, 
                    group_by,
                    col_linkage=None,
                    method='complete',
                    saveto=None,
                    figsize=[10,8],
                    sns_colour_palette="Pastel1"):
    """ Seaborn clustermap (hierarchical clustering heatmap) of normalised """                
    
    assert (featZ.index == meta.index).all()
    
    if type(group_by) != list:
        group_by = [group_by]
    n = len(group_by)
    if not (n == 1 or n == 2):
        raise IOError("Must provide either 1 or 2 'group_by' parameters")        
        
    # Store feature names
    fset = featZ.columns
        
    # Compute average value for strain for each feature (not each well)
    featZ_grouped = featZ.join(meta).groupby(group_by).mean().reset_index()
    
    var_list = list(featZ_grouped[group_by[0]].unique())

    # Row colors
    row_colours = []
    if len(var_list) > 1 or n == 1:
        var_colour_dict = dict(zip(var_list, sns.color_palette("tab10", len(var_list))))
        row_cols_var = featZ_grouped[group_by[0]].map(var_colour_dict)
        row_colours.append(row_cols_var)
    if n == 2:
        date_list = list(featZ_grouped[group_by[1]].unique())
        date_colour_dict = dict(zip(date_list, sns.color_palette("Blues", len(date_list))))
        #date_colour_dict=dict(zip(set(date_list),sns.hls_palette(len(set(date_list)),l=0.5,s=0.8)))
        row_cols_date = featZ_grouped[group_by[1]].map(date_colour_dict)
        row_cols_date.name = None
        row_colours.append(row_cols_date)  

    # Column colors
    bluelight_colour_dict = dict(zip(['prestim','bluelight','poststim'], 
                                     sns.color_palette(sns_colour_palette, 3)))
    feat_colour_dict = {f:bluelight_colour_dict[f.split('_')[-1]] for f in fset}
    
    # Plot clustermap
    plt.ioff() if saveto else plt.ion()
    plt.close('all')
    sns.set(font_scale=0.6)
    cg = sns.clustermap(data=featZ_grouped[fset], 
                        row_colors=row_colours,
                        col_colors=fset.map(feat_colour_dict),
                        #standard_scale=1, z_score=1,
                        col_linkage=col_linkage,
                        metric='euclidean', 
                        method=method,
                        vmin=-2, vmax=2,
                        figsize=figsize,
                        xticklabels=fset if len(fset) < 256 else False,
                        yticklabels=featZ_grouped[group_by].astype(str).agg('-'.join, axis=1),
                        cbar_pos=(0.98, 0.02, 0.05, 0.5), #None
                        cbar_kws={'orientation': 'horizontal',
                                  'label': None, #'Z-value'
                                  'shrink': 1,
                                  'ticks': [-2, -1, 0, 1, 2],
                                  'drawedges': False})
    cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_yticklabels(), rotation=0, 
                                  fontsize=15, ha='left', va='center')    
    #plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), fontsize=15)
    #cg.ax_heatmap.axes.set_xticklabels([]); cg.ax_heatmap.axes.set_yticklabels([])
    if len(fset) <= 256:
        plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    
    patch_list = []
    for l, key in enumerate(bluelight_colour_dict.keys()):
        patch = patches.Patch(color=bluelight_colour_dict[key], label=key)
        patch_list.append(patch)
    lg = plt.legend(handles=patch_list, 
                    labels=bluelight_colour_dict.keys(), 
                    title="Stimulus",
                    frameon=True,
                    loc='upper right',
                    bbox_to_anchor=(0.99, 0.99), 
                    bbox_transform=plt.gcf().transFigure,
                    fontsize=12, handletextpad=0.2)
    lg.get_title().set_fontsize(15)
    
    plt.subplots_adjust(top=0.98, bottom=0.02, 
                        left=0.02, right=0.9, 
                        hspace=0.01, wspace=0.01)
    #plt.tight_layout(rect=[0, 0, 1, 1], w_pad=0.5)
    
    # Add custom colorbar to right hand side
    # from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    # from mpl_toolkits.axes_grid1.colorbar import colorbar
    # # split axes of heatmap to put colorbar
    # ax_divider = make_axes_locatable(cg.ax_heatmap)
    # # define size and padding of axes for colorbar
    # cax = ax_divider.append_axes('right', size = '5%', pad = '2%')
    # # Heatmap returns an axes obj but you need to get a mappable obj (get_children)
    # colorbar(cg.ax_heatmap.get_children()[0], 
    #          cax = cax, 
    #          orientation = 'vertical', 
    #          ticks=[-2, -1, 0, 1, 2])
    # # locate colorbar ticks
    # cax.yaxis.set_ticks_position('right')
    
    # Save clustermap
    if saveto:
        saveto.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(saveto, dpi=300)
    else:
        plt.show()
    
    return cg

def plot_barcode_heatmap(featZ, 
                         meta, 
                         group_by,
                         pvalues_series=None,
                         p_value_threshold=0.05,
                         selected_feats=None,
                         figsize=[18,6],
                         saveto=None,
                         sns_colour_palette="Pastel1"):
    """  """
    
    assert set(featZ.index) == set(meta.index)
    if type(group_by) != list:
        group_by = [group_by]
        
    # Store feature names
    fset = featZ.columns
        
    # Compute average value for strain for each feature (not each well)
    featZ_grouped = featZ.join(meta).groupby(group_by).mean()#.reset_index()
       
    # Plot barcode clustermap
    plt.ioff() if saveto else plt.ion()
    plt.close('all')  
    # Make dataframe for heatmap plot
    heatmap_df = featZ_grouped[fset]
    
    var_list = list(heatmap_df.index)
    
    if pvalues_series is not None:
        assert all(f in fset for f in pvalues_series.index)
        
        heatmap_df = heatmap_df.append(-np.log10(pvalues_series[fset].astype(float)))
    
    # Map colors for stimulus type
    _stim = pd.DataFrame(data=[f.split('_')[-1] for f in fset], columns=['Stimulus'])
    _stim['Stimulus'] = _stim['Stimulus'].map({'prestim':1,'bluelight':2,'poststim':3})
    _stim = _stim.transpose().rename(columns={c:v for c,v in enumerate(fset)})
    heatmap_df = heatmap_df.append(_stim)
    
    # Add barcode - asterisk (*) to highlight selected features
    cm=list(np.repeat('inferno',len(var_list)))
    cm.extend(['Greys', sns_colour_palette])
    vmin_max = [(-2,2) for i in range(len(var_list))]
    vmin_max.extend([(0,20), (1,3)])
    plt.style.use(CUSTOM_STYLE) 
    sns.set_style('ticks')
    
    f = plt.figure(figsize= (20,len(var_list)+1))
    height_ratios = list(np.repeat(3,len(var_list)))
    height_ratios.extend([1,1])
    gs = GridSpec(len(var_list)+2, 1, wspace=0, hspace=0, height_ratios=height_ratios)
    cbar_ax = f.add_axes([.885, .275, .01, .68]) #  [left, bottom, width, height]
    
    # Stimulus colors
    stim_order = ['prestim','bluelight','poststim']
    bluelight_colour_dict = dict(zip(stim_order, sns.color_palette(sns_colour_palette, 3)))
        
    for n, ((ix, r), c, v) in enumerate(zip(heatmap_df.iterrows(), cm, vmin_max)):
        if n > len(var_list):
            c = sns.color_palette(sns_colour_palette,3)
        if type(ix) == tuple:
            ix = ' - '.join((str(i) for i in ix))
        axis = f.add_subplot(gs[n])
        sns.heatmap(r.to_frame().transpose().astype(float),
                    yticklabels=[ix],
                    xticklabels=[],
                    ax=axis,
                    cmap=c,
                    cbar=n==0, #only plots colorbar for first plot
                    cbar_ax=None if n else cbar_ax,
                    #cbar_kws={'shrink':0.8},
                    vmin=v[0], vmax=v[1])
        plt.yticks(rotation=0, fontsize=20)
        plt.ylabel("")
        #cbar_ax.set_yticklabels(labels = cbar_ax.get_yticklabels())#, fontdict=font_settings)
        # if n < len(var_list):
        #     axis.set_yticklabels(labels=(str(ix[0])+' - '+str(ix[1])), rotation=0)
        #axis.set_yticklabels(labels=[ix], rotation=0, fontsize=15)
        # if n == len(heatmap_df)-1:
        #     axis.set_yticklabels(labels=[ix], rotation=0) # fontsize=15
            
        patch_list = []
        for l, key in enumerate(bluelight_colour_dict.keys()):
            patch = patches.Patch(color=bluelight_colour_dict[key], label=key)
            patch_list.append(patch)
        lg = plt.legend(handles=patch_list, 
                        labels=bluelight_colour_dict.keys(), 
                        title="Stimulus",
                        frameon=False,
                        loc='lower right',
                        bbox_to_anchor=(0.99, 0.05), #(0.99,0.6)
                        bbox_transform=plt.gcf().transFigure,
                        fontsize=12, handletextpad=0.2)
        lg.get_title().set_fontsize(15)

        plt.subplots_adjust(top=0.95, bottom=0.05, 
                            left=0.08*len(group_by), right=0.88, 
                            hspace=0.01, wspace=0.01)
        #f.tight_layout(rect=[0, 0, 0.89, 1], w_pad=0.5)
    
    if selected_feats is not None:
        if len(selected_feats) > 0:
            for feat in selected_feats:
                try:
                    axis.text(heatmap_df.columns.get_loc(feat), 1, ' *', ha='center')
                except KeyError:
                    print('{} not in featureset'.format(feat))

    if saveto:
        saveto.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(saveto, dpi=600)
    else:
        plt.show()

def pcainfo(pca, zscores, PC=0, n_feats2print=10):
    """ A function to plot PCA explained variance, and print the most 
        important features in the given principal component (P.C.)
    """
        
    cum_expl_var_frac = np.cumsum(pca.explained_variance_ratio_)

    # Plot explained variance
    plt.style.use(CUSTOM_STYLE) 
    sns.set_style('ticks')
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
             control=None,
             saveDir=None,
             PCs_to_keep=10,
             n_feats2print=10,
             sns_colour_palette="tab10",
             hypercolor=False):
    """ Perform principal components analysis 
        - group_by : column in metadata to group by for plotting (colours) 
        - n_dims : number of principal component dimensions to plot (2 or 3)
        - var_subset : subset list of categorical names in featZ[group_by]
        - saveDir : directory to save PCA results
        - PCs_to_keep : number of PCs to project
        - n_feats2print : number of top features influencing PCs to store 
    """
    
    assert (featZ.index == meta.index).all()
    if var_subset is not None:
        assert all([strain in meta[group_by].unique() for strain in var_subset])
    else:
        var_subset = list(meta[group_by].unique())
              
    # Perform PCA on extracted features
    print("\nPerforming Principal Components Analysis (PCA)...")

    # Fit the PCA model with the normalised data
    pca = PCA() # OPTIONAL: pca = PCA(n_components=n_dims) 
    pca.fit(featZ)

    # Plot summary data from PCA: explained variance (most important features)
    plt.ioff() if saveDir else plt.ion()
    important_feats, fig = pcainfo(pca=pca, 
                                   zscores=featZ, 
                                   PC=0, 
                                   n_feats2print=n_feats2print)
    if saveDir:
        # Save plot of PCA explained variance
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

    # Compute explained variance ratio of component axes
    ex_variance=np.var(projected, axis=0) # PCA(n_components=n_dims).fit_transform(featZ)
    ex_variance_ratio = ex_variance/np.sum(ex_variance)
    
    # Store the results for first few PCs in dataframe
    projected_df = pd.DataFrame(data=projected[:,:PCs_to_keep],
                                columns=['PC' + str(n+1) for n in range(PCs_to_keep)],
                                index=featZ.index)
    
    # Create colour palette for plot loop
    if len(var_subset) > 10:
        if not control:
            raise IOError('Too many groups for plot color mapping!' + 
                          'Please provide a control group or subset of groups (n<10) to color plot')
        elif hypercolor:
            # Recycle palette colours to make up to number of groups
            print("\nWARNING: Multiple groups plotted with the same colour (too many groups)")
            colour_labels = sns.color_palette(sns_colour_palette, len(var_subset))
            palette = dict(zip(var_subset, colour_labels))           
        else:
            # Colour the control and make the rest gray
            palette = {var : "blue" if var == control else "darkgray" 
                       for var in meta[group_by].unique()}
            
    elif len(var_subset) <= 10:
        # Colour strains of interest
        colour_labels = sns.color_palette(sns_colour_palette, len(var_subset))
        palette = dict(zip(var_subset, colour_labels))
        
        if set(var_subset) != set(meta[group_by].unique()):
            # Make the rest gray
            gray_strains = [var for var in meta[group_by].unique() if var not in var_subset]
            gray_palette = {var:'darkgray' for var in gray_strains}
            palette.update(gray_palette)
            
    plt.close('all')
    plt.style.use(CUSTOM_STYLE) 
    plt.rcParams['legend.handletextpad'] = 0.5
    sns.set_style('ticks')
    if n_dims == 2:
        fig, ax = plt.subplots(figsize=[9,8])
        
        grouped = meta.join(projected_df).groupby(group_by)
        for key, group in grouped:
            group.plot(ax=ax, 
                       kind='scatter',
                       x='PC1', 
                       y='PC2', 
                       label=key, 
                       color=palette[key])
            
        if len(var_subset) <= 10:
            sns.kdeplot(x='PC1', 
                        y='PC2', 
                        data=meta.join(projected_df), 
                        hue=group_by, 
                        palette=palette,
                        fill=True, # TODO: Fill kde plot with plain colour by group
                        alpha=0.25,
                        thresh=0.05,
                        levels=2,
                        bw_method="scott", 
                        bw_adjust=1)        
            
        ax.set_xlabel('Principal Component 1 (%.1f%%)' % (ex_variance_ratio[0]*100), 
                      fontsize=20, labelpad=12)
        ax.set_ylabel('Principal Component 2 (%.1f%%)' % (ex_variance_ratio[1]*100), 
                      fontsize=20, labelpad=12)
        #ax.set_title("PCA by '{}'".format(group_by), fontsize=20)
        
        if len(var_subset) <= 10:
            plt.tight_layout() # rect=[0.04, 0, 0.84, 0.96]
            ax.legend(var_subset, frameon=True, loc='upper right', fontsize=15, markerscale=1.5)
        elif hypercolor:
            ax.get_legend().remove()
        else:
            control_patch = patches.Patch(color='blue', label=control)
            other_patch = patches.Patch(color='darkgray', label='other')
            ax.legend(handles=[control_patch, other_patch])
        ax.grid(False)
        
    elif n_dims == 3:
        fig = plt.figure(figsize=[10,10])
        mpl_axes_logger.setLevel('ERROR') # Work-around for 3D plot colour warnings
        ax = Axes3D(fig) # ax = fig.add_subplot(111, projection='3d')
                
        for g_var in var_subset:
            g_var_projected_df = projected_df[meta[group_by]==g_var]
            ax.scatter(xs=g_var_projected_df['PC1'], 
                       ys=g_var_projected_df['PC2'], 
                       zs=g_var_projected_df['PC3'],
                       zdir='z', s=30, c=palette[g_var], depthshade=False)
        ax.set_xlabel('Principal Component 1 (%.1f%%)' % (ex_variance_ratio[0]*100),
                      fontsize=15, labelpad=12)
        ax.set_ylabel('Principal Component 2 (%.1f%%)' % (ex_variance_ratio[1]*100), 
                      fontsize=15, labelpad=12)
        ax.set_zlabel('Principal Component 3 (%.1f%%)' % (ex_variance_ratio[2]*100),
                      fontsize=15, labelpad=12)
        #ax.set_title("PCA by '{}'".format(group_by), fontsize=20)
        if len(var_subset) <= 15:
            ax.legend(var_subset, frameon=True, fontsize=12)
        ax.grid(False)
    else:
        raise ValueError("Value for 'n_dims' must be either 2 or 3")

    # Save PCA plot
    if saveDir:
        pca_path = Path(saveDir) / ('pca_by_{}'.format(group_by) + 
                                    ('_colour' if hypercolor else '') + 
                                    ('.png' if n_dims == 3 else '.pdf'))
        plt.savefig(pca_path, format='png' if n_dims == 3 else 'pdf', 
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

def find_outliers_mahalanobis(featMatProjected, 
                              extremeness=2., 
                              figsize=[8,8], 
                              saveto=None):
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
    plt.style.use(CUSTOM_STYLE) 
    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('#F7FFFF')
    plt.scatter(np.array(projectedTable['PC1']), 
                np.array(projectedTable['PC2']), 
                c=MahalanobisDist) # colour PCA by Mahalanobis distance
    plt.title('Mahalanobis Distance for Outlier Detection', fontsize=20)
    plt.colorbar()
    ax.grid(False)
    
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
              figsize=[8,8],
              sns_colour_palette="tab10"):
    """ t-distributed stochastic neighbour embedding """
    
    assert (meta.index == featZ.index).all()
    assert type(perplexities) == list
    
    if var_subset is not None:
        assert all([strain in meta[group_by].unique() for strain in var_subset])
    else:
        var_subset = list(meta[group_by].unique())    
        
    print("\nPerforming t-distributed stochastic neighbour embedding (t-SNE)")
    for perplex in tqdm(perplexities, position=0):
        # 2-COMPONENT t-SNE
        tSNE_embedded = TSNE(n_components=n_components, 
                             init='random', 
                             random_state=42,\
                             perplexity=perplex, 
                             n_iter=3000).fit_transform(featZ)
        tSNE_df = pd.DataFrame(tSNE_embedded, 
                               columns=['tSNE_1','tSNE_2']).set_index(featZ.index)
        
        plt.ioff() if saveDir else plt.ion()
        plt.close('all')
        plt.style.use(CUSTOM_STYLE) 
        sns.set_style('ticks')
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('tSNE Component 1', fontsize=15, labelpad=12)
        ax.set_ylabel('tSNE Component 2', fontsize=15, labelpad=12)
        #ax.set_title('2-component tSNE (perplexity={0})'.format(perplex), fontsize=20)
        
        # Create colour palette for plot loop
        palette = dict(zip(var_subset, (sns.color_palette(sns_colour_palette, len(var_subset)))))
        
        for var in var_subset:
            tSNE_var = tSNE_df[meta[group_by]==var]
            sns.scatterplot(x='tSNE_1', y='tSNE_2', data=tSNE_var, color=palette[var], s=100)
        if len(var_subset) <= 15:
            plt.tight_layout() #rect=[0.04, 0, 0.84, 0.96]
            ax.legend(var_subset, frameon=True, loc='upper right', fontsize=15, markerscale=1.5)
        ax.grid(False)   
        
        if saveDir:
            saveDir.mkdir(exist_ok=True, parents=True)
            savePath = Path(saveDir) / 'tSNE_perplex={0}.pdf'.format(perplex)
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
              figsize=[8,8],
              sns_colour_palette="tab10"):
    """ Uniform manifold projection """
    
    assert (meta.index == featZ.index).all()
    assert type(n_neighbours) == list
    
    if var_subset is not None:
        assert all([strain in meta[group_by].unique() for strain in var_subset])
    else:
        var_subset = list(meta[group_by].unique())    

    print("\nPerforming uniform manifold projection (UMAP)")
    for n in tqdm(n_neighbours, position=0):
        UMAP_projection = umap.UMAP(n_neighbors=n,
                                    min_dist=min_dist,
                                    metric='correlation').fit_transform(featZ)
        
        UMAP_projection_df = pd.DataFrame(UMAP_projection, 
                                          columns=['UMAP_1', 'UMAP_2']).set_index(featZ.index)
        UMAP_projection_df.name = 'n={}'.format(str(n))
        
        # Plot 2-D UMAP
        plt.close('all')
        plt.style.use(CUSTOM_STYLE) 
        sns.set_style('ticks')
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('UMAP Component 1', fontsize=15, labelpad=12)
        ax.set_ylabel('UMAP Component 2', fontsize=15, labelpad=12)
        #ax.set_title('2-component UMAP (n_neighbours={0})'.format(n), fontsize=20)
                
        # Create colour palette for plot loop
        palette = dict(zip(var_subset, (sns.color_palette(sns_colour_palette, len(var_subset)))))
        
        # Plot UMAP projection
        for var in var_subset:
            UMAP_var = UMAP_projection_df[meta[group_by]==var]
            sns.scatterplot(x='UMAP_1', y='UMAP_2', data=UMAP_var, color=palette[var], s=100)
        if len(var_subset) <= 15:
            plt.tight_layout() #rect=[0.04, 0, 0.84, 0.96]
            ax.legend(var_subset, frameon=True, loc='upper right', fontsize=15, markerscale=1.5)
        ax.grid(False)
        
        if saveDir:
            saveDir.mkdir(exist_ok=True, parents=True)
            savePath = Path(saveDir) / 'UMAP_n_neighbours={0}.pdf'.format(n)
            plt.savefig(savePath, tight_layout=True, dpi=300)
        else:
            plt.show(); plt.pause(2)
        
    return UMAP_projection_df
