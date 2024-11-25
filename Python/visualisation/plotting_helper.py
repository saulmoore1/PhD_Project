#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper Functions

@author: sm5911
@date: 21/11/2020

"""

#import scipy.spatial as sp
#import scipy.cluster.hierarchy as hc

#%% Globals

CUSTOM_STYLE = 'visualisation/style_sheet_20210126.mplstyle'

#%% Functions

def sig_asterix(pvalues_array, ns=True):
    """ Convert p-values to asterisks for showing significance on plots 
        
        Parameters
        ----------
        pvalues_array : list, np.array, pd.Series
            1-D vector of p-values to convert to asterisk significance notation
            
        ns : bool
            True returns 'ns' for non-significant p-values, False returns an empty string ''
            
        Returns
        -------
        Asterisk significances for each element in p-value array
    """
    asterix = []
    for p in pvalues_array:
        if p < 0.001:
            asterix.append('***')
        elif p < 0.01:
            asterix.append('**')
        elif p < 0.05:
            asterix.append('*')
        else:
            if ns:
                asterix.append('n.s.')
            else:
                asterix.append('')
    return asterix

def hexcolours(n):
    """ Generate a list of n hexadecimal colours for plotting """
    
    import colorsys
    
    hex_list = []
    HSV = [(x*1/n,0.5,0.5) for x in range(n)]
    # Generate RGB hex code
    for RGB in HSV:
        RGB = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*RGB))
        hex_list.append('#%02x%02x%02x' % tuple(RGB)) 
        
    return(hex_list)

def hex2rgb(hex):
    """ Convert from hexadecimal to RGB colour format for plotting """
    
    hex = hex.lstrip('#')
    hlen = len(hex)
    
    return tuple(int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))

def plot_points(fig, ax, x, y, **kwargs):
    """ Plot points onto an image """
    
    from matplotlib import pyplot as plt

    ax.plot(x, y, **kwargs)
    plt.show()
    
    return(fig, ax)
        
def plot_pie(df, rm_empty=True, show=True, **kwargs):
    """ Plot pie chart from a labelled vector of values that sum to 1 """
    
    from matplotlib import pyplot as plt

    if rm_empty: # Remove any empty rows
        df = df.loc[df!=0]
    fig = plt.pie(df, autopct='%1.1f%%', **kwargs)
    plt.axis('equal')
    plt.tight_layout()
    
    if show:
        plt.show()
        
    return(fig)

def plot_day_variation(feat_df,
                       meta_df,
                       group_by,
                       control,
                       test_pvalues_df=None,
                       day_var='date_yyyymmdd',
                       feature_set=None,
                       max_feats2plt=None,
                       p_value_threshold=0.05,
                       saveDir=None,
                       figsize=[6,6],
                       sns_colour_palette="tab10",
                       dodge=False,
                       ranked=True,
                       drop_insignificant=True):
    """ """
    
    import numpy as np
    import seaborn as sns
    from tqdm import tqdm
    from pathlib import Path
    from matplotlib import pyplot as plt

    if feature_set is not None:
        assert all(f in feat_df.columns for f in feature_set)
    else:
        feature_set = [f for f in feat_df.columns]
    
    if max_feats2plt is not None:
        feature_set = feature_set[:max_feats2plt]
    
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
    
    import numpy as np
    from pathlib import Path
    from matplotlib import pyplot as plt
    
    if test_pvalues_df is not None:
        # Transpose dataframe axes to make strain name the index, and features as columns
        test_pvalues_df = test_pvalues_df.T

        # Proportion of features significantly different from control
        n_sigfeats = (test_pvalues_df < p_value_threshold).sum(axis=1)
        prop_sigfeats = (n_sigfeats / len(test_pvalues_df.columns)) * 100
        prop_sigfeats = prop_sigfeats.sort_values(ascending=False)
        
        n = len(prop_sigfeats.index)
        
        # Plot proportion significant features for each strain
        plt.ioff() if saveDir else plt.ion()
        plt.close('all')
        plt.style.use(CUSTOM_STYLE)
        fig = plt.figure(figsize=[6, n/4 if (n > 20 and n < 1000) else 7]) # width, height
        ax = fig.add_subplot(1,1,1)
        prop_sigfeats.plot.barh(x=prop_sigfeats.index,
                                y=prop_sigfeats.values,
                                color='gray',
                                ec='black') # fc
        ax.set_xlabel('% significant features', fontsize=16, labelpad=10)
        plt.xlim(0,100)
        
        y_arr = np.array([l.get_text() for l in ax.get_yticklabels()])
        
        for i, (l, v) in enumerate(n_sigfeats.items()):
            y = int(np.where(y_arr == l)[0])
            ax.text(prop_sigfeats.loc[l] + 2, y, str(v), color='k',
                    va='center', ha='left') #fontweight='bold'
        plt.text(0.85, 1, 'n = %d' % len(test_pvalues_df.columns), ha='center', va='bottom',
                 transform=ax.transAxes)
        plt.tight_layout(rect=[0.02, 0.02, 0.96, 1])
    
        if saveDir:
            Path(saveDir).mkdir(exist_ok=True, parents=True)
            savePath = Path(saveDir) / ('percentage_sigfeats_{}.png'.format(test_name) if test_name 
                                        is not None else 'percentage_sigfeats.png')
            print("Saving figure: %s" % savePath.name)
            plt.savefig(savePath, dpi=300)
        else:
            plt.show()

def errorbar_sigfeats(features, metadata, group_by, fset, control=None, rank_by='median', 
                      highlight_subset=None, max_feats2plt=10, figsize=[130,6], fontsize=4, 
                      tight_layout=None, color='dimgray', saveDir=None, saveName=None, 
                      highlight_colour='red', **kwargs):
    """ Plot mean feature value with errorbars (+/- 1.98 * std) for all groups in 
        metadata['group_by'] for each feature in feature set provided 
    """
    
    import numpy as np
    from pathlib import Path
    from tqdm import tqdm
    from matplotlib import pyplot as plt
    from scipy.stats import sem
    
    plt.ioff() if saveDir is not None else plt.ion()
    
    if highlight_subset is not None:
        assert all(s in metadata[group_by].unique() for s in highlight_subset)
    
    # Boxplots of significant features by ANOVA/LMM (all groups)
    grouped = metadata[[group_by]].join(features).groupby(group_by)
    
    mean_strain = grouped.mean()
    median_strain = grouped.median()
            
    max_feats2plt = len(fset) if max_feats2plt is None else max_feats2plt
    
    # Plot all strains (ranked by median) for top n significant features (ranked by ANOVA p-value)
    for f, feat in enumerate(tqdm(fset[:max_feats2plt])):
        # Errorbar plot
        if rank_by == 'median':
            order = median_strain[feat].sort_values(ascending=True).index.to_list()
            df_ordered = median_strain.reindex(order).reset_index(drop=False)[[group_by, feat]]  
        elif rank_by == 'mean':
            order = mean_strain[feat].sort_values(ascending=True).index.to_list()
            df_ordered = mean_strain.reindex(order).reset_index(drop=False)[[group_by, feat]]  

        df_ordered['error'] = [sem(features.loc[metadata[group_by]==strain, feat]) for strain in order]
        #error = [1.98 * features.loc[metadata[group_by]==strain, feat].std() for strain in order]

        if control is not None:
            df_ordered['colour'] = ['blue' if s == control else 'grey' for s in df_ordered[group_by]]
        else:
            df_ordered['colour'] = ['grey' for s in df_ordered[group_by]]

        if highlight_subset is not None:
            df_ordered.loc[np.where(df_ordered[group_by].isin(highlight_subset))[0],'colour'] = highlight_colour
                    
        plt.close('all')
        fig, ax = plt.subplots(figsize=figsize)
        ax.errorbar(x=group_by,
                    y=feat, 
                    yerr='error',
                    color='grey',
                    data=df_ordered, 
                    **kwargs)
        
        idxs = np.where(df_ordered['colour']!='grey')[0]
        values = df_ordered.loc[idxs,feat].values
        errors = df_ordered.loc[idxs,'error'].values
        colours = df_ordered.loc[idxs,'colour'].values
            
        for pos, y, err, colour in zip(idxs, values, errors, colours):
            ax.errorbar(pos, y, err, color=colour)
                
        _ = plt.xticks(rotation=90, ha='center', fontsize=fontsize, color=df_ordered['colour'])
        _ = [t.set_color(i) for (i,t) in zip(df_ordered['colour'], ax.xaxis.get_ticklabels())]
        #ax.tick_params(axis="x", color=colour)
                
        if rank_by == 'median':
            plt.axhline(median_strain.loc[control, feat], c='dimgray', ls='--')
            #med_of_med = median_ordered.median() # FOR PLOTTING MEDIAN OF MEDIANS
            #plt.axhline(med_of_med, c='', ls='--')
        elif rank_by == 'mean':
            plt.axhline(mean_strain.loc[control, feat], c='dimgray', ls='--')
            
        # plt.tick_params(
        #     axis='x',          # changes apply to the x-axis
        #     which='both',      # both major and minor ticks are affected
        #     bottom=False,      # ticks along the bottom edge are off
        #     top=False,         # ticks along the top edge are off
        #     labelbottom=True)  # labels along the bottom edge are off
        plt.title(feat, pad=10)
        
        if tight_layout is not None:
            plt.tight_layout(rect=tight_layout)
             
        if saveDir is not None:
            Path(saveDir).mkdir(exist_ok=True, parents=True)
            plt.savefig(Path(saveDir) / (saveName if saveName is not None else 
                                         (str(f + 1) + '_' + feat + '_errorbar.pdf')))
        
    return
    
def boxplots_sigfeats(features,
                      y_class,
                      control,
                      pvals,
                      z_class=None,
                      feature_set=None,
                      saveDir=None,
                      drop_insignificant=True,
                      p_value_threshold=0.05,
                      max_sig_feats=10,
                      max_strains=None,
                      sns_colour_palette="tab10",
                      verbose=True,
                      scale_outliers=False,
                      ylim_minmax=None,
                      append_ranking_fname=True):
    """ Box plots of most significantly different features between each strain and the control 
    
        Inputs
        ------
        features
        y_class
        control
        pvals
        feature_set
        saveDir
        drop_insignificant
        p_value_threshold
        sns_colour_palette
        colour_by
        verbose
    
    """    
        
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from tqdm import tqdm
    from matplotlib import pyplot as plt
    from matplotlib import transforms

    # features
    if feature_set is not None:
        if not type(feature_set) == list:
            try:
                feature_set = list(feature_set)
            except:
                raise IOError("Please provide selected features as a list")
        assert all(f in features.columns for f in feature_set)
        features = features[feature_set]
        pvals = pvals.loc[feature_set]

    assert saveDir is not None
    assert all(f in features.columns for f in pvals.index)
                    
    # Strain list
    if drop_insignificant:
        # Rank strains by number of sigfeats by t-test 
        ranked_nsig = (pvals < p_value_threshold).sum(axis=0).sort_values(ascending=False)

        # Select strains with at least one significant feature
        strain_list = ranked_nsig[ranked_nsig > 0].index.to_list()
    else:
        # Rank strain by minumum p-value (any feature)
        ranked_min_pval = pvals.min(axis=0).sort_values(ascending=True) 
        strain_list = ranked_min_pval.index.to_list()
        
    strain_list = [s for s in strain_list if s in list(y_class.unique())]
    
    max_sig_feats = features.shape[1] if max_sig_feats is None else max_sig_feats

    data = pd.concat([y_class, features], axis=1)
    
    if z_class is not None:
        data = pd.concat([z_class, data], axis=1)

    # Top10 features for each strain vs control    
    for s, strain in enumerate(tqdm(strain_list[:max_strains], position=0)):

        strain_pvals = pvals[strain]

        # drop NaNs + rank features by min pval
        strain_pvals = strain_pvals.dropna(axis=0)
        strain_pvals = strain_pvals.sort_values(ascending=True) # rank p-values (top features for strain)
                
        # drop insignificant features
        if drop_insignificant:
            strain_pvals = strain_pvals.loc[strain_pvals < p_value_threshold]
            if verbose:
                print("%d significant features found for %s" % (len(strain_pvals.index), str(strain)))

        if strain_pvals.shape[0] > max_sig_feats and verbose:
            print("\tPlotting only top %d features for %s\n" % (max_sig_feats, str(strain)))

        # Subset for strain + control only
        strain_data = data[np.logical_or(data[y_class.name]==control, data[y_class.name]==strain)]    
        
        # Create colour palette
        colour_labels = sns.color_palette(sns_colour_palette, 2)
        colour_dict = {control:colour_labels[0], str(strain):colour_labels[1]}
        
        if z_class is not None:
            cols = sns.color_palette("Greens", len(z_class.unique()))
            col_dict = dict(zip(list(z_class.unique()), cols))   
            
        order = list(strain_data[y_class.name].unique())
        order.remove(control)
        order.insert(0, control)
                                              
        # Boxplots of control vs test-strain for each top-ranked significant feature
        for f, feature in enumerate(strain_pvals.index.to_list()[:max_sig_feats]):
            
            plot_path = saveDir / (('{0}_'.format(s+1) if append_ranking_fname else '')+str(strain))/\
                        (('{0}_'.format(f+1) if append_ranking_fname else '')+feature+'.pdf')
            if plot_path.exists():
                continue
            else:
                plot_path.parent.mkdir(exist_ok=True, parents=True)
    
                plt.close('all')
                plt.style.use(CUSTOM_STYLE) 
                sns.set_style('ticks')
                fig = plt.figure(figsize=[10,8])
                ax = fig.add_subplot(1,1,1)
                sns.boxplot(x=y_class.name, 
                            y=feature, 
                            data=strain_data, 
                            order=order,
                            palette=colour_dict,
                            showfliers=False, 
                            showmeans=True if z_class is not None else False,
                            #meanline=True,
                            meanprops={"marker":"x", 
                                       "markersize":5,
                                       "markeredgecolor":"k"},
                            flierprops={"marker":"x", 
                                        "markersize":15, 
                                        "markeredgecolor":"r"})
                sns.stripplot(x=y_class.name, 
                              y=feature, 
                              data=strain_data,
                              s=10,
                              order=order,
                              hue=z_class if z_class is not None else None,
                              palette=col_dict if z_class is not None else None,
                              color=None if z_class is not None else 'gray',
                              marker=".",
                              edgecolor='k',
                              linewidth=.3) #facecolors="none"
                ax.axes.get_xaxis().get_label().set_visible(False) # remove x axis label
                # plt.ylabel(feature.replace('_',' '), fontsize=18, labelpad=25)
                plt.ylabel("")
                plt.title(feature.replace('_',' '), fontsize=18, pad=30)
    
                if z_class is not None:
                    plt.xlim(right=len(order)-0.3)
                    plt.legend(loc='upper right')
     
                if scale_outliers:
                    grouped_strain = strain_data.groupby(y_class.name)
                    y_bar = grouped_strain[feature].median() # median is less skewed by outliers
                    # compute IQR
                    Q1 = grouped_strain[feature].quantile(0.25)
                    Q3 = grouped_strain[feature].quantile(0.75)
                    IQR = Q3 - Q1
                    plt.ylim(min(y_bar) - 2.5 * max(IQR), max(y_bar) + 2.5 * max(IQR))
    
                #XXX ylim_minmax for speed only
                elif ylim_minmax is not None and 'speed_50th' in feature: 
                    assert isinstance(ylim_minmax, tuple)
                    plt.ylim(ylim_minmax[0], ylim_minmax[1])
                # elif 'motion_mode' in feature:
                #     plt.ylim(0, 1)
                    
                # Add p-value to plot
                p = strain_pvals.loc[feature]
                text = ax.get_xticklabels()[-1]
                assert text.get_text() == strain
                
                trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled
                plt.plot([0, 0, 1, 1], [0.98, 0.99, 0.99, 0.98], lw=1.5, c='k', transform=trans)
                p_text = 'P < 0.001' if p < 0.001 else 'P = %.3f' % p
                ax.text(0.5, 1, p_text, fontsize=12, ha='center', va='bottom', transform=trans)
                plt.subplots_adjust(left=0.15)
                
            # Save figure
            if saveDir:
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
                     max_feats2plt=None, 
                     max_groups_plot_cap=None,
                     sns_colour_palette="tab10",
                     figsize=[8,12],
                     **kwargs):
    """ Boxplots comparing all strains to control for each feature in feature set """
    
    import seaborn as sns
    from pathlib import Path
    from tqdm import tqdm
    from matplotlib import pyplot as plt
    from matplotlib import transforms
    
    if feature_set is not None:
        assert all(feat in feat_meta_df.columns for feat in feature_set)
        
        # Drop insignificant features
        if drop_insignificant and (test_pvalues_df is not None):
            feature_set = [feature for feature in feature_set if (test_pvalues_df[feature] < 
                                                                  p_value_threshold).any()]
        
        if max_feats2plt is not None and len(feature_set) > max_feats2plt:
            print("\nWARNING: Too many features to plot! Capping at %d plots"\
                  % max_feats2plt)
            feature_set = feature_set[:max_feats2plt]
    elif test_pvalues_df is not None:
        # Plot all sig feats between any strain and control
        feature_set = [feature for feature in test_pvalues_df.columns if
                       (test_pvalues_df[feature] < p_value_threshold).any()]
        test_pvalues_df.index = [i.replace("pvals_","") for i in test_pvalues_df.index]
    
    # OPTIONAL: Plot cherry-picked features
    #feature_set = ['speed_50th','curvature_neck_abs_50th','angular_velocity_neck_abs_50th']
            
    # Seaborn boxplots for each feature - saved to file
    plt.ioff() if saveDir else plt.ion()
    for f, feature in enumerate(tqdm(feature_set, position=0)):
        if test_pvalues_df is not None:
            sortedPvals = test_pvalues_df[feature].sort_values(ascending=True)
            strains2plt = list(sortedPvals.index)
        else:
            strains2plt = [s for s in list(feat_meta_df[group_by].unique()) if s != control_group]
        
        if max_groups_plot_cap is not None and len(strains2plt) > max_groups_plot_cap:
            print("\nCapping at %d strains" % max_groups_plot_cap)
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
                    meanprops={"marker":"x","markersize":3,"markeredgecolor":"k"},
                    flierprops={"marker":"x","markersize":3,"markeredgecolor":"r"},
                    palette=colour_dict) # **kwargs
        ax.set_xlabel(feature.replace('_',' '), fontsize=18, labelpad=10)
        ax.axes.get_yaxis().get_label().set_visible(False) # remove y axis label
        _, ylabels = plt.yticks()
        ax.set_yticklabels(ylabels, size=3)
        #ax.set_ylabel(group_by.replace('_',' '), fontsize=18, labelpad=10)
        
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
                    ax.text(1.02, i, text, fontsize=3, ha='left', va='center', transform=trans)
        plt.subplots_adjust(right=0.85) #top=0.9,bottom=0.1,left=0.2

        # Save boxplot
        if saveDir:
            saveDir.mkdir(exist_ok=True, parents=True)
            plot_path = Path(saveDir) / (str(f + 1) + '_' + feature + '.pdf')
            plt.savefig(plot_path)
        else:
            plt.show()

def all_in_one_boxplots(metadata,
                        features,
                        group_by='worm_strain',
                        hue=None,
                        control='N2',
                        control_hue=None,
                        save_dir=None,
                        ttest_path=None,
                        feature_set=None,
                        pvalue_threshold=0.05,
                        sigasterix=False,
                        order=None,
                        hue_order=None,
                        colour_dict=None,
                        override_palette_dict=None,
                        figsize=(30,6),
                        ylim_minmax=None,
                        vline_boxpos=None,
                        fontsize=15,
                        legend=True,
                        subplots_adjust={'bottom':0.32,'top':0.95,'left':0.05,'right':0.98}):

    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    from matplotlib import transforms
        
    # load t-test results for window
    if ttest_path is not None:
        ttest_df = pd.read_csv(ttest_path, header=0, index_col=0)
        pvals = ttest_df[[c for c in ttest_df.columns if 'pval' in c]]
        pvals.columns = [c.replace('pvals_','') for c in pvals.columns]
     
    strain_list = sorted(metadata[group_by].unique())
        
    if order is None:
        strain_list = [control] + [s for s in strain_list if s != control]
    else: 
        assert all(s in strain_list for s in order)
        strain_list = order
        
    if hue_order is None and hue is not None:    
        hue_order = sorted(metadata[hue].unique())
        assert len(hue_order) == 2
        hue_order = [control_hue] + [h for h in hue_order if h != control_hue]
    elif hue is not None:
        assert all(h in sorted(metadata[hue].unique()) for h in hue_order)
        
    if feature_set is not None:
        assert all(f in features.columns for f in feature_set)
    else:
        feature_set = list(features.columns)
    
    # all-in-one boxplots
    for feat in feature_set:
        save_path = save_dir / '{0}_{1}.svg'.format(group_by, feat)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        
        plt.close('all')
        sns.set_theme(style='white')
        fig, ax = plt.subplots(figsize=figsize)
        plot_df = metadata.join(features[[feat]])
        sns.boxplot(x=group_by, 
                    y=feat,
                    hue=hue,
                    hue_order=hue_order,
                    dodge=True if hue is not None else False,
                    showfliers=False,
                    showmeans=False,
                    order=strain_list, 
                    data=plot_df,
                    palette=colour_dict if colour_dict is not None else None)
        sns.stripplot(x=group_by,
                      y=feat,
                      hue=hue,
                      hue_order=hue_order,
                      dodge=True if hue is not None else False,
                      order=strain_list,
                      data=plot_df,
                      s=10,
                      color='gray',
                      marker=".",
                      edgecolor='k',
                      linewidth=.3)
        # plt.title(feat.replace('_',' '), fontsize=15, pad=20)
        plt.xticks(rotation=90, fontsize=fontsize)
        plt.yticks(fontsize=30)
        plt.ylabel('Speed (µm s$^{-1}$)', labelpad=30, fontsize=fontsize)
        ax.axes.get_xaxis().get_label().set_visible(False) # remove x axis label
        # ax.axes.get_yaxis().get_label().set_visible(False) # remove y axis label
        
        if override_palette_dict is not None:
            for box_pos, col in override_palette_dict.items():
                box = ax.artists[box_pos]
                box.set_facecolor(col)
                
        # Add p-value to plot
        if ttest_path is not None:
            feat_pvals = pvals.loc[feat]
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes) #y=scaled
            for i, strain in enumerate(strain_list):
                text = ax.get_xticklabels()[i]
                assert text.get_text() == strain
                if hue is None:
                    if strain == control:
                        continue
                    p = feat_pvals.loc[strain]
                    # plt.plot([0, 0, 1, 1], [0.98, 0.99, 0.99, 0.98], lw=1.5, c='k', transform=trans)
                    if sigasterix:
                        p_text = sig_asterix([p])[0]
                    else:
                        p_text = 'P < 0.001' if p < 0.001 else 'P = %.3f' % p
                    ax.text(i, 1.01, p_text, fontsize=30, ha='center', va='bottom', transform=trans)
                elif hue is not None:
                    for ii, h in enumerate(hue_order):
                        loc_text = strain + '-' + h
                        if loc_text == control + '-' + control_hue:
                            continue
                        p = feat_pvals.loc[loc_text]
                        if sigasterix:
                            p_text = sig_asterix([p])[0]
                        else: 
                            p_text = 'P < 0.001' if p < 0.001 else 'P = %.3f' % p
                        ax.text((i - 0.2 if ii==0 else i + 0.2), 1.01, p_text, fontsize=30, 
                                ha='center', va='bottom', transform=trans)
                        
        if ylim_minmax is not None and 'speed_50th' in feat: 
            assert isinstance(ylim_minmax, tuple)
            plt.ylim(ylim_minmax[0], ylim_minmax[1])
        
        if vline_boxpos is not None:
            vline_boxpos = [vline_boxpos] if not isinstance(vline_boxpos, list) else vline_boxpos
            for i in vline_boxpos:
                ax.axvline(i + 0.5, color='k')
                
        if not legend:
            ax.get_legend().remove()
        else:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:2], labels[:2], loc='best', frameon=False, fontsize=15)
                
        plt.subplots_adjust(bottom=subplots_adjust['bottom'], top=subplots_adjust['top'],
                            left=subplots_adjust['left'], right=subplots_adjust['right'])
        plt.savefig(save_path, bbox_inches='tight', transparent=True) #, dpi=600
        
    return

#%%
# Q1 = strain_data[feature].quantile(0.25)
# Q3 = strain_data[feature].quantile(0.75)
# IQR = Q3 - Q1

# # REMOVE OUTLIERS
# # Filtering Values between Q1-1.5IQR and Q3+1.5IQR
# # filtered = df.query('(@Q1 - 1.5 * @IQR) <= nb <= (@Q3 + 1.5 * @IQR)')


