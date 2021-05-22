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

def sig_asterix(pvalues_array):
    """ Convert p-values to asterisks for showing significance on plots 
        
        Parameters
        ----------
        pvalues_array : list, np.array, pd.Series
            1-D vector of p-values to convert to asterisk significance notation
            
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
                       max_features_plot_cap=None,
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
    
    from pathlib import Path
    from matplotlib import pyplot as plt
    
    if test_pvalues_df is not None:
        # Transpose dataframe axes to make strain name the index, and features as columns
        test_pvalues_df = test_pvalues_df.T

        # Proportion of features significantly different from control
        prop_sigfeats = ((test_pvalues_df < p_value_threshold).sum(axis=1) /\
                         len(test_pvalues_df.columns))*100
        prop_sigfeats = prop_sigfeats.sort_values(ascending=False)
        
        n = len(prop_sigfeats.index)
        
        # Plot proportion significant features for each strain
        plt.ioff() if saveDir else plt.ion()
        plt.close('all')
        plt.style.use(CUSTOM_STYLE)
        fig = plt.figure(figsize=[6, n/4 if (n > 20 and n < 1000) else 7]) # width, height
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
        plt.tight_layout(rect=[0.02, 0.02, 0.96, 1])
    
        if saveDir:
            Path(saveDir).mkdir(exist_ok=True, parents=True)
            savePath = Path(saveDir) / ('percentage_sigfeats_{}.png'.format(test_name) if test_name 
                                        is not None else 'percentage_sigfeats.png')
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
    
    import numpy as np
    import seaborn as sns
    from tqdm import tqdm
    from matplotlib import pyplot as plt
   
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
