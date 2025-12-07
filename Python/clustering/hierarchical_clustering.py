#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hierarchical Clustering and Heatmaps

@author: sm5911
@date: 01/03/2021

"""

#%% Globals

# Custom style for plotting
CUSTOM_STYLE = 'visualisation/style_sheet_20210126.mplstyle'

#%% Functions

def plot_clustermap(featZ, 
                    meta, 
                    group_by,
                    colour_by=None,
                    row_colours=True,
                    col_linkage=None,
                    method='complete',
                    metric='euclidean',
                    saveto=None,
                    figsize=[10,8],
                    sns_colour_palette="Pastel1",
                    sub_adj={'bottom':0,'left':0,'top':1,'right':1},
                    label_size=5,
                    show_xlabels=True,
                    bluelight_col_colours=True):
    """ Seaborn clustermap (hierarchical clustering heatmap)
    
        Inputs
        ------
        featZ - pd.DatFrame, dataframe of normalised feature results
    """          

    import seaborn as sns
    from matplotlib import patches
    from matplotlib import pyplot as plt
    
    assert (featZ.index == meta.index).all()
    
    if type(group_by) != list:
        group_by = [group_by]
    n = len(group_by)
        
    # Store feature names
    fset = featZ.columns
        
    # Compute average value for strain for each feature (not each well)
    featZ_grouped = featZ.join(meta).groupby(group_by, dropna=False).mean().reset_index()
    
    if colour_by is None:
        colour_by = group_by[0]
    assert colour_by in meta.columns
        
    var_list = list(featZ_grouped[colour_by].unique())

    # Row colors
    if row_colours is False:
        row_colours = None
    if row_colours is not None:
        row_colours = []
        if len(var_list) > 1 or n == 1:
            var_colour_dict = dict(zip(var_list, sns.color_palette("tab10", len(var_list))))
            row_cols_var = featZ_grouped[group_by[0]].map(var_colour_dict)
            row_colours.append(row_cols_var)
        if n == 2:
            date_list = list(featZ_grouped[group_by[1]].unique())
            date_colour_dict = dict(zip(date_list, sns.color_palette("Blues", len(date_list))))
            #date_colour_dict=dict(zip(set(date_list), sns.hls_palette(len(set(date_list)),l=0.5,s=0.8)))
            row_cols_date = featZ_grouped[group_by[1]].map(date_colour_dict)
            row_cols_date.name = None
            row_colours.append(row_cols_date)  

    # Column colors
    if bluelight_col_colours:
        bluelight_colour_dict = dict(zip(['prestim','bluelight','poststim'], 
                                         sns.color_palette(sns_colour_palette, 3)))
        feat_colour_dict = {f:bluelight_colour_dict[f.split('_')[-1]] for f in fset}
    
    if type(label_size) == tuple:
        x_label_size, y_label_size = label_size
    else:
        x_label_size = label_size
        y_label_size = label_size
        
    # Plot clustermap
    plt.close('all')
    sns.set(font_scale=0.8)
    cg = sns.clustermap(data=featZ_grouped[fset], 
                        row_colors=row_colours,
                        col_colors=fset.map(feat_colour_dict) if bluelight_col_colours else None,
                        #standard_scale=1, z_score=1,
                        col_linkage=col_linkage,
                        metric=metric, 
                        method=method,
                        vmin=-2, vmax=2,
                        figsize=figsize,
                        xticklabels=fset if show_xlabels else False,
                        yticklabels=featZ_grouped[group_by].astype(str).agg(' - '.join, axis=1),
                        #cbar_pos=(0.5, 0.01, 0.1, 0.01), # (left, bottom, width, height)
                        cbar_kws={'orientation': 'horizontal',
                                  'label': None, #'Z-value'
                                  #'shrink': 1,
                                  'ticks': [-2, -1, 0, 1, 2],
                                  'drawedges': False},
                        linewidths=0)  
    #col_linkage = cg.dendrogram_col.calculated_linkage
    
    if show_xlabels:
        labels = cg.ax_heatmap.xaxis.get_majorticklabels()
        plt.setp(labels, rotation=90, fontsize=x_label_size)
        
    cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_yticklabels(), rotation=0, 
                                  fontsize=y_label_size, ha='left', va='center') 
    #plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), fontsize=15)
    #cg.ax_heatmap.axes.set_xticklabels([]); cg.ax_heatmap.axes.set_yticklabels([])
    
    if bluelight_col_colours:
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
    
    plt.subplots_adjust(top=sub_adj['top'], bottom=sub_adj['bottom'], 
                        left=sub_adj['left'], right=sub_adj['right'], 
                        hspace=0.01, wspace=0.01)
    #plt.tight_layout(rect=[0, 0, 1, 1], w_pad=0.5)
    
    # Save clustermap
    if saveto:
        saveto.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(saveto, dpi=300)
    else:
        plt.show()
        
    data = featZ_grouped.iloc[featZ_grouped.index[cg.dendrogram_row.reordered_ind]
                              ].set_index('gene_name')
    data = data[data.columns[cg.dendrogram_col.reordered_ind]]
    
    return data

def plot_barcode_heatmap(featZ, 
                         meta, 
                         group_by,
                         strain_order=None,
                         pvalues_series=None,
                         p_value_threshold=0.05,
                         selected_feats=None,
                         figsize=[18,6],
                         saveto=None,
                         sns_colour_palette="bright",
                         label_size=20,
                         sub_adj={'top':0.95,'bottom':0.05,'left':0.08,'right':0.88}):
    """  """
    
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    from matplotlib import patches
    from matplotlib.gridspec import GridSpec
    
    assert set(featZ.index) == set(meta.index)
    if type(group_by) != list:
        group_by = [group_by]
        
    # Store feature names
    fset = featZ.columns
        
    # Compute average value for strain for each feature (not each well)
    featZ_mean = featZ.join(meta).groupby(group_by).mean()#.reset_index()
       
    # Make dataframe for heatmap plot
    heatmap_df = featZ_mean[fset]
    
    if strain_order is not None:
        heatmap_df = heatmap_df.reindex(strain_order)
    
    var_list = list(heatmap_df.index)
    
    if pvalues_series is not None:
        assert all(f in fset for f in pvalues_series.index)
        pvalues_series = -np.log10(pvalues_series[fset].astype(float))
        heatmap_df = pd.concat([heatmap_df, pd.DataFrame(pvalues_series).T])
    
    # Map colors for stimulus type
    _stim = pd.DataFrame(data=[f.split('_')[-1] for f in fset], columns=['Stimulus'])
    _stim['Stimulus'] = _stim['Stimulus'].map({'prestim':1,'bluelight':2,'poststim':3})
    _stim = _stim.transpose().rename(columns={c:v for c,v in enumerate(fset)})
    heatmap_df = pd.concat([heatmap_df, _stim])
    
    # Add barcode - asterisk (*) to highlight selected features
    cm=list(np.repeat('inferno',len(var_list)))
    cm.extend(['Greys', sns_colour_palette])
    vmin_max = [(-2,2) for i in range(len(var_list))]
    vmin_max.extend([(0,20), (1,3)])
    
    # Plot barcode clustermap
    plt.ioff() if saveto else plt.ion()
    plt.close('all')  
    #plt.style.use(CUSTOM_STYLE) 
    sns.set_style('ticks')
    
    f = plt.figure(figsize=figsize)
    height_ratios = list(np.repeat(3,len(var_list)))
    height_ratios.extend([3,3])
    gs = GridSpec(len(var_list)+2, 1, wspace=0, hspace=0, height_ratios=height_ratios)
    cbar_ax = f.add_axes([.95, .75, .02, .2]) #  [left, bottom, width, height]
    
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
                    vmin=v[0], vmax=v[1],
                    linewidths=0)
        plt.yticks(rotation=0, fontsize=label_size)
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

        plt.subplots_adjust(top=sub_adj['top'], bottom=sub_adj['bottom'], 
                            left=sub_adj['left']*len(group_by), right=sub_adj['right'], 
                            hspace=0.01, wspace=0.01)
        #f.tight_layout(rect=[0, 0, 0.89, 1], w_pad=0.5)
    
    if selected_feats is not None:
        if len(selected_feats) > 0:
            for feat in selected_feats:
                try:
                    axis.text(heatmap_df.columns.get_loc(feat), 1.1, ' *', ha='center')
                except KeyError:
                    print('{} not in featureset'.format(feat))

    if saveto:
        saveto.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(saveto, dpi=600)
    else:
        plt.show()
