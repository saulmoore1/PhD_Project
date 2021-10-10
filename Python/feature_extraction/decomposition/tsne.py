#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t-distributed Stochastic Neighbour Embedding (t-SNE)

@author: sm5911
@date: 01/03/2021

"""

#%% Imports

import sys

# Path to Github helper functions (USER-DEFINED path to local copy of Github repo)
PATH_LIST = ['/Users/sm5911/Tierpsy_Versions/tierpsy-tools-python/',
             '/Users/sm5911/Documents/GitHub/PhD_Project/Python/']
for sysPath in PATH_LIST:
    if sysPath not in sys.path:
        sys.path.insert(0, sysPath)

CUSTOM_STYLE = 'visualisation/style_sheet_20210126.mplstyle'

RAND_SEED = 20211010

#%% Functions

def plot_tSNE(featZ,
              meta,
              group_by,
              control=None,
              var_subset=None,
              saveDir=None,
              perplexities=[10],
              n_components=2,
              figsize=[8,8],
              sns_colour_palette="tab10",
              label_size=15,
              marker_size=100,
              n_colours=20):
    """ t-distributed stochastic neighbour embedding """
    
    import pandas as pd
    import seaborn as sns
    from tqdm import tqdm
    from pathlib import Path
    from sklearn.manifold import TSNE
    from matplotlib import pyplot as plt
    from matplotlib import patches
    
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
                             random_state=RAND_SEED,\
                             perplexity=perplex, 
                             n_iter=3000).fit_transform(featZ)
        tSNE_df = pd.DataFrame(tSNE_embedded, columns=['tSNE_1','tSNE_2']).set_index(featZ.index)
        
        # Create colour palette for plot loop
        if len(var_subset) > n_colours:
            if not control:
                raise IOError('Too many groups for plot color mapping!' + 
                              'Please provide a control group or subset of groups (n<20) to color plot')
            else:
                # Colour the control and make the rest gray
                palette = {var : "blue" if var == control else "darkgray" for var in meta[group_by].unique()}
                
        elif len(var_subset) <= n_colours:
            # Colour strains of interest
            colour_labels = sns.color_palette(sns_colour_palette, len(var_subset))
            palette = dict(zip(var_subset, colour_labels))
            
            if set(var_subset) != set(meta[group_by].unique()):
                # Make the rest gray
                gray_strains = [var for var in meta[group_by].unique() if var not in var_subset]
                gray_palette = {var:'darkgray' for var in gray_strains if not pd.isna(var)}
                palette.update(gray_palette)
            
        plt.ioff() if saveDir else plt.ion()
        plt.close('all')
        plt.style.use(CUSTOM_STYLE) 
        sns.set_style('ticks')
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('tSNE Component 1', fontsize=15, labelpad=12)
        ax.set_ylabel('tSNE Component 2', fontsize=15, labelpad=12)
        #ax.set_title('2-component tSNE (perplexity={0})'.format(perplex), fontsize=20)
                
        grouped = meta.join(tSNE_df).groupby(group_by)

        for key in list(palette.keys())[::-1]:
            if pd.isna(key):
                continue
            group = grouped.get_group(key)
            sns.scatterplot(x='tSNE_1', y='tSNE_2', data=group, color=palette[key], s=marker_size)

        # Construct legend from custom handles
        if len(var_subset) <= n_colours:
            plt.tight_layout() # rect=[0.04, 0, 0.84, 0.96]
            handles = []
            for key in var_subset:
                handles.append(patches.Patch(color=palette[key], label=key))
            # add 'other' for all other strains (in gray)
            if len(gray_palette.keys()) > 0:
                other_patch = patches.Patch(color='darkgray', label='other')
                handles.append(other_patch)  
            ax.legend(handles=handles, frameon=True, loc='upper right', fontsize=label_size, 
                      handletextpad=0.2)
        else:
            control_patch = patches.Patch(color='blue', label=control)
            other_patch = patches.Patch(color='darkgray', label='other')
            ax.legend(handles=[control_patch, other_patch])
        ax.grid(False)   
        
        if saveDir:
            saveDir.mkdir(exist_ok=True, parents=True)
            savePath = Path(saveDir) / 'tSNE_perplex={0}.pdf'.format(perplex)
            plt.savefig(savePath, tight_layout=True, dpi=300)
        else:
            plt.show(); plt.pause(2)
        
    return tSNE_df
    
