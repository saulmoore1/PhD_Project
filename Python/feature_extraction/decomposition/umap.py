#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uniform Manifold Projection (UMAP)

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

#%% Functions

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
    
    import umap
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    from pathlib import Path
    from tqdm import tqdm

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
