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

CUSTOM_STYLE = 'analysis/analysis_20210126.mplstyle'

#%% Functions

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
    
    import pandas as pd
    import seaborn as sns
    from tqdm import tqdm
    from pathlib import Path
    from sklearn.manifold import TSNE
    from matplotlib import pyplot as plt
    
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
    
