#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Principal Components Analysis (PCA)

(OPTIONAL) - Remove outliers for PCA (using Mahalanobis distance)

@author: sm5911
@date: 01/03/2021

"""
import sys

# Path to Github helper functions (USER-DEFINED path to local copy of Github repo)
PATH_LIST = ['/Users/sm5911/Tierpsy_Versions/tierpsy-tools-python/',
             '/Users/sm5911/Documents/GitHub/PhD_Project/Python/']
for sysPath in PATH_LIST:
    if sysPath not in sys.path:
        sys.path.insert(0, sysPath)

CUSTOM_STYLE = 'visualisation/style_sheet_20210126.mplstyle'

#%% Functions

def pcainfo(pca, zscores, PC=0, n_feats2print=10):
    """ A function to plot PCA explained variance, and print the most 
        important features in the given principal component (P.C.)
    """
    
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    
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
    
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from pathlib import Path
    from sklearn.decomposition import PCA
    from matplotlib import pyplot as plt
    from matplotlib import patches
    from matplotlib.axes._axes import _log as mpl_axes_logger
    from mpl_toolkits.mplot3d import Axes3D
    
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
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from pathlib import Path
    from sklearn.covariance import MinCovDet
    from matplotlib import pyplot as plt

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
        
    import numpy as np
    from scipy.stats import zscore
    from sklearn.decomposition import PCA
    from matplotlib import pyplot as plt

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


