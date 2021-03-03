#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load list of features for use in Tierpsy analysis
- Read Tierpsy 256 features from file

@author: sm5911
@date: 01/03/2021

"""

#%% Functions

def load_top256(top256_path, add_bluelight=True, remove_path_curvature=True, verbose=True):
    """ Load Tierpsy Top256 set of features describing N2 behaviour on E. coli OP50 bacteria 
        
        Parameters
        ----------
        top256_path : str
            Path to Tierpsy Top256 feature list
        add_bluelight : bool
            Append feature set separately for each bluelight condition
        remove_path_curvature : bool
            Omit path curvature-related feature from analysis
        verbose : bool
            Print progress to std out
            
        Returns
        -------
        top256 feature list
    """   
    import pandas as pd
    
    top256_df = pd.read_csv(top256_path, header=0)
    top256 = list(top256_df[top256_df.columns[0]])
    n = len(top256)
    if verbose:
        print("Feature list loaded (n=%d features)" % n)

    # Remove features from Top256 that are path curvature related
    top256 = [feat for feat in top256 if "path_curvature" not in feat]
    n_feats_after = len(top256)
    if verbose:
        print("Dropped %d features from Top%d that are related to path curvature"\
              % ((n - n_feats_after), n))
        
    if add_bluelight:
        bluelight_suffix = ['_prestim','_bluelight','_poststim']
        top256 = [col + suffix for suffix in bluelight_suffix for col in top256]

    return top256

