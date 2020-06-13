#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Investigation of random variables in syngenta dataset

Variables
-------------------------------
Duration in M9 buffer - starvation effects on features throughout the day
Temperature           - over time (internal rig + external cave) throughout day
Humidity              - over time (internal rig + external cave) throughout day

-------------------------------
@author: sm5911
@date: 13/05/2020

"""

#%% Imports

import sys
import os
import time
import warnings
#import logging
from tqdm import tqdm
import datetime as dt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
#import subprocess as sp

PATH_LIST = ['/Users/sm5911/Tierpsy_Versions/tierpsy-tools-python/']

for PATH in PATH_LIST:
    if PATH not in sys.path:
        sys.path.append(PATH)

from tierpsytools.read_data import hydra_metadata as tt_hm

#%% Parameters

project_root_dir = '/Users/sm5911/Documents/Syngenta_Analysis/'

screening_month = 'jan' # 'dec'

if screening_month.lower() == 'dec':
    # December 2019 screen
    aux_dir = "/Users/sm5911/Imperial College London/Minga, Eleni - AuxiliaryFiles"
    sum_dir = "/Users/sm5911/Imperial College London/Minga, Eleni - SummaryFiles"
    metadata_path = os.path.join(aux_dir, "full_metadata.csv")
    fileSums_path = os.path.join(sum_dir, "filenames_summaries_compiled.csv")
    featSums_path = os.path.join(sum_dir, "features_summaries_compiled.csv")
    feat_id_cols = ['file_id', 'well_name', 'is_good_well']
    screening_month = "2019_Dec"
    
elif screening_month.lower() == 'jan':
    # January 2020 screen
    aux_dir = "/Users/sm5911/Documents/Syngenta_Analysis/data/2020_Jan"
    sum_dir = aux_dir
    metadata_path = os.path.join(aux_dir, "full_metadata.csv")
    fileSums_path = os.path.join(sum_dir, "filenames_summaries_tierpsy_plate.csv")
    featSums_path = os.path.join(sum_dir, "features_summaries_tierpsy_plate.csv")
    feat_id_cols = ['file_id', 'well_name']
    screening_month = "2020_Jan"


top256feats_path = os.path.join(project_root_dir,\
                   'data/top256_tierpsy_no_blob_no_eigen_only_abs_no_norm.csv')

R_outpath = os.path.join(project_root_dir, "data", screening_month,
                         "control_results.csv")

p_value_threshold = 0.05

bluelight_condition = 'prestim'

drug_types = ['DMSO','NoCompound']   


#%% Functions

def dropBadWells(feats, meta):
    """ A function to remove erroneous 'bad well' entries from metadata and 
        feature summaries.
    """
    assert feats.shape[0] == meta.shape[0]
        
    n_before = feats.shape[0]
    
    # Find indexes of bad wells
    bad_well_cols = [col for col in meta.columns if 'is_bad_well' in col]
    bad_well_idx = []
    for col in bad_well_cols:
        bad_idx = list(np.where(meta[col] == True)[0])
        #print("%s: %d " % (col, len(bad_idx)))
        bad_well_idx.extend(bad_idx)        
    bad_well_idx = list(np.unique(bad_well_idx))
    
    # Filter results for all bad well instances
    meta = meta.drop(bad_well_idx)
    feats = feats.drop(bad_well_idx)
    
    assert feats.shape[0] == meta.shape[0]
    n_after = feats.shape[0]
    print("Total bad wells dropped: %d" % (n_before - n_after))
    
    return feats, meta

def addTimeWashed(df):
    """ A function to calculate the time worms were washed off food on the day 
        of tracking, aprroximated for each well using information in metadata 
        for 'wormsorter_start_time'.
        WARNING: Resets dataframe index
    """ 
    # Calculate time washed of food (start of no food)
    time_washed = pd.DataFrame(df.groupby(['date_yyyymmdd'])['wormsorter_start_time'].min())
    time_washed = time_washed.reset_index(drop=False)
    time_washed.columns = ['date_yyyymmdd','time_washed']
    
    df = pd.merge(left=df, right=time_washed, on='date_yyyymmdd')
    
    return df

def addDurationM9(df):
    """ A function to calculate the duration of time worms spent without food 
        in M9 buffer prior to dispensing with the worm sorter on the day of 
        tracking. The duration in M9 is taken to be the difference between the 
        time worms were washed off food and the middle wormsorter time 
        (ie. average of the time taken to dispense worms each set of plates).
    """
    # Calculate duration in M9 buffer (time without food)
    dt_0 = [dt.datetime.strptime(t,'%H:%M') for t in df['time_washed']]
    dt_1 = [dt.datetime.strptime(t,'%H:%M') for t in df['middle_wormsorter_time']]
    
    df['duration_M9_seconds'] = [(r - w).total_seconds() for r, w in zip(dt_1, dt_0)]

    return df

def addStarvationLevel(df):
    """ A function to create the categorical variable 'starvation_level' from
        'duration_M9_seconds' (continuous) by binning for the number of hours
        spent in M9 without food.
    """
    bins = [0,3600,7200,10800,14400]
    labels = ['0-1hrs','1-2hrs','2-3hrs','3-4hrs']
    starvation_levels = pd.cut(df['duration_M9_seconds'], bins, labels=labels)
    df['starvation_level'] = starvation_levels
    groups = df.groupby(['starvation_level','date_yyyymmdd'])
    groups.size()/df.shape[0]*100

    return df
    
#%%
        
if __name__ == "__main__":
    print("Running: %s" % os.path.basename(sys.argv[0]))
    tic = time.time()
    
    # Read in syngenta screen compiled metadata, features & filenames summaries  
    metadata = pd.read_csv(metadata_path)
    featSums = pd.read_csv(featSums_path, comment='#') #dtype={'':str})
    fileSums = pd.read_csv(fileSums_path, comment='#')
    
    # Align metadata nd feature summaries
    feats, metadata = tt_hm.read_hydra_metadata(feat=featSums,
                                                fname=fileSums,
                                                meta=metadata,
                                                add_bluelight=True,
                                                feat_id_cols=feat_id_cols)
    print(metadata.bluelight.unique())
    
#    feats, metadata = tt_hm.align_bluelight_conditions(feats, metadata, how='outer',
#                                                       return_separate_feat_dfs = False)
    
    # Drop bad wells from metadata + feature summaries
    feats, metadata = dropBadWells(feats, metadata)
    
    # Add columns for time washed off food, 'time_washed'
    metadata = addTimeWashed(metadata)
    
    # Calculate duration spent in M9 buffer, 'duration_in_M9'
    # NB: duration_in_M9 = middle_wormsorter_time - time_washed
    metadata = addDurationM9(metadata)

    # Create categories for level-of-starvation based by binning duration in M9
    metadata = addStarvationLevel(metadata)
    
#    # Remove features with over 10% nans
#    feats = feats.loc[:, feats.isna().sum(axis=0)/feats.shape[0]<0.10]
#    print('4: ', feats.shape)
#    # fill in remaining nans with mean values of strain
#    feats = [x for _,x in feats.groupby(by=metadata['worm_strain'], sort=True)]
#    metadata = [x for _,x in metadata.groupby(by=metadata['worm_strain'], sort=True)]
#    for i in range(len(feats)):
#        feats[i].fillna(feats[i].mean(), inplace=True)
#    print('5: ', [ft.shape for ft in feats])
#    
#    feats = pd.concat(feats, axis=0)
#    metadata = pd.concat(metadata, axis=0)

    # Combine metadata and feature summaries for plotting
    results = metadata.reset_index(drop=True).merge(feats.reset_index(drop=True),\
                                   left_index=True, right_index=True)    
        
    # results[results['date_yyyymmdd']==results['date_yyyymmdd'].unique()[0]]
        
    # Save results for control data to file for linear mixed models in R (lme4)
    R_results = results[results['drug_type'].isin(drug_types)]
    R_results.to_csv(R_outpath, index=False)
    print("Control results saved to file (for modelling in R)")
       
    # TODO: Use subprocess to call R script?
    # rscript_path = os.path.join(project_root_dir, "scripts/syngenta_variation.R")
    # start = time.time()
    # sp.call(["/Users/sm5911/anaconda3/bin/Rscript", "--vanilla", rscript_path])
    # stop = time.time()
    # print("R script runtime: %.2f" % (stop - start))
    
    
#%% Quick visualisation

    # Subset for prestimulus videos (before bluelight)
    results = results[results['bluelight'] == bluelight_condition]

    # Heatmap    
    heatmap_df = results.pivot_table(index='recording_time', columns='worm_strain',\
                                     values='duration_M9_seconds', aggfunc=np.median)
    sns.heatmap(heatmap_df, annot=True, fmt=".1f")
    plt.title("Duration in M9 buffer (seconds),\nby recording time and worm strain")
    plt.show(); plt.pause(2)
    plt.close()

    plt.figure(figsize=[10,8])
    sns.distplot(results['duration_M9_seconds'])
    #g = sns.FacetGrid(results,hue='worm_strain',size=6,aspect=2)
    #g.map(sns.distplot,'duration_M9_seconds')
    plt.title("Variation in duration in M9 (seconds)")
    plt.show(); plt.pause(2)
    plt.close()    


#%%
    # Read in Top256 Tierpsy features
    top256 = pd.read_csv(top256feats_path)['1']
    
    # Select features for which we have results (and omit 'path curvature' related features)
    top256 = [feat for feat in top256 if feat in results.columns]
    top256 = [feat for feat in top256 if "path_curvature" not in feat and "food_edge" not in feat]
    top256.append('speed_90th')
    
    
#%% Models

    plt.ioff() # Turn interactive plotting off
    for drug_name in drug_types:
        
        # Subset for durg type (in this case controls): ['DMSO','NoCompound']
        resultsDrug = results[results['drug_type'] == drug_name]
        print("\nSubsetting for '%s'" % drug_name)

        results_lm = []
        results_glm = []
        results_lmm = []
        print("Performing linear models..")
        for feature in tqdm(top256): 
            
            # Drop missing values for feature of interest
            resultsFeat = resultsDrug[[feature, 'worm_strain', 'duration_M9_seconds']].dropna()
            
            # Plot histogram of feature
            plt.figure(figsize=(10,8))
            sns.distplot(resultsFeat[feature], rug=False, label='All Strains',hist_kws={ "alpha": 0.25})
            strain_groups = list(resultsFeat['worm_strain'].unique())
            for strain in strain_groups:
                sns.distplot(resultsFeat[resultsFeat['worm_strain']==strain][feature],\
                             rug=True, label=strain, hist_kws={ "alpha": 0.25})
            plt.title("Histogram of %s for %s (%s, %s)" % (feature, strain, drug_name, bluelight_condition))
            plt.legend()
            out_path = os.path.join(project_root_dir, 'results', 'python_plots', drug_name,
                                    bluelight_condition, 'histogram', feature 
                                    + '_histogram.png')
            if not os.path.exists(os.path.dirname(out_path)):
                os.makedirs(os.path.dirname(out_path))                
            plt.savefig(out_path)
            plt.close('all')                   
            
            # Simple Linear Model (that ignores duration in M9, ie. starvation effects)
            # Assumption:
            # - residuals are normally distributed (ie. fit the t-distribution)
            # NB: If not, could try maximum likelihood estimation
            simple_model = '{} ~ 1 + worm_strain'.format(feature)
            lm = smf.ols(formula=simple_model, data=resultsFeat).fit()
            lm.summary()
            r2 = lm.rsquared_adj
            AIC = lm.aic 
            lm_pvals = lm.pvalues
            lm_params = lm.params
            lm_colnames = ['feature','R2','AIC']
            lm_colnames.extend([col + '_param' for col in list(lm_params.index)])
            lm_colnames.extend([col + '_pval' for col in list(lm_pvals.index)])
            results_lm.append((feature, r2, AIC, *lm_params, *lm_pvals))
     
            # Generalised Linear Model (that considers covariation w.r.t. duration in M9)
            maximal_model = "{} ~ worm_strain + duration_M9_seconds\
                             + worm_strain * duration_M9_seconds".format(feature)
            glm = smf.glm(formula=maximal_model, data=resultsFeat,
                          family=sm.families.Gaussian()).fit()
            glm.summary()
            AIC = glm.aic 
            glm_pvals = glm.pvalues
            glm_params = glm.params
            glm_colnames = ['feature','AIC']
            glm_colnames.extend([col + '_param' for col in list(glm_params.index)])
            glm_colnames.extend([col + '_pval' for col in list(glm_pvals.index)])
            results_glm.append((feature, AIC, *glm_params, *glm_pvals))
                      
            # Linear Mixed Model  
            # NB: Here we are fitting a model with two random effects for each 
            # worm strain: a random intercept, and a random slope (w.r.t. 
            # duration spent in M9). This means that each strain may have a 
            # different baseline response (in a given feature), as well as 
            # changing at a different rate with duration spent in M9. The 
            # formula specifies that duration in M9 is a covariate with a 
            # random coefficient. By default, formulas always include an 
            # intercept (which could be suppressed here using 0 + duration in 
            # M9 as the formula).
            
            # TODO: Should always use categorical variable as random effect? Not continuous?
            # TODO: Investigate: PlateID + rigID + Temperature + Humidity

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                mixed_model = "{} ~ duration_M9_seconds".format(feature)
                lmm = sm.MixedLM.from_formula(formula=mixed_model, # model formula
                                              data=resultsFeat,
                                              groups=resultsFeat['worm_strain'], # random effects
                                              re_formula='~duration_M9_seconds', # random effects structure
                                              vc_formula=None, 
                                              subset=None,
                                              use_sparse=False,
                                              missing='none').fit(method='cg')
                lmm.summary()
                AIC = lmm.aic
                lmm_pvals = lmm.pvalues
                lmm_params = lmm.params
                lmm_colnames = ['feature','AIC']
                lmm_colnames.extend([col + '_param' for col in list(lmm_params.index)])
                lmm_colnames.extend([col + '_pval' for col in list(lmm_pvals.index)])
                results_lmm.append((feature, AIC, *lmm_params, *lmm_pvals))
                
# TODO:         Plot the profile likelihood                
#                re = lmm.cov_re.iloc[0,0]
#                likev = lmm.profile_re(0, 're', dist_low=-0.1, dist_high=0.1)
#                plt.figure(figsize=(10,8))
#                plt.plot(likev[:,0], 2*likev[:,1])
#                plt.xlabel("Variance of random slope", size=17)
#                plt.ylabel("-2 times profile log likelihood", size=17)
#                out_path = os.path.join(project_root_dir, 'results', 'python_plots', drug_name,
#                                        bluelight_condition, 'linear_mixed_model',
#                                        feature + '_profile_likelihood_plot.png')
#                if not os.path.exists(os.path.dirname(out_path)):
#                    os.makedirs(os.path.dirname(out_path))                
#                plt.savefig(out_path)
#                plt.close('all')       

#            # HACK: statsmodels only supports linear regression for Gaussian-distributed outcomes
#            # No non-parametric equivalents exist. For other ways to fit mixed effects models:
#            # Could try Bayes: MCMC eg. PyStan, PyMC3, Edward
#            # Could try package: diamond, but you must specify hyperparameters (sigma)
#            # Could try ML: neural networks, tree ensembles
#            # Could use subprocess to run R and use lme4  
                
#            # Mixed-Effects Random Forests
#            from merf import MERF
#            # merf model formulae: 
#            # y_i = f(X_i) + Z_i * b_i + e_i
#            # b_i ~ N(0, D)
#            # e_i ~ N(0, R_i)
#            
#            mrf = MERF(n_estimators=300,max_iterations=100)
#            mrf = mrf.fit(X=results['worm_strain'],\
#                          Z=results['duration_M9_seconds'],\
#                          clusters=results['duration_M9_seconds'],\
#                          y=results[feature])
#            mrf.summary()
        
        # Compile linear model results dataframes
        results_lm = pd.DataFrame.from_records(results_lm, columns=lm_colnames)
        results_lm = results_lm.sort_values(by='AIC', ascending=True)
        sigFeats_lm = results_lm.loc[np.where(results_lm[results_lm.columns[4:]] < p_value_threshold)[0]]
        
        results_glm = pd.DataFrame.from_records(results_glm, columns=glm_colnames)
        results_glm = results_glm.sort_values(by='AIC', ascending=True)
        sigFeats_glm = results_glm.loc[np.where(results_glm[results_glm.columns[2:]] < p_value_threshold)[0]]
        
        results_lmm = pd.DataFrame.from_records(results_lmm, columns=lmm_colnames)
        results_lmm = results_lmm.sort_values(by='AIC', ascending=True)
        sigFeats_lmm = results_lmm.loc[np.where(results_lmm[results_lmm.columns[2:]] < p_value_threshold)[0]]
            
#%% Plots
                
        # Investigate effect of time of day on worm behaviour (feature by feature)
        # proxies_for_time_of_day = ['recording_time','imaging_run_number','duration_M9_seconds']
        print("\nPlotting feature summaries..")
        for feature in tqdm(top256):  

            #[col for col in results_lm.columns if 'worm_strain' in col and 'pval' in col]
            pval = np.float(results_lm[results_lm['feature']==feature][results_lm.columns[-1]])
               
            # Jointplots of feature by 'duration_M9_seconds'
            out_path = os.path.join(project_root_dir, 'results', 'python_plots', drug_name,
                                    bluelight_condition, 'jointplots', feature + '_jointplot.png')
            if not os.path.exists(out_path):
                if not os.path.exists(os.path.dirname(out_path)):
                    os.makedirs(os.path.dirname(out_path))    
                g = sns.jointplot(data=resultsDrug, x='duration_M9_seconds', y=feature,
                              kind='reg', color='g', height=8)
                fig = g.fig
                fig.subplots_adjust(top=0.9)
                fig.suptitle("Effect of duration in M9 on \n"+feature+" ("+drug_name+", "+bluelight_condition+")")
                ax = g.ax_joint
                ax.text(0,0,'p={:g}'.format(float('{:.2g}'.format(pval))))
                #plt.plot(results['duration_M9_seconds'],np.log(results[feature]),".")
                plt.savefig(out_path)
                plt.close()
            
            # Violinplots of feature by 'imaging_run_number'
            out_path = os.path.join(project_root_dir, 'results', 'python_plots', drug_name,
                                    bluelight_condition, 'violinplots', feature + '_violinplot.png')
            if not os.path.exists(out_path):
                if not os.path.exists(os.path.dirname(out_path)):
                    os.makedirs(os.path.dirname(out_path)) 
                plt.figure(figsize=[10,6])
                sns.violinplot(data=resultsDrug, x='imaging_run_number', y=feature, hue='worm_strain')
                plt.title("Effect of imaging run on \n"+feature+" ("+drug_name+", "+bluelight_condition+")")
                plt.savefig(out_path)
                plt.close()
                
            # Boxplot of feature by 'duration_M9_buffer' and 'worm_strain'
            out_path = os.path.join(project_root_dir, 'results', 'python_plots', drug_name,
                                    bluelight_condition, 'boxplots', feature + '_boxplot.png')
            if not os.path.exists(out_path):
                if not os.path.exists(os.path.dirname(out_path)):
                    os.makedirs(os.path.dirname(out_path))  
                plt.figure(figsize=[10,6])
                sns.boxplot(data=resultsDrug, x='imaging_run_number', y=feature, hue='worm_strain')
                plt.title("Effect of imaging run on \n"+feature+" ("+drug_name+", "+bluelight_condition+")")
                plt.savefig(out_path)
                plt.close()
    

#%%        
    toc = time.time()
    print("\nDone! (total time taken: %.1f seconds)" % (toc-tic))

#%% Linear Mixed Model - Try with dummy data
 
# BUG: Error with size of parameter vectors in linear mixed model
#ws = ['w1','w2']
#w1 = np.random.normal(3, 2.5, size=100)
#w2 = np.random.normal(5, 2.5, size=100)
#wdf = pd.DataFrame({w:f for w, f in zip(ws, [w1,w2])})
#wdf['random_effect'] = np.arange(1,101,1) # np.random.randint(30, size=100)
#wdf = pd.melt(wdf, value_vars=['w1','w2'], id_vars='random_effect',\
#              var_name='worm_strain', value_name='feature')
#lmm_dummy = smf.mixedlm(formula="feature ~ worm_strain", data=wdf,\
#                        groups='random_effect',\
#                        re_formula='1', vc_formula=None, subset=None,\
#                        use_sparse=False)
# FIXED: Remove missing values prior to invoking smf.mixedlm() OR add flag missing='drop'
    
    
# BUG: Plots of speed between N2 + Hawaiian worm strains is wrong somehow. 
#          There definitely should be a difference in speed between the strains
#          
# FIXED: Comparing with Eleni's code. Found bug: pd.merge to add a column to 
#           metadata resets the dataframe's index prior to concatenation with feats
    
