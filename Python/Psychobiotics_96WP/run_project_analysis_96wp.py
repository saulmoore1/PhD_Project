#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHENOTYPIC SCREENING FOR PSYCHOBIOTIC BACTERIA: COMPUTATIONAL ANALYSIS OF WORM 
BEHAVIOUR ON VARIOUS BACTERIAL FOODS (96WP ASSAY)

A script written to interpret and visualise results for the behavioural 
analysis of N2 C. elegans raised monoxenically on various bacterial food:

(1) C.elegans Microbiome: 50 bacterial isolates from the native microbiome 
                          (gut + substrate samples) of C. elegans
(2) Pan-Genome Library:   696 fully-sequenced strains of E. coli cultered from 
                          gut microbiota from various human + animal samples
(3) Keio Library:         ~2000 E. coli (K-12) single-gene deletion mutants

N2 performance on each strain is compared to performance on standard laboratory 
E. coli OP50 strain as a control. Control variation is investigated across 
experiment runs for time-dependence of various potential confounders: 
    Experiment date (day-effects), 
    Imaging camera/rig (imaging-effects), 
    L1 diapause duration (age-effects), 
    Temperature/Humidity (stochastic environmental effects)

Principal components analysis, tSNE and UMAP are performed to see if strains 
can be clustered into groups with different behaviours in phenotype space.

Depending on whether feature results are normally distributed, the following
parametric tests (or non-parametric equivalents) are performed (with the 
Benjamini-Hochberg correction applied to correct for multiple comparisons):

(1) T-tests/Wilcoxon ranksum tests to look for behavioural features that differ 
between each strain and the control. For each food, box plots are saved for the 
top features that most significantly differ from control observations.

(2) One-way ANOVA/Kruskal-Wallis tests to look for 'hit' strains that elicit 
different N2 behaviour. For significant features, post-hoc Tukey HSD tests are 
performed for pairwise comparisons between strains.

Box plots are produced for each significant feature in a representative list of 
Top256 features (Javer et al, 2018)

@author: sm5911
@date created: 26/10/2019
@date updated: 11/06/2020
"""

#%% IMPORTS & DEPENDENCIES

import os, sys, re, time, datetime, decimal, itertools, umap
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path, PosixPath
from scipy.stats import kruskal, ttest_ind, f_oneway, zscore
from statsmodels.stats import multitest as smm # AnovaRM
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib import transforms

# Custom imports
# Path to Github helper functions (USER-DEFINED path to local copy of Github repo)
PATH_LIST = ['/Users/sm5911/Tierpsy_Versions/tierpsy-tools-python/',
             '/Users/sm5911/Documents/GitHub/PhD_Project/Python/Psychobiotics_96WP']
for sysPath in PATH_LIST:
    if sysPath not in sys.path:
        sys.path.insert(0, sysPath)
        
from tierpsytools.hydra import compile_metadata as tt_cm # add_imgstore_name
from tierpsytools.read_data import compile_features_summaries as tt_cf # find_fname_summaries, compile_tierpsy_summaries
from tierpsytools.read_data import hydra_metadata as tt_hm # read_hydra_metadata

from my_helper import ranksumtest, savefig, pcainfo, plotPCA, MahalanobisOutliers, check_normality
from run_control_analysis_96wp import control_variation

bigtic = time.time() # record script start time

#%% LIBRARY TO ANALYSE


# Library to analyse:
ANALYSIS = 'microbiome'


# Libraries to choose from:
analyses = ['microbiome','keio','pangenome']

assert ANALYSIS in analyses
print("Analysing %s phenotypic screen:" % ANALYSIS.upper())


# NATIVE MICROBIOME STRAIN SET
if ANALYSIS.lower() == 'microbiome':
    PROJECT_ROOT_DIR = Path('/Volumes/behavgenom$/Saul/MicrobiomeScreen96WP')
    
    # List of experiment imaging dates (optional)
    IMAGING_DATES = ['20200213','20200214','20200221', '20200222']
    
    STRAINS_TO_EXCLUDE = None
    CONTROL_STRAIN = "OP50" # Control bacterial strain                                                 
    
    variables_list = ["date_yyyymmdd", "imaging_run_number", "imaging_plate_id",\
                      "master_stock_plate_ID", "instrument_name", "well_name",
                      "L1_diapause_seconds"]

    TIMEPOINT = 3 # Timepoint to analyse


# KEIO COLLECTION
elif ANALYSIS.lower() == 'keio':
    PROJECT_ROOT_DIR = Path('/Volumes/hermes$/KeioScreen_96WP')
    
    IMAGING_DATES = ['20200303']

    STRAINS_TO_EXCLUDE = ["0"]    
    CONTROL_STRAIN = "WT"

    variables_list = ["imaging_run_number", "imaging_plate_id",\
                      "master_stock_plate_ID", "instrument_name", "well_name",\
                      "dispense_method"]
    
    TIMEPOINT = None


# PAN-GENOMIC LIBRARY
elif ANALYSIS.lower() == 'pangenome':
    PROJECT_ROOT_DIR = Path('/Volumes/hermes$/PanGenomeTest_96WP')
    
    IMAGING_DATES = ['20191017','20191024','20191031']
    
    STRAINS_TO_EXCLUDE = None    
    CONTROL_STRAIN = "OP50"
    
    variables_list = ["date_yyyymmdd", "imaging_run_number", "imaging_plate_id",\
                      "instrument_name", "well_name"]
    
    TIMEPOINT = None

    
#%% GLOBAL PARAMETERS

RESULTS_DIR = PROJECT_ROOT_DIR / "Results"
RESULTS_PATH = RESULTS_DIR / "fullresults.csv"
PROCESS_FEATURE_SUMMARIES = False # Compile feature summary files?

METADATA_DIR = PROJECT_ROOT_DIR / "AuxiliaryFiles"
METADATA_PATH = METADATA_DIR / 'metadata.csv'
PROCESS_METADATA = False # Compile metadata? 

useTop256 = True # Use representative set of 256 features for analysis?
Top256_PATH = PROJECT_ROOT_DIR / 'AuxiliaryFiles' /\
              'top256_tierpsy_no_blob_no_eigen_only_abs_no_norm.csv'
filter_size_related_feats = False # Drop size-related features from analysis?                                                  
 
ANALYSE_CONTROL_VARIATION = True # Analyse variation in control data?
PATH_CONTROL = PROJECT_ROOT_DIR / 'Results' / 'Control' / 'control_results.csv'
                                   
nan_threshold = 0.2 # Threshold NaN proportion to drop feature from analysis  
p_value_threshold = 0.05 # Threshold p-value for statistical significance

check_feature_normality = True # Check if features obey Gaussian normality?
is_normal_threshold = 0.95 # Threshold normal features for parametric stats                                            

max_sigdiff_strains_plot_cap = 60
n_top_features = 5 # Number of top-ranked features to plot
show_plots = False # Display figures?                                                         

PCs_to_keep = 10 # Number of principle components to use for PCA 
n_PC_feats2plot = 10 # Number of top features influencing PC to plot
  
perplexities = [5,10,20,30] # Perplexity parameter for tSNE mapping
# NB: Similar to n-nearest neighbours, eg. expected cluster size

n_neighbours = [5,10,20,30] # N-neighbours parameter for UMAP projections                                            
min_dist = 0.3 # Minimum distance parameter for UMAP projections                                                                                                            


#%% FUNCTIONS

def concatenate_metadata(METADATA_DIR, IMAGING_DATES, SAVETO=False):
    """ COMPILE FULL METADATA FROM EXPERIMENT DAY METADATA 
        (OBTAIN MASKED VIDEO FILEPATHS + CAMERA SERIAL)
    """   
    print("Compiling from day-metadata in '%s'" % METADATA_DIR)
    
    AuxFileList = os.listdir(METADATA_DIR)
    ExperimentDates = sorted([expdate for expdate in AuxFileList if re.match(r'\d{8}', expdate)])
    
    if IMAGING_DATES:
        ExperimentDates = [expdate for expdate in ExperimentDates if expdate in IMAGING_DATES]
    else:
        IMAGING_DATES = ExperimentDates
    
    day_meta_list = []
    for expdate in IMAGING_DATES:
        day_meta_path = os.path.join(METADATA_DIR, expdate, expdate + '_day_metadata.csv')
                
        day_meta = pd.read_csv(day_meta_path, dtype={"comments":str})
        
        # Rename metadata columns for compatibility with TierpsyTools functions 
        day_meta = day_meta.rename(columns={'date_recording_yyyymmdd': 'date_yyyymmdd',
                                            'well_number': 'well_name',
                                            'plate_number': 'imaging_plate_id',
                                            'run_number': 'imaging_run_number'})
        day_meta = day_meta.drop(columns='camera_number')
        
        # Get path to RawVideo directory for day metadata
        rawDir = Path(day_meta_path.replace("AuxiliaryFiles","RawVideos")).parent
        
        # Get imgstore name and camera serial for metadata entries
        day_meta = tt_cm.add_imgstore_name(day_meta, rawDir)
        day_meta['filename'] = [rawDir.parent / day_meta.loc[i,'imgstore_name']\
                                for i in range(len(day_meta['filename']))]

        day_meta_list.append(day_meta)
    
    # Concatenate list of day metadata into full metadata
    metadata = pd.concat(day_meta_list, axis=0, ignore_index=True, sort=False)

    return metadata 

def add_timepoints(df, analysis='microbiome'):
    """ Add timepoint to analyse """
    
    if analysis == 'microbiome':
        run_to_repl = pd.DataFrame({'imaging_run_number': list(df['imaging_run_number'].unique()),
                                    'time_point': np.ceil(df['imaging_run_number'].unique()/2).astype(int)})
        df = pd.merge(df, run_to_repl, how='left', on='imaging_run_number')  
        
        return df

def calculate_L1_diapause(df):
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
        df['L1_diapause_seconds'] = [int(timedelta.total_seconds()) \
                                           for timedelta in L1_diapause_duration]
    else:
        missingInfo = [x for x in diapause_required_columns if x in df.columns\
                       and not df[x].any()]
        print("""WARNING: Could not calculate diapause duration.\n\t\
         Required column info: %s""" % missingInfo)

    return df

def cleanFeatureSummaries(features, metadata, featurelist=None, imputeNaN=True,
                          nan_threshold=0.2, filter_size_related_feats=False):
    """ Omit features with many NaN values, and impute remaining NaN values, """
    assert set(features.index) == set(metadata.index)

    if featurelist:
        features = features[featurelist]
    else:
        featurelist = features.columns
        
    assert all([feat in features.columns for feat in featurelist])
    
    print("Cleaning feature summary results..")

    # Drop rows from features that have no results (empty wells?)
    n_rows = len(features)
    features = features[features[featurelist].sum(axis=1) != 0]
    
    # Drop corresponding metadata
    metadata = metadata.iloc[features.index] 
    
    print("Dropped %d entries with missing feature summaries." % (n_rows - features.shape[0]))
    # NB: (no worms in those wells?)
    # Compare missing results with 'No worm' comments in metadata
    # Are there no results simply because there were no worms dispensed into those wells?
    
    # Drop feature columns with too many NaN values
    features = features.dropna(axis=1, thresh=nan_threshold)
    featurelist_noNaN = features.columns
    nan_cols = len(featurelist) - len(featurelist_noNaN)
    
    print("Dropped %d feature columns with too many NaNs" % nan_cols)
    # NB: All dropped features here have to do with the 'food_edge' 
    #     (which is undefined, so NaNs are expected)
    
    # Impute remaining NaN values (using global mean feature value for each food)
    if imputeNaN:
        n_nans = features.isna().sum(axis=0).sum()
        if n_nans > 0:
            print("Imputing %d missing values (%.2f%% data), using global mean value for each feature."\
                  % (n_nans, n_nans/features.count().sum()*100)) 
            features = features.fillna(features.mean(axis=0))
        else:
            print("No need to impute! No remaining NaN values found in feature summary results.")
    
    # Drop feature columns that contain only zeros
    features = features.drop(columns=features.columns[(features == 0).all()])
    featurelist_noNaNorAllZero = features.columns
    zero_cols = len(featurelist_noNaN) - len(featurelist_noNaNorAllZero)
    
    print("Dropped %d feature columns with all zeros" % zero_cols)
            
    # Filter size-related features from analysis (OPTIONAL)
    if filter_size_related_feats:
        size_feat_keys = ['blob','box','width','length','area']
        size_features = []
        for feature in featurelist:
            for key in size_feat_keys:
                if key in feature:
                    size_features.append(feature)  
        featurelist = [feat for feat in featurelist if feat not in size_features]

        print("Dropped %d features that are size-related" % len(size_features))
        features = features[featurelist]

    return features, metadata


#%% PROCESS METADATA

if not IMAGING_DATES or len(IMAGING_DATES) == 0:
    IMAGING_DATES = os.listdir(os.path.join(PROJECT_ROOT_DIR, "MaskedVideos"))     
    IMAGING_DATES = sorted([date for date in IMAGING_DATES if date != '.DS_Store'])
    assert IMAGING_DATES

# Process metadata
if PROCESS_METADATA:
    tic = time.time()
    print("\nProcessing metadata...")
    
    # COMPILE FULL METADATA FROM EXPERIMENT DAY METADATA
    metadata = concatenate_metadata(METADATA_DIR, IMAGING_DATES)
       
    # Calculate timepoints
    if TIMEPOINT:
        metadata = add_timepoints(metadata)

    # Calculate L1 diapause duration
    metadata = calculate_L1_diapause(metadata)
    
    # # Add columns of uppercase strain names
    # metadata['food_type_upper'] = metadata['food_type'].str.upper()
    # fullresults.drop(columns='food_type_upper', inplace=True)

    # Save metadata   
    metadata.to_csv(METADATA_PATH, index=False)        
    print("Saved compiled metadata to: '%s'\n(Time taken: %.1f seconds)"\
          % (METADATA_PATH, time.time() - tic))   
else:
    # Read metadata
    metadata = pd.read_csv(METADATA_PATH, dtype={"comments":str})
    print("\nMetadata file loaded.")

# Subset metadata to remove remaining entries with missing filepaths
is_filename = [isinstance(path, PosixPath) or isinstance(path, str) for path in metadata['filename']]
if any(list(~np.array(is_filename))):
    print("""WARNING: Could not find filepaths for %d entries in metadata.\n\t\
     Omitting these files from the analysis.""" % sum(list(~np.array(is_filename))))
     
    metadata = metadata[list(np.array(is_filename))]
    metadata.reset_index(drop=True, inplace=True) # reset index
    
# Record metadata column names
metadata_colnames = list(metadata.columns)


#%% PROCESS FEATURE SUMMARIES

tic = time.time()

combined_feats_path = RESULTS_DIR / "full_features.csv"
combined_fnames_path = RESULTS_DIR / "full_filenames.csv"

if not np.logical_and(combined_feats_path.is_file(), combined_fnames_path.is_file()):   
    print("\nProcessing feature summary results...")

    feat_files = [file for file in Path(RESULTS_DIR).rglob('features_summary*.csv')]
    fname_files = tt_cf.find_fname_summaries(feat_files)
    
    # Keep only features files for which matching filenames_summaries exist
    feat_files = [feat_fl for feat_fl,fname_fl in zip(feat_files, fname_files)
                  if fname_fl is not None]
    fname_files = [fname_fl for fname_fl in fname_files
                   if fname_fl is not None]
    
    tt_cf.compile_tierpsy_summaries(feat_files, fname_files,
                                    combined_feats_path, combined_fnames_path)
    
    feature_summaries = pd.read_csv(combined_feats_path, comment='#')
    filename_summaries = pd.read_csv(combined_fnames_path, comment='#')
    
    features, metadata = tt_hm.read_hydra_metadata(feature_summaries, 
                                                   filename_summaries,
                                                   metadata,
                                                   add_bluelight=False)
    
    features, metadata = cleanFeatureSummaries(features, 
                                               metadata,
                                               featurelist=None,
                                               imputeNaN=True,
                                               nan_threshold=nan_threshold,
                                               filter_size_related_feats=False)  
    # Save full results to file
    fullresults = metadata.join(features)
    fullresults.to_csv(RESULTS_PATH, index=False)
    print("Saved feature summary results to: '%s'\n(Time taken: %.1f seconds)"\
          % (RESULTS_PATH, time.time() - tic)) 
else:
    try:
        fullresults = pd.read_csv(RESULTS_PATH, dtype={"comments":str}) 
        print("Feature summary results loaded.")
    except Exception as EE:
        print("ERROR: %s" % EE)
        print("Please process feature summaries and provide correct path to results.")

# Record new columns added to metadata
newcols = ['featuresN_filename', 'file_id', 'is_good_well']
for col in newcols:
    if col not in metadata_colnames:
        metadata_colnames.append(col)

# Analysis is case-sensitive. Ensure that there is no confusion in strain names
assert len(fullresults['food_type'].unique()) == len(fullresults['food_type'].str.upper().unique())

# Record strain names for which we have results
BACTERIAL_STRAINS = [strain for strain in list(fullresults['food_type'].unique())]
if STRAINS_TO_EXCLUDE:
    BACTERIAL_STRAINS = [strain for strain in BACTERIAL_STRAINS if strain not in STRAINS_TO_EXCLUDE]


#%% SUBSET RESULTS
    
# Subset data for strains to investigate
fullresults = fullresults[fullresults['food_type'].isin(BACTERIAL_STRAINS)]

# Subset for imaging dates provided
fullresults = fullresults[fullresults['date_yyyymmdd'].isin(IMAGING_DATES)]

# Subset for a single timepoint only
if TIMEPOINT:
    fullresults = fullresults[fullresults['time_point'].isin(TIMEPOINT)]


#%% LOAD TOP256 FEATURES
        
if useTop256:
    # Read list of important features (as shown previously by Javer, 2018) and 
    # take first set of 256 features (it does not matter which set is chosen)
    top256_df = pd.read_csv(Top256_PATH)
    featurelist = list(top256_df[top256_df.columns[0]])
    n = len(featurelist)
    print("Feature list loaded (n=%d features)" % n)

    # Remove features from Top256 that are path curvature related
    featurelist = [feat for feat in featurelist if "path_curvature" not in feat]
    n_feats_after = len(featurelist)
    print("Dropped %d features from Top%d that are related to path curvature" %\
          ((n - n_feats_after), n)) 

    # Ensure results exist for features in featurelist
    featurelist = [feat for feat in featurelist if feat in fullresults.columns]
    assert len(featurelist) == n_feats_after
else:
    featurelist = [feat for feat in fullresults.columns if feat not in metadata_colnames]


#%% Analyse results for OP50 control across days
# High behavioural variation across days prevents conclusions about strain differences

# Extract control feature summary results
control_df = fullresults[fullresults['food_type'] == CONTROL_STRAIN]

if TIMEPOINT:
    PATH_CONTROL = Path(str(PATH_CONTROL).replace(".csv", "_timepoint{0}.csv".format(TIMEPOINT)))
# Save control data - make folder if it does not exist
PATH_CONTROL.parent.mkdir(exist_ok=True, parents=True)
control_df.to_csv(PATH_CONTROL, index=False)

#  Analyse control variation with respect to the defined variables
if ANALYSE_CONTROL_VARIATION:
    print("\nAnalysing control variation...\n")
    control_variation(df = control_df,
                      outDir = PATH_CONTROL.parent,
                      features_to_analyse = featurelist,
                      variables_to_analyse = variables_list,
                      # remove outliers using Mahalanobis distance (performed once only)
                      remove_outliers = True,
                      p_value_threshold = p_value_threshold,
                      PCs_to_keep = PCs_to_keep)
    plt.pause(2); plt.close('all')

#%% PERFORM TESTS FOR NORMALITY
    
# Look to see if response data are homoscedastic / normally distributed
if check_feature_normality:
    normtest_savepath = os.path.join(PROJECT_ROOT_DIR, "Results", "Stats", "shapiro_normality_test_results.csv")
    directory = os.path.dirname(normtest_savepath) # make folder if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    prop_features_normal, is_normal = check_normality(fullresults,\
                                                      metadata_colnames,\
                                                      BACTERIAL_STRAINS,\
                                                      p_value_threshold,\
                                                      is_normal_threshold)                
    # Save normailty test results to file
    prop_features_normal.to_csv(normtest_savepath, index=True, index_label='food_type', header='prop_normal')

else:
    is_normal = False # Default non-parameetric

#%% STATISTICS: t-tests/rank-sum tests
#   - To look for behavioural features that differ significantly on test foods vs control 

TEST_STRAINS = [strain for strain in BACTERIAL_STRAINS if strain != CONTROL_STRAIN]

if is_normal:
    TEST = ttest_ind
else:
    TEST = ranksumtest

# Record name of statistical test used (ttest/ranksumtest)
test_name = str(TEST).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]

# Record number of decimal places of threshold p-value, for print statements to std_out
p_decims = abs(int(decimal.Decimal(str(p_value_threshold)).as_tuple().exponent))

# Pre-allocate dataframes for storing test statistics and p-values
test_stats_df = pd.DataFrame(index=list(TEST_STRAINS), columns=featurelist)
test_pvalues_df = pd.DataFrame(index=list(TEST_STRAINS), columns=featurelist)
sigdifffeats_df = pd.DataFrame(index=test_pvalues_df.index, columns=['N_sigdiff_beforeBF','N_sigdiff_afterBF'])

# Compare each strain to OP50: compute test statistics for each feature
for t, food in enumerate(TEST_STRAINS):
    print("Computing %s tests for %s vs %s..." % (test_name, CONTROL_STRAIN, food))
        
    # Grab feature summary results for that food
    test_food_df = fullresults[fullresults['food_type_upper'] == food]
    
    # Drop non-data columns
    test_data = test_food_df.drop(columns=metadata_colnames)
    control_data = control_df.drop(columns=metadata_colnames)
               
    # Drop columns that contain only zeros
    n_cols = len(test_data.columns)
    test_data.drop(columns=test_data.columns[(test_data == 0).all()], inplace=True)
    control_data.drop(columns=control_data.columns[(control_data == 0).all()], inplace=True)
    
    # Use only shared feature summaries between control data and test data
    shared_colnames = control_data.columns.intersection(test_data.columns)
    test_data = test_data[shared_colnames]
    control_data = control_data[shared_colnames]

    zero_cols = n_cols - len(test_data.columns)
    if zero_cols > 0:
        print("Dropped %d feature summaries for %s (all zeros)" % (zero_cols, food))

    # Perform rank-sum tests comparing between foods for each feature (max features = 4539)
    test_stats, test_pvalues = TEST(test_data, control_data)

    # Add test results to out-dataframe
    test_stats_df.loc[food][shared_colnames] = test_stats
    test_pvalues_df.loc[food][shared_colnames] = test_pvalues
    
    # Record the names and number of significant features 
    sigdiff_feats = test_pvalues_df.columns[np.where(test_pvalues < p_value_threshold)]
    sigdifffeats_df.loc[food,'N_sigdiff_beforeBF'] = len(sigdiff_feats)
            
# Benjamini/Hochberg corrections for multiple comparisons
sigdifffeatslist = []
test_pvalues_corrected_df = pd.DataFrame(index=test_pvalues_df.index, columns=test_pvalues_df.columns)
for food in test_pvalues_df.index:
    # Locate pvalue results (row) for food
    food_pvals = test_pvalues_df.loc[food] # pd.Series object
    
    # Perform Benjamini/Hochberg correction for multiple comparisons on t-test pvalues
    _corrArray = smm.multipletests(food_pvals.values, alpha=p_value_threshold, method='fdr_bh',\
                                   is_sorted=False, returnsorted=False)
    
    # Get pvalues for features that passed the Benjamini/Hochberg (non-negative) correlation test
    pvalues_corrected = _corrArray[1][_corrArray[0]]
    
    # Add pvalues to dataframe of corrected ttest pvalues
    test_pvalues_corrected_df.loc[food, _corrArray[0]] = pvalues_corrected
    
    # Record the names and number of significant features (after Benjamini/Hochberg correction)
    sigdiff_feats = pd.Series(test_pvalues_corrected_df.columns[np.where(_corrArray[1] < p_value_threshold)])
    sigdiff_feats.name = food
    sigdifffeatslist.append(sigdiff_feats)
    sigdifffeats_df.loc[food,'N_sigdiff_afterBF'] = len(sigdiff_feats)

# Concatenate into dataframe of features for each food that differ significantly from behaviour on OP50
sigdifffeats_food_df = pd.concat(sigdifffeatslist, axis=1, ignore_index=True, sort=False)
sigdifffeats_food_df.columns = test_pvalues_corrected_df.index

# Save test statistics to file
stats_outpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Stats', test_name + '_results.csv')
sigfeats_outpath = stats_outpath.replace('_results.csv', '_significant_features.csv')

test_pvalues_corrected_df.to_csv(stats_outpath) # Save test results as CSV
sigdifffeats_food_df.to_csv(sigfeats_outpath, index=False) # Save feature list as text file
    
# Proportion of features significantly different from OP50
#plt.close()
propfeatssigdiff = ((test_pvalues_corrected_df < p_value_threshold).sum(axis=1)/len(featurelist))*100
propfeatssigdiff = propfeatssigdiff.sort_values(ascending=False)
fig = plt.figure(figsize=[7,10])
ax = fig.add_subplot(1,1,1)
propfeatssigdiff.plot.barh(ec='black') # fc
ax.set_xlabel('% significantly different features', fontsize=16, labelpad=10)
ax.set_ylabel('Bacterial Strain', fontsize=17, labelpad=10)
plt.xlim(0,100)
plt.tight_layout(rect=[0.02, 0.02, 0.96, 1])
plots_outpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'All',
                             'Percentage_features_sigdiff.eps')
directory = os.path.dirname(plots_outpath)
if not os.path.exists(directory):
    os.makedirs(directory)
print("Saving figure: %s" % os.path.basename(plots_outpath))
plt.savefig(plots_outpath, format='eps', dpi=600)
if show_plots:
    plt.show(); plt.pause(2); plt.close('all')


#%% STATISTICS: One-way ANOVA/Kruskal-Wallis + Tukey HSD post-hoc tests for pairwise differences 
#               between foods for each feature

if is_normal:
    TEST = f_oneway
else:
    TEST = kruskal
    
# Record name of statistical test used (kruskal/f_oneway)
test_name = str(TEST).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]

print("\nComputing %s tests between foods for each feature..." % test_name)

# Perform 1-way ANOVAs for each feature between test strains
test_pvalues_df = pd.DataFrame(index=['stat','pval'], columns=featurelist)
for f, feature in enumerate(featurelist):
    if f % 10 == 0:
        print("Analysing feature: %d/%d" % (f, len(featurelist)))  
        
    # Perform test and capture outputs: test statistic + p value
    test_stat, test_pvalue = TEST(*[fullresults[fullresults['food_type'].str.upper()==food][feature]\
                                       for food in fullresults['food_type'].str.upper().unique()])
    test_pvalues_df.loc['stat',feature] = test_stat
    test_pvalues_df.loc['pval',feature] = test_pvalue

# Perform Bonferroni correction for multiple comparisons on one-way ANOVA pvalues
_corrArray = smm.multipletests(test_pvalues_df.loc['pval'], alpha=p_value_threshold, method='fdr_bh',\
                               is_sorted=False, returnsorted=False)

# Get pvalues for features that passed the Benjamini-Hochberg (non-negative) correlation test
pvalues_corrected = _corrArray[1][_corrArray[0]]

# Add pvalues to one-way ANOVA results dataframe
test_pvalues_df = test_pvalues_df.append(pd.Series(name='pval_corrected'))
test_pvalues_df.loc['pval_corrected', _corrArray[0]] = pvalues_corrected

# Store names of features that show significant differences across the test bacteria
sigdiff_feats = test_pvalues_df.columns[np.where(test_pvalues_df.loc['pval_corrected'] < p_value_threshold)]
print("Complete!\n%d/%d (%.1f%%) features exhibit significant differences between foods (%s test, Benjamini-Hochberg)"\
      % (len(sigdiff_feats), len(test_pvalues_df.columns), len(sigdiff_feats)/len(test_pvalues_df.columns)*100, test_name))

# Compile list to store names of significant features
sigfeats_out = pd.Series(test_pvalues_df.columns[np.where(test_pvalues_df.loc['pval_corrected'] < p_value_threshold)])
sigfeats_out.name = 'significant_features_' + test_name
sigfeats_out = pd.DataFrame(sigfeats_out)

# Save test statistics to file
stats_outpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Stats', test_name + '_results.csv')
sigfeats_outpath = stats_outpath.replace('_results.csv', '_significant_features.csv')
directory = os.path.dirname(stats_outpath) # make folder if it does not exist
if not os.path.exists(directory):
    os.makedirs(directory)
test_pvalues_df.to_csv(stats_outpath) # Save test results as CSV
sigfeats_out.to_csv(sigfeats_outpath, index=False) # Save feature list as text file

topfeats = test_pvalues_df.loc['pval_corrected'].sort_values(ascending=True)[:10]
print("Top 10 significant features by %s test:\n" % test_name)
for feat in topfeats.index:
    print(feat)

# TODO: Perform post-hoc analyses (eg.Tukey HSD) for pairwise comparisons between foods for each feature?


#%% BOX PLOTS - INDIVIDUAL PLOTS OF TOP RANKED FEATURES (STATS) FOR EACH FOOD
# - Rank features by pvalue significance (lowest first) and select the Top 10 features for each food
# - Plot boxplots of the most important features for each food compared to OP50
# - Plot features separately with feature as title and in separate folders for each food

# Load test results (pvalues) for plotting
# NB: Non-parametric ranksum test preferred over t-test as many features may not be normally distributed
if is_normal:
    test_name = 'ttest_ind'
else:
    test_name = 'ranksumtest'
    
stats_inpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Stats', test_name + '_results.csv')
test_pvalues_df = pd.read_csv(stats_inpath, index_col=0)
print("Loaded %s results." % test_name)

plt.ioff()
plt.close('all')
print("\nPlotting box plots of top %d (%s) features for each food:\n" % (n_top_features, test_name))
for i, food in enumerate(test_pvalues_df.index):
    pvals = test_pvalues_df.loc[food]
    n_sigfeats = sum(pvals < p_value_threshold)
    n_nonnanfeats = np.logical_not(pvals.isna()).sum()
    if pvals.isna().all():
        print("No signficant features found for %s" % food)
    elif n_sigfeats > 0:       
        # Rank p-values in ascending order
        ranked_pvals = pvals.sort_values(ascending=True)
        # Drop NaNs
        ranked_pvals = ranked_pvals.dropna(axis=0)
        topfeats = ranked_pvals[:n_top_features] # Select the top ranked p-values
        topfeats = topfeats[topfeats < p_value_threshold] # Drop non-sig feats   
        ## OPTIONAL: Cherry-picked (relatable) features
        #topfeats = pd.Series(index=['curvature_midbody_abs_90th'])
        
        if n_sigfeats < n_top_features:
            print("Only %d significant features found for %s" % (n_sigfeats, food))
        #print("\nTop %d features for %s:\n" % (len(topfeats), food))
        #print(*[feat + '\n' for feat in list(topfeats.index)])

        # Subset feature summary results for test-food + OP50-control only
        plot_df = fullresults[np.logical_or(fullresults['food_type'].str.upper()==CONTROL_STRAIN,\
                                            fullresults['food_type'].str.upper()==food)] 
    
        # Colour/legend dictionary
        labels = list(plot_df['food_type'].str.upper().unique())
        colour_dict = {food:'#C2FDBE', CONTROL_STRAIN:'#0C9518'}
                                              
        # Boxplots of OP50 vs test-food for each top-ranked significant feature
        for f, feature in enumerate(topfeats.index):
            plt.close('all')
            sns.set_style('darkgrid')
            fig = plt.figure(figsize=[10,8])
            ax = fig.add_subplot(1,1,1)
            sns.boxplot(x="food_type_upper", y=feature, data=plot_df, palette=colour_dict,\
                        showfliers=False, showmeans=True,\
                        meanprops={"marker":"x", "markersize":5, "markeredgecolor":"k"},\
                        flierprops={"marker":"x", "markersize":15, "markeredgecolor":"r"})
            sns.swarmplot(x="food_type_upper", y=feature, data=plot_df, s=10, marker=".", color='k')
            ax.set_xlabel('Bacterial Strain (Food)', fontsize=15, labelpad=12)
            ax.set_ylabel(feature, fontsize=15, labelpad=12)
            ax.set_title(feature, fontsize=20, pad=40)

            # Add plot legend
            patches = []
            for l, key in enumerate(colour_dict.keys()):
                patch = mpatches.Patch(color=colour_dict[key], label=key)
                patches.append(patch)
                if key == food:
                    ylim = plot_df[plot_df['food_type'].str.upper()==key][feature].max()
                    pval = test_pvalues_df.loc[key, feature]
                    if isinstance(pval, float) and pval < p_value_threshold:
                        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                        ax.text(l - 0.1, 1, 'p={:g}'.format(float('{:.2g}'.format(pval))),\
                        fontsize=13, color='k', verticalalignment='bottom', transform=trans)
            plt.tight_layout(rect=[0.04, 0, 0.84, 0.96])
            plt.legend(handles=patches, labels=colour_dict.keys(), loc=(1.02, 0.8),\
                      borderaxespad=0.4, frameon=False, fontsize=15)

            # Save figure
            plots_outpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots',\
                                         food, '{0}_'.format(f + 1) + feature + '.eps')
            directory = os.path.dirname(plots_outpath)
            if not os.path.exists(directory):
                os.makedirs(directory)
            print("[%d] Saving figure: %s" % (i, os.path.basename(plots_outpath)))
            plt.savefig(plots_outpath, format='eps', dpi=300)
            if show_plots:
                plt.show(); plt.pause(2)
            plt.close(fig) # Close plot


#%% PLOT FEATURE SUMMARIES -- ALL FOODS (for Avelino's Top 256)
## - Investigate Avelino's top 256 features to look for any differences between foods (see paper: Javer et al, 2018)
## - Plot features that show significant differences from behaviour on OP50
#
## Number of foods to include in boxplots of Avelino's Top256 features
#n_top_foods2plt = 10
#
#if is_normal:
#    test_name = 'ttest_ind'
#else:
#    test_name = 'ranksumtest'
#    
#stats_inpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Stats', test_name + '_results.csv')
#test_pvalues_df = pd.read_csv(stats_inpath, index_col=0)
#print("Loaded %s results." % test_name)
#    
## Only plot features in Top256 list
#features2plot = [feature for feature in featurelist if feature in test_pvalues_df.columns]
#   
## Only plot features displaying significant differences between any food and OP50 control
#features2plot = [feature for feature in features2plot if (test_pvalues_df[feature] < p_value_threshold).any()]
#print("Dropped %d insignificant features from Top256." % (len(featurelist) - len(features2plot)))
#
## OPTIONAL: Plot cherry-picked features
##features2plot = ['speed_50th','curvature_neck_abs_50th','major_axis_50th','angular_velocity_neck_abs_50th']
#
## Seaborn boxplots with swarmplot overlay for each feature - saved to file
#tic = time.time()
#plt.ioff()
#sns.set(color_codes=True); sns.set_style('darkgrid')
#for f, feature in enumerate(features2plot):
#    plotpath_out = os.path.join(PROJECT_ROOT_DIR, "Results", "Plots", "All",\
#                                "Top256_Javer_2018", feature + '.eps')
#    foods2plt = list(test_pvalues_df[feature].sort_values(ascending=True)[:n_top_foods2plt].index)
#    foods2plt.insert(0, CONTROL_STRAIN)
#    plot_df = fullresults[fullresults['food_type'].isin(foods2plt)]
#    plot_df = plot_df.set_index("food_type").loc[foods2plt].reset_index()
#
#    # Seaborn boxplots for each feature (only top foods)
#    print("""Plotting Top%d significantly different foods for: '%s'""" % (n_top_foods2plt, feature))
#    fig = plt.figure(figsize=[12,7])
#    ax = fig.add_subplot(1,1,1)
#    sns.boxplot(x="food_type", y=feature, data=plot_df, showfliers=False, showmeans=True,\
#                meanprops={"marker":"x", "markersize":5, "markeredgecolor":"k"},\
#                flierprops={"marker":"x", "markersize":15, "markeredgecolor":"r"})
#    sns.swarmplot(x="food_type", y=feature, data=plot_df, s=10, marker=".", color='k')
#    ax.set_xlabel('Bacterial Strain (Test Food)', fontsize=15, labelpad=12)
#    ax.set_ylabel(feature, fontsize=15, labelpad=12)
#    ax.set_title(feature, fontsize=20, pad=20)                             # Set title
#    labels = [lab.get_text().upper() for lab in ax.get_xticklabels()]      # Get x-labels
#    ax.set_xticklabels(labels, rotation=45)                                # Rotate x-labels
#    labels.remove(CONTROL_STRAIN)
#    # TODO: Plot strains/labels in correct order!!!
#    for l, food in enumerate(labels):
#        #ylim = plot_df[plot_df['food_type']==food][feature].max()
#        pval = test_pvalues_df.loc[food, feature]
#        if isinstance(pval, float) and pval < p_value_threshold:
#            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
#            ax.text(l + 0.75, 0, 'p={:g}'.format(float('{:.2g}'.format(pval))),\
#                    fontsize=10, color='k', verticalalignment='bottom', transform=trans)
#    plt.subplots_adjust(top=0.9,bottom=0.2,left=0.1,right=0.95)
#    
#    # Save boxplots
#    directory = os.path.dirname(plotpath_out)
#    if not os.path.exists(directory):
#        os.makedirs(directory)
#    savefig(plotpath_out, tellme=False, saveFormat='eps')
#    if show_plots:
#        plt.show(); plt.pause(2)
#    plt.close(fig) # Close plot
#
#toc = time.time()
#print("Time taken: %.1f seconds" % (toc - tic))


#%% Plot of all foods for significant Top256 feats (flip x/y axis)

if is_normal:
    test_name = 'ttest_ind'
else:
    test_name = 'ranksumtest'
    
stats_inpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Stats', test_name + '_results.csv')
test_pvalues_df = pd.read_csv(stats_inpath, index_col=0)
print("Loaded %s results." % test_name)

#test_pvalues_df = test_pvalues_df[features2plot].loc['pval_corrected'].sort_values(ascending=True) 
   
# Only plot features in Top256 list
features2plot = [feature for feature in featurelist if feature in test_pvalues_df.columns]

# Only plot features displaying significant differences between any food and OP50 control
features2plot = [feature for feature in features2plot if (test_pvalues_df[feature] < p_value_threshold).any()]
print("Dropped %d insignificant features from Top256." % (len(featurelist) - len(features2plot)))

# OPTIONAL: Plot cherry-picked features
#features2plot = ['speed_50th','curvature_neck_abs_50th','major_axis_50th','angular_velocity_neck_abs_50th']

# Seaborn boxplots with swarmplot overlay for each feature - saved to file
tic = time.time()
plt.ioff()
sns.set(color_codes=True); sns.set_style('darkgrid')
for f, feature in enumerate(features2plot):
    plotpath_out = os.path.join(PROJECT_ROOT_DIR, "Results", "Plots", "All",\
                                "Top256_Javer_2018", str(f + 1) + '_' + feature + '.eps')
    sortedPvals = test_pvalues_df[feature].sort_values(ascending=True)
    foods2plt = list(sortedPvals.index)
    if len(foods2plt) > max_sigdiff_strains_plot_cap:
        foods2plt = list(sortedPvals[:max_sigdiff_strains_plot_cap].index)
        
    foods2plt.insert(0, CONTROL_STRAIN)
    plot_df = fullresults[fullresults['food_type'].str.upper().isin(foods2plt)]
    
    # Rank by median
    rankMedian = plot_df.groupby('food_type')[feature].median().sort_values(ascending=True)
    #plot_df = plot_df.set_index("food_type").loc[foods2plt].reset_index()
    plot_df = plot_df.set_index("food_type").loc[list(rankMedian.index)].reset_index()
    colour_dict = {strain: "r" if strain == "OP50" else "darkgray" for strain in plot_df['food_type'].str.upper().unique()}
    colour_dict2 = {strain: "b" for strain in list(sortedPvals[sortedPvals < p_value_threshold].index)}
    colour_dict.update(colour_dict2)
    
    # Seaborn boxplot for each feature (only top foods)
    if max_sigdiff_strains_plot_cap:
        print("Plotting Top %d strains for: %s" % (max_sigdiff_strains_plot_cap, feature))
    else:
        print("""Plotting all strains for: '%s'""" % feature)
    fig = plt.figure(figsize=[7,14])
    ax = fig.add_subplot(1,1,1)
    sns.boxplot(x=feature, y="food_type_upper", data=plot_df, showfliers=False,\
                showmeans=True,\
                meanprops={"marker":"x", "markersize":5, "markeredgecolor":"k"},\
                flierprops={"marker":"x", "markersize":15, "markeredgecolor":"r"},\
                palette=colour_dict)
    ax.set_xlabel(feature, fontsize=18, labelpad=10)
    ax.set_ylabel('Bacterial Strain', fontsize=18, labelpad=10)
    locs, labels = plt.yticks() # Get y-axis tick positions and labels
    labs = [lab.get_text().upper() for lab in labels]
    #trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    
    for l, food in enumerate(labs):
        if food == CONTROL_STRAIN:
            plt.axvline(x=rankMedian[CONTROL_STRAIN], c='dimgray', ls='--')
            continue
        pval = test_pvalues_df.loc[food, feature]
        if isinstance(pval, float) and pval < p_value_threshold:
            xmin, xmax = ax.get_xlim()
            xtext = xmin + 1*(xmax - xmin)
            ax.text(xtext, locs[l], 'p={:g}'.format(float('{:.2g}'.format(pval))),\
                    fontsize=10, color='k', 
                    horizontalalignment='left', verticalalignment='center')
    plt.subplots_adjust(top=0.9,bottom=0.1,left=0.2,right=0.85)
     
    # Save boxplot
    directory = os.path.dirname(plotpath_out)
    if not os.path.exists(directory):
        os.makedirs(directory)
    savefig(plotpath_out, tellme=False, saveFormat='eps')
    if show_plots:
        plt.show(); plt.pause(2)
    plt.close(fig) # Close plot

toc = time.time()
print("Time taken: %.1f seconds" % (toc - tic))
      

#%% HIERARCHICAL CLUSTERING (HEATMAP) - Top256 Features
# Clustermap of features by foods, to see if data cluster into
# groups for each food - does OP50 control form a nice group?
# NB: This has to be restricted to Top256 else plot will be too large

HCA_by_strain = True
                  
# Compute average value for strain for each feature (not each well)
if HCA_by_strain:
    strainMean_df = fullresults.groupby('food_type').mean().reset_index()
    
    # Subset for important strains
    # Read in stats results to rank strains by importance for plot cap on n_strains
    if is_normal:
        test_name = 'ttest_ind'
    else:
        test_name = 'ranksumtest'
    
    stats_inpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Stats', test_name + '_results.csv')
    test_pvalues_df = pd.read_csv(stats_inpath, index_col=0)
    print("Loaded %s results." % test_name)

    foods2plt = (test_pvalues_df == test_pvalues_df.min(axis=0)).sum(axis=1).sort_values(ascending=False)[:max_sigdiff_strains_plot_cap].index
    strainMean_df = strainMean_df[strainMean_df['food_type'].str.upper().isin(foods2plt)]
else:
    strainMean_df = fullresults

# Divide results dataframe into data + non-data
results_feats = strainMean_df.reindex(columns=featurelist)
results_meta = strainMean_df.reindex(columns=metadata_colnames) 

# Normalise the data (minus mean, divide by std)
zscores = (results_feats-results_feats.mean())/results_feats.std()
#zscores = results_feats.apply(zscore, axis=0)

plotroot = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'All', 'HCA')
colour_dictionary = dict(zip(BACTERIAL_STRAINS, sns.color_palette("gist_rainbow", len(BACTERIAL_STRAINS))))

# Heatmap (clustergram) of Top10 features per food (n=45)
plt.close('all')
row_colours = results_meta['food_type'].str.upper().map(colour_dictionary)
sns.set(font_scale=0.6)
g = sns.clustermap(zscores, #row_colors=row_colours,
                   standard_scale=1, # z_score=1
                   metric='euclidean', method='complete',\
                   figsize=[15,10], #xticklabels=3,
                   yticklabels=results_meta['food_type'],
                   xticklabels=False)

#plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), fontsize=10)
#patches = []
#for l, key in enumerate(colour_dictionary.keys()):
#    patch = mpatches.Patch(color=colour_dictionary[key], label=key)
#    patches.append(patch)
#plt.legend(handles=patches, labels=colour_dictionary.keys(),\
#           borderaxespad=0.4, frameon=False, loc=(-3, -13), fontsize=8)
plt.subplots_adjust(top=0.95,bottom=0.05,left=0.02,right=0.92,hspace=0.2,wspace=0.2)

# Save clustermap and features of interest
cluster_outpath = os.path.join(plotroot, 'Hierarchical_Clustering_Top256.eps')
directory = os.path.dirname(cluster_outpath)
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig(cluster_outpath, tight_layout=True, dpi=300, saveFormat='eps')
plt.show(); plt.pause(5)


#%% PRINCIPAL COMPONENTS ANALYSIS (PCA) - ALL FOODS Top256

# Divide results into: data (feature summaries) + non-data (metadata)
results_feats = fullresults.reindex(columns=featurelist)
results_meta = fullresults.reindex(columns=metadata_colnames) 

# Normalise the data (z-scores)
zscores = results_feats.apply(zscore, axis=0)

# Drop features with NaN values after normalising
zscores.dropna(axis=1, inplace=True)
print("Dropped %d features after normalisation (NaN)" % (len(results_feats.columns)-len(zscores.columns)))

print("Using Top256 feature list for dimensionality reduction...")
top256featcols = [feat for feat in zscores.columns if feat in featurelist]
zscores = zscores[top256featcols]

# NB: In general, for the curvature and angular velocity features we should only 
# use the 'abs' versions, because the sign is assigned based on whether the worm 
# is on its left or right side and this is not known for the multiworm tracking data

plt.ion() 
tic = time.time()
plotroot = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'All', 'PCA')

# Perform PCA on extracted features
print("\nPerforming Principal Components Analysis (PCA)...")

# Fit the PCA model with the normalised data
pca = PCA()
pca.fit(zscores)

# Plot summary data from PCA: explained variance (most important features)
important_feats, fig = pcainfo(pca=pca, zscores=zscores, PC=1, n_feats2print=n_PC_feats2plot)

# Save plot of PCA explained variance
plotpath = os.path.join(plotroot, 'PCA_explained.eps')
directory = os.path.dirname(plotpath)
if not os.path.exists(directory):
    os.makedirs(directory)
savefig(plotpath, tight_layout=True, tellme=True, saveFormat='eps')

# Project data (zscores) onto PCs
projected = pca.transform(zscores) # A matrix is produced
# NB: Could also have used pca.fit_transform() OR decomposition.TruncatedSVD().fit_transform()

# Store the results for first few PCs in dataframe
projected_df = pd.DataFrame(projected[:,:PCs_to_keep],\
                            columns=['PC' + str(n+1) for n in range(PCs_to_keep)])

# Add concatenate projected PC results to metadata
projected_df.set_index(fullresults.index, inplace=True) # Do not lose index position
projected_df = pd.concat([fullresults[metadata_colnames], projected_df], axis=1)

# TODO: Store PCA important features list + plot for each feature
#for feature in important_feats:
#    print("Plotting %s" % feature)

# Plot 2-Component PCA
plt.close()
plotpath_2d = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'All', 'PCA', 'PCA_2PCs_byStrain.eps')
title = """2-Component PCA - All Strains""" + """
(Top256 features)"""
plotPCA(projected_df, grouping_variable='food_type_upper', var_subset=BACTERIAL_STRAINS,\
        savepath=plotpath_2d, title=title, n_component_axes=2)
plt.pause(5)


#%% Remove outliers: Use Mahalanobis distance to exclude outliers from PCA

indsOutliers = MahalanobisOutliers(projected, showplot=True)
plt.pause(5); plt.close()

# Drop outlier observation(s)
print("Dropping %d outliers from analysis" % len(indsOutliers))
indsOutliers = results_feats.index[indsOutliers]
results_feats = results_feats.drop(index=indsOutliers)
fullresults = fullresults.drop(index=indsOutliers)

# Re-normalise data
zscores = results_feats.apply(zscore, axis=0)

# Drop features with NaN values after normalising
zscores.dropna(axis=1, inplace=True)
print("Dropped %d features after normalisation (NaN)" % (len(results_feats.columns)-len(zscores.columns)))

# Use Top256 features
print("Using Top256 feature list for dimensionality reduction...")
top256featcols = [feat for feat in zscores.columns if feat in featurelist]
zscores = zscores[top256featcols]

# Project data on PCA axes again
pca = PCA()
pca.fit(zscores)
projected = pca.transform(zscores) # project data (zscores) onto PCs
important_feats, fig = pcainfo(pca=pca, zscores=zscores, PC=1, n_feats2print=10)
plt.pause(5); plt.close()

# Store the results for first few PCs
projected_df = pd.DataFrame(projected[:,:10],\
                              columns=['PC' + str(n+1) for n in range(10)])
projected_df.set_index(fullresults.index, inplace=True) # Do not lose index position
projected_df = pd.concat([fullresults[metadata_colnames], projected_df], axis=1)


#%% Plot PCA - All bacterial strains (food)

topNstrains = 5

if is_normal:
    test_name = 'ttest_ind'
else:
    test_name = 'ranksumtest'
    
stats_inpath = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Stats', test_name + '_results.csv')
test_pvalues_df = pd.read_csv(stats_inpath, index_col=0)
print("Loaded %s results." % test_name)

propfeatssigdiff = ((test_pvalues_corrected_df < p_value_threshold).sum(axis=1)/len(featurelist))*100
propfeatssigdiff = propfeatssigdiff.sort_values(ascending=False)

topStrains = list(propfeatssigdiff[:topNstrains].index)
topStrains_projected_df = projected_df[projected_df['food_type'].str.upper().isin(topStrains)]
topStrains.insert(0, CONTROL_STRAIN)

otherStrains = [strain for strain in BACTERIAL_STRAINS if strain not in topStrains]
otherStrains_projected_df = projected_df[projected_df['food_type'].str.upper().isin(otherStrains)]

# Create colour palette for plot loop
#colour_dict_other = {strain: "r" if strain == "OP50" else "darkgray" for strain in otherStrains}
colour_dict_other = {strain: "darkgray" for strain in otherStrains}
topcols = sns.color_palette("Paired", len(topStrains))
colour_dict_top = {strain: topcols[i] for i, strain in enumerate(topStrains)}
#colour_dict.update(colour_dict2)
#palette = itertools.cycle(sns.color_palette("gist_rainbow", len(BACTERIAL_STRAINS)))

plt.close()
plotpath_2d = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'All', 'PCA', 'PCA_2PCs_byStrain.eps')
title = None #"""2-Component PCA (Top256 features)"""
plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=[10,10])

palette_other = itertools.cycle(list(colour_dict_other.values()))
for strain in otherStrains:
    strain_projected_df = projected_df[projected_df['food_type'].str.upper()==strain]
    sns.scatterplot(strain_projected_df['PC1'], strain_projected_df['PC2'],\
                    color=next(palette_other), s=50, alpha=0.65, linewidth=0)

palette_top = itertools.cycle(list(colour_dict_top.values()))
for strain in topStrains:
    strain_projected_df = projected_df[projected_df['food_type'].str.upper()==strain]
    sns.scatterplot(strain_projected_df['PC1'], strain_projected_df['PC2'],\
                    color=next(palette_top), s=70, edgecolor='k') # marker="^"
    
ax.set_xlabel('Principal Component 1', fontsize=20, labelpad=12)
ax.set_ylabel('Principal Component 2', fontsize=20, labelpad=12)
if title:
    ax.set_title(title, fontsize=20)

# Add plot legend
patches = []
for l, key in enumerate(colour_dict_top.keys()):
    patch = mpatches.Patch(color=colour_dict_top[key], label=key)
    patches.append(patch)
plt.legend(handles=patches, labels=colour_dict_top.keys(), frameon=False, fontsize=12)
ax.grid()

# Save PCA scatterplot of first 2 PCs
savefig(plotpath_2d, tight_layout=False, tellme=True, saveFormat='png') # rasterized=True
plt.show(); plt.pause(2)

## 3-PC
#plt.close()
#plotpath_3d = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'All', 'PCA', 'PCA_3PCs_byStrain.eps')
#title = """3-Component PCA - All Strains""" + """
#(Top256 features)"""
#plotPCA(projected_df, grouping_variable='food_type', var_subset=BACTERIAL_STRAINS,\
#        savepath=None, title=title, n_component_axes=3, rotate=False)
#plt.pause(2)

# TODO: Plot features that have greatest influence on PCA (eg. PC1)


# TODO: Fix t-SNE and UMAP params/legend

#%% t-distributed Stochastic Neighbour Embedding (t-SNE)

plotroot = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'All', 'tSNE')

# Perform tSNE on extracted features
print("\nPerforming t-distributed stochastic neighbour embedding (t-SNE)...")

for perplex in perplexities:
    # 2-COMPONENT t-SNE
    tSNE_embedded = TSNE(n_components=2, init='random', random_state=42,\
                         perplexity=perplex, n_iter=3000).fit_transform(zscores)
    tSNE_results_df = pd.DataFrame(tSNE_embedded, columns=['tSNE_1', 'tSNE_2'])
    
    # Combine tSNE results with metadata
    tSNE_results_df.set_index(fullresults.index, inplace=True) # Do not lose video snippet index position
    tSNE_results_df = pd.concat([fullresults[metadata_colnames], tSNE_results_df], axis=1)
    
    # Plot 2-D tSNE
    plt.close('all')
    plt.rc('xtick',labelsize=12)
    plt.rc('ytick',labelsize=12)
    fig = plt.figure(figsize=[10,10])
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('tSNE Component 1', fontsize=15, labelpad=12)
    ax.set_ylabel('tSNE Component 2', fontsize=15, labelpad=12)
    ax.set_title('2-component tSNE (Top256 features, perplexity={0})'.format(perplex), fontsize=20)
            
    # Create colour palette for plot loop
    palette = itertools.cycle(sns.color_palette("gist_rainbow", len(BACTERIAL_STRAINS))) # 'gist_rainbow'
    
    for food in BACTERIAL_STRAINS:
        food_tSNE_df = tSNE_results_df[tSNE_results_df['food_type'].str.upper()==food]
        sns.scatterplot(food_tSNE_df['tSNE_1'], food_tSNE_df['tSNE_2'], color=next(palette), s=100)
    plt.tight_layout(rect=[0.04, 0, 0.84, 0.96])
    #ax.legend(BACTERIAL_STRAINS, frameon=False, loc=(1, 0.1), fontsize=15)
    ax.grid()
    
    plotpath = os.path.join(plotroot, 'tSNE_perplex={0}.eps'.format(perplex))
    directory = os.path.dirname(plotpath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    savefig(plotpath, tight_layout=True, tellme=True, saveFormat='eps')
    
    plt.show(); plt.pause(2)

      
#%% Uniform Manifold Projection (UMAP)

plotroot = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'All', 'UMAP')

# Perform UMAP on extracted features
print("\nPerforming uniform manifold projection (UMAP)...")
assert type(n_neighbours) == list
for n in n_neighbours:
    UMAP_projection = umap.UMAP(n_neighbors=n,\
                                min_dist=min_dist,\
                                metric='correlation').fit_transform(zscores)
    
    UMAP_projection_df = pd.DataFrame(UMAP_projection, columns=['UMAP_1', 'UMAP_2'])
    UMAP_projection_df.shape
    
    # Add UMAP results to metadata
    UMAP_projection_df.set_index(fullresults.index, inplace=True) # Do not lose video snippet index position
    UMAP_projection_df = pd.concat([fullresults[metadata_colnames], UMAP_projection_df], axis=1)
    
    # Plot 2-D UMAP
    plt.close('all')
    sns.set_style('whitegrid')
    plt.rc('xtick',labelsize=12)
    plt.rc('ytick',labelsize=12)
    fig = plt.figure(figsize=[11,10])
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('UMAP Component 1', fontsize=15, labelpad=12)
    ax.set_ylabel('UMAP Component 2', fontsize=15, labelpad=12)
    ax.set_title('2-component UMAP (Top256 features, n_neighbours={0})'.format(n), fontsize=20)
            
    # Create colour palette for plot loop
    palette = itertools.cycle(sns.color_palette("gist_rainbow", len(BACTERIAL_STRAINS)))
    
    for food in BACTERIAL_STRAINS:
        food_UMAP_df = UMAP_projection_df[UMAP_projection_df['food_type'].str.upper()==food]
        sns.scatterplot(food_UMAP_df['UMAP_1'], food_UMAP_df['UMAP_2'], color=next(palette), s=100)
    plt.tight_layout(rect=[0.04, 0, 0.84, 0.96])
    #ax.legend(BACTERIAL_STRAINS, frameon=False, loc=(1, 0.1), fontsize=15)
    ax.grid()
    
    plotpath = os.path.join(plotroot, 'UMAP_n_neighbours={0}.eps'.format(n))
    directory = os.path.dirname(plotpath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    savefig(plotpath, tight_layout=True, tellme=True, saveFormat='eps')
    
    plt.show(); plt.pause(2)


#%% Plot PCA - Variation in all strains with respect to various confounding variables
    
#
# Include L1 diapause duration (recorded in metadata for microbiome assay only)
if TIMEPOINT:
    variables_list.insert(len(variables_list), "time_point")
if ANALYSIS == 'microbiome':
    variables_list.insert(len(variables_list), "L1_diapause_seconds")

for g, grouping_variable in enumerate(variables_list):
    print("\n%d - PCA of variation across '%s':" % (g+1, grouping_variable))

    # Plot PCA - Variation across experiment days (all strains)
    
    # 2-PC
    plt.close()
    plotpath_2d = os.path.join(PROJECT_ROOT_DIR, 'Results', 'Plots', 'All', 'PCA',\
                               'PCA_2PCs_by_{0}.eps'.format(grouping_variable))
    title = """2-Component PCA - {0}""".format(grouping_variable) + """
    (Top256 features)"""
    plotPCA(projected_df, grouping_variable=grouping_variable, var_subset=None,\
            savepath=plotpath_2d, title=title, n_component_axes=2)
    plt.pause(2)
    
    # 3-PC
    plt.close()
    plotpath_3d = plotpath_2d.replace("_2PCs_", "_3PCs_")
    title = title.replace("2-Component", "3-Component")
    plotPCA(projected_df, grouping_variable=grouping_variable, var_subset=None,\
            savepath=None, title=title, n_component_axes=3, rotate=False)
    plt.pause(2)


#%%
    
bigtoc = time.time()
print("Total time taken (seconds): %.1f" % (bigtoc - bigtic))


#%% 
# TODO: Find out why pangenome results only for first 96 wells?
# TODO: Could perform GLM (or similar) to look for age-bacteria interaction effects - but not so much interested in ageing
# TODO: Perform cPCA / LLE / LDA ???

