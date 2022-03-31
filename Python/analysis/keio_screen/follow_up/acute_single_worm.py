#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compile acute single worm metadata and feature summaries

@author: sm5911
@date: 30/03/2022

"""

#%% Imports

import pandas as pd
from pathlib import Path
from preprocessing.compile_hydra_data import compile_metadata, process_feature_summaries
from filter_data.clean_feature_summaries import clean_summary_results

#%% Globals

PROJECT_DIR = "/Volumes/hermes$/Keio_Acute_Single_Worm"
SAVE_DIR = "/Users/sm5911/Documents/Keio_Acute_Single_Worm"

IMAGING_DATES = ['20220206','20220209','20220212']
N_WELLS = 6

NAN_THRESHOLD_ROW = 0.8
NAN_THRESHOLD_COL = 0.05
MIN_NSKEL_PER_VIDEO = None
MIN_NSKEL_SUM = 50

WINDOW_DICT_SECONDS = {0:(290,300), 1:(305,315), 2:(315,325), 
                       3:(590,600), 4:(605,615), 5:(615,625), 
                       6:(890,900), 7:(905,915), 8:(915,925), 
                       9:(1190,1200), 10:(1205,1215), 11:(1215,1225), 
                       12:(1490,1500), 13:(1505.1515), 14:(1515,1525)}

#%% Functions

def acute_single_worm_stats(metadata, features, project_dir=PROJECT_DIR):
    """ Pairwise t-tests for each window comparing worm 'motion mode paused fraction' on 
        Keio mutants vs BW control 
    """

    # # categorical variables to investigate: 'gene_name' and 'window'
    # print("\nInvestigating variation in fraction of worms paused between hit strains and control " +
    #       "(for each window)")    

    # # assert there will be no errors due to case-sensitivity
    # assert len(metadata['gene_name'].unique()) == len(metadata['gene_name'].str.upper().unique())
        
    # # subset for windows in window_frame_dict
    # assert all(w in metadata['window'] for w in window_list)
    # metadata = metadata[metadata['window'].isin(window_list)]
    # features = features.reindex(metadata.index)

    # control_strain = args.control_dict['gene_name']
    # strain_list = list([s for s in metadata['gene_name'].unique() if s != control_strain])    

    # # print mean sample size
    # sample_size = df_summary_stats(metadata, columns=['gene_name', 'window'])
    # print("Mean sample size of strain/window: %d" % (int(sample_size['n_samples'].mean())))
    
    # # construct save paths (args.save_dir / topfeats? etc)
    # save_dir = get_save_dir(args)
    # stats_dir =  save_dir / "Stats" / args.fdr_method
        
    # control_meta = metadata[metadata['gene_name']==control_strain]
    # control_feat = features.reindex(control_meta.index)
    # control_df = control_meta.join(control_feat[[FEATURE]])
    
    # for strain in strain_list:
    #     print("\nPairwise t-tests for each window comparing fraction of worms paused " +
    #           "on %s vs control" % strain)
    #     strain_meta = metadata[metadata['gene_name']==strain]
    #     strain_feat = features.reindex(strain_meta.index)
    #     strain_df = strain_meta.join(strain_feat[[FEATURE]])
         
    #     stats, pvals, reject = pairwise_ttest(control_df, 
    #                                           strain_df, 
    #                                           feature_list=[FEATURE], 
    #                                           group_by='window', 
    #                                           fdr_method=args.fdr_method,
    #                                           fdr=0.05)
 
    #     # compile table of results
    #     stats.columns = ['stats_' + str(c) for c in stats.columns]
    #     pvals.columns = ['pvals_' + str(c) for c in pvals.columns]
    #     reject.columns = ['reject_' + str(c) for c in reject.columns]
    #     test_results = pd.concat([stats, pvals, reject], axis=1)
        
    #     # save results
    #     ttest_strain_path = stats_dir / 'pairwise_ttests' / '{}_window_results.csv'.format(strain)
    #     ttest_strain_path.parent.mkdir(parents=True, exist_ok=True)
    #     test_results.to_csv(ttest_strain_path, header=True, index=True)
        
    #     for window in window_list:
    #         print("%s difference in '%s' between %s vs %s in window %s (paired t-test, P=%.3f, %s)" %\
    #               (("SIGNIFICANT" if reject.loc[FEATURE, 'reject_{}'.format(window)] else "No"), 
    #               FEATURE, strain, control_strain, window, pvals.loc[FEATURE, 'pvals_{}'.format(window)],
    #               args.fdr_method))
        
    return

def acute_single_worm():
    
    
    window_list = metadata['window'].unique()
    assert all(w in WINDOW_DICT_SECONDS.keys() for w in window_list)
    grouped_window = metadata.groupby('window')
    for window in window_list:
        window_metadata = grouped_window.get_group(window)

#%% Main

if __name__ == "__main__":
    
    AUX_DIR = Path(PROJECT_DIR) / 'AuxiliaryFiles'
    RES_DIR = Path(PROJECT_DIR) / 'Results'

    # compile metadata
    metadata, metadata_path = compile_metadata(aux_dir=AUX_DIR,
                                               imaging_dates=IMAGING_DATES,
                                               add_well_annotations=N_WELLS==96,
                                               n_wells=N_WELLS,
                                               from_source_plate=False)
        
    # compile window summaries
    features, metadata = process_feature_summaries(metadata_path,
                                                   results_dir=RES_DIR,
                                                   compile_day_summaries=True,
                                                   imaging_dates=IMAGING_DATES,
                                                   align_bluelight=False,
                                                   window_summaries=True)
    
    # clean results
    features, metadata = clean_summary_results(features, 
                                               metadata,
                                               feature_columns=None,
                                               nan_threshold_row=NAN_THRESHOLD_ROW,
                                               nan_threshold_col=NAN_THRESHOLD_COL,
                                               max_value_cap=1e15,
                                               imputeNaN=True,
                                               min_nskel_per_video=MIN_NSKEL_PER_VIDEO,
                                               min_nskel_sum=MIN_NSKEL_SUM,
                                               drop_size_related_feats=False,
                                               norm_feats_only=False,
                                               percentile_to_use=None)

    assert not features.isna().sum(axis=1).any()
    assert not (features.std(axis=1) == 0).any()

    # save features
    features_path = Path(SAVE_DIR) / 'features.csv'
    features.to_csv(features_path, index=False) 

    # Save metadata
    metadata_path = Path(SAVE_DIR) / 'metadata.csv'
    metadata.to_csv(metadata_path, index=False)


    #acute_single_worm(metadata, features, PROJECT_DIR, SAVE_DIR)

