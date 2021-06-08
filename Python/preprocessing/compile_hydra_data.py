#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compile metadata and Tierpsy feature summaries results across days

@author: sm5911
@date: 8/2/21

"""

#%% Imports

import os
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

#%% Functions

def compile_day_metadata(aux_dir, day, from_source_plate=False, from_robot_runlog=False, verbose=True):
    """ Compile experiment day metadata from wormsorter and hydra rig metadata for a given day in 
        'AuxilliaryFiles' directory 
        
        Parameters
        ----------
        aux_dir : str
            Path to "AuxiliaryFiles" containing metadata  
        day : str, None
            Experiment day folder in format 'YYYYMMDD'
            
        Returns
        -------
        compiled_day_metadata
    """

    from tierpsytools.hydra.compile_metadata import (populate_96WPs, 
                                                     get_day_metadata,
                                                     #get_source_plate_metadata
                                                     number_wells_per_plate, 
                                                     day_metadata_check)

    day_dir = Path(aux_dir) / str(day)
    wormsorter_meta = day_dir / (str(day) + '_wormsorter.csv')
    hydra_meta = day_dir / (str(day) + '_manual_metadata.csv')
      
    # Expand wormsorter metadata to have separate row for each well
    plate_metadata = populate_96WPs(wormsorter_meta)
    
    # Create dataframe with metadata row for each well
# =============================================================================
#     if from_source_plate:
#         sourceplates_file = get_source_plate_metadata()
# =============================================================================
        
# =============================================================================
#     if from_robot_runlog:
#         from tierpsytools.hydra.compile_metadata import merge_robot_metadata, merge_robot_wormsorter
#         drug_metadata = merge_robot_metadata(sourceplates_file,
#                                              randomized_by='column',
#                                              saveto=None,
#                                              drug_by_column=True,
#                                              compact_drug_plate=False,
#                                              del_if_exists=False)
#         plate_metadata = merge_robot_wormsorter(day_dir, 
#                                                 drug_metadata,
#                                                 plate_metadata,
#                                                 bad_wells_csv=None,
#                                                 merge_on=['imaging_plate_id', 'well_name'],
#                                                 saveto=None,
#                                                 del_if_exists=False)
# =============================================================================

    day_metadata = get_day_metadata(complete_plate_metadata=plate_metadata, 
                                    hydra_metadata_file=hydra_meta,
                                    merge_on=['imaging_plate_id'],
                                    n_wells=96,
                                    run_number_regex='run\\d+_',
                                    saveto=None,
                                    del_if_exists=False,
                                    include_imgstore_name=True,
                                    raw_day_dir=None)
    
    # Check that day metadata is correct dimension
    day_metadata_check(day_metadata, day_dir)
    
    if verbose:
        print(number_wells_per_plate(day_metadata, day_dir))
    
    return day_metadata

def process_metadata(aux_dir, 
                     imaging_dates=None, 
                     add_well_annotations=True,
                     update_day_meta=False,
                     update_colnames=True):
    """ Compile metadata from individual day metadata CSV files
    
        Parameters
        ----------
        aux_dir : str
            Path to "AuxiliaryFiles" containing metadata 
        update_day_meta : bool
            Update existing day metadata files 
        imaging_dates : list of str, None
            List of day metadata imaging dates to compile
        add_well_annotations : bool
            Add annotations from WellAnnotator GUI
        update_colnames : bool
            Rename columns names for compatibility with 'tierpsytools' functions

        Returns
        -------
        Updated metadata
        compiled metadata path
    """
    
    from tierpsytools.hydra.match_wells_annotations import update_metadata_with_wells_annotations
    # from tierpsytools.hydra.match_wells_annotations import (import_wells_annotations_in_folder,
    #                                                         match_rawvids_annotations,
    #                                                         update_metadata)

    
    compiled_metadata_path = Path(aux_dir) / "metadata.csv"
    
    if compiled_metadata_path.exists():
        # Load metadata
        meta_df = pd.read_csv(compiled_metadata_path, dtype={"comments":str, 
                                                             "source_plate_id":str}, header=0)
        # subset for imaging dates
        if imaging_dates is not None:
            
            assert 'date_yyyymmdd' in meta_df.columns
            meta_df = meta_df[meta_df['date_yyyymmdd'].astype(str).isin(imaging_dates)]
            
        print("Metadata loaded.")
        
    else:
        # Compile metadata
        print("Metadata not found.\nCompiling from day metadata in: %s" % aux_dir)
        
        AuxFileList = os.listdir(aux_dir)
        dates = sorted([date for date in AuxFileList if re.match(r'\d{8}', date)])
        if imaging_dates:
            assert all(i in dates for i in imaging_dates)
        else:
            imaging_dates = dates
    
        day_meta_list = []
        for date in imaging_dates:
            day_meta_path = Path(aux_dir) / date / '{}_day_metadata.csv'.format(date)
                    
            day_meta = pd.read_csv(day_meta_path, dtype={"comments":str})
            day_meta['row_order'] = np.arange(len(day_meta))

            if update_colnames:
                # Rename metadata columns for compatibility with TierpsyTools functions 
                day_meta = day_meta.rename(columns={'date_recording_yyyymmdd': 'date_yyyymmdd',
                                                    'well_number': 'well_name',
                                                    'plate_number': 'imaging_plate_id',
                                                    'run_number': 'imaging_run_number',
                                                    'camera_number': 'camera_serial'})
             
            # Get path to RawVideo directory for day metadata
            rawDir = Path(str(day_meta_path).replace("AuxiliaryFiles", "RawVideos")).parent
            
            # Get imgstore name
            if 'imgstore_name' not in day_meta.columns:
                from tierpsytools.hydra.hydra_helper import add_imgstore_name                     

                # Delete camera_serial column as it will be recreated
                if 'camera_serial' in day_meta.columns:
                     day_meta = day_meta.drop(columns='camera_serial')
                     
                day_meta_col_order = list(day_meta.columns)

                # Add imgstore_name (+ camera_serial)
                day_meta = add_imgstore_name(day_meta, rawDir)
                
                day_meta_col_order.extend(['imgstore_name','camera_serial'])
                
                # Restore column order
                day_meta = day_meta[day_meta_col_order]
            else:
                assert not day_meta['imgstore_name'].isna().any()
                day_meta_col_order = list(day_meta.columns)
            
            # Get filename
            day_meta['filename'] = [rawDir.parent / day_meta.loc[i,'imgstore_name']\
                                    for i in range(day_meta.shape[0])]
            
            # save day metadata, keeping original row order
            day_meta = day_meta.sort_values(by='row_order').reset_index(drop=True)
            day_meta = day_meta.drop(columns='row_order')
            
            if update_day_meta:
                #day_meta_out_path = str(day_meta_path).replace(".csv", "_updated.csv")
                day_meta.to_csv(day_meta_path, index=False)
            
            # Append to compiled metadata list
            day_meta_list.append(day_meta)
        
        # Concatenate list of day metadata into full metadata
        meta_df = pd.concat(day_meta_list, axis=0, ignore_index=True, sort=False)

        # Ensure no NA values for filename or date recorded
        assert not any(meta_df['filename'].isna())
        
        # Ensure no NA values in any columns with 'date' in the name (and drop if all NA) for 
        # compatibility with fix_dtypes from tierpsytools platechecker
        check_na_cols = [col for col in meta_df.columns if 'date' in col]
        for col in check_na_cols:
            if all(meta_df[col].isna()):
                print("Removing column '%s' from metadata (all NA)" % col)
                meta_df = meta_df.drop(columns=[col])
            else:
                assert not any(meta_df[col].isna())
                
        # Convert 'date_yyyymmdd' column to string (factor)
        meta_df['date_yyyymmdd'] = meta_df['date_yyyymmdd'].astype(str)
        
        # Save metadata
        meta_df.to_csv(compiled_metadata_path, index=None) 
        print("Metadata saved to: %s" % compiled_metadata_path)
    
    # Add annotations to metadata
    if add_well_annotations:        
        annotated_metadata_path = Path(str(compiled_metadata_path).replace('.csv', 
                                                                           '_annotated.csv'))
        if not annotated_metadata_path.exists():
            print("Adding annotations to metadata")
            
            meta_df = update_metadata_with_wells_annotations(aux_dir=aux_dir, 
                                                             saveto=annotated_metadata_path, 
                                                             del_if_exists=False)

            # annotations_df = import_wells_annotations_in_folder(aux_dir=aux_dir)
            # matched_long = match_rawvids_annotations(rawvid_dir=aux_dir.parent / "RawVideos", 
            #                                          annotations_df=annotations_df)
            
            # if imaging_dates is not None:
            #     _idx = [i for i in matched_long.index if 
            #             matched_long.loc[i, 'imgstore'].split('_')[-2] in imaging_dates]
            #     matched_long = matched_long.loc[_idx, :]
        
            # # annotate metadata + save
            # meta_df = update_metadata(aux_dir=aux_dir, 
            #                           matched_long=matched_long, 
            #                           saveto=annotated_metadata_path,
            #                           del_if_exists=False)

            if imaging_dates is not None:
                imaging_dates = [float(i) for i in imaging_dates]
                meta_df = meta_df.loc[meta_df['date_yyyymmdd'].isin(imaging_dates),:]
                meta_df['date_yyyymmdd'] = meta_df['date_yyyymmdd'].astype(int).astype(str)
                # NB: Also omits missing video data for some wells (ie. due to single camera failure)

                meta_df.to_csv(annotated_metadata_path, index=None)
                print("Saving annotated metadata to: %s" % annotated_metadata_path)
    
            assert annotated_metadata_path.exists()
           
        # Load annotated metadata
        meta_df = pd.read_csv(annotated_metadata_path, dtype={"comments":str,
                                                              "source_plate_id":str}, header=0)
        if imaging_dates is not None:
            imaging_dates = [float(i) for i in imaging_dates]
            meta_df = meta_df.loc[meta_df['date_yyyymmdd'].isin(imaging_dates),:]
            meta_df['date_yyyymmdd'] = meta_df['date_yyyymmdd'].astype(int).astype(str)
        
        compiled_metadata_path = annotated_metadata_path
                          
    if not 'is_bad_well' in meta_df.columns:
        raise Warning("Bad well annotations not found in metadata!")
    else:
        prop_bad = meta_df['is_bad_well'].sum()/len(meta_df['is_bad_well'])
        print("%.1f%% of data are labelled as 'bad well' data" % (prop_bad*100))
                            
    return meta_df, compiled_metadata_path

def process_feature_summaries(metadata_path, 
                              results_dir, 
                              compile_day_summaries=True,
                              imaging_dates=None, 
                              align_bluelight=True):
    """ Compile feature summary results and join with metadata to produce
        combined full feature summary results
        
        Parameters
        ----------
        metadata : pd.DataFrame
            Experiment metadata
        results_dir : str, Path
            Path to 'Results' directory, containing Tierpsy feature summaries files
        compile_day_summaries : bool
            Compile from Tierpsy feature summaries for each experiment day
        imaging_dates : list of str, None
            List of imaging dates to compile Tierspy feature summaries from. If None, will use 
            'date_yyyymmdd' column of metadata
        align_bluelight : bool
            Align bluelight conditions (convert to wide format)
        
        Returns
        -------
        metadata, features
        
    """    

    from tierpsytools.read_data.hydra_metadata import read_hydra_metadata, align_bluelight_conditions
    from tierpsytools.read_data.compile_features_summaries import compile_tierpsy_summaries
    
    combined_feats_path = Path(results_dir) / "full_features.csv"
    combined_fnames_path = Path(results_dir) / "full_filenames.csv"
 
    if np.logical_and(combined_feats_path.is_file(), combined_fnames_path.is_file()):
        print("Found existing full feature summaries")
    else:
        print("Compiling feature summary results")    
        if compile_day_summaries:
            if imaging_dates:
                feat_files = []
                fname_files = []
                for date in imaging_dates:
                    date_dir = Path(results_dir) / date
                    feat_files.extend([f for f in Path(date_dir).rglob('features_summary*.csv')])
                    fname_files.extend([Path(str(f).replace("/features_","/filenames_"))
                                        for f in feat_files])
            else:
                feat_files = [f for f in Path(results_dir).rglob('features_summary*.csv')]
                fname_files = [Path(str(f).replace("/features_", "/filenames_"))
                               for f in feat_files]
        else:
            feat_files = list(Path(results_dir).glob('features_summary*.csv'))
            fname_files = list(Path(results_dir).glob('filenames_summary*.csv'))
               
        # TODO: check for multiple matches - throw warning
            
        # Keep only features files for which matching filenames_summaries exist
        feat_files = [ft for ft, fn in zip(np.unique(feat_files), np.unique(fname_files)) if fn is not None]
        fname_files = [fn for fn in np.unique(fname_files) if fn is not None]
        
        # Compile feature summaries for matched features/filename summaries
        compile_tierpsy_summaries(feat_files=feat_files, 
                                  fname_files=fname_files,
                                  compiled_feat_file=combined_feats_path,
                                  compiled_fname_file=combined_fnames_path)

    # Read metadata + record column order
    metadata = pd.read_csv(metadata_path, dtype={"comments":str, "source_plate_id":str})
    meta_col_order = metadata.columns.tolist()

    # Read features summaries + metadata and add bluelight column if aligning bluelight video results
    features, metadata = read_hydra_metadata(combined_feats_path, 
                                             combined_fnames_path,
                                             metadata_path, 
                                             add_bluelight=align_bluelight)
    # record new columns
    meta_col_order.extend(['bluelight','featuresN_filename','file_id','is_good_well','n_skeletons'])

    if align_bluelight:
        features, metadata = align_bluelight_conditions(feat=features, 
                                                        meta=metadata, 
                                                        how='outer',
                                                        merge_on_cols=['date_yyyymmdd',
                                                                       'imaging_run_number',
                                                                       'imaging_plate_id',
                                                                       'well_name'])
        # Update metadata column order 
        for col in ['bluelight','file_id','imgstore_name','n_skeletons']:
            meta_col_order.remove(col)
        meta_col_order.extend(['bluelight_prestim','bluelight_bluelight','bluelight_poststim',
                               'file_id_bluelight','file_id_poststim','file_id_prestim',
                               'imgstore_name_bluelight','imgstore_name_poststim','imgstore_name_prestim',
                               'n_skeletons_bluelight','n_skeletons_poststim','n_skeletons_prestim'])
        # TODO: Use set(meta_col_order)-set(metadata.columns) to avoid hard coding column names
        
    assert set(features.index) == set(metadata.index)
    
    return features, metadata[meta_col_order]

#%% Main
if __name__ == "__main__":
    # Accept command-line inputs
    parser = argparse.ArgumentParser(description='Compile metadata and feature summary results \
                                     (Hydra 96-well)')
    parser.add_argument('--project_dir', help="Project root directory, containing 'AuxiliaryFiles',\
                        'RawVideos', 'MaskedVideos' and 'Results' folders", type=str)
    parser.add_argument('--compile_day_summaries', help="Compile feature summaries from \
                        day summary results", action='store_false', default=True)
    parser.add_argument('--dates', help="List of imaging dates for day summaries to compile \
                        If None, will compile from features summaries for all imaging dates", 
                        nargs='+', default=None)
    parser.add_argument('--align_bluelight', help="Features as separate columns for each bluelight \
                        stimulus video?", type=bool, default=True)
    parser.add_argument('--add_well_annotations', help="Add 'is_bad_well' labels from WellAnnotator\
                        GUI", action='store_false', default=True)
    args = parser.parse_args()
    
    # Compile metadata
    metadata = process_metadata(aux_dir=Path(args.project_dir) / 'AuxiliaryFiles',
                                imaging_dates=args.dates,
                                add_well_annotations=args.add_well_annotations)
                
    # Process feature summary results
    features, metadata = process_feature_summaries(metadata, 
                                                   results_dir=Path(args.project_dir) / 'Results',
                                                   compile_day_summaries=args.compile_day_summaries,
                                                   imaging_dates=args.dates,
                                                   add_bluelight=args.align_bluelight)   
    print("\nMetadata:\n", metadata.head())
    print("\nFeatures:\n", features.head())
    