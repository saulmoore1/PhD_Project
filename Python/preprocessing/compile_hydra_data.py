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

def compile_day_metadata(aux_dir, 
                         day,
                         n_wells=96,
                         from_source_plate=False, 
                         from_robot_runlog=False):
    
    """ Compile experiment day metadata from wormsorter and hydra rig metadata for a given day in 
        'AuxiliaryFiles' directory 
        
        Parameters
        ----------
        aux_dir : str
            Path to "AuxiliaryFiles" containing metadata  
        day : str, None
            Experiment day folder in format 'YYYYMMDD'
        from_source_plate : bool
            Compile day metadata using source plate metadata ('source_plate.csv') and mapping for 
            each imaging plate ID ('source2imaging.csv')
        from_robot_runlog : bool
            Compile plate metadata from OpenTrons run log files (eg. for randomised plates)
            
        Returns
        -------
        day_metadata : pandas.DataFrame
            Compiled day metadata
    """
    from tierpsytools.hydra.compile_metadata import (populate_6WPs,
                                                     populate_24WPs,
                                                     populate_96WPs,
                                                     get_day_metadata,
                                                     get_source_metadata,
                                                     merge_basic_and_source_meta,
                                                     #merge_robot_metadata, 
                                                     #merge_robot_wormsorter,
                                                     number_wells_per_plate, 
                                                     day_metadata_check)
    
    day_dir = Path(aux_dir) / str(day)
    
    wormsorter_meta = day_dir / (str(day) + '_wormsorter.csv')
    hydra_meta = day_dir / (str(day) + '_manual_metadata.csv')
      
    # Expand wormsorter metadata to have separate row for each well (96-well or 6-well plate format)
    if n_wells == 6:    
        plate_metadata = populate_6WPs(wormsorter_meta)
    elif n_wells == 24:
        plate_metadata = populate_24WPs(wormsorter_meta)
    elif n_wells == 96:
        plate_metadata = populate_96WPs(wormsorter_meta)
    else:
        raise ValueError("Please choose n_wells from: [6, 24, 96]")
    
    from_source_plate = True if from_robot_runlog else from_source_plate
    
    if from_source_plate:
        print("Compiling from source plate metadata...")
        source_plate_meta = day_dir / 'source_plates.csv'
        plate_mapping_meta = day_dir / 'source2imaging.csv'

        source_metadata = get_source_metadata(sourceplates_file=source_plate_meta,
                                              imaging2source_file=plate_mapping_meta)
    
        plate_metadata = merge_basic_and_source_meta(plate_metadata, 
                                                     source_metadata, 
                                                     merge_on=['imaging_plate_id', 'well_name'],
                                                     saveto=None,
                                                     del_if_exists=False)
        
# =============================================================================
#     if from_robot_runlog:
#         robot_runlog_meta = day_dir / 'robot_runlog.csv'
#     
#         drug_metadata = merge_robot_metadata(robot_runlog_meta,
#                                              randomized_by='column',
#                                              saveto=None,
#                                              drug_by_column=True,
#                                              compact_drug_plate=False,
#                                              del_if_exists=False)
#         
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
                                    n_wells=n_wells,
                                    run_number_regex='run\\d+_',
                                    saveto=None,
                                    del_if_exists=False,
                                    include_imgstore_name=True,
                                    raw_day_dir=Path(str(day_dir).replace("AuxiliaryFiles",
                                                                          "RawVideos")))
    
    day_metadata_check(day_metadata, day_dir, plate_size=n_wells) # check day metadata is correct dimension
    print(number_wells_per_plate(day_metadata, day_dir))

    return day_metadata

def compile_metadata(aux_dir, 
                     imaging_dates, 
                     n_wells=96,
                     add_well_annotations=True,
                     from_source_plate=False,
                     from_robot_runlog=False):
    """ Compile metadata from individual day metadata files

        Parameters
        ----------
        aux_dir : str
            Path to "AuxiliaryFiles" directory containing project metadata files
        dates : list of str, None
            List of imaging dates to compile metadata from
        add_well_annotations : bool
            Add annotations from WellAnnotator GUI
        n_wells : int
            Choose from 6, 24, or 96. NB: 'camera_serial' information is required in metadata if 
            n_wells is not 96
    
        Returns
        -------
        metadata : pandas.DataFrame
            Compiled project metadata
    """
    
    from tierpsytools.hydra.hydra_helper import add_imgstore_name
    from tierpsytools.hydra.match_wells_annotations import update_metadata_with_wells_annotations

    metadata_path = Path(aux_dir) / 'metadata.csv'
           
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path, dtype={"comments":str, "source_plate_id":str}, header=0)
        print("Metadata loaded.")
        
        if imaging_dates is not None:
            metadata = metadata[metadata['date_yyyymmdd'].astype(str).isin(imaging_dates)]
            
    else:
        print("Metadata not found.\nCompiling from day metadata in: %s" % aux_dir)
        
        dates_list = [p.name for p in Path(aux_dir).glob('*')]
        dates_found = sorted([date for date in dates_list if re.match(r'\d{8}', date)])
        if imaging_dates:
            assert all(i in dates_found for i in imaging_dates)
        else:
            imaging_dates = dates_found
    
        # from tierpsytools.hydra.compile_metadata import concatenate_days_metadata
        # metadata = concatenate_days_metadata(aux_dir, imaging_dates, saveto=None)
        
        day_meta_list = []
        for date in imaging_dates:
            day_meta_path = Path(aux_dir) / date / '{}_day_metadata.csv'.format(date)
            
            if day_meta_path.exists():
                day_meta = pd.read_csv(day_meta_path, dtype={"comments":str})            
            else:
                day_meta = compile_day_metadata(aux_dir=aux_dir, 
                                                day=date, 
                                                n_wells=n_wells, 
                                                from_source_plate=from_source_plate, 
                                                from_robot_runlog=from_robot_runlog)

            raw_dir = Path(str(day_meta_path).replace("AuxiliaryFiles", "RawVideos")).parent
                
            # add 'imgstore_name' if not present in metadata
            if 'imgstore_name' not in day_meta.columns:
                
                # delete 'camera_serial' column as it will be recreated
                if 'camera_serial' in day_meta.columns:
                     day_meta = day_meta.drop(columns='camera_serial')

                # add 'imgstore_name' and 'camera_serial'
                day_meta = add_imgstore_name(day_meta, raw_dir, n_wells=n_wells)

            # add filename = raw_dir / imgstore_name
            assert not day_meta['imgstore_name'].isna().any()
            day_meta['filename'] = [raw_dir.parent / day_meta.loc[i, 'imgstore_name'] for i in
                                    range(day_meta.shape[0])]
            
            # append day metadata to list
            day_meta_list.append(day_meta)
        
        # concatenate list of day metadata into full metadata
        metadata = pd.concat(day_meta_list, axis=0, ignore_index=True, sort=False)

        # Ensure no NA values in any columns with 'date' in the name 
        # for compatibility with fix_dtypes from tierpsytools platechecker
        check_na_cols = [col for col in metadata.columns if 'date' in col]
        assert not any(any(metadata[col].isna()) for col in check_na_cols)
                
        # Convert 'date_yyyymmdd' column to string (factor)
        metadata['date_yyyymmdd'] = metadata['date_yyyymmdd'].astype(str)
        
        # drop any wells annotations columns that might exist as will throw an error when re-added
        metadata = metadata.drop(columns=['is_bad_well', 'well_label'], errors='ignore')
        
        # save metadata
        metadata.to_csv(metadata_path, index=None) 
        print("Metadata saved to: %s" % metadata_path)
    
    # add annotations to metadata
    if add_well_annotations and n_wells == 96:        
        annotated_metadata_path = Path(str(metadata_path).replace('.csv', '_annotated.csv'))
        
        if not annotated_metadata_path.exists():
            print("Adding annotations to metadata")
            metadata = update_metadata_with_wells_annotations(aux_dir=aux_dir, 
                                                              saveto=annotated_metadata_path, 
                                                              del_if_exists=False)

            if imaging_dates is not None:
                metadata = metadata.loc[metadata['date_yyyymmdd'].astype(str).isin(imaging_dates)]
                metadata.to_csv(annotated_metadata_path, index=None)
    
            assert annotated_metadata_path.exists()
            metadata_path = annotated_metadata_path
   
        # Load annotated metadata
        metadata = pd.read_csv(metadata_path, header=0,
                               dtype={"comments":str, "source_plate_id":str})
        if imaging_dates is not None:
            metadata = metadata.loc[metadata['date_yyyymmdd'].astype(str).isin(imaging_dates)]
                                  
        if not 'is_bad_well' in metadata.columns:
            raise Warning("Bad well annotations not found in metadata!")
        else:
            prop_bad = metadata['is_bad_well'].sum() / len(metadata['is_bad_well'])
            print("%.1f%% of data are labelled as 'bad well' data" % (prop_bad * 100))
            
    return metadata, metadata_path

# TODO: Deprecated. Remove this function
def process_metadata(aux_dir, 
                     imaging_dates=None, 
                     add_well_annotations=True,
                     update_day_meta=False,
                     update_colnames=False, 
                     n_wells=96):
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
        n_wells : int
            Choose from either 96 or 6. NB: 'camera_serial' information is required 
            in metadata if n_wells is not 96

        Returns
        -------
        Updated metadata
        compiled metadata path
    """
    
    from tierpsytools.hydra.hydra_helper import add_imgstore_name
    from tierpsytools.hydra.match_wells_annotations import update_metadata_with_wells_annotations
    # from tierpsytools.hydra.match_wells_annotations import (import_wells_annotations_in_folder,
    #                                                         match_rawvids_annotations, update_metadata)
    
    compiled_metadata_path = Path(aux_dir) / "metadata.csv"
    
    if compiled_metadata_path.exists():
        # Load metadata
        meta_df = pd.read_csv(compiled_metadata_path, dtype={"comments":str, 
                                                             "source_plate_id":str}, header=0)
        # subset for imaging dates
        if imaging_dates is not None:
            assert type(imaging_dates) == list
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

            # if update_colnames:
            #     # Rename old metadata column names for compatibility with TierpsyTools functions 
            #     day_meta = day_meta.rename(columns={'date_recording_yyyymmdd': 'date_yyyymmdd',
            #                                         'well_number': 'well_name',
            #                                         'plate_number': 'imaging_plate_id',
            #                                         'run_number': 'imaging_run_number',
            #                                         'camera_number': 'camera_serial'})
             
            # day_meta = day_meta.rename(columns={'channel':'old_channel'})
            day_meta_col_order = list(day_meta.columns)
            day_meta['row_order'] = np.arange(len(day_meta))

            # Get path to RawVideo directory for day metadata
            rawDir = Path(str(day_meta_path).replace("AuxiliaryFiles", "RawVideos")).parent
            
            # Get imgstore name
            if 'imgstore_name' not in day_meta.columns:

                # Delete camera_serial column as it will be recreated
                if 'camera_serial' in day_meta_col_order:
                     day_meta = day_meta.drop(columns='camera_serial')

                # day_meta['well_name'] = day_meta['old_channel'].map({1:'A1', 2:'B1', 
                #                                                      3:'A2', 4:'B2', 
                #                                                      5:'A3', 6:'B3'})
                
                # add imgstore_name (camera_serial added if n_wells=96)
                day_meta = add_imgstore_name(day_meta, rawDir, n_wells=n_wells)

            else:
                assert not day_meta['imgstore_name'].isna().any()
            
            # Get filename
            day_meta['filename'] = [rawDir.parent / day_meta.loc[i,'imgstore_name']\
                                    for i in range(day_meta.shape[0])]
            
            # save day metadata, keeping original row order
            day_meta = day_meta.sort_values(by='row_order').reset_index(drop=True)
            day_meta = day_meta.drop(columns='row_order')
            
            newcols = list(set(day_meta.columns) - set(day_meta_col_order))
            day_meta_col_order.extend(newcols)
            
            if update_day_meta:
                day_meta[day_meta_col_order].to_csv(day_meta_path, index=False)
            
            # Append to compiled metadata list
            day_meta_list.append(day_meta[day_meta_col_order])
        
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
        
        # drop any wells annotations columns that might exist as will throw an error when re-added
        meta_df = meta_df.drop(columns=['is_bad_well', 'well_label'], errors='ignore')
        
        # Save metadata
        meta_df.to_csv(compiled_metadata_path, index=None) 
        print("Metadata saved to: %s" % compiled_metadata_path)
    
    # Add annotations to metadata
    if add_well_annotations and n_wells == 96:        
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
                              align_bluelight=True,
                              window_summaries=False,
                              n_wells=96):
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
        window_summaries : bool
            Compile from windowed features summaries files
        
        Returns
        -------
        features, metadata
        
    """    

    from tierpsytools.read_data.compile_features_summaries import compile_tierpsy_summaries
    from tierpsytools.read_data.hydra_metadata import read_hydra_metadata, align_bluelight_conditions
    from preprocessing.compile_window_summaries import find_window_summaries, compile_window_summaries
    
    combined_feats_path = Path(results_dir) / ("full_features.csv" if not window_summaries else
                                               "full_window_features.csv")
    combined_fnames_path = Path(results_dir) / ("full_filenames.csv" if not window_summaries else
                                                "full_window_filenames.csv")
 
    if np.logical_and(combined_feats_path.is_file(), combined_fnames_path.is_file()):
        print("Found existing full feature summaries")
    else:
        print("Compiling feature summary results")   
        if window_summaries:
            print("\nFinding window summaries files..")
            fname_files, feat_files = find_window_summaries(results_dir=results_dir, 
                                                            dates=imaging_dates)
    
            # compile window summaries files
            print("\nCompiling window summaries..")
            compiled_filenames, compiled_features = compile_window_summaries(fname_files=fname_files, 
                                                                             feat_files=feat_files,
                                                                             compiled_fnames_path=combined_fnames_path,
                                                                             compiled_feats_path=combined_feats_path,
                                                                             results_dir=Path(results_dir), 
                                                                             window_list=None,
                                                                             n_wells=n_wells)
        else:
            if compile_day_summaries:
                if imaging_dates is not None:
                    assert type(imaging_dates) == list
                    feat_files = []
                    fname_files = []
                    for date in imaging_dates:
                        date_dir = Path(results_dir) / date
                        feat_files.extend(list(Path(date_dir).rglob('features_summary*.csv')))
                        fname_files.extend(list(Path(date_dir).rglob('filenames_summary*.csv')))
                else:
                    feat_files = list(Path(results_dir).rglob('features_summary*.csv'))
                    fname_files = [Path(str(f).replace("/features_", "/filenames_")) for f in feat_files]
            else:
                feat_files = list(Path(results_dir).glob('features_summary*.csv'))
                fname_files = list(Path(results_dir).glob('filenames_summary*.csv'))

            # Keep only features files for which matching filenames_summaries exist
            feat_files = [ft for ft, fn in zip(np.unique(feat_files), np.unique(fname_files)) if fn is not None]
            fname_files = [fn for fn in np.unique(fname_files) if fn is not None]           
                
            feat_files = [ft for ft in feat_files if not 'window' in str(ft)]
            fname_files = [fn for fn in fname_files if not 'window' in str(fn)]
                    
            # Compile feature summaries for matched features/filename summaries
            compile_tierpsy_summaries(feat_files=feat_files, 
                                      fname_files=fname_files,
                                      compiled_feat_file=combined_feats_path,
                                      compiled_fname_file=combined_fnames_path)

    # Read metadata + record column order
    metadata = pd.read_csv(metadata_path, dtype={"comments":str, "source_plate_id":str,
                                                 "imaging_run_number":str})
    meta_col_order = metadata.columns.tolist()

    feat_id_cols = ['file_id', 'n_skeletons', 'well_name', 'is_good_well']

    # if there are no well annotations in metadata, omit 'is_good_well' from feat_id_cols
    if 'is_good_well' not in meta_col_order: 
        feat_id_cols = [f for f in feat_id_cols if f != 'is_good_well']
    if window_summaries:
        feat_id_cols.append('window')
        
    # Read features summaries + metadata and add bluelight column if aligning bluelight video results
    features, metadata = read_hydra_metadata(combined_feats_path, 
                                             combined_fnames_path,
                                             metadata_path, 
                                             feat_id_cols=feat_id_cols,
                                             add_bluelight=align_bluelight)

    if align_bluelight:
        features, metadata = align_bluelight_conditions(feat=features, 
                                                        meta=metadata, 
                                                        how='outer',
                                                        merge_on_cols=['date_yyyymmdd',
                                                                       'imaging_run_number',
                                                                       'imaging_plate_id',
                                                                       'well_name'])
        meta_col_order.remove('imgstore_name')
            
    assert set(features.index) == set(metadata.index)    
    
    # record new columns
    assert len(set(meta_col_order) - set(metadata.columns)) == 0 # ensure no old columns were dropped
    new_cols = list(set(metadata.columns) - set(meta_col_order))
    meta_col_order.extend(new_cols)
    
    return features, metadata[meta_col_order]

#%% Main

if __name__ == "__main__":
    # Accept command-line inputs
    parser = argparse.ArgumentParser(description='Compile metadata and feature summary results \
                                     (Hydra 96-well or 6-well)')
    parser.add_argument('--project_dir', help="Project root directory, containing 'AuxiliaryFiles',\
                        'RawVideos', 'MaskedVideos' and 'Results' folders", type=str)
    parser.add_argument('--compile_day_summaries', help="Compile feature summaries from \
                        day summary results", type=bool, action='store_false', default=True)
    parser.add_argument('--dates', help="List of imaging dates for day summaries to compile \
                        If None, will compile from features summaries for all imaging dates", 
                        nargs='+', default=None)
    parser.add_argument('--align_bluelight', help="Features as separate columns for each bluelight \
                        stimulus video?", type=bool, action='store_false', default=True)
    parser.add_argument('--add_well_annotations', help="Add 'is_bad_well' labels from WellAnnotator\
                        GUI", type=bool, action='store_false', default=True)
    parser.add_argument('--window_summaries', help="If True, compile window summaries files", 
                        type=bool, action='store_true', default=False)
    parser.add_argument('--n_wells', help="Number of wells imaged by Hydra rig (96-well or 6-well)",
                        default=96)
    args = parser.parse_args()
    
    # Compile metadata
    metadata = process_metadata(aux_dir=Path(args.project_dir) / 'AuxiliaryFiles',
                                imaging_dates=args.dates,
                                add_well_annotations=args.add_well_annotations,
                                n_wells=args.n_wells)
                
    # Process feature summary results
    features, metadata = process_feature_summaries(metadata, 
                                                   results_dir=Path(args.project_dir) / 'Results',
                                                   compile_day_summaries=args.compile_day_summaries,
                                                   imaging_dates=args.dates,
                                                   align_bluelight=args.align_bluelight,
                                                   window_summaries=args.window_summaries,
                                                   n_wells=args.n_wells)   
    print("\nMetadata:\n", metadata.head())
    print("\nFeatures:\n", features.head())
    