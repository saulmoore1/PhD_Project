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
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path, PosixPath

# Path to Github helper functions (USER-DEFINED path to local copy of Github repo)
PATH_LIST = ['/Users/sm5911/Tierpsy_Versions/tierpsy-tools-python/']
for sysPath in PATH_LIST:
    if sysPath not in sys.path:
        sys.path.insert(0, sysPath)

from tierpsytools.hydra.compile_metadata import add_imgstore_name                     
from tierpsytools.read_data.compile_features_summaries import compile_tierpsy_summaries
from tierpsytools.read_data.hydra_metadata import (read_hydra_metadata, 
                                                   align_bluelight_conditions)
from tierpsytools.hydra.match_wells_annotations import (import_wells_annoations_in_folder,
                                                        match_rawvids_annotations,
                                                        update_metadata)

CUSTOM_STYLE = '/Users/sm5911/Documents/GitHub/PhD_Project/Python/Psychobiotics_96WP/analysis_20210126.mplstyle'

#%% Functions
def process_metadata(aux_dir, 
                     imaging_dates=None, 
                     align_bluelight=True, 
                     add_well_annotations=True):
    """ Compile metadata from individual day metadata 
        - Add 'imgstore_name'
        - Add well annotations from WellAnnotator GUI
        - Add camera serial number 
        - Add duration on food
    """
    
    compiled_metadata_path = Path(aux_dir) / "metadata.csv"
    
    if not compiled_metadata_path.exists():
        print("Compiling from day-metadata in '%s'" % aux_dir)
        
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
            
            # # Rename metadata columns for compatibility with TierpsyTools functions 
            # day_meta = day_meta.rename(columns={'date_recording_yyyymmdd': 'date_yyyymmdd',
            #                                     'well_number': 'well_name',
            #                                     'plate_number': 'imaging_plate_id',
            #                                     'run_number': 'imaging_run_number',
            #                                     'camera_number': 'camera_serial'})
             
            # Get path to RawVideo directory for day metadata
            rawDir = Path(str(day_meta_path).replace("AuxiliaryFiles", "RawVideos")).parent
            
            # Get imgstore name
            if 'imgstore_name' not in day_meta.columns:
                if 'camera_serial' in day_meta.columns:
                    # Delete camera_serial column as it will be recreated
                     day_meta = day_meta.drop(columns='camera_serial')
                day_meta = add_imgstore_name(day_meta, rawDir)
            
            # If imgstore_name column is incomplete
            elif day_meta['imgstore_name'].isna().any():
                day_meta = day_meta.drop(columns=['imgstore_name', 'camera_serial'])
                day_meta = add_imgstore_name(day_meta, rawDir)                    
            
            # Get filename
            day_meta['filename'] = [rawDir.parent / day_meta.loc[i,'imgstore_name']\
                                    for i in range(len(day_meta['filename']))]
            
            # Overwrite day metadata, keping original row order
            print("Updating day metadata for: %s" % date)
            day_meta = day_meta.sort_values(by='row_order').reset_index(drop=True)
            day_meta = day_meta.drop(columns='row_order')
            day_meta.to_csv(day_meta_path, index=False)
            
            day_meta_list.append(day_meta)
        
        # Concatenate list of day metadata into full metadata
        meta_df = pd.concat(day_meta_list, axis=0, ignore_index=True, sort=False)

        # Ensure no missing filenames
        assert not any(list(~np.array([isinstance(path, PosixPath) or\
                       isinstance(path, str) for path in meta_df['filename']])))
        
        # Save metadata
        meta_df.to_csv(compiled_metadata_path, index=None) 
        print("Metadata saved to: %s" % compiled_metadata_path)
        
    if add_well_annotations:
        annotated_metadata_path = Path(str(compiled_metadata_path).replace('.csv', 
                                                                           '_annotated.csv'))
        if not annotated_metadata_path.exists():
            # load metadata
            meta_df = pd.read_csv(compiled_metadata_path, dtype={"comments":str}, header=0)
            print("Adding annotations to metadata")

            annotations_df = import_wells_annoations_in_folder(aux_dir=aux_dir)
            
            rawDir = aux_dir.parent / "RawVideos"
            matched_long = match_rawvids_annotations(rawvid_dir=rawDir, 
                                                     annotations_df=annotations_df)
            if imaging_dates:
                _idx = [i for i in matched_long.index if 
                        matched_long.loc[i, 'imgstore'].split('_')[-2] in imaging_dates]
                matched_long = matched_long.loc[_idx, :]
        
            # annotate metadata + save
            meta_df = update_metadata(aux_dir=aux_dir, 
                                      matched_long=matched_long, 
                                      saveto=annotated_metadata_path)
        else:
            # load annotated metadata
            meta_df = pd.read_csv(annotated_metadata_path, dtype={"comments":str}, header=0)
            print("Loaded annotated metadata")
            
            if not 'is_bad_well' in meta_df.columns:
                raise Warning("Bad well annotations not found in metadata!")
                
            if imaging_dates:
                meta_df = meta_df.loc[meta_df['date_yyyymmdd'].isin(imaging_dates),:]
                print("Extracted metadata for imaging dates provided")
            
        prop_bad = meta_df.is_bad_well.sum()/len(meta_df.is_bad_well)
        print("%.1f%% of data are labelled as 'bad well' data" % (prop_bad*100))
                
    return meta_df

def process_feature_summaries(metadata, 
                              results_dir, 
                              compile_day_summaries=False,
                              imaging_dates=None, 
                              add_bluelight=True):
    """ Compile feature summary results and join with metadata to produce
        combined full feature summary results.    
    """    
    combined_feats_path = results_dir / "full_features.csv"
    combined_fnames_path = results_dir / "full_filenames.csv"
 
    if np.logical_and(combined_feats_path.is_file(), 
                      combined_fnames_path.is_file()):
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
               
        # Keep only features files for which matching filenames_summaries exist
        feat_files = [feat_fl for feat_fl,fname_fl in zip(np.unique(feat_files),
                      np.unique(fname_files)) if fname_fl is not None]
        fname_files = [fname_fl for fname_fl in np.unique(fname_files)
                       if fname_fl is not None]
        
        # Compile feature summaries for matched features/filename summaries
        compile_tierpsy_summaries(feat_files=feat_files, 
                                  compiled_feat_file=combined_feats_path,
                                  compiled_fname_file=combined_fnames_path,
                                  fname_files=fname_files)

    # Read features/filename summaries
    feature_summaries = pd.read_csv(combined_feats_path, comment='#')
    filename_summaries = pd.read_csv(combined_fnames_path, comment='#')
    print("Feature summary results loaded.")

    if imaging_dates:
        if not all(filename_summaries.loc[i,'filename'].split('/')[-2].split('_')[-2] in 
                   imaging_dates for i in filename_summaries.index):
            raise Warning("Incorrect feature summaries for imaging dates provided.\n" +
                          "Please delete and recompile:\n%s\n%s" % (combined_feats_path, 
                                                                    combined_fnames_path))
    
    features, metadata = read_hydra_metadata(feature_summaries, 
                                             filename_summaries,
                                             metadata,
                                             add_bluelight=add_bluelight)
    if add_bluelight:
        features, metadata = align_bluelight_conditions(feat=features, 
                                                        meta=metadata, 
                                                        how='outer',
                                                        merge_on_cols=['date_yyyymmdd',
                                                                       'imaging_run_number',
                                                                       'imaging_plate_id',
                                                                       'well_name'])
    assert set(features.index) == set(metadata.index)
    
    return features, metadata

#%% Main
if __name__ == "__main__":
    # Accept command-line inputs
    parser = argparse.ArgumentParser(description='Compile metadata and feature summary results \
                                     (Hydra 96-well)')
    parser.add_argument('--project_dir', help="Project root directory,\
                        containing 'AuxiliaryFiles', 'RawVideos',\
                        'MaskedVideos' and 'Results' folders",
                        default='/Volumes/hermes$/KeioScreen_96WP', type=str)
    parser.add_argument('--compile_day_summaries', help="Compile feature summaries from \
                        day summary results", default=True, action='store_false')
    parser.add_argument('--dates', help="List of imaging dates for day summaries to compile \
                        If None, will compile from features summaries for all imaging dates", 
                        nargs='+', default=['20210126', '20210127'])
    parser.add_argument('--align_bluelight', help="Features as separate columns for each bluelight \
                        stimulus video?", default=True, type=bool)
    parser.add_argument('--add_well_annotations', help="Add 'is_bad_well' labels \
                        from WellAnnotator GUI", default=True, action='store_false')
    args = parser.parse_args()
    
    PROJECT_DIR = Path(args.project_dir)
    COMPILE_DAY = args.compile_day_summaries
    IMAGING_DATES = args.dates
    BLUELIGHT = args.align_bluelight
    ADD_WELL_ANNOTATIONS = args.add_well_annotations

    # Compile metadata
    metadata = process_metadata(aux_dir=PROJECT_DIR / 'AuxiliaryFiles',
                                imaging_dates=IMAGING_DATES,
                                align_bluelight=BLUELIGHT,
                                add_well_annotations=ADD_WELL_ANNOTATIONS)
                
    # Process feature summary results
    features, metadata = process_feature_summaries(metadata, 
                                                   results_dir=PROJECT_DIR / 'Results',
                                                   compile_day_summaries=COMPILE_DAY,
                                                   imaging_dates=IMAGING_DATES,
                                                   add_bluelight=BLUELIGHT)   
    print("\nMetadata:\n", metadata.head())
    print("\nFeatures:\n", features.head())
