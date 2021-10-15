#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compile wiindow summaries

@author: lferiani (modified by sm5911)
@date: 14/10/2021
"""

def parse_window_number(fname):
    """
    Parse the filename to find the number between 'window_' and '.csv'
    (with .csv being at the end of the name)
    """
    # using re.findall and not match because I'm lazy
    regex = r'(?<=window_)\d+(?=\.csv)'
    window_str = re.findall(regex, fname.name)[0]
    return int(window_str)


def concatenate_filenames_and_feats(fnames_df_list, featsums_df_list):
    """Concatenate lists of dataframes into one dataframe,
    create unique file id by just making sure file_id keeps growing.
    Unique id is created just by adding to each file the highest cumulative id
    seen so far.
    Example:
        df1 contains ids 0,1,2, df2 contains ids 0,1, and df3 ids 0,1,2.
        df2's ids in the concatenated df will become 3,4,
        df3's ids in the concatenated df will become 5,6,7
        """
    # concatenate a list of dataframes into a larger dataframe
    offset_id = 0
    for fnames_df, featsums_df in zip(fnames_df_list, featsums_df_list):
        fnames_df['file_id'] = fnames_df['file_id'] + offset_id
        featsums_df['file_id'] = featsums_df['file_id'] + offset_id
        offset_id = max(fnames_df['file_id'].max(),
                        featsums_df['file_id'].max()) + 1

    fnames_df = pd.concat(fnames_df_list, ignore_index=True,
                          axis=0, sort=False)
    featsums_df = pd.concat(featsums_df_list, ignore_index=True,
                            axis=0, sort=False)

    return fnames_df, featsums_df


def concatenate_day_summaries(sum_root_dir, list_days,
                              window='whole',
                              include=None, exclude=None):
    """concatenate_day_summaries_nowindow
    Loop through the subfolders of `sum_root_dir` listed in `list_days`.
    Find ``filenames_summary`` and ``features_summary`` files
    that span whole videos
    (i.e. filter out if there's any that was obtained with time windows).
    Safely concatenate the ``filenames_summary``s and ``features_summary``s.
    Filter summaries according to keywords in `include` and `exclude`.
    If only `include` is provided:
        all files without ``included`` keywords are dropped.
    If only `exclude` is provided:
        any file with ``excluded`` keywords is dropped.
    If both `include` and `exclude` are provided:
        only files containing any of the ``included`` keywords AND NONE of the
        ``excluded`` keywords are kept.

    Parameters
    ----------
    sum_root_dir : pathlib Path
        Path to the folder containing the features summaries.
        The feature sumaries are expected to be placed in day subfolders
        (specified by list_days).
    list_days : list of strings
        Day folders to load features summaries from.
        Usually in the format YYYYMMDD. Dictates the order the days sufolders
        will be looped through.
    window : str or int
        Out of the summary files, select only the ones calculated on this
        window. If "whole", only find the summary files done on
        the entire videos.
    include : list of strings, optional
        Words that have to be present in an imgstore name for it to be kept.
        The default is None.
    exclude : list of strings, optional
        If an imgstore contains any of these words, it will be discarded.
        Takes precedence over `include` (i.e. if an imgstore contains an
        ``included`` keyword but also an ``excluded`` one
        it will be discarded).
        The default is None.


    Returns
    -------
    filenames_df, features_df.

    """

    filenames_summaries = []
    features_summaries = []
    for day in list_days:
        # find non window summaries
        daydir = sum_root_dir / day
        fnames_raw = daydir.rglob('filenames_summary*.csv')
        featsums_raw = daydir.rglob('features_summary*.csv')
        # filter based on `window`
        if window == 'whole':
            fnames = [f for f in fnames_raw if 'window' not in f.name]
            featsums = [f for f in featsums_raw if 'window' not in f.name]
        else:
            fnames = [f for f in fnames_raw
                      if 'window_{}.csv'.format(window) in f.name]
            featsums = [f for f in featsums_raw
                        if 'window_{}.csv'.format(window) in f.name]

        # check only one of each
        if len(fnames) > 1 or len(featsums) > 1:
            print(fnames)
            print(featsums)
        assert len(fnames) == 1 and len(featsums) == 1, \
            ('Multiple whole-video summaries (or none) in '
             '{}, not supported'.format(daydir))
        # grow lists
        filenames_summaries.extend(fnames)
        features_summaries.extend(featsums)

    # check
    for fnames, featsums in zip(filenames_summaries, features_summaries):
        assert fnames.name.replace('filenames', 'features') == featsums.name, \
            'mismatch!'

    # load
    fnames_df_list = []
    featsums_df_list = []
    for fnames, featsums in tqdm(zip(filenames_summaries, features_summaries)):
        fn_df, fs_df = read_tierpsy_feat_summaries(featsums, fnames,
                                                   asfloat32=True)
        fnames_df_list.append(fn_df)
        featsums_df_list.append(fs_df)
    for i, fs_df in enumerate(featsums_df_list[1:]):
        if len(fs_df.columns.difference(featsums_df_list[i-1].columns)) > 0:
            print(fs_df.columns.difference(featsums_df_list[i-1].columns))

    try:
        filenames_df, features_df = concatenate_filenames_and_feats(
            fnames_df_list, featsums_df_list)
    except:
        import pdb; pdb.set_trace()

    # filter indices according to include/exclude
    if include is None:
        idx_inc = pd.Series(True, index=filenames_df.index)
    else:
        include = [include] if isinstance(include, str) else include
        idx_inc = filenames_df['filename'].str.contains('|'.join(include),
                                                         regex=True)
    if exclude is None:
        idx_exc = pd.Series(False, index=filenames_df.index)
    else:
        exclude = [exclude] if isinstance(exclude, str) else exclude
        idx_exc = filenames_df['filename'].str.contains('|'.join(exclude),
                                                         regex=True)
    idx_keep = idx_inc & ~idx_exc
    # apply filtering
    filenames_df = filenames_df[idx_keep]
    features_df = pd.merge(filenames_df['file_id'],
                           features_df,
                           how='left',
                           on='file_id')

    return filenames_df, features_df