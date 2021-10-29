#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 22:27:11 2021

@author: lferiani
"""
import tables
import os.path
import numpy as np
import pandas as pd
from tqdm import tqdm
from tierpsy.analysis.compress.selectVideoReader import selectVideoReader


def hdf52raw(hdf5_fname):
    """convert masked or featuresN file to the metadata.yaml"""
    raw_str = str(hdf5_fname)  # in case of pathlib
    if 'MaskedVideos' in raw_str:
        raw_str = raw_str.replace('MaskedVideos', 'RawVideos')
        raw_str = raw_str.replace('metadata.hdf5', 'metadata.yaml')
    else:
        raw_str = raw_str.replace('Results', 'RawVideos')
        raw_str = raw_str.replace('metadata_featuresN.hdf5', 'metadata.yaml')
    return raw_str


def read_well_boundaries(hdf5_fname, well_name):
    """read_well_boundaries
      Input: name of hdf5 filename, name of the well we want the boundaries of
      output: ymin, ymax, xmin, xmax.
      use output: roi_img = img[ymin:ymax, xmin:xmax]
      """
    # read df and check well exists
    wells_df = pd.read_hdf(hdf5_fname, '/fov_wells')
    assert any(well_name == wells_df['well_name']), (
        f'{well_name} not in this video')
    wells_df = wells_df.set_index('well_name')

    # now read the image shape for the video
    with tables.File(hdf5_fname, 'r') as fid:
        wells_node_attrs = fid.get_node('/fov_wells')._v_attrs
        img_shape = wells_node_attrs['img_shape']

    # read the boundaries of the well we want
    well_props = wells_df.loc[well_name]
    xmin = max(well_props['x_min'], 0)
    ymin = max(well_props['y_min'], 0)
    xmax = min(well_props['x_max'], img_shape[1])
    ymax = min(well_props['y_max'], img_shape[0])

    return ymin, ymax, xmin, xmax


def well_reader(hdf5_fname, well_name):
    """
    well_reader Generator yielding the image data of a well frame by frame

    Parameters
    ----------
    hdf5_fname : str or Path
        path to the hdf5 file
        (a masked metadata.hdf5 or a results metadata_featuresN.hdf5)
    well_name : str
        name of the well to read the data of

    Yields
    -------
    numpy array
        one frame cropped at the well's boundaries
    """
    # take hdf5 file and well name
    # get wall boundaries
    # find raw video
    # get the reader
    # while loop with generator returning only the right portion of the frame

    # read boundaries
    r_min, r_max, c_min, c_max = read_well_boundaries(hdf5_fname, well_name)

    # get raw video
    raw_fname = hdf52raw(hdf5_fname)
    assert os.path.exists(raw_fname), f'Cannot find {raw_fname}'

    # initalise reader
    vid = selectVideoReader(raw_fname)

    # start with reading a frame.
    # then at every iteration read the next one but do not return it yet
    # this avoids having an empty frame at the end when
    # the reader returns (0, None)
    status, image = vid.read()

    while status > 0:
        status, next_image = vid.read()
        roi = image[r_min:r_max, c_min:c_max].copy()
        # prep for next loop
        image = next_image
        yield roi


if __name__ == '__main__':

    featuresN_file = (
        '/Users/lferiani/Hackathon/multiwell_tierpsy/playground/Results/'
        '20191205/syngenta_screen_run1_bluelight_20191205_151104.22956805/'
        'metadata_featuresN.hdf5'
        )

    well_name = 'E5'

    # create the generator
    well_img_gen = well_reader(featuresN_file, well_name)

    # iterate, let's store the data as we go.
    # a 5 minute video gets you about 4GB of data
    import time
    tic = time.time()

    # one alternative:
    # img_stack = []
    # for frame_counter, well_image in enumerate(tqdm(well_img_gen)):
    #     # do something, here I'm just storing the frame in the stack
    #     img_stack.append(well_image[None, :, :])

    # or
    img_stack = [well_image[None, :, :] for well_image in tqdm(well_img_gen)]

    # convert the list of images to an n_frames-by-height-by-width array
    img_stack = np.concatenate(img_stack, axis=0)

    print(f'time elapsed: {time.time() - tic}')
