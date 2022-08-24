#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read Zeiss microscope (CZI) files, apply BREN (or GLVA) method focus measure to find best 
focussed images based RFP channel only, and save the RFP channel image and corresponding 
GFP channel image for each series

Inputs
-----------------
image_root_dir (str): path to parent directory of images to be processed

Supported Methods
-----------------
GLVA: Graylevel variance (Krotkov, 86)
BREN: Brenner's (Santos, 97)

-----------------
@author: Saul Moore (sm5911)
@date: 09/08/2022

"""

#%% Imports

import os, sys, time
import numpy as np
import pandas as pd
# from czifile import imread
# from tifffile import imwrite
from aicsimageio import AICSImage
import javabridge
import bioformats #unicodedata
# from matplotlib import pyplot as plt
from cv2 import subtract
from pathlib import Path

#%% Globals

# max RAM to use (Gb)
maxRAM = 6

# save most focussed images in output directory?
# saveBestFocus = True        

# method for calculating best focussed image: ['BREN','GLVA']
method = 'BREN'

# image file type (format): ['tif','czi']    
imageFormat = 'czi'

# size filter to omit small images that appear to be duplicates (< 1024x1024 pixels) - thumbnails??
imageSizeThreshXY = [1024,1024]

#%% Functions

def fmeasure(im, method='BREN'):
    """ Python implementation of MATLAB's fmeasure module
    
        params
        ------
        im (array): 2D numpy array
        method (str): focus measure algorithm
        
        supported methods
        --------------------
        GLVA: Graylevel variance (Krotkov, 86)
        BREN: Brenner's (Santos, 97)
    """
               
    # measure focus
    if method == 'GLVA':
        FM = np.square(np.std(im))
        return FM
    
    elif method == 'BREN':
        M, N = im.shape
        DH = np.zeros((M, N))
        DV = np.zeros((M, N))
        DV[:M-2,:] = subtract(im[2:,:],im[:-2,:])
        DH[:,:N-2] = subtract(im[:,2:],im[:,:-2])
        FM = np.maximum(DH, DV)
        FM = np.square(FM)
        FM = np.mean(FM)
        return FM
    
    else:
        raise Exception('Method not supported.')
 
    
def findImageFiles(image_root_dir, imageFormat):
    """ Return dataframe of image filepaths for images in the given directory 
        that match the given file regex string. 
    """
    
    # search for CZI or TIFF images in directory
    if type(imageFormat) != str:
        raise Exception('ERROR: Image format (str) not provided.')
    else:
        if imageFormat.lower().endswith('czi'):
            file_regex_list = ['*.czi','*/*.czi']
        elif imageFormat.lower().endswith('tif'):
            file_regex_list = ['*.tif','*/*.tif']

    # find image files
    print("Finding image files in %s" % image_root_dir)
    image_list = []
    for file_regex in file_regex_list:
        images = list(Path(image_root_dir).rglob(file_regex))
        image_list.extend(images)
    
    if len(image_list) == 0:
        raise Exception('ERROR: Unable to locate images. Check directory!')
    else:
        print("%d image files found." % len(image_list))
        return pd.DataFrame(image_list, columns=["filepath"])

def crop_image_nonzero(img):
    """ A function to delete the all-zero rows and columns of a 
        matrix of size n x m, to crop an image down to size (non-zero pixels) """
        
    hor_profile = img.any(axis=0)
    ver_profile = img.any(axis=1)
    
    hor_first = np.where(hor_profile != 0)[0].min()
    hor_last = np.where(hor_profile != 0)[0].max() + 1
    ver_first = np.where(ver_profile != 0)[0].min()
    ver_last = np.where(ver_profile != 0)[0].max() + 1
    
    img = img[ver_first:ver_last, hor_first:hor_last]

    return img
        
def findFocussedCZI(file, output_dir, method='BREN', imageSizeThreshXY=None, show=False):
    """ Find most focussed CZI image from dataframe of 'filepaths' to CZI image
        stacks of 96-well plate well images.
       
        params
        ------
        df (DataFrame): pandas DataFrame containing 'filepath' column of full 
                        paths to CZI files
        output_dir (str): output directory path to save results

        method (str): focus measure algorithm
        
        supported methods
        -----------------
        GLVA: Graylevel variance (Krotkov, 86)
        BREN: Brenner's (Santos, 97)
        
        imageSizeThreshXY (list/array) [int,int]: minimum threshold X,Y image size
    """
    
    file_info = []

    reader = AICSImage(str(file))
    scenes = list(reader.scenes)
    
    # parse the CZI file, calculate focus measures + store results
    for s, sc in enumerate(scenes):
        print("%d/%d well: %s (%.1f%%)" % (s+1, len(scenes), sc.split('-')[-1], ((s+1)/len(scenes))*100))
        
        reader.set_scene(sc)
        zslices = reader.shape[2]
        
        # find most focussed RFP image in z-stack for image series 
        for zc in range(zslices):
            
            RFP_img = reader.data[0,1,zc,:,:]
            
            if imageSizeThreshXY is not None:
                x, y = RFP_img.shape
                assert x > imageSizeThreshXY[0] and y > imageSizeThreshXY[1]
            
            # measure focus of RFP image (uint16)
            fm = fmeasure(RFP_img, method)
            
            # store image info
            file_info.append([file, sc, zc, fm])

    # create dataframe from list of recorded data
    colnames = ['filepath','seriesID','zslice','focus_measure']
    file_df = pd.DataFrame.from_records(file_info, columns=colnames)
    
    # get images with max focus for each well/GFP concentration
    focussed_images_df = file_df[file_df['focus_measure'] == 
                         file_df.groupby(['seriesID'])['focus_measure'].transform(max)]
    print("%d most focussed RFP images found." % focussed_images_df.shape[0])

    # save focus measures to file
    fm_outPath = os.path.join(output_dir, 'focus_measures_RFP.csv')
    focussed_images_df.to_csv(fm_outPath, index=False)

    # save most focussed images
    print("Saving GFP and RFP images separately for most focussed RFP images...")
    
    # create most focussed folder for file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    n_focussed = focussed_images_df.shape[0]
    for i in range(n_focussed):
        sc = focussed_images_df.iloc[i]['seriesID']
        zc = focussed_images_df.iloc[i]['zslice']
        print("%d/%d well: %s (%.1f%%)" % (i+1, n_focussed, sc.split('-')[-1], ((i+1)/n_focussed)*100))
        
        reader.set_scene(sc)

        # We do NOT want to rescale images if comparing between them
        GFP_img = reader.data[0,0,zc,:,:]
        RFP_img = reader.data[0,1,zc,:,:]
        assert GFP_img.size == RFP_img.size
        assert GFP_img.dtype == np.uint16 and RFP_img.dtype == np.uint16
    
        # paths to output TIFF images
        outPath_GFP = os.path.join(output_dir, 'GFP', 
                                   '%s_z%d' % (sc.split('-')[-1], zc+1) + '_GFP.tif')
        outPath_RFP = os.path.join(output_dir, 'RFP', 
                                   '%s_z%d' % (sc.split('-')[-1], zc+1) + '_RFP.tif')
        
        # Save as TIFF
        bioformats.write_image(pathname=outPath_GFP, 
                                pixels=GFP_img,
                                pixel_type=bioformats.PT_UINT16)
        bioformats.write_image(pathname=outPath_RFP,
                                pixels=RFP_img,
                                pixel_type=bioformats.PT_UINT16)
    
    return focussed_images_df
    
#%% Main
    
if __name__ == "__main__":

    tic = time.time()   

    ######################
        
    # Start Java virtual machine (for parsing CZI files with bioformats)
    javabridge.start_vm(class_path=bioformats.JARS, max_heap_size='{}G'.format(maxRAM))

    classpath = javabridge.JClassWrapper('java.lang.System').getProperty('java.class.path')
    assert pd.Series([os.path.isfile(path) for path in classpath.split(os.pathsep)]).all()    

    ######################
    ##### PARAMETERS #####
    
    # set root directory
    if len(sys.argv) > 1:
        image_root_dir = sys.argv[1]
    else: 
        # local copy
        raise Warning("No directory path provided! " + 
                      "Please provide path to parent directory of CZI image files")
        #image_root_dir = '/Users/sm5911/Documents/PanGenomeMulti/multi-channel_imaging_examples'
    
    ####################
    ##### COMMANDS #####
    
    # find image files
    images_df = findImageFiles(image_root_dir, imageFormat)
    
    # output directory
    n = len(images_df['filepath'])
    for f, file in enumerate(images_df['filepath']):
        if (f+1) % 1 == 0:
            print("Processing file %d/%d (%.1f%%)" % (f+1, n, ((f+1)/n)*100))

        assert str(file).endswith('czi')
    
        # find + save most focussed images
        print("Finding most focussed RFP images..")
        focussed_images_df = findFocussedCZI(file=file,
                                             output_dir=str(file).split('.')[0] + '_focussed',
                                             method=method,
                                             imageSizeThreshXY=imageSizeThreshXY)
        
    ####################
    
    # Terminate Java virtual machine
    javabridge.kill_vm()

    toc = time.time()
    print('Complete!\nTotal time taken: %.2f seconds.' % (toc - tic))
    
    
