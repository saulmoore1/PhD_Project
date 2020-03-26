#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Get most focussed images

Select best images based on focus measure

@author: Saul Moore (sm5911)
@date: 16/03/2020

"""

#%% Imports

import os, sys, glob, time
import bioformats, javabridge #czifile
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from shutil import copyfile
from cv2 import subtract


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
            file_regex = '*/*.czi'
        elif imageFormat.lower().endswith('tif'):
            file_regex = '*/*.tif'

        # find image files
        print("Finding image files in %s" % image_root_dir)
        images = glob.glob(os.path.join(image_root_dir, file_regex))
        
        if len(images) == 0:
            raise Exception('ERROR: Unable to locate images. Check directory!')
        else:
            print("%d image files found." % len(images))
            return pd.DataFrame(images, columns=["filepath"])
        

def findFocussedCZI(df, output_dir, method='BREN', show=False):
    """ Find most focussed CZI image from dataframe of 'filepaths' to CZI image
        stacks of 96-well plate well images.
       
        params
        ------
        df (DataFrame): pandas DataFrame containing 'filepath' column of full 
                        paths to CZI files
        method (str): focus measure algorithm
        
        supported methods
        -----------------
        GLVA: Graylevel variance (Krotkov, 86)
        BREN: Brenner's (Santos, 97)
    """
    
    file_df_list = []
    df = df.sort_values(by=['filepath']).reset_index(drop=True)    
    n = len(df['filepath'].unique())
    for f, file in enumerate(df['filepath']):
        print("Processing file %d/%d (%.1f%%)" % (f+1, n, ((f+1)/n)*100))
        if f == 1:
            raise Exception('STOP!')

        # extract metadata from filename
        fname, dname = os.path.basename(file), os.path.basename(os.path.dirname(file))
        plateID = fname.split("_")[0].split("PG")[1]
        GFP_mM = fname.split("mM_")[0].split("_")[-1]
        
        # get the actual image reader
        rdr = bioformats.get_image_reader(None, path=file)
        #with bioformats.ImageReader(path=file, url=None, perform_init=True) as reader:
        #    img_arr = reader.read(file)
        #image_arrays = czifile.imread(file)

        # for "whatever" reason the number of total image series can only be accessed this way
        try:
            totalseries = np.int(rdr.rdr.getSeriesCount())
        except:
            totalseries = 1 # in case there is only ONE series

        # parse the CZI file
        try:
            file_info = []
            # Loop over wells (series)
            for sc in range(0, totalseries):
                rdr.rdr.setSeries(sc)
                
                # get number of z-slices
                zSlices = rdr.rdr.getImageCount()

                # Loop over z-slices
                for zc in range(0, zSlices):
                    img = rdr.read(c=None, z=0, t=0, series=sc, index=zc, rescale=False)
                                            
                    # measure focus
                    fm = fmeasure(img, method)
                    
                    # store image info
                    file_info.append([file, plateID, GFP_mM, sc, zc, fm])

            # create dataframe from list of recorded data
            colnames = ['filepath','plateID','GFP_mM','wellID','z_slice_number','focus_measure']
            file_df = pd.DataFrame.from_records(file_info, columns=colnames)

            # store file info
            file_df_list.append(file_df)
            
            # get images with max focus for each well/GFP concentration
            focussed_images_df = file_df[file_df['focus_measure'] == \
                                 file_df.groupby(['GFP_mM','wellID'])['focus_measure']\
                                 .transform(max)]

            # save most focussed images
            outDir = os.path.join(output_dir, dname)
            if not os.path.exists(outDir):
                os.makedirs(outDir)
            
            print("Saving most focussed images..")
            for i in range(focussed_images_df.shape[0]):
                if i % 8 == 0:
                    print("%d/%d" % (i+1, focussed_images_df.shape[0]))

                img_info = focussed_images_df.iloc[i]
                
                wellID = img_info['wellID']
                zSlice = img_info['z_slice_number']
                rdr.rdr.setSeries(wellID)
                
                # I imagine we do NOT want to rescale the images as we are comparing between them
                img = rdr.read(c=None, z=0, t=0, series=wellID,\
                               index=zSlice, rescale=False)
               
                if show:
                    plt.close('all')
                    plt.imshow(img); plt.pause(5)
            
                # save as TIFF image
                outPath = os.path.join(outDir, fname.split('.')[0] +\
                                        '_s%dz%d' % (wellID, zSlice) + '.tif')
                
                # Save as TIFF (bioformats)
                bioformats.write_image(outPath, img, pixel_type=bioformats.PT_UINT16)
                #from libtiff import TIFF
                #tiff = TIFF.open(outPath, mode='w')
                #tiff.write_image(img)
                #tiff.close()
                    
        except Exception as EE:
            print('ERROR: Cannot read CZI file: %s\n%s' % (file, EE))
        
    # concatenate dataframe from CZI file info
    df = pd.concat(file_df_list, axis=0, ignore_index=True)

    # save focus measures to file
    df.to_csv(os.path.join(output_dir, 'focus_measures.csv'))

    focussed_images_df = df[df['focus_measure'] == \
                         df.groupby(['plateID','GFP_mM','wellID'])['focus_measure']\
                         .transform(max)].reset_index(drop=True)

    return focussed_images_df


def findFocussedTIF(df, output_dir, method='BREN'):
    """ Find most focussed TIF images from dataframe of 'filepaths' 
       
        params
        ------
        df (DataFrame): pandas DataFrame containing 'filepath' column
        method (str): focus measure algorithm
        
        supported methods
        -----------------
        GLVA: Graylevel variance (Krotkov, 86)
        BREN: Brenner's (Santos, 97)
    """

    # add columns to df    
    new_cols = ['GFP_mM','wellID','plateID','z_slice_number','focus_measure']
    cols = list(df.columns)  
    cols.extend(new_cols)
    df = df.reindex(columns=cols)
    df = df.sort_values(by=['filepath']).reset_index(drop=True)
    
    n = len(df['filepath'])
    for f, file in enumerate(df['filepath']):
        if f % 10 == 0:
            print("Processing file %d/%d (%.1f%%)" % (f, n, (f/n)*100))
        
        # extract metadata from filename
        fname = os.path.basename(file)
        ind = fname.find('_s') + 2
        df.loc[df['filepath']==file,['plateID',\
                                     'wellID',\
                                     'z_slice_number',\
                                     'GFP_mM']] = \
                                     [fname.split("_")[0].split("PG")[1],\
                                      fname[ind:ind+2],\
                                      fname[ind+3:ind+5],\
                                      fname.split("mM_")[0].split("_")[-1]]
                
        # read image and convert to greyscale
        im = np.array(Image.open(file).convert('L'))        
        #plt.imshow(im)
        
        # measure focus - method='BREN' works best according to Andre's tests
        FM = fmeasure(im, method)
        
        # record focus measure
        df.loc[df['filepath']==file, 'focus_measure'] = FM
   
    # get images with max focus for each well for each GFP concentration
    focussed_images_df = df[df['focus_measure'] == \
                         df.groupby(['GFP_mM','wellID'])['focus_measure'].transform(max)]
    
    assert len(focussed_images_df['filepath']) == \
           len(focussed_images_df['filepath'].unique())
           
    # save most focussed images
    n = len(focussed_images_df['filepath'])
    print("Saving most focussed images..")
    for f, file in enumerate(sorted(focussed_images_df['filepath'])):
        if (f+1) % 10 == 0:
            print("Saving image %d/%d (%.1f%%)" % (f+1, n, ((f+1)/n)*100))

        # create a directory to save copy
        fname, dname = os.path.basename(file), os.path.basename(os.path.dirname(file))
        outDir = os.path.join(output_dir, dname)

        if not os.path.exists(outDir):
            os.makedirs(outDir)
            
        outPath = os.path.join(outDir, fname)
        copyfile(file, outPath)
   
    # save focus measures to file
    outPath = os.path.join(output_dir, 'focus_measures.csv')
    print("Saving focus measures to '%s'" % outPath)
    df.to_csv(outPath, index=False)
    
    return focussed_images_df


def findFocussedImages(df, output_dir, method, imageFormat, show=False):
    
    df = df[df['filepath'].str.endswith(imageFormat)]

    if imageFormat.lower().endswith('czi'):
        focussed_images_df = findFocussedCZI(df, output_dir, method, show)
        
    elif imageFormat.lower().endswith('tif'):
        focussed_images_df = findFocussedTIF(df, output_dir, method)
        
    return focussed_images_df

    
#%% Main
    
if __name__ == "__main__":

    tic = time.time()   

    # Start Java virtual machine (for parsing CZI files with bioformats)
    os.environ["JAVA_HOME"] = "/Library/Internet Plug-Ins/JavaAppletPlugin.plugin/Contents/Home"
    javabridge.start_vm(class_path=bioformats.JARS, max_heap_size='6G')

    classpath = javabridge.JClassWrapper('java.lang.System').getProperty('java.class.path')
    assert pd.Series([os.path.isfile(path) for path in classpath.split(os.pathsep)]).all()    

    # set root directory
    if len(sys.argv) > 1:
        image_root_dir = sys.argv[1]
    else:
#        image_root_dir = '/Volumes/behavgenom$/Andre/_collaborations/2020-filipe-fluorescence/Data/200218_GFP_TIFF_images'
        image_root_dir = '/Volumes/hermes$/PanGenome_SG_JL/' 
#        image_root_dir = '/Users/sm5911/Documents/CellProfiler/WormToolBox/data/CZI_Images' # local copy
#        raise Exception("No directory path provided.")    
    
    # save most focussed images in output directory?
    saveBestFocus = True        
    
    # method for calculating best focussed image: ['BREN','GLVA']
    method = 'BREN'
    
    # Image file type (format) to look for: ['tif','czi']    
    imageFormat = 'czi'

    # find image files
    images_df = findImageFiles(image_root_dir, imageFormat)

    # output directory
    output_dir = "/Users/sm5911/Documents/CellProfiler/WormToolBox/data/CZI_Images_focussed"
    #output_dir = os.path.dirname(os.path.commonprefix(list(images_df['filepath'].values))) + '_focussed'
            
    # find most focussed images
    print("Finding most focussed images..")
    focussed_images_df = findFocussedImages(df=images_df,\
                                            output_dir=output_dir,\
                                            method=method,\
                                            imageFormat=imageFormat,\
                                            show=True)
                                
    # Terminate Java virtual machine
    javabridge.kill_vm()

    toc = time.time()
    print('Complete!\nTotal time taken: %.2f seconds.' % (toc - tic))        


