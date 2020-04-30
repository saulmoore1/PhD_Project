#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Get best focussed TIF/CZI images based on focus measure

input
-----
image_root_dir (str): path to parent directory of images to be processed


supported methods
-----------------
GLVA: Graylevel variance (Krotkov, 86)
BREN: Brenner's (Santos, 97)

----------------------------
@author: Saul Moore (sm5911)
@date: 16/03/2020

"""

#%% Imports

import os, sys, glob, time
import numpy as np
import pandas as pd
import bioformats, javabridge, unicodedata #czifile
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
   
     
def getMetadata(filePath):
    """ Get metadata associated with image file as xml.

    Input: filePath: name and path to image
    
    Output: omeXMLObject: xml object based on OME standard ready to be parsed
    
    """
    xml=bioformats.get_omexml_metadata(filePath)
    # xml is in unicode and may contain characters like 'mu'
    # these characters will fail the xml parsing, thus recode the information
    xml_normalized = unicodedata.normalize('NFKD',xml).encode('ascii','ignore')
    
    omeXMLObject = bioformats.OMEXML(xml_normalized)
    
#    # Parse XML with beautiful soup
#    from bs4 import BeautifulSoup as bs    
#    soup = bs(xml_normalized)
#    prettysoup = bs.prettify(soup)
#            
#    # Extract metadata from XML
#    import re
#    image_series = re.findall(r'Image:[0-9]+', prettysoup)
#    image_series = [im.split(':')[-1] for im in image_series]

    return omeXMLObject


def readMetadata(omeXMLObject, imageID=0):
    """ Parses most common meta data out of OME-xml structure.
    
    Input: omeXMLObject: OME-XML object with meta data
           imageID: the image in multi-image files the meta data should 
                    come from. Default=0
     
    Output: meta: dict with meta data
     
    Warning: If some keys are not found, software replaces the values 
             with default values.
    """       
    meta={'AcquisitionDate': omeXMLObject.image(imageID).AcquisitionDate}
    meta['Name']=omeXMLObject.image(imageID).Name
    meta['SizeC']=omeXMLObject.image(imageID).Pixels.SizeC
    meta['SizeT']=omeXMLObject.image(imageID).Pixels.SizeT
    meta['SizeX']=omeXMLObject.image(imageID).Pixels.SizeX
    meta['SizeY']=omeXMLObject.image(imageID).Pixels.SizeY
    meta['SizeZ']=omeXMLObject.image(imageID).Pixels.SizeZ

    # Most values are not included in bioformats parser. 
    # Thus, we have to find them ourselves
    # The normal find procedure is problematic because each item name 
    # is proceeded by a schema identifier
    try:
        pixelsItems=omeXMLObject.image(imageID).Pixels.node.items()  
        meta.update(dict(pixelsItems))
    except:
        print('Could not read meta data in LoadImage.read_standard_meta\n\
               used default values')
        meta['PhysicalSizeX']=1.0
        meta['PhysicalSizeXUnit']='mum'
        meta['PysicalSizeY']=1.0
        meta['PhysicalSizeYUnit']='mum'
        meta['PysicalSizeZ']=1.0
        meta['PhysicalSizeZUnit']='mum'
        for c in range(meta['SizeC']):
            meta['Channel_'+str(c)]='Channel_'+str(c)
    return meta


def findFocussedCZI(df, output_dir, method='BREN', imageSizeThreshXY=None, show=False):
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
    
    file_df_list = []
    df = df.sort_values(by=['filepath']).reset_index(drop=True)    
    n = len(df['filepath'].unique())
    for f, file in enumerate(df['filepath']):
        print("\nProcessing file %d/%d (%.1f%%)" % (f+1, n, ((f+1)/n)*100))
        
        ##### TEST CODE BREAKER #####
        #if f == 1:
        #    raise Exception('STOP!')
        #############################

        # extract metadata from filename
        fname, dname = os.path.basename(file), os.path.basename(os.path.dirname(file))
        frname = fname.split('.')[0]
        plateID = frname.split("_")[0].split("PG")[1]
        GFP_mM = frname.split("mM_")[0].split("_")[-1]
        
        # get the actual image reader
        rdr = bioformats.get_image_reader(None, path=file)
        #with bioformats.ImageReader(path=file, url=None, perform_init=True) as reader:
        #    img_arr = reader.read(file)
        #image_arrays = czifile.imread(file)

        # get total image series count
        try:
            # for "whatever" reason the number of total image series can only be accessed this way
            totalseries = np.int(rdr.rdr.getSeriesCount())
        except:
            totalseries = 1 # in case there is only ONE series

        # OPTIONAL: Get metadata (obtain instrument info)
        #omeXMLObject = getMetadata(file)
        #meta = readMetadata(omeXMLObject)
        
        # parse the CZI file
        file_info = []
        too_small_log = []
        # Loop over wells (series)
        for sc in range(0, totalseries):
            
            # Set reader to series
            rdr.rdr.setSeries(sc)
            
            # Filter small images
            if imageSizeThreshXY:
                x, y = rdr.rdr.getSizeX(), rdr.rdr.getSizeY()
                if (x <= imageSizeThreshXY[0] and y <= imageSizeThreshXY[1]):
                    too_small_log.append(sc)
                else:
                    # get number of z-slices
                    zSlices = rdr.rdr.getImageCount()

                    # Loop over z-slices    
                    for zc in range(0, zSlices):
                        img = rdr.read(c=None, z=0, t=0, series=sc, index=zc,\
                                       rescale=False)
                        
                        # measure focus of raw image (uint16)
                        fm = fmeasure(img, method)
                                                
                        # store image info
                        file_info.append([file, plateID, GFP_mM, sc, zc, fm])

        if len(too_small_log) > 0:
            print("WARNING: %d image series were omitted (image size too small)"\
                  % len(too_small_log))
        
        # create dataframe from list of recorded data
        colnames = ['filepath','plateID','GFP_mM','seriesID','z_slice_number','focus_measure']
        file_df = pd.DataFrame.from_records(file_info, columns=colnames)
        
        # store file info
        file_df_list.append(file_df)
        
        # get images with max focus for each well/GFP concentration
        focussed_images_df = file_df[file_df['focus_measure'] == \
                             file_df.groupby(['seriesID'])['focus_measure']\
                             .transform(max)]
        print("%d focussed images found." % focussed_images_df.shape[0])

        # save most focussed images
        print("Saving most focussed images..")
        
        # Add dname to outDir when analysing multiple replicate folders at a time
        if df.shape[0] == len([i for i in df['filepath'] if dname in i]):
            # We are analysing a single replicate folder
            outDir = os.path.join(output_dir, frname)
        else:
            # We are analysing multiple replicate folders
            outDir = os.path.join(output_dir, dname, frname)
        if not os.path.exists(outDir):
            os.makedirs(outDir)
            
        for i in range(focussed_images_df.shape[0]):
            if (i+1) % 8 == 0:
                print("%d/%d" % (i+1, focussed_images_df.shape[0]))

            # Extract image metadata from filename
            img_info = focussed_images_df.iloc[i]            
            seriesID = img_info['seriesID']
            zSlice = img_info['z_slice_number']
            
            # We do NOT want to rescale images if comparing between them
            rdr.rdr.setSeries(seriesID)
            img = rdr.read(c=None, z=0, t=0, series=seriesID,\
                           index=zSlice, rescale=False)
            
            assert img.dtype == np.uint16 
            
            # Convert image to 8-bit (Warning: some information will be lost)
            #import cv2
            #img2 = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
            #assert img2.dtype == np.uint8
            
            if show:
                plt.close('all')
                plt.imshow(img); plt.pause(5)
                
            # save as TIFF image
            outPath = os.path.join(outDir, frname +\
                                    '_s%dz%d' % (seriesID, zSlice) + '.tif')
            
            # Save as TIFF (bioformats)
            bioformats.write_image(pathname=outPath,\
                                   pixels=img,\
                                   pixel_type=bioformats.PT_UINT16)
                            
    # concatenate dataframe from CZI file info
    df = pd.concat(file_df_list, axis=0, ignore_index=True)

    # save focus measures to file
    outDir = os.path.join(output_dir, 'focus_measures.csv')
    #df.to_csv()

    focussed_images_df = df[df['focus_measure'] == \
                         df.groupby(['plateID','GFP_mM','seriesID'])['focus_measure']\
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
        if (f+1) % 10 == 0:
            print("Processing file %d/%d (%.1f%%)" % (f+1, n, ((f+1)/n)*100))
        
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

        # save copy            
        outPath = os.path.join(outDir, fname)
        copyfile(file, outPath)
   
    # save focus measures to file
    outPath = os.path.join(output_dir, 'focus_measures.csv')
    print("Saving focus measures to '%s'" % outPath)
    df.to_csv(outPath, index=False)
    
    return focussed_images_df


def findFocussedImages(df, output_dir, method, imageFormat, imageSizeFilterXY, show=False):
    
    df = df[df['filepath'].str.endswith(imageFormat)]

    if imageFormat.lower().endswith('czi'):
        focussed_images_df = findFocussedCZI(df, output_dir, method, imageSizeFilterXY, show)
        
    elif imageFormat.lower().endswith('tif'):
        focussed_images_df = findFocussedTIF(df, output_dir, method)
        
    return focussed_images_df

    
#%% Main
    
if __name__ == "__main__":

    tic = time.time()   

    ######################
    
    # Start Java virtual machine (for parsing CZI files with bioformats)
    # TODO: Sort python-bioformats/javabridge compatibility with macOS Catalina
    os.environ["JAVA_HOME"] = "/Library/Internet Plug-Ins/JavaAppletPlugin.plugin/Contents/Home"
    javabridge.start_vm(class_path=bioformats.JARS, max_heap_size='6G')

    classpath = javabridge.JClassWrapper('java.lang.System').getProperty('java.class.path')
    assert pd.Series([os.path.isfile(path) for path in classpath.split(os.pathsep)]).all()    

    ######################
    ##### PARAMETERS #####
    
    # set root directory
    if len(sys.argv) > 1:
        image_root_dir = sys.argv[1]
    else: 
        # local copy
        image_root_dir = '/Users/sm5911/Documents/fluorescence_data_local_copy'
        #raise Exception("No directory path provided.")    
    
    # save most focussed images in output directory?
    saveBestFocus = True        
    
    # method for calculating best focussed image: ['BREN','GLVA']
    method = 'BREN'
    
    # image file type (format): ['tif','czi']    
    imageFormat = 'czi'
    
    # Size filter to use only small images (< 1000x1000 pixels)
    imageSizeThreshXY = [1024,1024]

    ####################
    ##### COMMANDS #####
    
    # find image files
    images_df = findImageFiles(image_root_dir, imageFormat)
    
    # filter image file list
    filterStr = None
    if filterStr:
        images_df = images_df[images_df['filepath'].isin([f for f in\
                              images_df['filepath'] if filterStr in f])]
        print("Filtering for images that contain: '%s' (%d files)" % (filterStr, images_df.shape[0]))

    # output directory
    output_dir = os.path.dirname(os.path.commonprefix(list(images_df['filepath'].values))) + '_focussed'
    
    # find + save most focussed images
    print("Finding most focussed images..")
    focussed_images_df = findFocussedImages(df=images_df,\
                                            output_dir=output_dir,\
                                            method=method,\
                                            imageFormat=imageFormat,\
                                            imageSizeFilterXY=imageSizeThreshXY,\
                                            show=False)
        
    ####################
    
    # Terminate Java virtual machine
    javabridge.kill_vm()

    toc = time.time()
    print('Complete!\nTotal time taken: %.2f seconds.' % (toc - tic))        
