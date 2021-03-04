#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Change path

@author: sm5911
@date: 04/03/2021

"""

#%% Globals

# Path dictionary = {key, (dir, suffix)}
PATH_DICT = {'raw' : ('/RawVideos/', '.yaml'),
             'masked' : ('/MaskedVideos/', '.hdf5'),
             'features' : ('/Results/', '_featuresN.hdf5'),
             'skeletons' : ('/Results/', '_skeletons.hdf5'),
             'intensities' : ('/Results/', '_intensities.hdf5')}
    
#%% Functions    

def change_path(input_path, to):
    """ Edit Tierpsy filepath 
        
        Parameters
        ----------
        input_path : str, pathlib.PosixPath
            Path to change
        to : str
            Choose from: 'raw', 'masked', 'features', 'skeletons', 'intensities'
            
        Returns
        -------
        output_path : pathlib.PosixPath
    """
        
    if not isinstance(input_path, str):
        input_path = str(input_path)
    
    # determine input file type
    in_type = None
    for k, (d, s) in PATH_DICT.items():
        if d in input_path:
            in_type = k
    assert in_type is not None
        
    root, suffix  = input_path.split(PATH_DICT[in_type][0])
    assert suffix.endswith(PATH_DICT[in_type][1])
    
    # change path
    suffix = suffix.replace(PATH_DICT[in_type][1], PATH_DICT[to.lower()][1])
    output_path = root + PATH_DICT[to.lower()][0] + suffix
    
    return output_path
