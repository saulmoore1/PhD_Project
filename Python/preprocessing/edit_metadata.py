#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute additional information for analysis and append to metadata

@author: sm5911
@date: 01/03/2021

"""

#%% Functions

def duration_L1_diapause(df):
    """ Calculate L1 diapause duration (if possible) and append to results """
    
    import pandas as pd
    from datetime import datetime
    
    assert type(df) == type(pd.DataFrame())
    
    diapause_required_columns = ['date_bleaching_yyyymmdd','time_bleaching',\
                                 'date_L1_refed_yyyymmdd','time_L1_refed_OP50']
        
    if all(x in df.columns for x in diapause_required_columns) and \
       all(df[x].any() for x in diapause_required_columns):
        # Extract bleaching dates and times
        bleaching_datetime = [datetime.strptime(date_str + ' ' +\
                              time_str, '%Y%m%d %H:%M:%S') for date_str, time_str\
                              in zip(df['date_bleaching_yyyymmdd'].astype(str),\
                              df['time_bleaching'])]
        # Extract dispensing dates and times
        dispense_L1_datetime = [datetime.strptime(date_str + ' ' +\
                                time_str, '%Y%m%d %H:%M:%S') for date_str, time_str\
                                in zip(df['date_L1_refed_yyyymmdd'].astype(str),\
                                df['time_L1_refed_OP50'])]
        # Estimate duration of L1 diapause
        L1_diapause_duration = [dispense - bleach for bleach, dispense in \
                                zip(bleaching_datetime, dispense_L1_datetime)]
        
        # Add duration of L1 diapause to df
        df['L1_diapause_seconds'] = [int(timedelta.total_seconds()) for \
                                     timedelta in L1_diapause_duration]
    else:
        missingInfo = [x for x in diapause_required_columns if x in df.columns\
                       and not df[x].any()]
        print("""WARNING: Could not calculate L1 diapause duration.\n\t\
         Required column info: %s""" % missingInfo)

    return df

def duration_on_food(df):
    """ Calculate time worms since worms dispensed on food for each video 
        entry in metadata """

    import pandas as pd
    from datetime import datetime
    
    assert type(df) == type(pd.DataFrame())
    
    duration_required_columns = ['date_yyyymmdd','time_recording',
                                 'date_worms_on_test_food_yyyymmdd',
                                 'time_worms_on_test_food']
    
    if all(x in df.columns for x in duration_required_columns) and \
        all(df[x].any() for x in duration_required_columns):
        # Extract worm dispensing dates and times
        dispense_datetime = [datetime.strptime(date_str + ' ' +\
                             time_str, '%Y%m%d %H:%M:%S') for date_str, time_str in
                             zip(df['date_worms_on_test_food_yyyymmdd'].astype(int).astype(str),
                             df['time_worms_on_test_food'])]
        # Extract imaging dates and times
        imaging_datetime = [datetime.strptime(date_str + ' ' +\
                            time_str, '%Y%m%d %H:%M:%S') for date_str, time_str in 
                            zip(df['date_yyyymmdd'].astype(int).astype(str), df['time_recording'])]
        # Estimate duration worms have spent on food at time of imaging
        on_food_duration = [image - dispense for dispense, image in \
                            zip(dispense_datetime, imaging_datetime)]
            
        # Add duration on food to df
        df['duration_on_food_seconds'] = [int(timedelta.total_seconds()) for \
                                          timedelta in on_food_duration]
   
    return df
