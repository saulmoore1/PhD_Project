#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual Labelling of Food Regions in Assay

@author: sm5911
@date: 01/03/2021

"""

#%% Imports


#%% Functions

# # MANUAL LABELLING OF FOOD REGIONS (To check time spent feeding throughout video)

# # Return list of pathnames for masked videos in the data directory under given imaging dates
# maskedfilelist = []
# date_total = []
# for i, expDate in enumerate(IMAGING_DATES):
#     tmplist = lookforfiles(os.path.join(DATA_DIR, "MaskedVideos", expDate), ".*.hdf5$")
#     date_total.append(len(tmplist))
#     maskedfilelist.extend(tmplist)
# print("%d masked videos found for imaging dates provided:\n%s" % (len(maskedfilelist), [*zip(IMAGING_DATES, date_total)]))

# first_snippets = [snip for snip in maskedfilelist if ('/%.6d.hdf5' % snippet) in snip]
# print("\nManual labelling:\n%d masked video snippets found for %d assay recordings (duration: 2hrs)"\
#       % (len(maskedfilelist), len(first_snippets)))

# # Manual labelling of food regions in each assay using 1st video snippet, 1st frame
# if MANUAL_LABELLING:
#     plt.ion() # Interactive plotting (for user input when labelling plots)
#     tic = time.time()
#     for i, maskedfilepath in enumerate(first_snippets):    
#         # Manually outline + assign labels to food regions + save coordinates + trajectory overlay to file           
#         manuallabelling(maskedfilepath, n_poly=1, out_dir='MicrobiomeAssay', save=True, skip=True)       
#     print("Manual labelling complete!\n(Time taken: %d seconds.)\n" % (time.time() - tic))

# # TODO: Automate food labelling using (U-Net) ML algorithm -- pre-trained already by Luigi



