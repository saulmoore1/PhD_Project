#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:47:29 2020

@author: sm5911
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
#%%
img_path = '/Users/lferiani/Desktop/ffmpeg_playground/PG10_0mM_GFP_s121z2.tif'
img = cv2.imread(img_path, -1)
img_float = img.astype(np.float)
img_float = np.log10(img_float)
img_float -= img_float.min()
img_float /= img_float.max()
ff = np.fft.fft2(img_float)
keep_modes_frac = 1/64
cut_rows_min = int(keep_modes_frac/2*ff.shape[0])
print(cut_rows_min)
cut_rows_max = int(ff.shape[0] - cut_rows_min)
cut_cols_min = int(keep_modes_frac/2*ff.shape[1])
cut_cols_max = int(ff.shape[1] - cut_rows_min)
ff[cut_rows_min:cut_rows_max, cut_cols_min:cut_cols_max] *= 0
img_filtered = np.fft.ifft2(ff)
img_filtered = abs(img_filtered)
#%%
plt.close('all')
plt.figure()
plt.imshow(img_filtered)
plt.title('filtered')
plt.show()
plt.figure()
plt.imshow(img_float * img_filtered)
plt.title('times')
plt.show()
#%%
x = np.linspace(-10, 10, 1000)
y = np.sin(x) + np.random.randn(x.shape[0])
y_ff = np.fft.fft(y)
y_ff_filt = y_ff.copy()
y_ff_filt[5:-5] = 0
y_filt = np.fft.ifft(y_ff_filt)
plt.figure()
plt.plot(x, y)
plt.plot(x, y_filt)
plt.figure()
plt.plot(y_ff)
plt.plot(y_ff_filt)
