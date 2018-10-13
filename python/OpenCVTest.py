#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 19:40:59 2018

@author: amelie
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


GoldR = cv2.imread('/Users/amelie/PythonRoverRuckus/images/GoldR.jpg')
height, width, depth = GoldR.shape
imagescale = 400/width
newheight, newwidth = int(height*imagescale), int(width*imagescale)
newGoldR = cv2.resize(GoldR, (newwidth, newheight))
plt.imshow(cv2.cvtColor(newGoldR, cv2.COLOR_BGR2RGB))
gold = np.uint8([[[180, 105, 35]]])
gold_hsv = cv2.cvtColor(gold, cv2.COLOR_RGB2HSV)
newGoldRHSV = cv2.cvtColor(newGoldR, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
lower_gold = np.array([6,150,150])
upper_gold = np.array([24,255,255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(newGoldRHSV, lower_gold, upper_gold)

    # Bitwise-AND mask and original image
res = cv2.bitwise_and(newGoldR,newGoldR, mask= mask)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_HSV2RGB))

plt.imshow(mask, cmap ="gray" , interpolation = "nearest")
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
