#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 19:40:59 2018

@author: amelie
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

#gets image
GoldC = cv2.imread('/Users/amelie/PythonRoverRuckus/images/GoldC1.jpg')
#gets height, width and depth of pic
height, width, depth = GoldC.shape
#resizes image to have less pixels
imagescale = 400/width
newheight, newwidth = int(height*imagescale), int(width*imagescale)
newGoldC = cv2.resize(GoldC, (newwidth, newheight))
#plotting resized image and converting to RGB
plt.imshow(cv2.cvtColor(newGoldC, cv2.COLOR_BGR2RGB))
#defining gold in rgb colorspace. got this by inspecting the previous plot
gold = np.uint8([[[180, 105, 35]]])
#convert color to hsv
gold_hsv = cv2.cvtColor(gold, cv2.COLOR_RGB2HSV)
#convert image to hsv
newGoldCHSV = cv2.cvtColor(newGoldC, cv2.COLOR_BGR2HSV)

    # define range of gold color in HSV
    #picked numbers around teh gold_hsv
lower_gold = np.array([6,150,150])
upper_gold = np.array([24,255,255])

# Threshold the HSV image to get only gold colors
mask = cv2.inRange(newGoldCHSV, lower_gold, upper_gold)

    # Bitwise-AND mask and original image
    #mask is black and white
    #bitwise_and puts the image behind the mask
    #in comp, we just need the black/white mask
res = cv2.bitwise_and(newGoldC,newGoldC, mask= mask)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_HSV2RGB))

plt.imshow(mask, cmap ="gray" , interpolation = "nearest")
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))


M = cv2.moments(mask)
#Finds the center of the cube
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
#finds the edges
#edges = cv2.Canny(mask, 100, 200)
#plt.imshow(edges, cmap = 'gray')


kernel = np.ones((5,5), np.uint8)
dilation = cv2.dilate(mask,kernel,iterations =1)
erosion = cv2.erode(dilation,kernel, iterations=1)
plt.imshow(erosion, cmap = 'gray')
plt.subplot(221), plt.imshow(mask , cmap = 'gray')
plt.subplot(222), plt.imshow(dilation , cmap = 'gray')
plt.subplot(223), plt.imshow(erosion , cmap = 'gray')
plt.subplot(224), plt.imshow(mask , cmap = 'gray')
