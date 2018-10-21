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
Silver = cv2.imread('/Users/amelie/PythonRoverRuckus/images/IMG_20181021_145709232.jpg')

#gets height, width and depth of pic
height, width, depth = Silver.shape

#resizes image to have less pixels
imagescale = 400/width
newheight, newwidth = int(height*imagescale), int(width*imagescale)
newSilver = cv2.resize(Silver, (newwidth, newheight))
#plotting resized image and converting to RGB
#plt.imshow(cv2.cvtColor(newSilver, cv2.COLOR_BGR2RGB))
#defining gold in rgb colorspace. got this by inspecting the previous plot
silver = np.uint8([[[180, 15, 200]]])
#convert color to hsv
silver_hsv = cv2.cvtColor(silver, cv2.COLOR_RGB2HSV)
#convert image to hsv
newSilverHSV = cv2.cvtColor(newSilver, cv2.COLOR_BGR2HSV)
#plt.subplot(221),plt.imshow(cv2.cvtColor(newSilver, cv2.COLOR_BGR2RGB))
plt.subplot(221), plt.imshow(newSilverHSV)

    # define range of gold color in HSV
    #picked numbers around the gold_hsv
lower_silver = np.array([90,20,130])
upper_silver = np.array([255,120,255])

# Threshold the HSV image to get only gold colors
mask = cv2.inRange(newSilverHSV, lower_silver, upper_silver)
plt.subplot(222), plt.imshow(mask , cmap = 'gray')

    # Bitwise-AND mask and original image
    #mask is black and white
    #bitwise_and puts the image behind the mask
    #in comp, we just need the black/white mask
res = cv2.bitwise_and(newSilver,newSilver, mask= mask)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_HSV2RGB))

#plt.imshow(mask, cmap ="gray" , interpolation = "nearest")
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))


M = cv2.moments(mask)
#Finds the center of the cube
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
#finds the edges
#edges = cv2.Canny(mask, 100, 200)
#plt.imshow(edges, cmap = 'gray')


#erosion deletes islands
#dilations delete holes

kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(mask,kernel, iterations=1)
dilation = cv2.dilate(erosion,kernel,iterations =1)
kernel = np.ones((20,20), np.uint8)
dilation1 = cv2.dilate(dilation,kernel,iterations =1)
erosion2 = cv2.erode(dilation1,kernel,iterations=1)
plt.imshow(erosion, cmap = 'gray')
plt.subplot(221), plt.imshow(cv2.cvtColor(Silver, cv2.COLOR_BGR2RGB))
plt.subplot(223), plt.imshow(mask , cmap = 'gray')
plt.subplot(224), plt.imshow(erosion2 , cmap = 'gray')
#plt.subplot(223), plt.imshow(erosion , cmap = 'gray')
#plt.subplot(224), plt.imshow(mask , cmap = 'gray')


#finds edges
edges = cv2.Canny(erosion2,100,200)
plt.subplot(221), plt.imshow(cv2.cvtColor(Silver, cv2.COLOR_BGR2RGB))
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(erosion2,cmap = 'gray')
plt.title('Erosion Image'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()


#finds circles
#img = cv2.imread('opencv_logo.png',0)

cimg = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
#copies a gray scale edges map to a bgr picture
circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,10,200,param1=50,param2=30,minRadius=0,maxRadius=80)

circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
plt.subplot(121), plt.imshow(cv2.cvtColor(Silver, cv2.COLOR_BGR2RGB))

plt.subplot(122), plt.imshow(cimg)
plt.show()
