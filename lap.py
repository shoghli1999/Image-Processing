import sys

import cv2
import numpy as np
#define the lap filter
lap_filter = [[0,1,0], [1,-4,1], [0,1,0]]
#read the image
if len(sys.argv) < 2:
    sys.exit('usage: python lap.py <image file>')
img = cv2.imread(sys.argv[1])
#get the dimensions of the image
n,m,d = img.shape
#initialize the edges image
edges_img = img.copy()
#loop over all pixels in the image
for row in range(3, n-2):
    for col in range(3, m-2):
        #create little local 3x3 box
        local_pixels = img[row-1:row+2, col-1:col+2, 0]
        #apply the lap filter
        lap_transformed_pixels = lap_filter*local_pixels
        #remap the lap score
        lap_score = (lap_transformed_pixels.sum()**2)**0.5
        edges_img[row, col] = [lap_score]
cv2.imshow('lap',edges_img)
cv2.waitKey(0)
