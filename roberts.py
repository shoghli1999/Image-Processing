import cv2
import numpy as np
#define the vertical filter
vertical_filter = [[1,0], [0,-1]]
#define the horizontal filter
horizontal_filter = [[0,1], [-1,0]]
#read image
img = cv2.imread(r'C:\Users\MOSALAS\Pictures\elephant.jpg')
#get the dimensions of the image
n,m,d = img.shape
#initialize the edges image
edges_img = img.copy()
#loop over all pixels in the image
for row in range(2, n-1):
    for col in range(2, m-1): 
        #create little local 2x2 box
        local_pixels = img[row-1:row+1, col-1:col+1,0]
        #apply the vertical filter
        vertical_transformed_pixels = vertical_filter*local_pixels
        #remap the vertical score
        vertical_score = vertical_transformed_pixels.sum()
        #apply the horizontal filter
        horizontal_transformed_pixels = horizontal_filter*local_pixels
        #remap the horizontal score
        horizontal_score = horizontal_transformed_pixels.sum()
        #combine the horizontal and vertical scores into a total edge score
        edge_score = (vertical_score**2 + horizontal_score**2)**.5
        #insert this edge score into the edges image
        edges_img[row, col] = [edge_score]
cv2.imshow('roberts',edges_img)
cv2.waitkey(0)
