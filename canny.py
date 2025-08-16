import cv2
import numpy as np

img = cv2.imread(r'C:\Users\MOSALAS\Pictures\elephant.jpg')
edges_img = cv2.Canny(img,100,300)

cv2.imshow('canny',edges_img)
cv2.waitkey(0)
