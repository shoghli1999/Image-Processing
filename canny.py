import sys

import cv2
import numpy as np

if len(sys.argv) < 2:
    sys.exit('usage: python canny.py <image file>')
img = cv2.imread(sys.argv[1])
edges_img = cv2.Canny(img,100,300)

cv2.imshow('canny',edges_img)
cv2.waitKey(0)
