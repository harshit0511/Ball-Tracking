#BOWEN LIN bl2514
import cv2
import copy
import math
import numpy as np
from matplotlib import pyplot as plt
import functions as ed

fn = '5x5filter_1d.jpg'
img = cv2.imread(fn, 0)

height, width = img.shape


#use sobel operator to get x and y derivatives
sobelox = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobeloy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

imgdx = ed.derivative(img, sobelox, height, width, 3, 3)
imgdy = ed.derivative(img, sobeloy, height, width, 3, 3)
#cv2.imwrite('dx.jpg', imgdx)
#cv2.imwrite('dy.jpg', imgdy)


#get image gradient magnitude
imgmagnitude = ed.magnitude(imgdx, imgdy, height, width)
imgmagnitudedisplay = ed.adjust(imgmagnitude, height, width)
cv2.imwrite('magnitudemap.jpg', imgmagnitudedisplay)


#get image gradient orientation
imgorientation, imgedgethin = ed.orientation(imgdx, imgdy, height, width, imgmagnitude)
cv2.imwrite('orientation_map.jpg', imgorientation)
cv2.imwrite('edge_map_thin.jpg', imgedgethin)


#apply threshold and edgelink
threshold_low = 25
threshold_high = 50
imgedge = ed.doublethreshold(imgedgethin, height, width, threshold_low, threshold_high)
cv2.imwrite('edge_map.jpg', imgedge)
