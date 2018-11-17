import numpy as np
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import os
from skimage import data
from skimage import color
from skimage.util import view_as_blocks
from skimage.feature import corner_harris, corner_subpix, corner_peaks, CENSURE
from sklearn.datasets import load_sample_image
import cv2.cv2
from skimage import img_as_float,data


#print(os.path.join('resources\\s1' , 'max_001.png'))


# Convert path to Windows format
#path_on_windows = PureWindowsPath(mypath)

#img=img_as_float(cv2.imread('max_001.png'))

from skimage.io import imread
con_sk=img_as_float(imread('max_001.png'))
con_cv=img_as_float(cv2.imread('max_001.png')) 


img=load_sample_image('china.jpg')
plt.imshow(con_sk, cmap='Set3')
plt.imshow(con_cv, cmap='Set3')

"""
img=load_sample_image('china.jpg')

detector = CENSURE()
detector.detect(img)

coords = corner_peaks(corner_harris(img), min_distance=5)
coords_subpix = corner_subpix(img, coords, window_size=13)

plt.subplot(121)
plt.title('CENSURE feature detection')
plt.imshow(img, cmap='Set3')
plt.scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
              2 ** detector.scales, facecolors='none', edgecolors='r')

plt.subplot(122)
plt.title('Harris Corner Detection')
plt.imshow(img, cmap='Set3')  # show me the leaf
plt.plot(coords[:, 1], coords[:, 0], '.b', markersize=5)
plt.show()

"""