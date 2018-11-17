import numpy as np
from sklearn.feature_extraction import image
from skimage import img_as_float,data
import cv2.cv2



one_image = img_as_float(cv2.imread('max_001.png'))

#one_image = np.arange(4 * 4 * 3).reshape((4, 4, 3))
#one_image[:, :, 0]  # R channel of a fake RGB picture





patches = image.extract_patches_2d(one_image, (1,1), max_patches=2,
    random_state=0)
print(patches.shape)
print(patches)
print("")
print(patches[:, :, :, 0])
#patches[:, :, :, 0]

patches = image.extract_patches_2d(one_image, (1, 1))
print(patches.shape)

#patches[4, :, :, 0]

#reconstructed = image.reconstruct_from_patches_2d(patches, (4, 4, 3))
reconstructed = image.reconstruct_from_patches_2d(patches, (366, 366, 3))

np.testing.assert_array_equal(one_image, reconstructed)