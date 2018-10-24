import cv2
import glob
import numpy as np

X_data = []
files = glob.glob ("C:/Users/Max/git/reconocimiento/resources/*.PNG")
for myFile in files:
    print(myFile)
    image = cv2.imread (myFile)
    X_data.append (image)

print('X_data shape:', np.array(X_data).shape)