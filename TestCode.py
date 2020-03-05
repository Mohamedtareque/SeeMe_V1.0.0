import numpy as np 
import cv2

height = 48
width = 48

img = cv2.imread('testSet/Happy.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("image shape ", img.shape)

img = np.asarray(img)
img=cv2.resize(img,(48,48))

img = np.expand_dims(img, axis=-1)

print("image shape after resize", img.shape)

img= img.astype('float32')
img /= 255.0


input_data = np.expand_dims(img, axis=0)

print("input data shape", input_data.shape)