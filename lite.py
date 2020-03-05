# =============================================================================
# Import Libraries
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
#import tf.lite as tflite
import argparse
import cv2

# =============================================================================
# load Data
# =============================================================================

"""
Describtion :- return list of labels 
input:- file directory file.txt 
output:- return list of labels 
        ex:.['angry', 'disgusted', 'fear', 'happy',
                           'neutral', 'sad', 'surprised']

"""
def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]


if __name__ == '__main__':

#Load TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(model_path="model.tflite")
  interpreter.allocate_tensors()
# Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

# check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32

# images 
  height = 48
  width = 48
#load test image 
  img = cv2.imread('testSet/Fear.png')
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#convert image to gray from BGR 1 channel

  #print("image shape ", img.shape)
  img = np.asarray(img)#convert img to array 
  img=cv2.resize(img,(height,width))#resizing image to 48x48 2D array

  img = np.expand_dims(img, axis=-1)#
  #print("image shape after resize", img.shape)
  img= img.astype('float32')
  img /= 255.0 #normalize data


  input_data = np.expand_dims(img, axis=0)
  #print("input data shape", input_data.shape)

  if floating_model:
    input_data = (np.float32(input_data) - 127.5) / 127.5

  interpreter.set_tensor(input_details[0]['index'], input_data)

  interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
  output_data = interpreter.get_tensor(output_details[0]['index'])
  results = np.squeeze(output_data)
  print("result",results)
  top_k = results.argsort()[-2:][::-1]
  labels = load_labels("labels.txt")
  print("top_k",top_k)


  for i in top_k:
    if floating_model:
      print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
    else:
      print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
