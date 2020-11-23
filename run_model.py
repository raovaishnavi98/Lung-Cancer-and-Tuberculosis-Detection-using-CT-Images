import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
import cv2 as cv
import numpy as np
from tkinter import Tk

Tk().withdraw()
from tkinter.filedialog import askopenfilename

def run_model(image):
  model = tf.keras.models.load_model('extra_layer_2_cancer.h5')
  out=model.predict(image)
  print()
  print(out)
  print()
  ans = int(out[0][0])
  if ans == 1:
      print("Cancer Positive")
  else:
      print("Cancer Negative")
  print()

#test='dataset_final/test_data/0/image_2125.jpg' #Negative Image
#test='dataset_final/test_data/1/image_4049.jpg' #Positive Image

#test='dataset_final/test_data/1/image_291489.jpg'

test = askopenfilename(initialdir = "/media/sf_New_folder/dataset_final/test_data/1/",) 
#test='test/image_12604.jpg'
image = cv.imread(test)
#print(image.shape)
img = (np.expand_dims(image,0))
#print(img.shape)

run_model(img)

#cv.imshow('image',image)
#cv.waitKey(0)
