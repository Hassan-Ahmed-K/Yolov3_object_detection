#Importing Required Liberary
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
import os
import numpy as np

#Function To preprocess the Image

def load_and_preprocess_image(path,shape):
  image=tf.io.read_file(path) # Read Image File From The Given Path
  width,height=load_img(path).size
  image = tf.image.decode_image(image, channels=3)  # Decode image in to 3-channel Numpy array
  image=tf.image.resize(image, shape) # Resize The give Image
  image/=255 # Normalize The Given Array
  return image,width,height


# Function TO crop and Save the Detected Objects

base_dir = "/home/hassan-ahmed-khan/Ai/Yolo/Object_Detection/ouput"
def cropped_detected_object(image_path,boxes,labels,save = False):
    img = Image.open(image_path) # Opening Image By Using PIL liberary
    for i in range(len(boxes)):
        box = tuple(boxes[i]) 
        cropped_img = img.crop(box) # Cropping The Image
        if(save):
          try:
            os.mkdir(os.path.join(base_dir,labels[i])) # Making Lable Directory 
          except:
            print(f'"{labels[i]}" folder already exist')
          # Save the Crop Image
          cropped_img.save(f'/home/hassan-ahmed-khan/Ai/Yolo/Object_Detection/ouput/{labels[i]}/{labels[i]}({len(os.listdir(os.path.join(base_dir,labels[i])))}).jpg')
          
     
def load_and_preprocess_frame(frame,shape):
  width = frame.shape[1]
  height =frame.shape[0]
  image= np.array(tf.image.resize(frame, shape))
  image = image/255 # Normalize The Given Array
  return image,width,height