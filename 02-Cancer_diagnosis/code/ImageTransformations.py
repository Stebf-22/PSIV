import cv2
from torchvision import transforms
from scipy import ndimage
import random 
import torch
import numpy as np

def erode(img, kernel = (5,5),iters = 1):
  assert len(img.shape) == 2
  kernel_ = np.ones(kernel, np.uint8)
  return cv2.erode(img.copy(), kernel_,iterations=iters)

def dilate(img, kernel=(5,5),iters=1):
  assert len(img.shape) == 2
  kernel_ = np.ones(kernel, np.uint8)
  return cv2.dilate(img.copy(), kernel_,iterations=iters)


def opening(img, kernelSize = 5,iters=1):
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
  return cv2.morphologyEx(img.copy(), cv2.MORPH_OPEN, kernel,iterations=iters)
 

def closing(img, kernelSize = 5,iters=1):
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
  return cv2.morphologyEx(img.copy(), cv2.MORPH_CLOSE, kernel,iterations=iters)

def morphGrad(img, kernelSize=(13,6)):
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
  return cv2.morphologyEx(img.copy(), cv2.MORPH_GRADIENT, kernel)

def resize(img,des_size = (30,60)):
    return cv2.resize(img, des_size)

def rotate(img, angle):
    return ndimage.rotate(img, angle)

def GaussianNoise(img, mean = 0.2, stdev = 0.3):
    noise = stdev * np.random.randn(*img.shape) + mean
    #noise = img.data.new(img.size())
    return img + noise

def scale(img, fx,fy):
    #print(img.shape)
    return resize(cv2.resize(img,img.shape,fx=fx,fy=fy,interpolation=cv2.INTER_CUBIC))

def otsu_transformation(img):
    threshold, image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image

def dowm_sample_up_sample(img):
    img = resize(img,( 20, 20))
    return resize(img)

def doTransformations(img):
     img = resize(img)
     #### Rotation Augmentation ####
     rotation_probability = random.random()
     max_rotation, min_rotation = 15 , 0
     rotation_angle = random.random() * (max_rotation - min_rotation) + min_rotation
     if rotation_probability <= 0.15:
         img = rotate(img, int(rotation_angle))
     #print(img.shape)
     #### SCALE IMAGE ####
     #print(img.shape)
     down_up_prob = random.random()
     if down_up_prob <= 0.5:
      img = dowm_sample_up_sample(img)
     
     #### DILATE / ERODE image ###
     max_kernel , min_kernel = 2, 1
     kernel_size_X = int(random.random()*(max_kernel-min_kernel) ) + min_kernel
     kernel_size_Y = int(random.random()*(max_kernel-min_kernel) ) + min_kernel
     kernel = (kernel_size_X, kernel_size_Y)

     dilate_erode_probability = random.random()
     if dilate_erode_probability<0.5:
         img = dilate(img, kernel) if dilate_erode_probability < 0.25 else erode(img, kernel)
     #print(img.shape)
     ### GAUSSIAN NOISE ###
     max_mean , min_mean = 0.05, 0
     max_sdev, min_sdev = 0.1, 0.01
     mean = random.random()* (max_mean - min_mean) + min_mean
     sdev = random.random() * (max_sdev - min_sdev) + min_sdev
     noise_prob = random.random()
     if noise_prob>0.5:
         img = GaussianNoise(img, mean=mean, stdev=sdev)
     
     ret, imgf = cv2.threshold(img.astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
     return (imgf)
