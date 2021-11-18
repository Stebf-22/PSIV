import os
import IOFunctions as IO
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import torch
import copy
import cv2 as cv
from VolumeCutBrowser import *
#from nibabel.affines import apply_affine


def crop_image(img,affine , center, radius,  eps= 30, show=True, log = True, alreadry_roi = False, idx_roi = 0):

    #img, trf = IO.load_nifti_img(img)
    if not alreadry_roi:
        mat = np.linalg.inv(affine)
        point = mat.dot(center)
        point2 = radius
        if log: print(point , point2)
        pointA = point + point2
        pointB = point - point2
        if log: print(pointA, pointB, point2)
        
        #eps=20
        x1, x2 = int(min([pointB[0], pointA[0]])), int(max([pointB[0], pointA[0]]))
        y1, y2 =int(min([pointB[1], pointA[1]])), int(max([pointB[1], pointA[1]]))
        z1,z2 =int(min([pointB[2], pointA[2]])), int(max([pointB[2], pointA[2]]))
        ROI =img[x1-eps:x2+eps, y1-eps: y2-eps, z1:z2]
        if show:
            mat = img[:, :, int(point[2])]
            mat2 = copy.deepcopy(img[:, :, int(point[2])])
            mat[x1-eps:x2+eps , y1-eps:y2 + eps] = 0
            plt.figure()
            f, axarr = plt.subplots(1,2) 
            axarr[0].imshow(mat)
            axarr[1].imshow(mat2)
            plt.show()
            VolumeCutBrowser(ROI)
        return ROI
    else:
        if show:
            plt.imshow(img[:,:,idx_roi])
            plt.show()
        return img


def read_img_and_points(path, prefix = ''):
    img = [x for x in os.listdir(path) if 'nii' in x and prefix in x][0]
    img, trf = IO.load_nifti_img(path + img)
    if 'Seg' not in prefix:
        #print(os.listdir(path))
        acsv = [x for x in os.listdir(path) if 'acsv' in x][0]
        affine = trf['affine']
        with open(path + acsv, 'r') as hdlr:
            points = [x for x in hdlr.readlines() if 'point|' in x]
            separate = lambda x: np.asarray([int(float(y)) for y in x.split('|')[1:5]] )
            center = separate(points[0])
            radius = separate(points[1])
        return img, affine, center, radius
    return img,1,1,1,




if __name__ == '__main__':
    ### Per llegir FULL image no fer servir prefix no fer servir already_roy ni idx_roy
    ### Per llegir segmentacions fer servir el que no abans
   img, affine, center, radius =  read_img_and_points(
  "/home/user/PSIV/02-Cancer_diagnosis/data/NoduleSegmentation/LENS_P3_16_07_2015/INSP_SIN/", prefix='Seg')
   crop_image(img, affine, center, radius, eps=0, alreadry_roi=True, idx_roi = 455)

