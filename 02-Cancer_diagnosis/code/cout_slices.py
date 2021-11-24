import torch
from torch.utils.data import Dataset, DataLoader
import os 
import load_patients as lp
import IOFunctions as io
import pandas as pd
import numpy as np
import nib
import os
from ImageReader import read_img_and_points
from ImageReader import crop_image
from tqdm import tqdm



def count_slices(path, prefix, segmentation = False, log = True):
    path = path if path[-1] == '/' else path +'/'
    patients = [path + x for x in os.listdir(path) if 'txt' not in x]
    data = None
    prefix2 = '' if not segmentation else '/INSP_SIN/'
    counter = 0
    for patient in tqdm(patients):
        data = read_img_and_points(patient + prefix2 , prefix=prefix)
        ROI = crop_image(*data, show=False, log=False, padding=0)
        counter +=   ROI.shape[2]
    if log: print(f"The toal number of slices to be segmented is {counter}")
    return counter


if __name__ == '__main__':
    count_slices("/home/user/PSIV/02-Cancer_diagnosis/data/NoduleSegmentation", "CTD",segmentation=True)