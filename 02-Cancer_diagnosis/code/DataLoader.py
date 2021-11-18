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


class SegmentationDataloader:


    def __init__(self, path_segmentation, path_diagnosi):

        self.path_segmentation = path_segmentation
        self.path_diagnosi = path_diagnosi
        self.sort = lambda y: sorted(y, key = lambda x: int(x.split('_')[1]) if not('P' in x.split('_')[1]) else int(x.split('_')[1][1:]))
        self.segmentation_dirs = [x for x in os.listdir(self.path_segmentation) if not 'txt' in x]
        self.data= self._read_images()
        print(self.data)
    def _read_images(self, idx):
        res = []

        patient = self.segmentation_dirs[idx]
        base = {'data': None, 'GT': None, 'GT2': None}
        #print(os.listdir(self.path_segmentation + patient))
        for elements in os.listdir(self.path_segmentation + patient + '/INSP_SIN/'):
            if 'ROI' in elements:
                data = read_img_and_points((self.path_segmentation + patient + '/INSP_SIN/'), prefix = 'CTD')
                ROI = crop_image(*data, show=False, eps = 0)
                base['data'] =  ROI
            if 'Seg_Nodule1' in elements:
                data = read_img_and_points((self.path_segmentation + patient + '/INSP_SIN/'), prefix = 'Seg_Nodule1')
                base['GT'] =  data[0]
            if 'Seg_Nodule2' in elements:
                base['GT2'] =  read_img_and_points(self.path_segmentation + patient + '/INSP_SIN/' + elements, prefix = "Sef_Nodule2")         
        return base
    
    def __getitem__(self, idx):
        return self._read_images(idx)
    
    def __len__(self ):
        return len(self.segmentation_dirs)
    
    
        #return res



if __name__ == '__main__':
    path_segmentation = "02-Cancer_diagnosis/data/NoduleSegmentation/"
    path_diagnosi = "/home/user/PSIV/02-Cancer_diagnosis/data/Diagnosis"
    SegmentationDataloader(path_segmentation, path_diagnosi)
