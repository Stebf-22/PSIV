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
from cout_slices import count_slices
import matplotlib.pyplot as plt
from ImageTransformations import resize

class SegmentationDataset(Dataset):


    def __init__(self, path_segmentation, path_diagnosi):

        self.path_segmentation = path_segmentation
        self.path_diagnosi = path_diagnosi
        self.sort = lambda y: sorted(y, key = lambda x: int(x.split('_')[1]) if not('P' in x.split('_')[1]) else int(x.split('_')[1][1:]))
        self.segmentation_dirs = [x for x in os.listdir(self.path_segmentation) if not 'txt' in x]
        #self.data= self._read_images()
        self.actual=None
        self.actual_ofset = 0
        self.image_counter = 0
    def _read_images(self, idx):
        res = []

        patient = self.segmentation_dirs[idx]
        base = {'data': None, 'GT': None, 'GT2': None}
        #print(os.listdir(self.path_segmentation + patient))
        affine=None
        for elements in sorted(os.listdir(self.path_segmentation + patient + '/INSP_SIN/')):
            if 'ROI' in elements:
                data, affine, center, radius = read_img_and_points((self.path_segmentation + patient + '/INSP_SIN/'), prefix = 'CTD')
                ROI = crop_image(data, affine, center, radius, show=False, eps = 20, log=False)
                #print(elements)
                #for x in range(res.shape[2]):
                #    res[:,:,x] = resize(ROI[:,:,x], (430, 329) )  
                base['data'] =  ROI
            if 'Seg_Nodule1' in elements:
                data = read_img_and_points((self.path_segmentation + patient + '/INSP_SIN/'), prefix = 'Seg_Nodule1')
                ROI = crop_image(data[0], affine, center, radius, show=False, eps = 20, log=False)
                base['GT'] =  ROI
            if 'Seg_Nodule2' in elements:
                data = read_img_and_points((self.path_segmentation + patient + '/INSP_SIN/'), prefix = 'Seg_Nodule2')
                ROI = crop_image(data[0], affine, center, radius, show=False, eps = 20, log=False)
                base['GT2'] =  ROI

                 
    
        self.actual= base
        return base
    
    def __getitem__(self, idx):
        pure_idx = idx
        if self.actual == None:
            img = self._read_images(self.image_counter)
            print(f"idx is {idx}")
            res = {'img': resize(img['data'][:,:,idx],(240,240)) , 'GT': resize(img['GT'][:,:,idx],(64,64)) }
        else:
            print(idx)
            idx -= self.actual_ofset
            print(idx, self.actual['data'].shape[2])
            if idx >= self.actual['data'].shape[2]:
                self.actual_ofset += idx
                
                print(f"idx is :", idx, "actual ofset: ", self.actual_ofset)
                pure_idx -= self.actual_ofset
                idx = pure_idx
                self._read_images(self.image_counter)
                self.image_counter += 1
                print(f"here")
            print(f"Here")
            img = self.actual
            res = {'img': resize(img['data'][:,:,idx],(240,240)) , 'GT': resize(img['GT'][:,:,idx],(64,64)) }
        return res
    
    def __len__(self ):
        #return count_slices("/home/user/PSIV/02-Cancer_diagnosis/data/NoduleSegmentation", "CTD",segmentation=True)
        return 1098
    
    
        #return res



if __name__ == '__main__':
    path_segmentation = "02-Cancer_diagnosis/data/NoduleSegmentation/"
    path_diagnosi = "/home/user/PSIV/02-Cancer_diagnosis/data/Diagnosis"
    dataset = SegmentationDataset(path_segmentation, path_diagnosi)
    dataloader = DataLoader(dataset, batch_size = 30)
    d = {}
    for x in range(30,31):
        data = dataset[x]
        f, axarr = plt.subplots(1,2) 
        axarr[0].imshow(data['img'])
        axarr[1].imshow(data['GT'])
        plt.show()
        break
        
