from ImageTransformations import otsu_transformation
from UnetModel import UNet
from ImageReader import read_img_and_points
from ImageReader import crop_image
import torch
import os
import numpy as np

class Segment:

    def __init__(self, model_type, diagnosi_path, trained_model_path):

        self.model_type = model_type
        self.diagnosi_path = diagnosi_path
        self.model = otsu_transformation if model_type == 'otsu' else UNet().load_state_dict(torch.loaf(trained_model_path))
    
    def _read_image(self, path):
        data, affine, center, radius = read_img_and_points(path)
        ROI = crop_image(data, affine, center, radius, show=False, padding = 20, log=False) 
        return ROI
    
    def segment2D(self, img, transformations=None):
        return self.model(img.astype(np.uint8)) if transformations == None else self.model(transformations(img))
    def _segmentROI(self, ROI, transformations=None):
        segmented = np.zeros_like(ROI)
        for index in range(ROI.shape[-1]):
            segmented[:,:,index] = self.segment2D(ROI[:,:, index], transformations=transformations)
    def segment_all(self, transformations=None, save=True):
        segmented = []
        if 'SegmentedDiagnosi' not in os.listdir('02-Cancer_diagnosis/data'): os.mkdir("02-Cancer_diagnosis/data/SegmentedDiagnosi/")
        patients = [ self.diagnosi_path +  x for x in  os.listdir(self.diagnosi_path) if 'csv' not in x]
        for patient in patients:
            pat_name = patient.split('/')[-1]
            if  pat_name not in os.listdir('02-Cancer_diagnosis/data/SegmentedDiagnosi'): os.mkdir("02-Cancer_diagnosis/data/SegmentedDiagnosi/" + pat_name)
            ROI = self._read_image(patient+'/')
            segROI = self._segmentROI(ROI, transformations)
            if save: np.save(f"02-Cancer_diagnosis/data/SegmentedDiagnosi/{pat_name}/segmented.npy", segROI)
            else: segmented.append(segROI)
        return segmented

if __name__ == '__main__':
    Segment('otsu', "02-Cancer_diagnosis/data/Diagnosis/", None).segment_all()

