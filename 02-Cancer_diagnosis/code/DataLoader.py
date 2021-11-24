import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import csv
from ImageReader import read_img_and_points, crop_image
from ImageTransformations import resize


class SegmentationDataset(Dataset):

    def __init__(self, path_segmentation, path_diagnosi, transforms=None):

        self.path_segmentation = path_segmentation
        self.path_diagnosi = path_diagnosi
        self.sort = lambda y: sorted(y, key=lambda x: int(x.split('_')[1]) if not ('P' in x.split('_')[1]) else int(
            x.split('_')[1][1:]))
        self.segmentation_dirs = [x for x in os.listdir(self.path_segmentation) if not 'txt' in x]
        self.actual = None
        self.actual_offset = 0
        self.image_counter = 0
        self.transforms = transforms if transforms is not None else lambda x: x

    def _read_images(self, idx):
        res = []

        patient = self.segmentation_dirs[idx]
        base = {'data': None, 'GT': None, 'GT2': None}
        affine = None
        for elements in sorted(os.listdir(self.path_segmentation + patient + '/INSP_SIN/')):
            if 'ROI' in elements:
                data, affine, center, radius = read_img_and_points((self.path_segmentation + patient + '/INSP_SIN/'),
                                                                   prefix='CTD')
                ROI = crop_image(data, affine, center, radius, show=False, padding=20, log=False)
                base['data'] = ROI

            if 'Seg_Nodule1' in elements:
                data = read_img_and_points((self.path_segmentation + patient + '/INSP_SIN/'), prefix='Seg_Nodule1')
                ROI = crop_image(data[0], affine, center, radius, show=False, padding=20, log=False)
                base['GT'] = ROI

            if 'Seg_Nodule2' in elements:
                data = read_img_and_points((self.path_segmentation + patient + '/INSP_SIN/'), prefix='Seg_Nodule2')
                ROI = crop_image(data[0], affine, center, radius, show=False, padding=20, log=False)
                base['GT2'] = ROI

        self.actual = base

        return base

    def __getitem__(self, idx):
        pure_idx = idx

        if self.actual is None:
            img = self._read_images(self.image_counter)
            print(f"idx is {idx}")
            res = {'img': resize(img['data'][:, :, idx], (240, 240)), 'GT': resize(img['GT'][:, :, idx], (64, 64))}

        else:
            print(idx)
            idx -= self.actual_ofset
            print(idx, self.actual['data'].shape[2])
            if idx >= self.actual['data'].shape[2]:
                self.actual_ofset += idx

                print(f"idx is :", idx, "actual offset: ", self.actual_ofset)
                pure_idx -= self.actual_ofset
                idx = pure_idx
                self._read_images(self.image_counter)
                self.image_counter += 1

            img = self.actual
            res = {'img': resize(img['data'][:, :, idx], (240, 240)), 'GT': resize(img['GT'][:, :, idx], (64, 64))}

        return {k: self.transforms(v) for k, v in res.items()}

    def __len__(self):
        # return count_slices("/home/user/PSIV/02-Cancer_diagnosis/data/NoduleSegmentation", "CTD",segmentation=True)
        return 1098


class DiagnosisEnd2End(Dataset):

    def __init__(self, diagnosis_path: str, nodules_path: str, is_segmented: bool = False):
        super().__init__()

        self.diagnosis_path: list = [diagnosis_path + patient + "/" for patient in os.listdir(diagnosis_path)
                                     if os.path.isdir(diagnosis_path + patient)
                                     ]

        self.nodules: dict = {
            # patient_id: type, lobe, histological_diagnosis
            int(patient[0]): 1 if patient[4] == "Malignant" else 0 for patient in csv.reader(open(nodules_path))
            if patient[0] != "patient_id"
        }

        self.anonymous_nodules: dict = {x: y for x, y in enumerate(self.nodules.keys())}

        self.is_segmented: bool = is_segmented

    def __getitem__(self, idx: int):
        patient = self.anonymous_nodules[idx]
        patient_data = dict()

        if self.is_segmented:
            patient_data['ROI'] = np.load(self.diagnosis_path[patient], allow_pickle=True)

        else:
            scan, affine, center, radius = read_img_and_points(self.diagnosis_path[patient])

            roi = crop_image(scan, affine, center, radius, show=False, padding=5, log=False)

            patient_data['ROI'] = np.array([resize(roi[:, :, x], (240, 240)) for x in range(roi.shape[2])])

        patient_data['GT'] = self.nodules[patient]

        return patient_data

    def __len__(self):
            return len(self.diagnosis_path)


if __name__ == "__main__":

    diagnosis_path = "02-Cancer_diagnosis/data/Diagnosis/"
    nodules_path = "02-Cancer_diagnosis/data/Diagnosis/Radiolung_NoduleDiagnosis.csv"

    data_laoder = DiagnosisEnd2End(diagnosis_path, nodules_path)
