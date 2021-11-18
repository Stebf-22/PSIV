import os
import IOFunctions as IO
import numpy as np
from tqdm import tqdm
import pandas as pd


def read_METADATA_csv(path):
    if path.split('.')[-1] == 'csv':
        return pd.read_csv(path)
    raise ValueError

def apply_transformation():
    pass

def load_diagnosis_data(path: str = '../data/Diagnosis/', limit: int = -1, start: int = 0):
    """
    Diagnosis loader tool

    :param path: Path to data parent folder
    :param limit: Subset of complete dataset to load

    :return: patients dict, Rl_Nd
    """

    patients = [ x for x in  os.listdir(path) if not ('csv' in x) ]
    csv_path = [ x for x in  os.listdir(path) if 'csv' in x]
    data = {}

    for patient in tqdm(patients[start : limit]):
        if os.path.isdir(path + patient):
            data[patient] = IO.load_nifti_img(
                path + patient + [f'/{route}' for route in os.listdir(path + patient) if 'nii' in route][0])
    return data


def load_nodule_data(path: str = '../data/NoduleSegmentation/'):
    """
    NoduleSegmentation tool

    :param path: Path to data

    :return: data
    """

    patients = os.listdir(path)
    data = {}
    for patient in tqdm(patients):
        data[patient] = {}
        if os.path.isdir(path + patient):
            files = os.listdir(path + patient + '/INSP_SIN/')
            data[patient]['ROI'] = IO.load_nifti_img(
                path + patient + [f'/INSP_SIN/{n}' for n in files if 'ROI' in n][0])

            for i, nodule in enumerate([path + patient + '/INSP_SIN/' + n for n in files if 'Nodule' in n]):
                data[patient][f'Nodule{i}'] = IO.load_nifti_img(nodule)
        else:
            pass

    return data


if __name__ == '__main__':

    print(load_diagnosis_data(limit=2, start=0))