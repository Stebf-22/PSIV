import os
import IOFunctions as io
from tqdm import tqdm
import pandas as pd


def read_METADATA_csv(path):
    if path.split('.')[-1] == 'csv':
        return pd.read_csv(path)
    raise ValueError


def apply_transformation():
    pass


def load_diagnosis_data(path: str = '../data/Diagnosis/', limit: int = -1, start: int = 0) -> dict:
    """
    Diagnosis loader tool

    :param path: Path to data parent folder
    :param limit: Subset of complete dataset to load
    :param start: Initial point of subset to load


    :return: patients dict
    """

    patients = [x for x in os.listdir(path) if not ('csv' in x)]
    data = dict()

    for patient in tqdm(patients[start:limit]):
        if os.path.isdir(path + patient):
            data[patient] = io.load_nifti_img(
                path + patient + [f'/{route}' for route in os.listdir(path + patient) if 'nii' in route][0])
    return data


def load_nodule_data(path: str = '../data/NoduleSegmentation/') -> dict:
    """
    Nodule Segmentation tool

    :param path: Path to data

    :return: data
    """

    patients = os.listdir(path)
    data = dict()

    for patient in tqdm(patients):
        data[patient] = dict()

        if os.path.isdir(path + patient):
            files = os.listdir(path + patient + '/INSP_SIN/')
            data[patient]['ROI'] = io.load_nifti_img(
                path + patient + [f'/INSP_SIN/{n}' for n in files if 'ROI' in n][0])

            for i, nodule in enumerate([path + patient + '/INSP_SIN/' + n for n in files if 'Nodule' in n]):
                data[patient][f'Nodule{i}'] = io.load_nifti_img(nodule)
        else:
            continue

    return data
