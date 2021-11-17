import os
import IOFunctions as IO
import numpy as np
from tqdm import tqdm
import pandas as pd
from typing import Iterable


def load_diagnosis_data(path: str = '../data/Diagnosis/', limit: int = -1) -> Iterable[np.ndarray, pd.DataFrame]:
    """
    Diagnosis loader tool

    :param path: Path to data parent folder
    :param limit: Subset of complete dataset to load

    :return: patients dict, Rl_Nd
    """

    patients = os.listdir(path)
    data = {}

    for patient in tqdm(patients[:limit]):
        if os.path.isdir(path + patient):
            data[patient] = IO.load_nifti_img(
                path + patient + [f'/{route}' for route in os.listdir(path + patient) if 'nii' in route][0])

        # Shine case: radiolung_nodule_diagnosis
        else:
            Rl_Nd = pd.read_csv(path + patient)

    return data, Rl_Nd


def load_nodule_data(path: str = '../data/NoduleSegmentation/') -> dict:
    """
    NoduleSegmentation tool

    :param path: Path to data

    :return: data
    """

    patients = os.listdir(path)
    data = {}
    for patient in patients:
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
