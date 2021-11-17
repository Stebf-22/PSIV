import numpy as np
import os
import IOFunctions as io
#import scipy
#from scipy.ndimage import affine_transform
#from numpy.linalg import inv
import pandas as pd

def load_diagnosis_data():
    path = '../data/Diagnosis/'
    patients = os.listdir(path)
    data = {}
    for patient in patients:
        if os.path.isdir(path+patient):            
            data[patient] = io.load_nifti_img(path+patient+[f'/{n}' for n in os.listdir(path+patient) if 'nii' in n][0])

        else:
            Rl_Nd = pd.read_csv(path+patient)

    return data, Rl_Nd

def load_Nodule_data():
    path = '../data/NoduleSegmentation/'
    patients = os.listdir(path)
    data = {}
    for patient in patients:
        data[patient] = {}
        if os.path.isdir(path+patient):
            files = os.listdir(path+patient+'/INSP_SIN/')
            data[patient]['ROI'] = io.load_nifti_img(path+patient+[f'/INSP_SIN/{n}' for n in files if 'ROI' in n][0])

            for i, nodule in enumerate([path+patient+'/INSP_SIN/'+n for n in files if 'Nodule' in n]):
                data[patient][f'Nodule{i}'] = io.load_nifti_img(nodule)
        else:
            pass


    return data