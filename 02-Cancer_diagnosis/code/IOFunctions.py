import nibabel as nib
import numpy as np
import os

from typing import Tuple


def load_nifti_img(filepath, dtypes=np.int16) -> Tuple[np.ndarray, dict]:
    """
    **NIFTI Image Loader**

    :param filepath: path to the input NIFTI image
    :param dtypes: data type of the nifti numpy array
    :return: return numpy array
    """

    nim = nib.load(filepath)
    out_nii_array = np.array(nim.get_fdata(), dtype=dtypes)
    meta = {'affine': nim.affine,
            'dim': nim.header['dim'],
            'pixdim': nim.header['pixdim']
            }
    
    return out_nii_array, meta


def write_nifti_img(input_nii_array, meta, save_dir, filename) -> None:
    """
    **NIFTI Image writer**

    :param input_nii_array: np.ndarray containing .nii volume
    :param meta: .nii metadata (optional). If ommitted default (identity) values are used
    :param savedir: is the full path to the output file, e.g. "/home/user/Desktop/BD"
    :param filename: is the output file, e.g. "/LIDC-IDRI-0001_GT1.nii.gz"
    :return: None
    """

    # Create directory if not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    affine = meta['affine']
    pixdim = meta['pixdim']
    dim = meta['dim']

    img = nib.Nifti1Image(input_nii_array, affine=affine)
    img.header['dim'] = dim
    img.header['pixdim'] = pixdim

    save_name = os.path.join(save_dir, filename)
    print('saving: ', save_name)
    nib.save(img, save_name)
