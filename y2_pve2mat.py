from scipy.ndimage import binary_erosion, binary_dilation
from scipy.io import savemat
from nibabel import processing
import nibabel as nib
import numpy as np
import glob
import os

file_list = glob.glob("original*.nii.gz")
file_list.sort()
for file_path in file_list:
    print("*"*50)
    print(file_path)
    file_name = os.path.basename(file_path)[:-7]
    print(file_name)
    mri_file = nib.load(file_path)
    mri_data = mri_file.get_fdata()
    print(mri_data.shape)

    mdic = {"data": mri_data}

    save_name = file_name+".mat"
    savemat(save_name, mdic)
    print("Mat:", save_name)
    print("--------------------------------------")