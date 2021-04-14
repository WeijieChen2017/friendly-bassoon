import os
import glob
import numpy as np
from nibabel import load, save, Nifti1Image

minc_list = glob.glob("./*.mnc")
minc_list.sort()

for minc_path in minc_list:
    print(minc_path)  
    minc_file = load(minc_path)
    minc_name = os.path.basename(minc_path)[7:9]

    affine = np.array([[0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [1, 0, 0, 0],
                       [0, 0, 0, 1]])

    minc_data = minc_file.get_fdata()
    # set_to_zero = [6, 7, 8, 9, 10, 11]
    # for value in set_to_zero:
    #     minc_data[minc_data == value] = 0

    # minc_data[minc_data == 1] = 0.05
    # minc_data[minc_data == 2] = 1
    # minc_data[minc_data == 3] = 0.25
    # minc_data[minc_data == 4] = 0.1
    # minc_data[minc_data == 5] = 0.1

    out_file = Nifti1Image(minc_data, affine=affine)
    save_name = "MINC_0"+minc_name+'_ORI.nii.gz'
    save(out_file, save_name)
    print(save_name)