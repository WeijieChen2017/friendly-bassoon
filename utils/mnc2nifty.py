import os
import glob
import numpy as np
from nibabel import load, save, Nifti1Image

minc_list = glob.glob("./*.mnc")
minc_list.sort()

for minc_path in minc_list:
    print(minc_path)  
    minc_file = load(minc_path)
    minc_name = os.path.basename(minc_path)[:-4]
    basename = minc_file.get_filename().split(os.extsep, 1)[0]

    affine = np.array([[0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [1, 0, 0, 0],
                       [0, 0, 0, 1]])

    out = Nifti1Image(minc_file.get_fdata(), affine=affine)
    save(out, minc_name+'.nii.gz')
    print(minc_name+'.nii.gz')