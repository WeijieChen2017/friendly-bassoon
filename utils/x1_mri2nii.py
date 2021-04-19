import os
import glob
import numpy as np
from scipy.ndimage import zoom
from nibabel import load, save, Nifti1Image

minc_list = glob.glob("./*.mnc")
minc_list.sort()

for minc_path in minc_list:
    print(minc_path)  
    minc_file = load(minc_path)
    minc_name = os.path.basename(minc_path)[7:9]

    affine = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

    minc_data = minc_file.get_fdata()
    print(minc_data.shape)
    dz, dx, dy = minc_data.shape
    minc_rot = np.zeros((dx, dy, dz))

    for idx in range(dz):
        minc_rot[:, :, idx] = np.rot90(minc_data[idx, :, :])

    qx, qy, qz = (256, 256, 180)
    zoom_data = zoom(minc_rot, (qx/dx, qy/dy, qz/dz))

    out_file = Nifti1Image(zoom_data, affine=affine)
    save_name = "MINC_0"+minc_name+'_MRI.nii.gz'
    save(out_file, save_name)
    print(save_name)