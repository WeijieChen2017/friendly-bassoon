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

    affine = np.array([[0.712, 0, 0, 0],
                       [0, 0.860, 0, 0],
                       [0, 0, 0.847, 0],
                       [0, 0, 0, 1]])

    minc_data = minc_file.get_fdata()
    print(minc_data.shape)

    dx, dz, dy = minc_data.shape
    minc_rot = np.zeros((dx, dy, dz))
    for idx in range(dz):
        minc_rot[:, :, idx] = np.rot90(minc_data[:, idx, :])

    dx, dz, dy = minc_rot.shape
    minc_again = np.zeros((dx, dy, dz))
    for idx in range(dz):
        minc_again[:, :, idx] = minc_rot[:, idx, :]

    qx, qy, qz = (256, 256, 180)
    zoom_data = zoom(minc_again, (qx/dx, qy/dy, qz/dz))

    out_file = Nifti1Image(zoom_data, affine=affine)
    save_name = "MINC_0"+minc_name+'_V2.nii.gz'
    save(out_file, save_name)
    print(save_name)