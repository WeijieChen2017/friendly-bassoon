import os
import glob
import nibabel as nib
import numpy as np
import nibabel.processing

mri_list = glob.glob("./data/nick_MRI/*.nii.gz")
mri_list.sort()

def maxmin_norm(data):
    MAX = np.amax(data)
    MIN = np.amin(data)
    data = (data - MIN)/(MAX-MIN)
    return data

for mri_path in mri_list:
    print("&"*60)
    print(mri_path)
    mri_name = os.path.basename(mri_path)
    mri_name = mri_name[:mri_name.find(".")]
    mri_dir = os.path.dirname(mri_path)+"/"
    mri_file = nib.load(mri_path)
    scale_factor = np.amax(mri_file.get_fdata())
    
    file_1 = nib.processing.conform(mri_file, out_shape=(960, 960, 68), voxel_size=(0.25, 0.25, 2.4))
    name_1 = "x250y250z2400"
    
    file_2 = nib.processing.smooth_image(file_1, fwhm=3)
    name_2 = "x250y250z2400f3"
    
    file_3 = nib.processing.conform(file_2, out_shape=(240, 240, 68), voxel_size=(1, 1, 2.4))
    name_3 = "x1000y1000z2400f3"

    for package in [[file_1, name_1], [file_2, name_2], [file_3, name_3]]:
        nii_file = package[0]
        tag = package[1]

        nii_new_file = nib.Nifti1Image(nii_file.get_fdata()/scale_factor, nii_file.affine, nii_file.header)
        save_name = mri_dir + mri_name + "_" + tag + ".nii.gz"
        nib.save(nii_new_file, save_name)
        print(save_name)