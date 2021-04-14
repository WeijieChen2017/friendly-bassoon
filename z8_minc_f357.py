import os
import glob
import nibabel as nib
import numpy as np
import nibabel.processing

mri_list = glob.glob("./data/minc_v1/*.nii.gz") #
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
    scale_factor = np.amax(mri_file.get_fdata()) #362, 434, 362
    name_0 = "ORI"
    
    file_1 = nib.processing.conform(mri_file, out_shape=(720, 840, 180), voxel_size=(0.5, 0.5, 2))
    name_1 = "GT"
    
    file_2l = nib.processing.smooth_image(file_1, fwhm=3)
    name_2l = "Large3"
    
    file_2s = nib.processing.conform(file_2l, out_shape=(180, 210, 360), voxel_size=(2, 2, 1))
    name_2s = "Small3"

    file_3l = nib.processing.smooth_image(file_1, fwhm=5)
    name_3l = "Large5"

    file_3s = nib.processing.conform(file_3l, out_shape=(180, 210, 360), voxel_size=(2, 2, 1))
    name_3s = "Small5"

    file_4l = nib.processing.smooth_image(file_1, fwhm=7)
    name_4l = "Large7"

    file_4s = nib.processing.conform(file_4l, out_shape=(180, 210, 360), voxel_size=(2, 2, 1))
    name_4s = "Small7"

    # for package in [[file_1, name_1]]:
    for package in [[mri_file, name_0], [file_1, name_1],
                    [file_2l, name_2l], [file_2s, name_2s],
                    [file_3l, name_3l], [file_3s, name_3s],
                    [file_4l, name_4l], [file_4s, name_4s]]:
        nii_file = package[0]
        tag = package[1]

        nii_new_file = nib.Nifti1Image(nii_file.get_fdata()/scale_factor, nii_file.affine, nii_file.header)
        save_name = mri_dir + mri_name + "_" + tag + ".nii.gz"
        nib.save(nii_new_file, save_name)
        print(save_name)