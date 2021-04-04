import os
import glob
import nibabel as nib
import numpy as np
import nibabel.processing

mri_list = glob.glob("./data/mri_4688/*.nii.gz")
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
    mri_file_4x = nib.processing.conform(mri_file, out_shape=(960, 960, 284), voxel_size=(0.25, 0.25, 0.6))
    save_name = mri_dir + mri_name + "_x960y960z284.nii.gz"

    mri_file_xy4x = nib.processing.conform(mri_file, out_shape=(960, 960, 71), voxel_size=(0.25, 0.25, 2.4))
    save_name = mri_dir + mri_name + "_x960y960z71.nii.gz"
    
    mri_smooth = nib.processing.smooth_image(mri_file_4x, fwhm=3)
    save_name = mri_dir + mri_name + "_x960y960z284f3.nii.gz"
    
    mri_smooth_1x = nib.processing.conform(mri_smooth, out_shape=(240, 240, 71), voxel_size=(1, 1, 2.4))
    save_name = mri_dir + mri_name + "_x240y240z71f3.nii.gz"

    for package in [[mri_file_4x, "x960y960z284"], [mri_smooth, "_x960y960z284f3"],
                    [mri_smooth_1x, "_x240y240z71f3"], [mri_file_xy4x, "_x960y960z71"]]:
        nii_file = package[0]
        tag = package[1]

        nii_new_file = nib.Nifti1Image(maxmin_norm(nii_file.get_fdata()), nii_file.affine, nii_file.header)
        save_name = mri_dir + mri_name + "_" + tag + ".nii.gz"
        nib.save(nii_new_file, save_name)
        print(save_name)