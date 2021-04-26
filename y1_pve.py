from nibabel import load, save, Nifti1Image
import nibabel.processing
import os
import glob
import nibabel as nib
import numpy as np

file_list = glob.glob("*.nii.gz")
file_list.sort()
for file_path in file_list:
    print(file_path)
    file_name = os.path.basename(file_path)[:-7]
    print(file_name)
    mri_file = nib.load(file_path)
    mri_data = mri_file.get_fdata()
    print(mri_data.shape)
    mri_data[mri_data<0] = 0
    mean = np.mean(mri_data)
    sigma = np.amax(mri_data)*0.02
    gaussian = np.random.normal(mean, sigma, (mri_data.shape[0], mri_data.shape[1], mri_data.shape[2]))
    print(gaussian.shape)
    out_file = Nifti1Image(mri_data+gaussian, affine=mri_file.affine, header=mri_file.header)
    out_file = nibabel.processing.smooth_image(out_file, fwhm=3)
    save_name = "pve"+file_name[3:]+".nii.gz"
    save(out_file, save_name)
    print(save_name)