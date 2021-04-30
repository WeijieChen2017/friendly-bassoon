import os
import glob
import nibabel as nib
import numpy as np
import nibabel.processing

mri_list = glob.glob("./*_recon.nii.gz")
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
    mri_name = mri_name[:mri_name.find(".")][:-6]
    mri_dir = os.path.dirname(mri_path)+"/"
    mri_file = nib.load(mri_name+".nii.gz")
    recon_file = nib.load(mri_path)
    
    file_1 = nib.processing.conform(mri_file, out_shape=(1024, 1024, 89), voxel_size=(0.2344, 0.2344, 1.825))
    name_1 = "GTH"
    
    file_2l = nib.processing.conform(recon_file, out_shape=(1024, 1024, 89), voxel_size=(0.2344, 0.2344, 1.825))
    name_2l = "BIC"
    
    file_2s = recon_file
    name_2s = "INP"

    for package in [[file_1, name_1],
                    [file_2l, name_2l], [file_2s, name_2s]]:
        nii_file = package[0]
        tag = package[1]

        nii_new_file = nib.Nifti1Image(nii_file.get_fdata()/scale_factor, nii_file.affine, nii_file.header)
        save_name = mri_dir + mri_name + "_" + tag + ".nii.gz"
        nib.save(nii_new_file, save_name)
        print(save_name)