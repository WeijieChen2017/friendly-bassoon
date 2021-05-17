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
    case_index = mri_name[-3:]
    print(case_index)
    exit()
    mri_dir = os.path.dirname(mri_path)+"/"
    mri_file = nib.load(mri_name+".nii.gz")
    recon_file = nib.load(mri_path)
    
    file_1 = nib.processing.conform(mri_file, out_shape=(1024, 1024, 89), voxel_size=(0.2344, 0.2344, 1.825))
    file_1_data = file_1.get_fdata()
    file_1_data[file_1_data < 0] = 0
    name_1 = "GTH"
    
    file_2l = nib.processing.conform(recon_file, out_shape=(1024, 1024, 89), voxel_size=(0.2344, 0.2344, 1.825))
    file_2l_data = file_2l.get_fdata()
    file_2l_data[file_2l_data < 0] = 0
    name_2l = "BIC"
    
    file_2s = recon_file
    file_2s_data = file_2s.get_fdata()
    file_2s_data[file_2s_data < 0] = 0
    name_2s = "INP"

    scale_factor = np.max((np.amax(file_1_data), np.amax(file_2l_data), np.amax(file_2s_data)))
    print("Norm factor:", scale_factor)
    file_1 = nib.Nifti1Image(file_1_data/scale_factor, file_1.affine, file_1.header)
    file_2l = nib.Nifti1Image(file_2l_data/scale_factor, file_2l.affine, file_2l.header)
    file_2s = nib.Nifti1Image(file_2s_data/scale_factor, file_2s.affine, file_2s.header)

    for package in [[file_1, name_1],
                    [file_2l, name_2l], [file_2s, name_2s]]:
        nii_file = package[0]
        tag = package[1]

        # nii_new_file = nib.Nifti1Image(nii_file.get_fdata()/scale_factor, nii_file.affine, nii_file.header)
        save_name = mri_dir + "SEE3" + "_" + tag + ".nii.gz"
        nib.save(nii_file, save_name)
        print(save_name)