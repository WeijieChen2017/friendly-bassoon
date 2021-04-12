import os
import glob
import nibabel as nib
import numpy as np
import nibabel.processing

mri_list = glob.glob("./data/duetto/*_ori.nii.gz")
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
    
    ori_file = nib.load(mri_path)
    scale_factor = np.amax(ori_file.get_fdata())
    osp_path = mri_dir+mri_name[:-4]+"_osp.nii.gz"
    osp_file = nib.load(osp_path)
    
    ori_file = nib.processing.conform(ori_file, out_shape=(960, 960, 71), voxel_size=(0.25, 0.25, 2.4))
    ori_name = "GT"
    
    osp_small_file = nib.processing.conform(osp_file, out_shape=(240, 240, 71), voxel_size=(1, 1, 2.4))
    osp_small_name = "osp_small"

    osp_large_file = nib.processing.conform(osp_file, out_shape=(960, 960, 71), voxel_size=(0.25, 0.25, 2.4))
    osp_large_name = "osp_large"

    for package in [[ori_file, ori_name], [osp_small_file, osp_small_name], [osp_large_file, osp_large_name]]:
        nii_file = package[0]
        tag = package[1]

        nii_new_file = nib.Nifti1Image(nii_file.get_fdata()/scale_factor, nii_file.affine, nii_file.header)
        save_name = mri_dir + mri_name + "_" + tag + ".nii.gz"
        nib.save(nii_new_file, save_name)
        print(save_name)