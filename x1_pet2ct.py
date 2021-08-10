import os
import glob
import nibabel as nib
import numpy as np
import nibabel.processing

pet_list = glob.glob("./data/score_1/*_NAC.nii.gz")
pet_list.sort()

def maxmin_norm(data):
    MAX = np.amax(data)
    MIN = np.amin(data)
    data = (data - MIN)/(MAX-MIN)
    return data

for pet_path in pet_list:
    print("&"*60)
    print(pet_path)
    # pet_dir = os.path.dirname(pet_path)+"/"
    ct_path = pet_path[:-11]+"_CTAC.nii.gz"

    pet_file = nib.load(pet_path)
    ct_file = nib.load(ct_path)

    print(pet_file.get_fdata().shape, np.amax(pet_file.get_fdata()), np.amin(pet_file.get_fdata()))
    print(ct_file.get_fdata().shape, np.amax(ct_file.get_fdata()), np.amin(ct_file.get_fdata()))
    
    pet_big = nib.processing.conform(pet_file, out_shape=(512, 512, 64), voxel_size=(1.367, 1.367, 3.27))
    name_big = "NACB"

    pet_small = nib.processing.conform(pet_file, out_shape=(128, 128, 64), voxel_size=(5.468, 5.468, 3.27))
    name_small = "NACS"
    # name_0 = "PF3_ORI"

    # pet_1x = nib.processing.conform(pet_blur, out_shape=(300, 300, 103), voxel_size=(1, 1, 2.4))
    # name_1 = "PF3_x1000y1000z2400"

    # pet_4x = nib.processing.conform(pet_blur, out_shape=(1200, 1200, 103), voxel_size=(0.25, 0.25, 2.4))
    # name_2 = "PF3_x250y250z2400"

    for package in [[pet_big, name_big], [pet_small, name_small]]:
        nii_file = package[0]
        tag = package[1]

        nii_new_data = nii_file.get_fdata()
        nii_new_data[nii_new_data<0] = 0
        nii_new_file = nib.Nifti1Image(nii_new_data, nii_file.affine, nii_file.header)
        save_name = pet_path[:-11] + "_" + tag + ".nii.gz"
        nib.save(nii_new_file, save_name)
        print(save_name)