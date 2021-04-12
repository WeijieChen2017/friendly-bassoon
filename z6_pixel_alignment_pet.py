import os
import glob
import nibabel as nib
import numpy as np
import nibabel.processing

pet_list = glob.glob("./data/eval_PET/*_PET.nii.gz")
pet_list.sort()

def maxmin_norm(data):
    MAX = np.amax(data)
    MIN = np.amin(data)
    data = (data - MIN)/(MAX-MIN)
    return data

for pet_path in pet_list:
    print("&"*60)
    print(pet_path)
    pet_name = os.path.basename(pet_path)
    pet_name = pet_name[:pet_name.find(".")][:-4]
    pet_dir = os.path.dirname(pet_path)+"/"
    
    pet_file = nib.load(pet_path)
    name_0 = "PF3_ORI"

    pet_blur = nib.processing.smooth_image(pet_file, fwhm=3)

    pet_1x = nib.processing.conform(pet_blur, out_shape=(300, 300, 103), voxel_size=(1, 1, 2.4))
    name_1 = "PF3_x1000y1000z2400"

    pet_4x = nib.processing.conform(pet_blur, out_shape=(1200, 1200, 103), voxel_size=(0.25, 0.25, 2.4))
    name_2 = "PF3_x250y250z2400"

    for package in [[pet_file, name_0], [pet_1x, name_1], [pet_4x, name_2]]:
        nii_file = package[0]
        tag = package[1]

        nii_new_file = nib.Nifti1Image(maxmin_norm(nii_file.get_fdata()), nii_file.affine, nii_file.header)
        save_name = pet_dir + pet_name + "_" + tag + ".nii.gz"
        nib.save(nii_new_file, save_name)
        print(save_name)