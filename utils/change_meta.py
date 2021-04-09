import os
import glob
import nibabel as nib
import numpy as np
import nibabel.processing

nii_list = glob.glob("./eval_nick_NLL/*.nii.gz")
nii_list.sort()

for nii_path in nii_list:
    print("&"*60)
    print(nii_path)
    nii_name = os.path.basename(nii_path)
    nii_name = nii_name[:nii_name.rfind("_")]
    nii_file = nib.load(nii_path)
    
    templ_path = "./eval_nick/"+nii_name+"_x250y250z2400.nii.gz"
    templ_file = nib.load(templ_path)

    nii_new_file = nib.Nifti1Image(nii_file.get_fdata(), templ_file.affine, templ_file.header)
    nib.save(nii_new_file, "./"+os.path.basename(nii_path))
    print(nii_path)