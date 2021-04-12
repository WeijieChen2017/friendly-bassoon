from scipy.ndimage import zoom
from scipy.io import loadmat
import numpy as np
import nibabel as nib
import glob
import os

nii_list = glob.glob("../data/mat/*.nii.gz")
nii_list.sort()
for nii_path in nii_list:
    print("-----------------------------------------------")
    nii_file = nib.load(nii_path)
    tmpl_affine = nii_file.affine
    tmpl_header = nii_file.header
    nii_name = os.path.basename(nii_path)[:-7]
    osp_path = os.path.dirname(nii_path)+"/"+nii_name+"_recon_OSP_F4.mat"
    ori_path = os.path.dirname(nii_path)+"/"+nii_name+".mat"

    for package in [[osp_path, "osp"], [ori_path, "ori"]]:

        mat_path = package[0]
        save_tag = package[1]

        mdict = loadmat(mat_path)
        try:
            mat_data = mdict["reconImg"]
        except Exception:
            pass  # or you could use 'continue'
        try:
            mat_data = mdict["data"]
        except Exception:
            pass  # or you could use 'continue'

        save_data = process_data(mat_data)
        save_file = nib.Nifti1Image(save_data, affine=tmpl_affine, header=tmpl_header)
        save_name = os.path.dirname(nii_path)+nii_name+"_"+save_tag+".nii.gz"
        nib.save(save_file, save_name)
        print(save_name)
