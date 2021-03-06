from scipy.ndimage import zoom
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from nibabel import processing
import glob
import argparse
import os

def maxmin_norm(data):
    MAX = np.amax(data)
    MIN = np.amin(data)
    data = (data - MIN)/(MAX-MIN)
    return data

def create_index(dataA, n_slice):
    h, w, z = dataA.shape
    index = np.zeros((z,n_slice))
    
    for idx_z in range(z):
        for idx_c in range(n_slice):
            index[idx_z, idx_c] = idx_z-(n_slice-idx_c+1)+n_slice//2+2
    index[index<0]=0
    index[index>z-1]=z-1
    return index

def main():
    parser = argparse.ArgumentParser(
        description='''This is a beta script for Partial Volume Correction in PET/MRI system. ''',
        epilog="""All's well that ends well.""")

    parser.add_argument('--nameDataset', metavar='', type=str, default="reverse",
                        help='Name for the dataset needed to be sliced.(reverse)<str>')

    args = parser.parse_args()
    name_dataset = args.nameDataset
    nii_list = glob.glob("./data/"+name_dataset+"/*.nii")+glob.glob("./data/"+name_dataset+"/*.nii.gz")
    nii_list.sort()
    n_channel = 3

    for nii_path in nii_list:
        print("@"*60)
        print(nii_path)
        nii_file = nib.load(nii_path)
        nii_name = os.path.basename(nii_path)
        nii_name = nii_name[:nii_name.find(".")]
        nii_header = nii_file.header
        nii_affine = nii_file.affine
        nii_data = np.asanyarray(nii_file.dataobj)
        nii_data_norm = maxmin_norm(nii_data)

        # reverse
        nii_data_norm = 1-nii_data_norm
        nii_data_norm[nii_data_norm == 1] = 0

        save_path = "./z3/"+name_dataset+"/"
        for path in [save_path]:
            if not os.path.exists(path):
                os.makedirs(path)

        # enhance
        power_hub = [0.5,1,2,3]
        for power in power_hub:
            norm_mri_p = nii_data_norm ** power
            norm_mri_p = maxmin_norm(norm_mri_p)
            file_inv = nib.Nifti1Image(norm_mri_p, nii_file.affine, nii_file.header)
            save_name = save_path+nii_name+"_invp"+str(power)+".nii.gz"
            nib.save(file_inv, save_name)
            print(save_name)
        # norm_mri[otsu_data>0] = 255-norm_mri[otsu_data>0]
        
        # cut_th_0 = 100
        # cut_point_0 = np.percentile(norm_mri, cut_th_0)
    #     cut_point_1 = np.percentile(norm_mri, 99.9)
        
        # norm_mri[norm_mri < cut_point_0] = cut_point_0
    #     norm_mri[norm_mri > cut_point_1] = cut_point_1
        # norm_mri = maxmin_norm(norm_mri)*255

        # nii_smooth = processing.smooth_image(nii_file, fwhm=3, mode='nearest')
        # nii_smooth_zoom = zoom(np.asanyarray(nii_smooth.dataobj), zoom=(1/8, 1/8, 1))
        # nii_smooth_zoom_norm = maxmin_norm(nii_smooth_zoom)
        # print("nii_data_norm", nii_data_norm.shape)
        # print("nii_smooth_zoom_norm", nii_smooth_zoom_norm.shape)
        # nii_smooth_norm = maxmin_norm(np.asanyarray(nii_smooth.dataobj)) * 255

if __name__ == "__main__":
    main()
