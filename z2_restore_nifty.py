from skimage.metrics import mean_squared_error, normalized_root_mse, peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
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

    parser.add_argument('--nameDataset', metavar='', type=str, default="hybrid",
                        help='Name for the dataset needed to be sliced.(hybrid)<str>')

    args = parser.parse_args()
    name_dataset = args.nameDataset
    nii_list = glob.glob("./data/"+name_dataset+"/*.nii")+glob.glob("./data/"+name_dataset+"/*.nii.gz")
    nii_list.sort()
    metric_hub = [mean_squared_error, normalized_root_mse, peak_signal_noise_ratio, structural_similarity]

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

        dx, dy, dz = nii_data.shape
        save_path = "./SR_nii/"+name_dataset+"/"
        for path in [save_path]:
            if not os.path.exists(path):
                os.makedirs(path)

        for package in [["LR", ""], ["SR", "_rlt"]]:
            modality = package[0]
            load_tag = package[1]
            curr_data = np.zeros(nii_data.shape)
            for idx_z in range(dz):
                curr_path = "./"+modality+"/"+name_dataset+"/"+nii_name+"_{0:03d}".format(idx_z)+load_tag+".npy"
                curr_img = np.load(curr_path)
                curr_scale_factor = dx / curr_img.shape[0]
                curr_data[:, :, idx_z] = zoom(curr_img[:, :, curr_img.shape[2]//2], zoom=curr_scale_factor)            
            
            curr_data_norm = maxmin_norm(curr_data)
            for metric in metric_hub:
                print(metric_hub.__name__, metric(nii_data_norm, curr_data_norm))
            curr_file = nib.Nifti1Image(curr_data, nii_affine, nii_header)
            nib.save(curr_file, save_path+nii_name+"_"+modality+".nii.gz")
            print(save_path+nii_name+"_"+modality+".nii.gz saved.")

if __name__ == "__main__":
    main()

