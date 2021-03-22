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

    parser.add_argument('--nameDataset', metavar='', type=str, default="pet",
                        help='Name for the dataset needed to be sliced.(pet)<str>')

    args = parser.parse_args()
    name_dataset = args.nameDataset
    nii_list = glob.glob("./eval/"+name_dataset+"/*.nii")+glob.glob("./eval/"+name_dataset+"/*.nii.gz")
    nii_list.sort()
    n_channel = 3

    model_list = glob.glob("./weights/*epoch_69.pth")
    model_list.sort()

    for model_path in model_list:
        up_factor = int(model_path[:-15]) 
        weight_name = os.path.basename(model_path)
        command = "python eval.py"
        command += " --upscale_factor "
        command += up_factor
        command += " --model "
        command += "weights/"+weight_name
        print(command)


    # for nii_path in nii_list:
    #     print("@"*60)
    #     print(nii_path)
    #     nii_file = nib.load(nii_path)
    #     nii_name = os.path.basename(nii_path)
    #     nii_name = nii_name[:nii_name.find(".")]
    #     nii_header = nii_file.header
    #     nii_affine = nii_file.affine
    #     nii_data = np.asanyarray(nii_file.dataobj)
    #     nii_data = zoom(nii_data, scale=(1, 1, 3))
    #     nii_data_norm = maxmin_norm(nii_data)

    #     save_path = "./z4/"+name_dataset+"/"
    #     for path in [save_path]:
    #         if not os.path.exists(path):
    #             os.makedirs(path)



if __name__ == "__main__":
    main()
