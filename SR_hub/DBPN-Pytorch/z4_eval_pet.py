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

    parser.add_argument('--nameDataset', metavar='', type=str, default="pet_nii",
                        help='Name for the dataset needed to be sliced.(pet)<str>')

    args = parser.parse_args()
    name_dataset = args.nameDataset
    nii_list = glob.glob("./Input/"+name_dataset+"/*.nii")+glob.glob("./Input/"+name_dataset+"/*.nii.gz")
    nii_list.sort()
    n_channel = 3

    model_list = glob.glob("./weights/*epoch_69.pth")
    model_list.sort()

    for model_path in model_list:
        up_factor = int(model_path[-15]) 
        weight_name = os.path.basename(model_path)
        command = "python eval.py"
        command += " --upscale_factor "
        command += str(up_factor)
        command += " --model "
        command += "weights/"+weight_name
        print(command)
        os.system(command)

        for nii_path in nii_list:
            print("@"*60)
            print(nii_path)
            nii_file = nib.load(nii_path)
            nii_name = os.path.basename(nii_path)
            nii_name = nii_name[:nii_name.find(".")]
            nii_header = nii_file.header
            nii_affine = nii_file.affine
            nii_data = np.asanyarray(nii_file.dataobj)

            dx, dy, dz = nii_data.shape
            curr_data = np.zeros((dx, dy, dz*3))
            save_path = "./eval/"+weight_name[:-4]+"/"
            for path in [save_path]:
                if not os.path.exists(path):
                    os.makedirs(path)

            for idx_z in range(dz*3):
                curr_path = "./Results/pet/"+nii_name+"_{0:03d}".format(idx_z)+"_113.npy"
                curr_img = np.load(curr_path)
                curr_data[:, :, idx_z] = zoom(curr_img[:, :, curr_img.shape[2]//2], zoom=(1/up_factor, 1/up_factor))            
            
            pvc_data = zoom(curr_data, zoom=(1, 1, 1/3))
            pvc_sum = np.sum(pvc_data)
            pvc_data = pvc_data / pvc_sum * np.sum(nii_data)

            pvc_file = nib.Nifti1Image(pvc_data, nii_affine, nii_header)
            nib.save(pvc_file, save_path+nii_name+".nii.gz")
            print(save_path+nii_name+".nii.gz")





if __name__ == "__main__":
    main()
