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

    parser.add_argument('--nameDataset', metavar='', type=str, default="hybrid",
                        help='Name for the dataset needed to be sliced.(hybrid)<str>')

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
        nii_smooth = processing.smooth_image(nii_file, fwhm=3, mode='nearest')
        nii_smooth_zoom = zoom(np.asanyarray(nii_smooth.dataobj), zoom=(1/8, 1/8, 1))
        nii_smooth_zoom_norm = maxmin_norm(nii_smooth_zoom)
        print("nii_data_norm", nii_data_norm.shape)
        print("nii_smooth_zoom_norm", nii_smooth_zoom_norm.shape)
        # nii_smooth_norm = maxmin_norm(np.asanyarray(nii_smooth.dataobj)) * 255

        dx, dy, dz = nii_data.shape
        save_path_X = "./z1_8x/"+name_dataset+"/"
        save_path_Y = "./z1_8x/"+name_dataset+"/"
        for path in [save_path_X, save_path_Y]:
            if not os.path.exists(path):
                os.makedirs(path)

        for package in [[nii_data_norm, save_path_Y, "_Y"], [nii_smooth_zoom_norm, save_path_X, "_X"]]:
            data = package[0]
            savepath = package[1]
            suffix = package[2]

            index = create_index(data, n_channel)
            img = np.zeros((data.shape[0], data.shape[1], n_channel))
            for idx_z in range(dz):
                for idx_c in range(n_channel):
                    # img[:, :, idx_c] = zoom(nii_data[:, :, int(index[idx_z, idx_c])], zoom=resize_f)
                    img[:, :, idx_c] = data[:, :, int(index[idx_z, idx_c])]
                name2save = savepath+nii_name+"_{0:03d}".format(idx_z)+suffix+".npy"
                np.save(name2save, img)
            print("#"*20)
            print("Last:", savepath+nii_name+"_{0:03d}".format(idx_z)+suffix+".npy")
            print(str(idx_z)+" images have been saved.")
            

if __name__ == "__main__":
    main()

