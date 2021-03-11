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
        save_path = "./SR/"+name_dataset+"/"
        for path in [save_path]:
            if not os.path.exists(path):
                os.makedirs(path)

        for idx_z in range(dz):
            LR_path = "./LR/"+name_dataset+"/"+nii_name+"_{0:03d}".format(idx_z)+".npy"
            HR_path = "./HR/"+name_dataset+"/"+nii_name+"_{0:03d}".format(idx_z)+".npy"
            SR_path = "./SR/"+name_dataset+"/"+nii_name+"_{0:03d}".format(idx_z)+"_rlt.npy"

            img_LR = np.load(LR_path)
            img_HR = np.load(HR_path)
            img_SR = np.load(SR_path)

            print(img_LR.shape)
            print(img_HR.shape)
            print(img_SR.shape)

            exit()
            img[:, :, idx_c] = zoom(nii_data[:, :, int(index[idx_z, idx_c])], zoom=scale_factor)
            name2save = savepath+nii_name+"_{0:03d}".format(idx_z)+".npy"
            img = maxmin_norm(img) * 255.0
            np.save(name2save, img)
            print(name2save, img.shape)
        print(str(idx_z)+" images have been saved.")
        print("#"*20)

if __name__ == "__main__":
    main()

