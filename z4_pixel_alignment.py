import os
import glob
import nibabel as nib
import nibabel.processing

mri_list = glob.glob("./data/mri_4688/*.nii.gz")
mri_list.sort()

for mri_path in mri_list:
    print("&"*60)
    print(mri_path)
    mri_name = os.path.basename(mri_path)
    mri_name = mri_name[:mri_name.find(".")]
    mri_dir = os.path.dirname(mri_path)+"/"
    
    mri_file = nib.load(mri_path)
    mri_file_4x = nib.processing.conform(mri_file, out_shape=(960, 960, 284), voxel_size=(0.25, 0.25, 0.6))
    save_name = mri_dir + mri_name + "_25.nii.gz"
    nib.save(mri_file_4x, save_name)
    print(save_name)
    
    mri_smooth = nib.processing.smooth_image(mri_file_4x, fwhm=3)
    mri_smooth_1x = nib.processing.conform(mri_smooth, out_shape=(240, 240, 71), voxel_size=(1, 1, 2.4))
    save_name = mri_dir + mri_name + "_1f3.nii.gz"
    nib.save(mri_smooth_1x, save_name)
    print(save_name)
