import cv2
import glob
import copy
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import nibabel.processing
from nibabel import load, save, Nifti1Image

file_list = glob.glob("./sCT_1_*.nii.gz")
file_list.sort()
for file_path in file_list:
    print(file_path)
    case_idx = file_path[-10:-7]
    print(case_idx)
    mri_path = "./sCT_1_"+case_idx+".nii.gz"
    gm_path = "./c1sCT_1_"+case_idx+".nii.gz"
    wm_path = "./c2sCT_1_"+case_idx+".nii.gz"
    csf_path = "./c3sCT_1_"+case_idx+".nii.gz"
    skull_path = "./c4sCT_1_"+case_idx+".nii.gz"
    other_path = "./c5sCT_1_"+case_idx+".nii.gz"

    mri_file = nib.load(mri_path)
    gm_file = nib.load(gm_path)
    wm_file = nib.load(wm_path)
    csf_file = nib.load(csf_path)
    skull_file = nib.load(skull_path)
    other_file = nib.load(other_path)

    mri_data = mri_file.get_fdata()
    gm_data = gm_file.get_fdata()
    wm_data = wm_file.get_fdata()
    csf_data = csf_file.get_fdata()
    skull_data = skull_file.get_fdata()
    other_data = other_file.get_fdata()
    print("data shape", mri_data.shape)
    
    kernel = np.ones((3,3),np.uint8)
    gm_k3m = copy.deepcopy(gm_data)
#     gm_k3m[gm_k3m == 0] = 1
#     gm_k3m[gm_k3m == 1] = 0
    k3m_data = np.zeros(gm_data.shape)
    for idx in range(mri_data.shape[2]):
#         print(idx)
        img = gm_k3m[:, :, idx]
        k3m_data[:, :, idx] = cv2.erode(img, kernel, iterations = 1)
    gm_diff = np.abs(gm_data - k3m_data)
    gm_data = k3m_data
    wm_data = wm_data + gm_diff

    gm_mask = gm_data >= 0.5
    wm_mask = wm_data >= 0.5
    csf_mask = csf_data >= 0.5
    skull_mask = skull_data >= 0.5
    other_mask = other_data >= 0.5

    syn_data = np.zeros(mri_data.shape)
    gm_value = [10, 12] # k
    wm_value = [4, 6] # k
    csf_value = [4, 8] # 100
    skull_value = [15, 25] # 100
    other_value = [15, 25] # 100
    lesion_small = [18, 22] # k
    lesion_large = [23, 25] # k

    gm_val = int(np.random.rand()*(gm_value[1]-gm_value[0])+gm_value[0])*1000
    wm_val = int(np.random.rand()*(wm_value[1]-wm_value[0])+wm_value[0])*1000
    csf_val = int(np.random.rand()*(csf_value[1]-csf_value[0])+csf_value[0])*100
    skull_val = int(np.random.rand()*(skull_value[1]-skull_value[0])+skull_value[0])*100
    other_val = int(np.random.rand()*(other_value[1]-other_value[0])+other_value[0])*100
    large_val = int(np.random.rand()*(lesion_small[1]-lesion_small[0])+lesion_small[0])*1000
    small_val = int(np.random.rand()*(lesion_large[1]-lesion_large[0])+lesion_large[0])*1000

    syn_data[gm_mask] = gm_val
    syn_data[wm_mask] = wm_val
    syn_data[csf_mask] = csf_val
    syn_data[skull_mask] = skull_val
    syn_data[other_mask] = other_val

    print("gm_val", gm_val)
    print("wm_val", wm_val)
    print("csf_val", csf_val)
    print("skull_val", skull_val)
    print("other_val", other_val)
    print("large_val", large_val)
    print("small_val", small_val)

    center_x, center_y, center_z = 256, 256, 135
    span_x, span_y, span_z = 100, 100, 45
    range_x = [center_x-span_x//2, center_x+span_x//2]
    range_y = [center_y-span_y//2, center_y+span_y//2]
    range_z = [center_z-span_z//2, center_z+span_z//2]

    # radius_large = [15, 20]
    # radius_small = [5, 10]
    # rad_large = int(np.random.rand()*10+20)
    # rad_small = int(np.random.rand()*10+5)
    # print("rad_large", rad_large)
    # print("rad_small", rad_small)

    # dist_between = 75
    # loc_large = [int(np.random.rand()*span_x+range_x[0]),
    #              int(np.random.rand()*span_y+range_y[0]),
    #              int(np.random.rand()*span_z+range_z[0])]
    # loc_small = [int(np.random.rand()*span_x+range_x[0]),
    #              int(np.random.rand()*span_y+range_y[0]),
    #              int(np.random.rand()*span_z+range_z[0])]
    # loc_large = np.array(loc_large)
    # loc_small = np.array(loc_small)

    # while np.linalg.norm(loc_large-loc_small) <= dist_between:
    #     loc_large = [int(np.random.rand()*span_x+range_x[0]),
    #                  int(np.random.rand()*span_y+range_y[0]),
    #                  int(np.random.rand()*span_z+range_z[0])]
    #     loc_small = [int(np.random.rand()*span_x+range_x[0]),
    #                  int(np.random.rand()*span_y+range_y[0]),
    #                  int(np.random.rand()*span_z+range_z[0])]
    #     loc_large = np.array(loc_large)
    #     loc_small = np.array(loc_small)

    # print("loc_large", loc_large)
    # print("loc_small", loc_small)

    # for idx_x in range(mri_data.shape[0]):
    #     for idx_y in range(mri_data.shape[1]):
    #         for idx_z in range(mri_data.shape[2]):
    #             point = np.array([idx_x+1, idx_y+1, idx_z+1])
    #             for package in [[loc_large, rad_large, large_val],
    #                             [loc_small, rad_small, small_val]]:
    #                 loc = package[0]
    #                 rad = package[1]
    #                 val = package[2]
    #                 if np.linalg.norm(np.abs(loc-point)) <= rad:
    #                     syn_data[idx_x+1, idx_y+1, idx_z+1] = val

    physical = [mri_file.header["pixdim"][1]*mri_data.shape[0],
                mri_file.header["pixdim"][2]*mri_data.shape[1],
                mri_file.header["pixdim"][3]*mri_data.shape[2]]
    print("physical dist", physical)

    target_pixel = [256, 256, 89]
    target_physical = (round(physical[0]/target_pixel[0], 4),
                       round(physical[1]/target_pixel[1], 4),
                       round(physical[2]/target_pixel[2], 4))
    print("target physical dist per pixel", target_physical)

    out_file = Nifti1Image(syn_data, affine=mri_file.affine, header=mri_file.header)
    pet_style = nib.processing.conform(out_file, out_shape=(256, 256, 89), voxel_size=target_physical)
    save_name = "./k3m_1_"+case_idx+".nii.gz"
    save(pet_style, save_name)
    print(save_name)