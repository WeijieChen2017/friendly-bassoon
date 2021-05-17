import cv2
import glob
import copy
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import nibabel.processing
from nibabel import load, save, Nifti1Image

def specify_value(syn_data, value_hub, mask_hub):
    
    if value_hub is None:
    
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
        
        print("gm_val", gm_val)
        print("wm_val", wm_val)
        print("csf_val", csf_val)
        print("skull_val", skull_val)
        print("other_val", other_val)
        print("large_val", large_val)
        print("small_val", small_val)
        
        value_hub = [gm_val, wm_val, csf_val, skull_val, other_val, large_val, small_val]
    
    gm_val, wm_val, csf_val, skull_val, other_val, large_val, small_val = value_hub
    gm_mask, wm_mask, csf_mask, skull_mask, other_mask = mask_hub
    
    syn_data[gm_mask] = gm_val
    syn_data[wm_mask] = wm_val
    syn_data[csf_mask] = csf_val
    syn_data[skull_mask] = skull_val
    syn_data[other_mask] = other_val


    
    return syn_data, value_hub

def shrink_from_edge(gm_data, mask_hub, k):
    
    mask_hub_se = copy.deepcopy(mask_hub)
    kernel = np.ones((k, k),np.uint8)
    k3m_data = np.zeros(gm_data.shape)
    for idx in range(gm_data.shape[2]):
        img = gm_data[:, :, idx]
        k3m_data[:, :, idx] = cv2.erode(img, kernel, iterations = 1)
        
    gm_diff = np.abs(gm_data - k3m_data)
    gm_diff_mask = gm_diff >= 0.5

    # Nearest neighbor for partical volume
    nn_map = np.zeros(gm_data.shape)
    nn_map[mask_hub[0]] = 1
    nn_map[mask_hub[1]] = 2
    nn_map[mask_hub[2]] = 3
    nn_map[mask_hub[3]] = 4
    nn_map[mask_hub[4]] = 5
    nn_map[gm_diff_mask] = 255
    
    for idx in range(gm_data.shape[0]):
        for idy in range(gm_data.shape[1]):
            for idz in range(gm_data.shape[2]):
                if nn_map[idx, idy, idz] == 255:
                    vote = np.zeros(6)
                    for dx in range(-1,2):
                        for dy in range(-1, 2):
                            for dz in range(-1, 2):
                                seg_type = int(nn_map[idx+dx, idy+dy, idz+dz])
                                if seg_type <= 5:
                                    vote[seg_type] += 1
                    if np.argmax(vote) > 0:
                        mask_hub_se[np.argmax(vote)-1][idx, idy, idz] = True
    return mask_hub_se

def pre_process(file_path):
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
    data_shape = mri_data.shape
    print("data shape", data_shape)
    
    physical = [mri_file.header["pixdim"][1]*mri_data.shape[0],
                mri_file.header["pixdim"][2]*mri_data.shape[1],
                mri_file.header["pixdim"][3]*mri_data.shape[2]]
    print("physical dist", physical)

    target_pixel = [256, 256, 89]
    target_physical = (round(physical[0]/target_pixel[0], 4),
                       round(physical[1]/target_pixel[1], 4),
                       round(physical[2]/target_pixel[2], 4))
    print("target physical dist per pixel", target_physical)
    
    gm_mask = gm_data >= 0.5
    wm_mask = wm_data >= 0.5
    csf_mask = csf_data >= 0.5
    skull_mask = skull_data >= 0.5
    other_mask = other_data >= 0.5
    
    mask_hub = [gm_mask, wm_mask, csf_mask, skull_mask, other_mask]
    
    return case_idx, mri_file, gm_data, data_shape, mask_hub, target_physical

def save_syn_pet(syn_data, mri_file, target_physical, case_idx, save_tag):
    out_file = Nifti1Image(syn_data, affine=mri_file.affine, header=mri_file.header)
    pet_style = nib.processing.conform(out_file, out_shape=(256, 256, 89), voxel_size=target_physical)
    out_data = pet_style.get_fdata()
    out_data[out_data < 0] = 0
    out_file = Nifti1Image(out_data, affine=pet_style.affine, header=pet_style.header)
    save_name = "./"+save_tag+"_"+case_idx+".nii.gz"
    save(out_file, save_name)
    print(save_name)

def add_tumor(syn_data, large_val, small_value, tumor_geo):
    
    if tumor_geo is None:
        center_x, center_y, center_z = 256, 256, 135
        span_x, span_y, span_z = 100, 100, 45
        range_x = [center_x-span_x//2, center_x+span_x//2]
        range_y = [center_y-span_y//2, center_y+span_y//2]
        range_z = [center_z-span_z//2, center_z+span_z//2]

        radius_large = [20, 30]
        radius_small = [5, 15]
        rad_large = int(np.random.rand()*10+20)
        rad_small = int(np.random.rand()*10+5)
        print("rad_large", rad_large)
        print("rad_small", rad_small)

        dist_between = 75
        loc_large = [int(np.random.rand()*span_x+range_x[0]),
                     int(np.random.rand()*span_y+range_y[0]),
                     int(np.random.rand()*span_z+range_z[0])]
        loc_small = [int(np.random.rand()*span_x+range_x[0]),
                     int(np.random.rand()*span_y+range_y[0]),
                     int(np.random.rand()*span_z+range_z[0])]
        loc_large = np.array(loc_large)
        loc_small = np.array(loc_small)

        while np.linalg.norm(loc_large-loc_small) <= dist_between:
            loc_large = [int(np.random.rand()*span_x+range_x[0]),
                         int(np.random.rand()*span_y+range_y[0]),
                         int(np.random.rand()*span_z+range_z[0])]
            loc_small = [int(np.random.rand()*span_x+range_x[0]),
                         int(np.random.rand()*span_y+range_y[0]),
                         int(np.random.rand()*span_z+range_z[0])]
            loc_large = np.array(loc_large)
            loc_small = np.array(loc_small)
        
        print("loc_large", loc_large)
        print("loc_small", loc_small)
        
        tumor_geo = [rad_large, rad_small, loc_large, loc_small]

    rad_large, rad_small, loc_large, loc_small = tumor_geo

    for idx_x in range(mri_data.shape[0]):
        for idx_y in range(mri_data.shape[1]):
            for idx_z in range(mri_data.shape[2]):
                point = np.array([idx_x+1, idx_y+1, idx_z+1])
                for package in [[loc_large, rad_large, large_val],
                                [loc_small, rad_small, small_val]]:
                    loc = package[0]
                    rad = package[1]
                    val = package[2]
                    if np.linalg.norm(np.abs(loc-point)) <= rad:
                        syn_data[idx_x+1, idx_y+1, idx_z+1] = val
    return syn_data, tumor_geo

file_list = glob.glob("./sCT_1_*.nii.gz")
file_list.sort()
for file_path in file_list:

    case_idx, mri_file, gm_data, data_shape, mask_hub, target_physical = pre_process(file_path)
    
    syn_data = np.zeros(data_shape)
    syn_data, value_hub = specify_value(syn_data, None, mask_hub)
#     syn_data, tumor_geo = add_tumor(syn_data, value_hub[5], value_hub[6], None)
    save_syn_pet(syn_data, mri_file, target_physical, case_idx, "original")
    
    syn_data_se = np.zeros(data_shape)
    mask_hub_se = shrink_from_edge(gm_data, mask_hub, 5)
    syn_data_se, value_hub = specify_value(syn_data_se, value_hub, mask_hub_se)
#     syn_data_se, tumor_geo = add_tumor(syn_data_se, value_hub[5], value_hub[6], tumor_geo)
    save_syn_pet(syn_data_se, mri_file, target_physical, case_idx, "erode5_nn")
