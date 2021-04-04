from __future__ import print_function
import argparse

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dbpn import Net as DBPN
from dbpn_v1 import Net as DBPNLL
from dbpn_iterative import Net as DBPNITER
from data import get_eval_set
from functools import reduce

from scipy.misc import imsave
import scipy.io as sio
import nibabel as nib
import numpy as np
import time
import glob
# import cv2

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

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--self_ensemble', type=bool, default=False)
parser.add_argument('--chop_forward', type=bool, default=False)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--input_dir', type=str, default='Input')
parser.add_argument('--output', default='Results/', help='Location to save checkpoint models')
parser.add_argument('--test_dataset', type=str, default='25f')
parser.add_argument('--model_type', type=str, default='DBPN-RES-MR64-3')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--model', default='weights/Mar27L1122-WCHENDBPN-RES-MR64-3NIFTY_4x_epoch_999.pth', help='sr pretrained base model')

opt = parser.parse_args()

gpus_list=range(opt.gpus)
print(opt)

cuda = opt.gpu_mode
dtype = torch.FloatTensor

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
# test_set = get_eval_set(os.path.join(opt.input_dir,opt.test_dataset), opt.upscale_factor)
# testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
if opt.model_type == 'DBPNLL':
    model = DBPNLL(num_channels=3, base_filter=64,  feat = 256, num_stages=10, scale_factor=opt.upscale_factor) ###D-DBPN
elif opt.model_type == 'DBPN-RES-MR64-3':
    model = DBPNITER(num_channels=3, base_filter=64,  feat = 256, num_stages=3, scale_factor=opt.upscale_factor) ###D-DBPN
else:
    model = DBPN(num_channels=3, base_filter=64,  feat = 256, num_stages=7, scale_factor=opt.upscale_factor) ###D-DBPN
    
if cuda:
    model = torch.nn.DataParallel(model, device_ids=gpus_list)

model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])

print(model)

def eval():
    model.eval()

    pet_list = glob.glob(os.path.join(opt.input_dir,opt.test_dataset)+"/*_25f.nii.gz")
    # print(os.path.join(opt.input_dir,opt.test_dataset)+"/*.nii.gz")
    pet_list.sort()

    for pet_path in pet_list:
        print("&"*60)
        print(pet_path)
        input_nii = nib.load(pet_path[:-11]+"_x960y960z71.nii.gz") # 1200
        bicubic_nii = nib.load(pet_path[:-11]+"_x960y960z71f3.nii.gz") # 300
        _, name = os.path.split(pet_path[:-11])
    # for batch in testing_data_loader:

        # input_nii = batch[0] # nifty format
        # bicubic_nii = batch[1]
        # name = batch[2]
        n_channel = 3

        templ_header = input_nii.header
        templ_affine = input_nii.affine
        xy1200_data = input_nii.get_fdata()
        xy1200_norm = maxmin_norm(xy1200_data)
        xy300_norm = maxmin_norm(bicubic_nii.get_fdata())
        pet_recon = np.zeros(xy1200_data.shape)
        pet_diff = np.zeros(xy1200_data.shape)
        pet_z = xy300_norm.shape[2]
        index = create_index(dataA=xy300_norm, n_slice=n_channel)

        xy300_slice = np.zeros((1, 3, xy300_norm.shape[0], xy300_norm.shape[1]))
        # xy1200_slice = np.zeros((xy1200_norm.shape[0], xy1200_norm.shape[1], 1))
        for idx_z in range(pet_z):
            # print(idx_z)

            for idx_c in range(n_channel):
                xy300_slice[0, idx_c, :, :] = xy300_norm[:, :, int(index[idx_z, idx_c])]
                # xy1200_slice[idx_c, :, :] = xy1200_norm[:, :, int(index[idx_z, idx_c])]

            with torch.no_grad():
                input = torch.cuda.FloatTensor(xy300_slice)
                # bicubic = torch.cuda.FloatTensor(xy1200_slice)
                input = Variable(input)
                # bicubic = Variable(input)

            # if cuda:
            #     input = input.cuda(gpus_list[0])
            #     bicubic = bicubic.cuda(gpus_list[0])

                t0 = time.time()
                if opt.chop_forward:
                    with torch.no_grad():
                        prediction = chop_forward(input, model, opt.upscale_factor)
                else:
                    if opt.self_ensemble:
                        with torch.no_grad():
                            prediction = x8_forward(input, model)
                    else:
                        with torch.no_grad():
                            prediction = model(input)

                prediction = np.asarray(prediction.cpu())
                pet_diff[:, :, idx_z] = np.squeeze(prediction[:, 1, :, :])

            # if opt.residual:
            #     prediction = prediction + bicubic

            t1 = time.time()
            print("===> Processing: %s || Timer: %.4f sec." % (str(idx_z), (t1 - t0)))

        # sum_recon = np.sum(pet_recon)
        # pet_recon = pet_recon / sum_recon * np.sum(xy1200_data)
        pet_recon = xy1200_data + pet_diff

        save_dir = os.path.join(opt.output,opt.test_dataset)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_fn = save_dir +'/'+ name
        recon_file = nib.Nifti1Image(pet_recon, templ_affine, templ_header)
        diff_file = nib.Nifti1Image(pet_diff, templ_affine, templ_header)
        nib.save(recon_file, save_fn + "_recon.nii.gz")
        nib.save(diff_file, save_fn + "_diff.nii.gz")
        print(save_fn + "_recon.nii.gz")
        print(save_fn + "_diff.nii.gz")

        # save_img(prediction.cpu().data, name[0])

def save_img(img, img_name):
    save_img = img.squeeze().clamp(-1, 1).numpy().transpose(1,2,0)
    # save img
    save_dir=os.path.join(opt.output,opt.test_dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_fn = save_dir +'/'+ img_name
    np.save(save_fn, save_img)
    # cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])

def x8_forward(img, model, precision='single'):
    def _transform(v, op):
        if precision != 'single': v = v.float()

        v2np = v.data.cpu().numpy()
        if op == 'vflip':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'hflip':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 'transpose':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()
        
        ret = torch.Tensor(tfnp).cuda()

        if precision == 'half':
            ret = ret.half()
        elif precision == 'double':
            ret = ret.double()

        with torch.no_grad():
            ret = Variable(ret)

        return ret

    inputlist = [img]
    for tf in 'vflip', 'hflip', 'transpose':
        inputlist.extend([_transform(t, tf) for t in inputlist])

    outputlist = [model(aug) for aug in inputlist]
    for i in range(len(outputlist)):
        if i > 3:
            outputlist[i] = _transform(outputlist[i], 'transpose')
        if i % 4 > 1:
            outputlist[i] = _transform(outputlist[i], 'hflip')
        if (i % 4) % 2 == 1:
            outputlist[i] = _transform(outputlist[i], 'vflip')
    
    output = reduce((lambda x, y: x + y), outputlist) / len(outputlist)

    return output
    
def chop_forward(x, model, scale, shave=8, min_size=80000, nGPUs=opt.gpus):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, nGPUs):
            with torch.no_grad():
                input_batch = torch.cat(inputlist[i:(i + nGPUs)], dim=0)
            if opt.self_ensemble:
                with torch.no_grad():
                    output_batch = x8_forward(input_batch, model)
            else:
                with torch.no_grad():
                    output_batch = model(input_batch)
            outputlist.extend(output_batch.chunk(nGPUs, dim=0))
    else:
        outputlist = [
            chop_forward(patch, model, scale, shave, min_size, nGPUs) \
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    with torch.no_grad():
        output = Variable(x.data.new(b, c, h, w))

    output[:, :, 0:h_half, 0:w_half] \
        = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

##Eval Start!!!!
eval()
