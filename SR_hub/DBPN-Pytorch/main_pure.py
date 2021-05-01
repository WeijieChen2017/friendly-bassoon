from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dbpn import Net as DBPN
from dbpn_v1 import Net as DBPNLL
from dbpns import Net as DBPNS
from dbpn_iterative import Net as DBPNITER
from data import get_training_set
import pdb
import socket
import time

import nibabel as nib
import numpy as np
import random
import glob

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=10, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./dataset')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--hr_train_dataset', type=str, default='alan_2021')
parser.add_argument('--model_type', type=str, default='DBPN-RES-MR64-3')
parser.add_argument('--residual', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=40, help='Size of cropped HR image')
parser.add_argument('--pretrained_sr', default='DBPN-RES-MR64-3_4x.pth', help='sr pretrained base model')
# parser.add_argument('--pretrained_sr', default='DBPN-RES-MR64-3_4x.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='alan2021_', help='Location to save checkpoint models')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
print(gpus_list)
# gpus_list = [0, 1]
hostname = str(socket.gethostname())
cudnn.benchmark = True
dtype = torch.FloatTensor
print(opt)

def get_patch(img_in, img_tar, img_bic, patch_size, scale, ix=-1, iy=-1):
    img_x = img_in.shape[0]
    img_y = img_in.shape[1]
    tar_x = img_x * scale
    tar_y = img_y * scale

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, img_x - ip + 1)
    if iy == -1:
        iy = random.randrange(0, img_y - ip + 1)

    [tx, ty] = [scale * ix, scale * iy]

    img_in = img_in[ix:ix+ip, iy:iy+ip]
    img_tar = img_tar[tx:tx+tp, ty:ty+tp]
    img_bic = img_bic[tx:tx+tp, ty:ty+tp]
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, img_bic, info_patch

def augment(img_in, img_tar, img_bic, flip_h=True, flip_v=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        for img in [img_in, img_tar, img_bic]:
            for idx_c in range(0):
                img[:, :, idx_c] = np.fliplr(img)
        info_aug['flip_h'] = True

    if random.random() < 0.5 and flip_v:
        for img in [img_in, img_tar, img_bic]:
            for idx_c in range(0):
                img[:, :, idx_c] = np.flipup(img)
        info_aug['flip_v'] = True

    if rot:
        cnt_rot = int(random.random()//0.25)
        for img in [img_in, img_tar, img_bic]:
            for idx_c in range(0):
                img[:, :, idx_c] = np.rot90(img, cnt_rot)
        info_aug['trans'] = True
            
    return img_in, img_tar, img_bic, info_aug

def train(epoch):
    epoch_loss = 0
    model.train()
    nii_list = glob.glob("./dataset/"+opt.hr_train_dataset+"/*_GTH.nii.gz")
    nii_list.sort()
    # print(os.path.join(opt.input_dir,opt.test_dataset)+"/*.nii.gz")
    n_dataset = len(nii_list)
    for idx_t in range(n_dataset):

        nii_path = nii_list[idx_t]
        input_nii = nib.load(nii_path.replace("GTH", "INP")).get_fdata()
        target_nii = nib.load(nii_path).get_fdata()
        bicubic_nii = nib.load(nii_path.replace("GTH", "BIC")).get_fdata()

        # input_nii = nib.load(image_dir+"MINC_"+train_hub[idx_t]+"_v1_"+"Small"+suffix_hub[idx_f]+".nii.gz").get_fdata()
        # target_nii = nib.load(image_dir+"MINC_"+train_hub[idx_t]+"_v1_GT.nii.gz").get_fdata()
        # bicubic_nii = nib.load(image_dir+"MINC_"+train_hub[idx_t]+"_v1_"+"Large"+suffix_hub[idx_f]+".nii.gz").get_fdata()
        # print(input_nii.shape, target_nii.shape, bicubic_nii.shape)

        cntz = input_nii.shape[2]
        input_batch = np.zeros((opt.batchSize, 3, opt.patch_size, opt.patch_size))
        target_batch = np.zeros((opt.batchSize, 3, opt.patch_size*opt.upscale_factor, opt.patch_size*opt.upscale_factor))
        bicubic_batch = np.zeros((opt.batchSize, 3, opt.patch_size*opt.upscale_factor, opt.patch_size*opt.upscale_factor))
        
        cnt_iter = (cntz-2) // opt.batchSize
        seq_order = np.linspace(1, cntz-2, num=cntz-2, dtype=np.int32)
        random.shuffle(seq_order)
        print(seq_order)

        for idx_s in range(cnt_iter):
            for idx_b in range(opt.batchSize):
                iz = seq_order[idx_s*opt.batchSize+idx_b]
                input = input_nii[:, :, iz-1:iz+2]
                target = target_nii[:, :, iz-1:iz+2]
                bicubic = bicubic_nii[:, :, iz-1:iz+2]
                # print(input.shape, target.shape, bicubic.shape)

                input, target, bicubic, _ = get_patch(input,target,bicubic, opt.patch_size, opt.upscale_factor)
                input, target, bicubic, _ = augment(input, target, bicubic)

                for idx_c in range(3):
                    input_batch[idx_b, idx_c, :, :] = input[:, :, idx_c]
                    target_batch[idx_b, idx_c, :, :] = target[:, :, idx_c]
                    bicubic_batch[idx_b, idx_c, :, :] = bicubic[:, :, idx_c]

            input = torch.cuda.FloatTensor(input_batch)
            target = torch.cuda.FloatTensor(target_batch)
            bicubic = torch.cuda.FloatTensor(bicubic_batch)
            
            input = Variable(input)
            target = Variable(target)
            bicubic = Variable(bicubic)
                
            if cuda:
                input = input.cuda(gpus_list[0])
                target = target.cuda(gpus_list[0])
                bicubic = bicubic.cuda(gpus_list[0])

            optimizer.zero_grad()
            t0 = time.time()
            prediction = model(input)
            # print("prediction", prediction.size())

            if opt.residual:
                prediction = prediction + bicubic

            loss = criterion(prediction, target)
            t1 = time.time()
            epoch_loss += loss.data
            loss.backward()
            optimizer.step()

            # print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, idx_t, n_dataset, loss.data, (t1 - t0)))
            print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, idx_t*cnt_iter+idx_s, n_dataset*cnt_iter, loss.data, (t1 - t0)))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / n_dataset))



    # for iteration, batch in enumerate(training_data_loader, 1):
    #     input = Variable(batch[0]).type(dtype)
    #     target = Variable(batch[1]).type(dtype)
    #     bicubic = Variable(batch[2]).type(dtype)

    #     # print("input, target, bicubic", input.size(), target.size(), bicubic.size())
    #     if cuda:
    #         input = input.cuda(gpus_list[0])
    #         target = target.cuda(gpus_list[0])
    #         bicubic = bicubic.cuda(gpus_list[0])

    #     optimizer.zero_grad()
    #     t0 = time.time()
    #     prediction = model(input)
    #     # print("prediction", prediction.size())

    #     if opt.residual:
    #         prediction = prediction + bicubic

    #     loss = criterion(prediction, target)
    #     t1 = time.time()
    #     epoch_loss += loss.data
    #     loss.backward()
    #     optimizer.step()

    #     print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.data, (t1 - t0)))

    # print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.cuda(gpus_list[0])
            target = target.cuda(gpus_list[0])

        prediction = model(input)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def checkpoint(epoch):
    model_out_path = opt.save_folder+opt.hr_train_dataset+hostname+opt.model_type+opt.prefix+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.hr_train_dataset, opt.upscale_factor, opt.patch_size, opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Building model ', opt.model_type)
if opt.model_type == 'DBPNLL':
    model = DBPNLL(num_channels=3, base_filter=64,  feat = 256, num_stages=10, scale_factor=opt.upscale_factor) 
elif opt.model_type == 'DBPN-RES-MR64-3':
    model = DBPNITER(num_channels=3, base_filter=64,  feat = 256, num_stages=3, scale_factor=opt.upscale_factor)
else:
    model = DBPN(num_channels=3, base_filter=64,  feat = 256, num_stages=7, scale_factor=opt.upscale_factor) 
    
model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion = nn.MSELoss()

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])
    criterion = criterion.cuda(gpus_list[0])

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

for epoch in range(opt.start_iter, opt.nEpochs + 1):
    train(epoch)

    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch+1) % (opt.nEpochs/2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
            
    if (epoch+1) % (opt.snapshots) == 0:
        checkpoint(epoch)

