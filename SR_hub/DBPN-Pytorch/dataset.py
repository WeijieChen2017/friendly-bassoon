import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from random import randrange
from scipy.ndimage import zoom

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".npy"])

def load_img(filepath):
    # img = Image.open(filepath).convert('RGB')
    img = np.load(filepath)
    #y, _, _ = img.split()
    return img

def rescale_img(img_in, scale):
    # img_in = zoom(img_in, zoom=(scale, scale, 1))
    # new_size_in = tuple([int(x * scale) for x in size_in])
    # img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return zoom(img_in, zoom=(scale, scale, 1))

def get_patch(img_in, img_tar, img_bic, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy,ix,iy + ip, ix + ip))
    img_tar = img_tar.crop((ty,tx,ty + tp, tx + tp))
    img_bic = img_bic.crop((ty,tx,ty + tp, tx + tp))
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, img_bic, info_patch

def augment(img_in, img_tar, img_bic, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_bic = ImageOps.flip(img_bic)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            img_bic = ImageOps.mirror(img_bic)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            img_bic = img_bic.rotate(180)
            info_aug['trans'] = True
            
    return img_in, img_tar, img_bic, info_aug
    
class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, patch_size, upscale_factor, data_augmentation, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        origin = load_img(self.image_filenames[index][:-6]+"_X.npy")
        target = load_img(self.image_filenames[index][:-6]+"_Y.npy")
        # print(input.shape, self.image_filenames[index][:-6]+"_X.npy")
        # print(target.shape, self.image_filenames[index][:-6]+"_Y.npy")
        # input = target.resize((int(target.size[0]/self.upscale_factor),int(target.size[1]/self.upscale_factor)), Image.BICUBIC)       
        bicubic = rescale_img(origin, self.upscale_factor)
        print(bicubic.shape)
        
        origin, target, bicubic, _ = get_patch(origin,target,bicubic,self.patch_size, self.upscale_factor)
        
        if self.data_augmentation:
            origin, target, bicubic, _ = augment(origin, target, bicubic)
        
        if self.transform:
            origin = self.transform(origin)
            bicubic = self.transform(bicubic)
            target = self.transform(target)
                
        return origin, target, bicubic

    def __len__(self):
        return len(self.image_filenames)

class DatasetFromFolderEval(data.Dataset):
    def __init__(self, lr_dir, upscale_factor, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        self.image_filenames = [join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index][:-6]+"_X.npy")
        _, file = os.path.split(self.image_filenames[index])

        bicubic = rescale_img(input, self.upscale_factor)
        
        if self.transform:
            input = self.transform(input)
            bicubic = self.transform(bicubic)
            
        return input, bicubic, file
      
    def __len__(self):
        return len(self.image_filenames)
