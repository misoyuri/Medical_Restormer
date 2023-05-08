from torch.utils import data as data
from torchvision.transforms.functional import normalize
from torchvision import transforms

from basicsr.data.transforms import paired_random_crop, random_augmentation
from basicsr.utils import FileClient, img2tensor, padding
from basicsr.utils import scandir
import random
import numpy as np
import torch
import cv2
from glob import glob
import os

from os import path as osp
from scipy.ndimage import gaussian_filter

class Dataset_Medical_Paired_GaussianDenoising(data.Dataset):
    def __init__(self, opt):
        super(Dataset_Medical_Paired_GaussianDenoising, self).__init__()
        print("Dataset_Medical_Paired_GaussianDenoising!")
        self.opt = opt
        self.in_ch = opt['in_ch']

        # file client (io backend)
        self.file_client = None
        # self.io_backend_opt = opt['io_backend']

        self.gt_path = opt['dataroot_gt']
        self.lq_path = opt['dataroot_lq']

        self.img_lq = sorted(glob(os.path.join(self.lq_path, "*.npy")))
        
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
        ])
        #################################
        
    def __getitem__(self, index):

        #############
        
        # if self.file_client is None:
        #     self.file_client = FileClient(
        #         self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.img_lq)

        img_lq_path = self.img_lq[index]
        # print("img_lq_path: ", img_lq_path)
        # print("self.gt_path: ", self.gt_path)
        # print(img_lq_path.split("/"))
        # print( os.path.join(self.gt_path, img_lq_path.split("/")[-1][:-10]+".npy"))
        img_gt_path = os.path.join(self.gt_path, img_lq_path.split("/")[-1][:-10]+".npy")
        
        img_lq = np.load(img_lq_path)
        img_gt = np.load(img_gt_path)

        img_lq = np.expand_dims(img_lq, axis=-1)
        img_gt = np.expand_dims(img_gt, axis=-1)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, None)
            # flip, rotation
            img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

        else:            
            np.random.seed(seed=0)
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': f"lq_{index}.png",
            'gt_path': f"gt_{index}.png"
        }

    def __len__(self):
        return len(self.lq_path)