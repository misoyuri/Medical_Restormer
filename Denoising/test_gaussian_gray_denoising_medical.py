## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

import torch.nn as nn
import torch
import torch.nn.functional as F

from basicsr.models.archs.restormer_arch import Restormer
from basicsr.utils import img2tensor
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import utils
from pdb import set_trace as stx
import yaml
from yaml import CLoader as Loader

from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H%M%S")
# except ImportError:
#     from yaml import Loader

np.random.seed(seed=0)  # for reproducibility

    
parser = argparse.ArgumentParser(description='Gasussian Grayscale Denoising using Restormer')

# parser.add_argument('--input_dir', default='./Denoising/Datasets/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', required =True, type=str, help='Directory for results')
parser.add_argument('--path_lq', required =True, type=str, help='Path to weights')
parser.add_argument('--weights', required =True, type=str, help='Path to weights')
parser.add_argument('--config', required =True, type=str, help='config file path')
args = parser.parse_args()

####### Load yaml #######
x = yaml.load(open(args.config, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

factor = 8
print("Model Test: Medical Images")
model_restoration = Restormer(**x['network_g'])    
checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])

print("===>Testing using weights: ", args.weights)
print("------------------------------------------------")
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

result_dir = os.path.join(args.result_dir, f"Medical_{current_time}")
result_dir_lq_png = os.path.join(result_dir, "lq", 'png')
result_dir_lq_npy = os.path.join(result_dir, "lq", 'npy')
result_dir_output_png = os.path.join(result_dir, "output", 'png')
result_dir_output_npy = os.path.join(result_dir, "output", 'npy')
result_dir_grid_png = os.path.join(result_dir, "grid", 'png')

os.makedirs(result_dir, exist_ok=True)
os.makedirs(result_dir_lq_npy, exist_ok=True)
os.makedirs(result_dir_lq_png, exist_ok=True)
os.makedirs(result_dir_output_png, exist_ok=True)
os.makedirs(result_dir_output_npy, exist_ok=True)
os.makedirs(result_dir_grid_png, exist_ok=True)

lq_pathes = glob(os.path.join(args.path_lq, "*.npy"))
with torch.no_grad():
    for idx, (img_lq_path) in enumerate(lq_pathes):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        
        img_lq_name = img_lq_path.split("/")[-1][:-10]
        
        img_lq = np.load(img_lq_path)
        img_lq = np.expand_dims(img_lq, axis=-1)
        
        input_ = img2tensor(img_lq, bgr2rgb=False, float32=True).unsqueeze(0).cuda()

        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        restored = model_restoration(input_)

        # Unpad images to original dimensions
        restored = restored[:,:,:h,:w]
        restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        
        # Setting path for Saving.
        save_file_lq_png_path = os.path.join(result_dir_lq_png, f"{img_lq_name}.png")
        save_file_output_png_path = os.path.join(result_dir_output_png, f"{img_lq_name}.png")
        save_file_grid_png_path = os.path.join(result_dir_grid_png, f"{img_lq_name}.png")

        save_file_lq_npy_path = os.path.join(result_dir_lq_npy, f"{img_lq_name}.npy")
        save_file_output_npy_path = os.path.join(result_dir_output_npy, f"{img_lq_name}.npy")
        
        # Saving npy images
        np.save(save_file_lq_npy_path, img_lq)
        np.save(save_file_output_npy_path, restored)
        
        # Saving png images
        restored = (restored - restored.min()) / (restored.max() - restored.min()) 
        restored = np.clip(restored, 0, 1)
        
        img_lq = (img_lq - img_lq.min()) / (img_lq.max() - img_lq.min())
        img_lq = np.clip(img_lq, 0, 1)
        
        
        fig, axes = plt.subplots(ncols=2)
        axes[0].imshow(img_lq, cmap='gray')
        axes[0].set_title("Noised")
        axes[0].axis('off')
        
        axes[1].imshow(restored, cmap='gray')
        axes[1].set_title("Denoised")
        axes[1].axis('off')
        
        fig.tight_layout()
        plt.savefig(save_file_grid_png_path)
        plt.close(fig)
        plt.clf()
        
        cv2.imwrite(save_file_lq_png_path, img_as_ubyte(img_lq))
        cv2.imwrite(save_file_output_png_path, img_as_ubyte(restored))
        