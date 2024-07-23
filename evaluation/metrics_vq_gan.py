import os
import numpy as np
import torch
import time
from scipy import linalg # For numpy FID
from pathlib import Path
from PIL import Image
import models.models as models
import matplotlib.pyplot as plt
import torchvision.models
import cv2
import torch
import torchvision.transforms as transforms
import pytorch_msssim
import lpips
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
# --------------------------------------------------------------------------#
# This code is to calculate and save SSIM PIPS PSNR RMSE
# --------------------------------------------------------------------------#


class metrics:
    def __init__(self, opt, dataloader_val):
        self.opt = opt
        self.val_dataloader = dataloader_val
        self.path_to_save = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        self.path_to_save_PIPS = os.path.join(self.opt.checkpoints_dir, self.opt.name, "PIPS")
        self.path_to_save_SSIM = os.path.join(self.opt.checkpoints_dir, self.opt.name, "SSIM")
        self.path_to_save_PSNR = os.path.join(self.opt.checkpoints_dir, self.opt.name, "PSNR")
        self.path_to_save_RMSE = os.path.join(self.opt.checkpoints_dir, self.opt.name, "RMSE")

        Path(self.path_to_save_SSIM).mkdir(parents=True, exist_ok=True)
        Path(self.path_to_save_PIPS).mkdir(parents=True, exist_ok=True)
        Path(self.path_to_save_PSNR).mkdir(parents=True, exist_ok=True)
        Path(self.path_to_save_RMSE).mkdir(parents=True, exist_ok=True)

    def compute_metrics(self, model):
        pips, ssim, psnr, rmse  = [], [], [], []
        loss_fn_alex = lpips.LPIPS(net='vgg')
        loss_fn_alex = loss_fn_alex.to('cuda:0')
        model.eval()
        transform1 = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [0, 1]normalized to [—1, 1]
        ])
        transform2 = transforms.Compose([
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])  # normalized to [0, 1]
        ])

        total_samples = len(self.val_dataloader)
        with torch.no_grad():
            for i, data_i in enumerate(self.val_dataloader):
                image = data_i["image"]
                recon = model(image)

                input1 = (recon + 1) / 2
                input2 = (image + 1) / 2

                # SSIM
                ssim_value = pytorch_msssim.ssim(input1, input2)
                ssim.append(ssim_value.mean().item())
                # PIPS lpips
                d = loss_fn_alex(input1, input2)
                pips.append(d.mean().item())
                # PSNR, RMSE
                mse = torch.nn.functional.mse_loss(input1, input2)
                max_pixel_value = 1.0
                psnr_value = 10 * torch.log10((max_pixel_value ** 2) / mse)
                rmse_value = torch.sqrt(mse)
                psnr.append(psnr_value.mean().item())
                rmse.append(rmse_value.mean().item())

        model.train()

        avg_pips = sum(pips) / total_samples
        avg_ssim = sum(ssim) / total_samples
        avg_psnr = sum(psnr) / total_samples
        avg_rmse = sum(rmse) / total_samples
        avg_pips = np.array(avg_pips)
        avg_ssim = np.array(avg_ssim)
        avg_psnr = np.array(avg_psnr)
        avg_rmse = np.array(avg_rmse)

        return avg_pips, avg_ssim, avg_psnr, avg_rmse

    def update_metrics(self, model, cur_iter):
        print("--- Iter %s: computing PIPS SSIM PSNR RMSE---" % (cur_iter))
        cur_pips, cur_ssim, cur_psnr, cur_rmse= self.compute_metrics(model)
        self.update_logs(cur_pips, cur_iter, 'PIPS')
        self.update_logs(cur_ssim, cur_iter, 'SSIM')
        self.update_logs(cur_psnr, cur_iter, 'PSNR')
        self.update_logs(cur_rmse, cur_iter, 'RMSE')

        print("--- Metrics at Iter %s: " % cur_iter, "{:.2f}".format(cur_pips),"{:.2f}".format(cur_ssim),"{:.2f}".format(cur_psnr),"{:.2f}".format(cur_rmse))

    def update_logs(self, cur_data, epoch, mode):
        try:
            np_file = np.load(os.path.join(self.path_to_save, mode, f"{mode}_log.npy"), allow_pickle=True)
            first = list(np_file[0, :])
            sercon = list(np_file[1, :])
            first.append(epoch)
            sercon.append(cur_data)
            np_file = [first, sercon]
        except:
            np_file = [[epoch], [cur_data]]

        np.save(os.path.join(self.path_to_save, mode, f"{mode}_log.npy"), np_file)
        np_file = np.array(np_file)
        plt.figure()
        plt.plot(np_file[0, :], np_file[1, :])
        plt.grid(b=True, which='major', color='#666666', linestyle='--')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
        plt.savefig(os.path.join(self.path_to_save, mode, f"plot_{mode}"), dpi=600)
        plt.close()

    def metrics_test(self, model):
        print("--- test: computing FID ---")
        pips, ssim, psnr, rmse = self.compute_metrics(model.module.netG, model.module.netEMA, model)
        print("--- PIPS at test : ", "{:.2f}".format(pips))
        print("--- SSIM at test : ", "{:.5f}".format(ssim))
        print("--- PSNR at test : ", "{:.2f}".format(psnr))
        print("--- RMSE at test : ", "{:.2f}".format(rmse))