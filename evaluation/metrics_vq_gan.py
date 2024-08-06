import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import lpips
import requests
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from einops import rearrange
from torchvision import transforms as T
# --------------------------------------------------------------------------#
# This code is to calculate and save SSIM PIPS PSNR RMSE
# --------------------------------------------------------------------------#


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window


class metrics:
    def __init__(self, path, dataloader_val, num_classes=37):
        self.val_dataloader = dataloader_val
        self.root_dir = path
        self.path_to_save = os.path.join(self.root_dir, 'metrics')
        self.path_to_save_PIPS = os.path.join(self.path_to_save, "PIPS")
        self.path_to_save_SSIM = os.path.join(self.path_to_save, "SSIM")
        self.path_to_save_PSNR = os.path.join(self.path_to_save, "PSNR")
        self.path_to_save_RMSE = os.path.join(self.path_to_save, "RMSE")
        self.path_to_save_FID = os.path.join(self.path_to_save, "FID")
        self.path_to_save_L1 = os.path.join(self.path_to_save, "L1")

        Path(self.path_to_save_SSIM).mkdir(parents=True, exist_ok=True)
        Path(self.path_to_save_PIPS).mkdir(parents=True, exist_ok=True)
        Path(self.path_to_save_PSNR).mkdir(parents=True, exist_ok=True)
        Path(self.path_to_save_RMSE).mkdir(parents=True, exist_ok=True)
        Path(self.path_to_save_L1).mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        #self.inception_model = inception_v3(pretrained=True, transform_input=False).eval().cuda()
        #self.inception_model = self.load_inception_model()

    def load_inception_model(self):
        url = "https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth"
        response = requests.get(url)
        with open("inception_v3.pth", "wb") as f:
            f.write(response.content)

        model = inception_v3(pretrained=False, transform_input=False)
        model.load_state_dict(torch.load("inception_v3.pth"))
        model = model.eval().cuda()
        model_features = nn.Sequential(*list(model.children())[:-2])

        return model_features

    def compute_metrics(self, model, encoder=None, seg=False):
        pips, ssim, psnr, rmse, fid, l1 = [], [], [], [], [], []
        model.eval()
        with torch.no_grad():
            for i, data_i in enumerate(self.val_dataloader):
                label_save = data_i['label']
                image, label = self.preprocess_input(data_i)
                if seg:
                    recon = model(image, label)
                else:
                    recon = model(image)

                input1 = (recon + 1) / 2
                input2 = (image + 1) / 2

                # SSIM
                #ssim_value, _ = self.ssim_3d(input1, input2)
                #ssim.append(ssim_value.item())
                # PIPS lpips
                d = self.pips_3d(input1, input2)
                pips.append(d.mean().item())
                # PSNR, RMSE
                psnr_value = self.psnr_3d(input1, input2)
                rmse_value = self.rmse_3d(input1, input2)
                psnr.append(psnr_value.item())
                rmse.append(rmse_value.item())

                # FID
                fid_value = self.calculate_fid(input1, input2)
                fid.append(fid_value.item())
                #l1
                l1_value = F.l1_loss(image, recon).item()
                l1.append(l1_value)

                break
        model.train()

        avg_pips = torch.mean(torch.tensor(pips)).item()
        #avg_ssim = torch.mean(torch.tensor(ssim)).item()
        avg_psnr = torch.mean(torch.tensor(psnr)).item()
        avg_rmse = torch.mean(torch.tensor(rmse)).item()
        avg_fid = torch.mean(torch.tensor(fid)).item()
        avg_l1 = torch.mean(torch.tensor(l1)).item()

        return avg_pips, avg_ssim, avg_psnr, avg_rmse, avg_fid, avg_l1

    def compute_metrics_test(self, model):
        pips, ssim, psnr, rmse, fid, l1 = [], [], [], [], [], []
        model.eval()
        total_samples = len(self.val_dataloader)
        with torch.no_grad():
            for i, data_i in enumerate(self.val_dataloader):
                image = data_i['image'].to('cuda:0')
                x_recon, z, vq_output = model(image, evaluation=True)

                input = (image + 1) / 2
                recon = (x_recon + 1) / 2


                input_list = F.pad(input, (2, 2, 2, 2))
                recon_list = F.pad(recon, (2, 2, 2, 2))

                input_gif = rearrange(input_list, '(i j) c f h w -> c f (i h) (j w)', i=1)
                recon_gif = rearrange(recon_list, '(i j) c f h w -> c f (i h) (j w)', i=1)
                path_video = os.path.join(self.root_dir, 'video_results')
                os.makedirs(path_video, exist_ok=True)

                image_path = os.path.join(path_video, f'{i}_input.gif')
                label_path = os.path.join(path_video, f'{i}_recon.gif')
                video_tensor_to_gif(input_gif, image_path)
                video_tensor_to_gif(recon_gif, label_path)

                # SSIM
                ssim_value, _ = self.ssim_3d(input, recon)
                ssim.append(ssim_value.item())
                # PIPS lpips
                d = self.pips_3d(input, recon)
                pips.append(d.mean().item())
                # PSNR, RMSE
                psnr_value = self.psnr_3d(input, recon)
                rmse_value = self.rmse_3d(input, recon)
                psnr.append(psnr_value.item())
                rmse.append(rmse_value.item())

                l1_value = F.l1_loss(x_recon, recon).item()
                l1.append(l1_value)

                if i == 200:
                    break

                # FID
                #fid_value = self.calculate_fid(input1, input2)
                #fid.append(fid_value.item())

        model.train()

        avg_pips = sum(pips) / len(pips)
        avg_ssim = sum(ssim) / len(ssim)
        avg_psnr = sum(psnr) / len(psnr)
        avg_rmse = sum(rmse) / len(rmse)
        avg_l1 = sum(l1) / len(l1)
        #avg_fid = sum(fid) / total_samples

        return avg_pips, avg_ssim, avg_psnr, avg_rmse, avg_l1 #, #avg_fid

    def compute_metrics_test_spade(self, model):
        pips, ssim, psnr, rmse, fid, l1 = [], [], [], [], [], []
        model.eval()
        total_samples = len(self.val_dataloader)
        with torch.no_grad():
            for i, data_i in enumerate(self.val_dataloader):
                image, seg = self.preprocess_input(data_i)
                x_recon, z, vq_output = model(image, seg, evaluation=True)

                input = (image + 1) / 2
                recon = (x_recon + 1) / 2

                input_list = F.pad(input, (2, 2, 2, 2))
                recon_list = F.pad(recon, (2, 2, 2, 2))

                input_gif = rearrange(input_list, '(i j) c f h w -> c f (i h) (j w)', i=1)
                recon_gif = rearrange(recon_list, '(i j) c f h w -> c f (i h) (j w)', i=1)
                path_video = os.path.join(self.root_dir, 'video_results')
                os.makedirs(path_video, exist_ok=True)

                image_path = os.path.join(path_video, f'{i}_input.gif')
                label_path = os.path.join(path_video, f'{i}_recon.gif')
                video_tensor_to_gif(input_gif, image_path)
                video_tensor_to_gif(recon_gif, label_path)

                # SSIM
                ssim_value, _ = self.ssim_3d(input, recon)
                ssim.append(ssim_value.item())
                # PIPS lpips
                d = self.pips_3d(input, recon)
                pips.append(d.mean().item())
                # PSNR, RMSE
                psnr_value = self.psnr_3d(input, recon)
                rmse_value = self.rmse_3d(input, recon)
                psnr.append(psnr_value.item())
                rmse.append(rmse_value.item())

                l1_value = F.l1_loss(x_recon, recon).item()
                l1.append(l1_value)

                if i == 200:
                    break

                # FID
                # fid_value = self.calculate_fid(input1, input2)
                # fid.append(fid_value.item())

        model.train()

        avg_pips = sum(pips) / len(pips)
        avg_ssim = sum(ssim) / len(ssim)
        avg_psnr = sum(psnr) / len(psnr)
        avg_rmse = sum(rmse) / len(rmse)
        avg_l1 = sum(l1) / len(l1)
        # avg_fid = sum(fid) / total_samples

        return avg_pips, avg_ssim, avg_psnr, avg_rmse, avg_l1  # , #avg_fid

    def compute_metrics_during_training(self, image, recon):
        pips, ssim, psnr, rmse, fid, l1 = [], [], [], [], [], []

        input1 = (recon + 1) / 2
        input2 = (image + 1) / 2

        # SSIM
        #ssim_value, _ = self.ssim_3d(input1, input2)
        #ssim.append(ssim_value.item())
        # PIPS lpips
        d = self.pips_3d(input1, input2)
        pips.append(d.mean().item())

        # PSNR, RMSE
        psnr_value = self.psnr_3d(input1, input2)
        rmse_value = self.rmse_3d(input1, input2)
        psnr.append(psnr_value.item())
        rmse.append(rmse_value.item())

        # FID
        #fid_value = self.calculate_fid(input1, input2)
        #fid.append(fid_value.item())
        # l1
        l1_value = F.l1_loss(image, recon).item()
        l1.append(l1_value)

        avg_pips = torch.mean(torch.tensor(pips)).item()
        avg_ssim = torch.mean(torch.tensor(ssim)).item()
        avg_psnr = torch.mean(torch.tensor(psnr)).item()
        avg_rmse = torch.mean(torch.tensor(rmse)).item()
        #avg_fid = torch.mean(torch.tensor(fid)).item()
        avg_l1 = torch.mean(torch.tensor(l1)).item()

        return avg_pips, avg_psnr, avg_rmse, avg_l1 #fid ssim

    def pips_3d(self, img1, img2):
        assert img1.shape == img2.shape
        b, c, d, h, w = img1.shape
        loss_lpips = lpips.LPIPS(net='vgg').to('cuda:0')
        total_loss = 0.0
        for i in range(d):
            total_loss += loss_lpips(img1[:, :, i, :, :], img2[:, :, i, :, :])
        return total_loss / d

    def calculate_fid(self, img1, img2):

        assert img1.shape == img2.shape
        b, c, d, h, w = img1.shape
        total_fid = 0.0
        for i in range(d):
            x1 = img1[:, :, i, :, :].repeat(1, 3, 1, 1)
            x2 = img2[:, :, i, :, :].repeat(1, 3, 1, 1)
            x1 = F.interpolate(x1, size=(299, 299), mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, size=(299, 299), mode='bilinear', align_corners=False)
            total_fid += self.get_fid(x1, x2)

        return total_fid / d

    def get_activations(self, images, model):

        pred = model(images)
        pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1)).squeeze(-1).squeeze(-1)
        return pred

    def get_fid(self, im1, im2):

        act1 = self.get_activations(im1, self.inception_model)
        act2 = self.get_activations(im2, self.inception_model)

        mu1 = torch.mean(act1, dim=0)
        mu2 = torch.mean(act2, dim=0)
        sigma1 = torch.cov(act1.T)
        sigma2 = torch.cov(act2.T)

        fid_score = self.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

        return fid_score

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        diff = mu1 - mu2

        covmean, _ = sqrtm(sigma1.cpu().numpy() @ sigma2.cpu().numpy(), disp=False)
        if not np.isfinite(covmean).all():
            covmean = np.eye(sigma1.shape[0])

        covmean = torch.from_numpy(covmean).to(mu1.device)

        fid = diff.dot(diff) + torch.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    def psnr_3d(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr

    def rmse_3d(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        rmse = torch.sqrt(mse)
        return rmse

    def ssim_3d(self, img1, img2, window_size=11, size_average=True, val_range=None):
        if val_range is None:
            max_val = 255 if torch.max(img1) > 128 else 1
            min_val = -1 if torch.min(img1) < -0.5 else 0
            L = max_val - min_val
        else:
            L = val_range

        padd = 0
        img1 = img1.cpu()
        img2 = img2.cpu()
        (_, channel, depth, height, width) = img1.size()
        window = create_window(window_size, channel).to(img1.device)

        mu1 = F.conv3d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv3d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv3d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv3d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv3d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            return ssim_map.mean(), cs
        else:
            return ssim_map.mean(1).mean(1).mean(1), cs

    def preprocess_input(self, data):

        label = data['label'].cuda().long()
        img = data['image'].cuda().float()
        # create one-hot label map
        bs, _, t, h, w = label.size()
        nc = self.num_classes
        input_label = torch.FloatTensor(bs, nc, t, h, w).zero_().cuda()
        input_semantics = input_label.scatter_(1, label, 1.0)

        return img, input_semantics

    def update_metrics(self, image, recon, cur_iter):
        print(f"--- Iter {cur_iter}: computing PIPS SSIM PSNR RMSE FID ---" )
        cur_pips,  cur_psnr, cur_rmse,  cur_l1 = self.compute_metrics_during_training(image, recon)
        self.update_logs(cur_pips, cur_iter, 'PIPS')
        #self.update_logs(cur_ssim, cur_iter, 'SSIM')
        self.update_logs(cur_psnr, cur_iter, 'PSNR')
        self.update_logs(cur_rmse, cur_iter, 'RMSE')
        self.update_logs(cur_l1, cur_iter, 'L1')

        print("--- Metrics at Iter %s: " % cur_iter, "{:.2f}".format(cur_pips),
              "{:.2f}".format(cur_psnr), "{:.2f}".format(cur_rmse), "{:.2f}".format(cur_l1))

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
        pips, ssim, psnr, rmse, l1 = self.compute_metrics_test(model)
        print("--- PIPS at test : ", "{:.2f}".format(pips))
        print("--- SSIM at test : ", "{:.5f}".format(ssim))
        print("--- PSNR at test : ", "{:.2f}".format(psnr))
        print("--- RMSE at test : ", "{:.2f}".format(rmse))
        #print("--- FID at test : ", "{:.2f}".format(fid))
        print("--- L1 at test : ", "{:.2f}".format(l1))

    def metrics_test_spade(self, model):
        pips, ssim, psnr, rmse, l1 = self.compute_metrics_test_spade(model)
        print("--- PIPS at test : ", "{:.2f}".format(pips))
        print("--- SSIM at test : ", "{:.5f}".format(ssim))
        print("--- PSNR at test : ", "{:.2f}".format(psnr))
        print("--- RMSE at test : ", "{:.2f}".format(rmse))
        #print("--- FID at test : ", "{:.2f}".format(fid))
        print("--- L1 at test : ", "{:.2f}".format(l1))

    def preprocess_input(self, data):

        label = data['label'].cuda().long()
        img = data['image'].cuda().float()
        # create one-hot label map
        bs, _, t, h, w = label.size()
        nc = self.num_classes
        input_label = torch.FloatTensor(bs, nc, t, h, w).zero_().cuda()
        input_semantics = input_label.scatter_(1, label, 1.0)

        return img, input_semantics

    def image_saver(self, fake, real, label, milestone):

        B, C, D, H, W = label.shape
        frame_idx = torch.randint(0, D, [B]).cuda()
        frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
        frames = torch.gather(fake, 2, frame_idx_selected).squeeze(2)
        all_label_list = F.pad(label, (2, 2, 2, 2))
        all_image_list = F.pad(real, (2, 2, 2, 2))
        label_frames = torch.gather(all_label_list, 2, frame_idx_selected).squeeze(2)
        image_frames = torch.gather(all_image_list, 2, frame_idx_selected).squeeze(2)
        path_image_root = os.path.join(self.root_dir, 'images_results')
        os.makedirs(path_image_root, exist_ok=True)
        path_sampled = os.path.join(path_image_root, f'{milestone}-sample.jpg')
        path_label = os.path.join(path_image_root, f'{milestone}-label.jpg')
        path_image = os.path.join(path_image_root, f'{milestone}-image.jpg')
        save_image(frames, path_sampled)
        save_image(label_frames, path_label)
        save_image(image_frames, path_image)


def save_image(image_tensor, path, cols=3):
    B, C, H, W = image_tensor.shape
    plt.figure(figsize=(50, 50))
    for i in range(B):
        plt.subplot(B // cols + 1, cols, i + 1)
        img = image_tensor[i].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())  # 归一化到 [0, 1]
        plt.imshow(img, cmap='gray' if C == 1 else None)
        plt.axis('off')
    plt.savefig(path)
    plt.close()


def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    tensor = ((tensor - tensor.min()) / (tensor.max() - tensor.min())) * 1.0
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs,
                   duration=duration, loop=loop, optimize=optimize)
    return images
