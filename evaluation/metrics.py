import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import lpips
import torch
import torch.nn.functional as F
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


class Metrics:
    def __init__(self, path, dataloader_val, num_classes=37):
        self.val_dataloader = dataloader_val
        self.root_dir = path
        self.path_to_save = os.path.join(self.root_dir, 'metrics')
        self.path_to_save_PIPS = os.path.join(self.path_to_save, "PIPS")
        self.path_to_save_SSIM = os.path.join(self.path_to_save, "SSIM")
        self.path_to_save_PSNR = os.path.join(self.path_to_save, "PSNR")
        self.path_to_save_RMSE = os.path.join(self.path_to_save, "RMSE")
        self.path_to_save_FID = os.path.join(self.path_to_save, "FID")

        Path(self.path_to_save_SSIM).mkdir(parents=True, exist_ok=True)
        Path(self.path_to_save_PIPS).mkdir(parents=True, exist_ok=True)
        Path(self.path_to_save_PSNR).mkdir(parents=True, exist_ok=True)
        Path(self.path_to_save_RMSE).mkdir(parents=True, exist_ok=True)
        Path(self.path_to_save_FID).mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        #self.inception_model = inception_v3(pretrained=True, transform_input=False).eval().cuda()

    def sample(self, model):
        model.eval()
        with torch.no_grad():
            for i, data_i in enumerate(self.val_dataloader):
                label_save = data_i['label']
                image, label = self.preprocess_input(data_i)
                generated = model.sample(cond=label)
                generated = (generated + 1) / 2
                real = (image + 1) / 2
                break

        return generated, real, label_save

    def compute_metrics(self, model, encoder=None):
        pips, ssim, psnr, rmse, fid = [], [], [], [], []
        model.eval()
        total_samples = len(self.val_dataloader)
        with torch.no_grad():
            for i, data_i in enumerate(self.val_dataloader):
                image, seg = self.preprocess_input(data_i)
                if encoder is not None:
                    with torch.no_grad():
                        seg = encoder.encode(seg, quantize=False, include_embeddings=True)
                        # normalize to -1 and 1
                        seg = ((seg - encoder.codebook.embeddings.min()) /
                               (encoder.codebook.embeddings.max() -
                                encoder.codebook.embeddings.min())) * 2.0 - 1.0
                        assert seg.size()[-1] == 64  # torch.Size([1, 8, 8, 64, 64])

                generated = model.sample(cond=seg)
                input1 = (generated + 1) / 2
                input2 = (image + 1) / 2
                # SSIM
                ssim_value, _ = self.ssim_3d(input1, input2)
                ssim.append(ssim_value.item())
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
                break
        model.train()

        avg_pips = sum(pips) / len(pips)
        avg_ssim = sum(ssim) / len(ssim)
        avg_psnr = sum(psnr) / len(psnr)
        avg_rmse = sum(rmse) / len(rmse)
        #avg_fid = sum(fid) / total_samples

        return avg_pips, avg_ssim, avg_psnr, avg_rmse #, #avg_fid

    def compute_metrics_test(self, model, encoder=None):
        pips, ssim, psnr, rmse, fid = [], [], [], [], []
        model.eval()
        total_samples = len(self.val_dataloader)
        save_npy=False
        save_slice_image = False
        save_gif = False

        with torch.no_grad():
            for i, data_i in enumerate(self.val_dataloader):
                label_save = data_i["label"]
                image, seg = self.preprocess_input(data_i)
                if encoder is not None:
                    with torch.no_grad():
                        seg = encoder.encode(seg, quantize=False, include_embeddings=True)
                        # normalize to -1 and 1
                        seg = ((seg - encoder.codebook.embeddings.min()) /
                               (encoder.codebook.embeddings.max() -
                                encoder.codebook.embeddings.min())) * 2.0 - 1.0
                        assert seg.size()[-1] == 64  # torch.Size([1, 8, 8, 64, 64])

                generated = model.sample(cond=seg, get_middle_process=True)  # (-1,1)
                input1 = generated     #(generated + 1) / 2
                input2 = image      #(image + 1) / 2
                generated_np = input1.cpu().numpy()
                image_np = input2.cpu().numpy()
                label_np = label_save.cpu().numpy()
                # print(generated_np.shape, image_np.shape, label_np.shape) (1, 1, 32, 256, 256) (1, 1, 32, 256, 256) (1, 1, 32, 256, 256)
                path_video = os.path.join(self.root_dir, 'video_results')
                os.makedirs(path_video, exist_ok=True)

                if save_npy:
                    sample_np_path = os.path.join(path_video, 'fake', f'{i}_sample.npy')
                    image_np_path = os.path.join(path_video, 'real', f'{i}_image.npy')
                    label_np_path = os.path.join(path_video, 'label', f'{i}_label.npy')
                    os.makedirs(os.path.join(path_video, 'fake'), exist_ok=True)
                    os.makedirs(os.path.join(path_video, 'real'), exist_ok=True)
                    os.makedirs(os.path.join(path_video, 'label'), exist_ok=True)
                    np.save(sample_np_path, generated_np, allow_pickle=True, fix_imports=True)
                    np.save(image_np_path, image_np, allow_pickle=True, fix_imports=True)
                    np.save(label_np_path, label_np, allow_pickle=True, fix_imports=True)
                    if i > 50:
                        save_npy = False

                if save_slice_image:
                    slice_index = 16  # Specify which slice you want to save

                    # Path to save images
                    path_images = os.path.join(self.root_dir, 'slices')
                    os.makedirs(path_images, exist_ok=True)

                    # For generated_np
                    plt.imshow(generated_np[0, 0, slice_index, :, :], cmap='gray')  # Grayscale image
                    plt.axis('off')
                    plt.savefig(os.path.join(path_images, f'{i}_generated_slice_{slice_index}.png'),
                                bbox_inches='tight',
                                pad_inches=0)
                    plt.close()

                    # For image_np
                    plt.imshow(image_np[0, 0, slice_index, :, :], cmap='gray')  # Grayscale image
                    plt.axis('off')
                    plt.savefig(os.path.join(path_images, f'{i}_image_slice_{slice_index}.png'), bbox_inches='tight',
                                pad_inches=0)
                    plt.close()

                    # For label_np (assuming RGB)
                    plt.imshow(label_np[0, 0, slice_index, :, :])  # Color image, transpose (H, W, C)
                    plt.axis('off')
                    plt.savefig(os.path.join(path_images, f'{i}_label_slice_{slice_index}.png'), bbox_inches='tight',
                                pad_inches=0)
                    plt.close()

                if save_gif:
                    all_videos_list = F.pad(generated, (2, 2, 2, 2))
                    all_label_list = F.pad(label_save, (2, 2, 2, 2))
                    all_image_list = F.pad(image, (2, 2, 2, 2))

                    sample_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i=1)
                    label_gif = rearrange(all_label_list, '(i j) c f h w -> c f (i h) (j w)', i=1)
                    image_gif = rearrange(all_image_list, '(i j) c f h w -> c f (i h) (j w)', i=1)

                    sample_path = os.path.join(path_video, f'{i}_sample.gif')
                    image_path = os.path.join(path_video, f'{i}_image.gif')
                    label_path = os.path.join(path_video, f'{i}_label.gif')
                    video_tensor_to_gif(sample_gif, sample_path)
                    video_tensor_to_gif(image_gif, image_path)
                    video_tensor_to_gif(label_gif, label_path)

                    if i > 50:
                        save_gif = False

                # SSIM
                ssim_value, _ = self.ssim_3d(input1, input2)
                ssim.append(ssim_value.item())
                # PIPS lpips
                d = self.pips_3d(input1, input2)
                pips.append(d.mean().item())
                # PSNR, RMSE
                psnr_value = self.psnr_3d(input1, input2)
                rmse_value = self.rmse_3d(input1, input2)
                psnr.append(psnr_value.item())
                rmse.append(rmse_value.item())

                if i == 200:
                    print("test finished")
                    break

        avg_pips = sum(pips) / len(pips)
        avg_ssim = sum(ssim) / len(ssim)
        avg_psnr = sum(psnr) / len(psnr)
        avg_rmse = sum(rmse) / len(rmse)
        #avg_fid = sum(fid) / total_samples

        return avg_pips, avg_ssim, avg_psnr, avg_rmse #, #avg_fid


    def pips_3d(self, img1, img2):
        assert img1.shape == img2.shape
        b, c, d, h, w = img1.shape
        loss_lpips = lpips.LPIPS(net='vgg').to('cuda:0')
        total_loss = 0.0
        for i in range(d):
            total_loss += loss_lpips(img1[:, :, i, :, :], img2[:, :, i, :, :])
        return total_loss / d

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
        (_, channel, depth, height, width) = img1.size()
        window = create_window(window_size, channel=channel).to(img1.device)

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

    def calculate_fid(self, img1, img2):
        def get_activations(images, model):
            pred = model(images)
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1)).squeeze(-1).squeeze(-1)
            return pred

        act1 = get_activations(img1, self.inception_model)
        act2 = get_activations(img2, self.inception_model)

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

    def preprocess_input(self, data):

        label = data['label'].cuda().long()
        img = data['image'].cuda().float()
        # create one-hot label map
        bs, _, t, h, w = label.size()
        nc = self.num_classes
        input_label = torch.FloatTensor(bs, nc, t, h, w).zero_().cuda()
        input_semantics = input_label.scatter_(1, label, 1.0)

        return img, input_semantics

    def update_metrics(self, model, cur_iter, encoder=None):
        print("--- Iter %s: computing PIPS SSIM PSNR RMSE FID ---" % (cur_iter))
        cur_pips, cur_ssim, cur_psnr, cur_rmse= self.compute_metrics(model, encoder)
        self.update_logs(cur_pips, cur_iter, 'PIPS')
        self.update_logs(cur_ssim, cur_iter, 'SSIM')
        self.update_logs(cur_psnr, cur_iter, 'PSNR')
        self.update_logs(cur_rmse, cur_iter, 'RMSE')
        #self.update_logs(cur_fid, cur_iter, 'FID')

        print("--- Metrics at Iter %s: " % cur_iter, "{:.2f}".format(cur_pips), "{:.2f}".format(cur_ssim),
              "{:.2f}".format(cur_psnr), "{:.2f}".format(cur_rmse))   #, "{:.2f}".format(cur_fid))

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

    def metrics_test(self, model, encoder=None):
        pips, ssim, psnr, rmse= self.compute_metrics_test(model, encoder=encoder)
        print("--- PIPS at test : ", "{:.2f}".format(pips))
        print("--- SSIM at test : ", "{:.5f}".format(ssim))
        print("--- PSNR at test : ", "{:.2f}".format(psnr))
        print("--- RMSE at test : ", "{:.2f}".format(rmse))
        #print("--- FID at test : ", "{:.2f}".format(fid))

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


