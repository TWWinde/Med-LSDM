import os
from pathlib import Path
import torch
from matplotlib import pyplot as plt


class results_saver():
    def __init__(self, save_path, val_dataloader):
        self.val_dataloader = val_dataloader
        self.path_to_save_PIPS = os.path.join(save_path, "PIPS")
        self.path_to_save_SSIM = os.path.join(save_path, "SSIM")
        self.path_to_save_PSNR = os.path.join(save_path, "PSNR")
        self.path_to_save_RMSE = os.path.join(save_path, "RMSE")

        Path(self.path_to_save_SSIM).mkdir(parents=True, exist_ok=True)
        Path(self.path_to_save_PIPS).mkdir(parents=True, exist_ok=True)
        Path(self.path_to_save_PSNR).mkdir(parents=True, exist_ok=True)
        Path(self.path_to_save_RMSE).mkdir(parents=True, exist_ok=True)

    def __call__(self, model, save_path):
        pips, ssim, psnr, rmse = [], [], [], []
        with torch.no_grad():
            for i, data_i in enumerate(self.val_dataloader):
                image, label = self.preprocess_input(data_i)
                sampled_image = model.sample(cond=label)
                all_videos_list = list(sampled_image)
                all_videos_list = torch.cat(all_videos_list, dim=0)

        assert len(label) == len(generated)
        for i in range(len(label)):
            im = tens_to_lab_color(label[i], self.num_cl)
            self.save_im(im, "label", name[i])
            im = tens_to_im(generated[i]) * 255
            self.save_im(im, "image", name[i])

    def preprocess_input(self, data):

        # move to GPU and change data types
        data = data['label'].long()
        img = data['image']
        # create one-hot label map
        label_map = data
        bs, _, t, h, w = label_map.size()
        nc = self.num_classes
        input_label = torch.FloatTensor(bs, nc, t, h, w).zero_().cuda()
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        return img, input_semantics

    def save_im(self, im, mode, name):
        im = Image.fromarray(im.astype(np.uint8))
        #print(name.split("/")[-1])
        im.save(os.path.join(self.path_to_save[mode], name.split("/")[-1]).replace('.jpg', '.png'))

        with torch.no_grad():
            milestone = self.step // self.save_and_sample_every
            num_samples = self.num_sample_rows ** 2
            batches = num_to_groups(num_samples, self.batch_size)

            all_videos_list = list(map(lambda n: self.ema_model.sample(cond=seg, batch_size=n), batches))
            all_videos_list = torch.cat(all_videos_list, dim=0)

        all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))

        one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i=self.num_sample_rows)
        path_video = os.path.join(self.results_folder, 'video_results')
        os.makedirs(path_video, exist_ok=True)
        video_path = os.path.join(path_video, f'{milestone}.gif')
        video_tensor_to_gif(one_gif, video_path)
        log = {**log, 'sample': video_path}

        # Selects one random 2D image from each 3D Image
        B, C, D, H, W = all_videos_list.shape
        frame_idx = torch.randint(0, D, [B]).cuda()
        frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
        frames = torch.gather(all_videos_list, 2, frame_idx_selected).squeeze(2)
        all_label_list = F.pad(label, (2, 2, 2, 2))
        all_image_list = F.pad(input_image, (2, 2, 2, 2))
        label_frames = torch.gather(all_label_list, 2, frame_idx_selected).squeeze(2)
        image_frames = torch.gather(all_image_list, 2, frame_idx_selected).squeeze(2)
        path_image_root = os.path.join(self.results_folder, 'images_results')
        os.makedirs(path_image_root, exist_ok=True)
        path_sampled = os.path.join(path_image_root, f'{milestone}-sample.jpg')
        path_label = os.path.join(path_image_root, f'{milestone}-label.jpg')
        path_image = os.path.join(path_image_root, f'{milestone}-image.jpg')

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

        save_image(frames, path_sampled)
        save_image(label_frames, path_label)
        save_image(image_frames, path_image)


