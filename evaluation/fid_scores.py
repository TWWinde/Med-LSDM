
# ---------------------------------------------------------------------------------------------#
# This code is an adapted version of https://github.com/batmanlab/HA-GAN/tree/master/evaluation
# ---------------------------------------------------------------------------------------------#
import sys
sys.path.append('/misc/no_backups/d1502/medicaldiffusion')
import time
from argparse import ArgumentParser
from collections import OrderedDict
import numpy as np
from scipy import linalg
import torch
import torch.nn as nn
from evaluation.resnet3D import resnet50
import os
from PIL import Image
from torch.utils.data import Dataset

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True

parser = ArgumentParser()
parser.add_argument('--path', type=str, default='/data/private/autoPET/medicaldiffusion_results/test_results/ddpm/AutoPET/output_with_segconv_64out/video_results')
parser.add_argument('--real_suffix', type=str, default='eval_600_size_256_resnet50_fold')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--dims', type=int, default=2048)
parser.add_argument('--latent_dim', type=int, default=1024)
parser.add_argument('--basename', type=str, default="256_1024_Alpha_SN_v4plus_4_l1_GN_threshold_600_fold")
parser.add_argument('--fold', type=int)
parser.add_argument('--num_samples', type=int, default=16)


def trim_state_dict_name(ckpt):
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


class Flatten(torch.nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)


def get_activations_from_dataloader(model, data_loader, args):

    pred_arr = np.empty((args.num_samples, args.dims))
    for i, batch in enumerate(data_loader):
        if i % 10 == 0:
            print('\rPropagating batch %d' % i, end='', flush=True)
        batch = batch.float().cuda()
        with torch.no_grad():
            pred = model(batch)

        if i* args.batch_size > pred_arr.shape[0]:
            pred_arr[i * args.batch_size:] = pred.cpu().numpy()
        else:
            pred_arr[i * args.batch_size:(i + 1) * args.batch_size] = pred.cpu().numpy()
    print(' done')
    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def post_process(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def get_feature_extractor():
    model = resnet50(shortcut_type='B')
    model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), Flatten())  # (N, 512)
    # ckpt from https://drive.google.com/file/d/1399AsrYpQDi1vq6ciKRQkfknLsQQyigM/view?usp=sharing
    ckpt = torch.load("/data/private/autoPET/medicaldiffusion_results/pretrain/resnet_50.pth")
    ckpt = trim_state_dict_name(ckpt["state_dict"])
    model.load_state_dict(ckpt)  # No conv_seg module in ckpt
    model = nn.DataParallel(model).cuda()
    model.eval()
    print("Feature extractor weights loaded")
    return model


def calculate_fid_real(args, data_loader):
    """Calculates the FID of two paths"""
    assert os.path.exists(args.path)

    model = get_feature_extractor()

    #args.num_samples = len(dataset)
    #print("Number of samples:", args.num_samples)

    act = get_activations_from_dataloader(model, data_loader, args)
    np.save("/data/private/autoPET/medicaldiffusion_results/results/fid/pred_arr_real_train_size_" + str(
        args.img_size) + "_resnet50_GSP_fold" + str(args.fold) + ".npy", act)
    # np.save("./results/fid/pred_arr_real_train_600_size_"+str(args.img_size)+"_resnet50_fold"+str(args.fold)+".npy", act)
    # calculate_mmd(args, act)
    m, s = post_process(act)

    m1 = np.load("./results/fid/m_real_" + args.real_suffix + str(args.fold) + ".npy")
    s1 = np.load("./results/fid/s_real_" + args.real_suffix + str(args.fold) + ".npy")

    fid_value = calculate_frechet_distance(m1, s1, m, s)
    print('FID: ', fid_value)
    # np.save("./results/fid/m_real_train_600_size_"+str(args.img_size)+"_resnet50_fold"+str(args.fold)+".npy", m)
    # np.save("./results/fid/s_real_train_600_size_"+str(args.img_size)+"_resnet50_fold"+str(args.fold)+".npy", s)
    # np.save("./results/fid/m_real_train_size_"+str(args.img_size)+"_resnet50_GSP_fold"+str(args.fold)+".npy", m)
    # np.save("./results/fid/s_real_train_size_"+str(args.img_size)+"_resnet50_GSP_fold"+str(args.fold)+".npy", s)


def calculate_fid_fake(args):
    # assert os.path.exists("./results/fid/m_real_"+args.real_suffix+str(args.fold)+".npy")
    act = generate_samples(args)
    m2, s2 = post_process(act)

    m1 = np.load("./results/fid/m_real_" + args.real_suffix + str(args.fold) + ".npy")
    s1 = np.load("./results/fid/s_real_" + args.real_suffix + str(args.fold) + ".npy")

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    print('FID: ', fid_value)


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, real=True):

        self.folder_path = folder_path
        self.real = real
        self.head = 'image' if real else 'sample'
        self.image_files = [f for f in os.listdir(folder_path) if
                                 os.path.isfile(os.path.join(folder_path, f)) and self.head in f]

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, idx):

        img_name = self.image_files[idx]
        img_path = os.path.join(self.folder_path, img_name)
        with Image.open(img_path) as img:
            frames = []
            try:
                while True:
                    # 将当前帧转换为 RGB 格式（必要时）
                    # 将帧转换为 NumPy 数组
                    frame_array = np.array(img)
                    frames.append(frame_array)
                    # 移动到下一帧
                    img.seek(img.tell() + 1)
            except EOFError:
                # 所有帧处理完毕
                pass

            gif_array = np.stack(frames, axis=0)
        image = np.expand_dims(np.array(gif_array), axis=0)
        print(image.shape)

        return image


if __name__ == '__main__':
    args = parser.parse_args()
    start_time = time.time()
    path = "/data/private/autoPET/medicaldiffusion_results/test_results/ddpm/AutoPET/output_with_segconv/video_results"
    dataset_real = ImageFolderDataset(folder_path=path, real=True)
    data_loader_real = torch.utils.data.DataLoader(dataset_real, batch_size=32, shuffle=False, num_workers=4)
    dataset_fake = ImageFolderDataset(folder_path=path, real=True)
    data_loader_fake = torch.utils.data.DataLoader(dataset_fake, batch_size=32, shuffle=False, num_workers=4)
    calculate_fid_real(args, data_loader_real)
    calculate_fid_fake(args)
    print("Done. Using", (time.time() - start_time) // 60, "minutes.")
