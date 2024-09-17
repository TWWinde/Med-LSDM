# ---------------------------------------------------------------------------------------------#
# Some code is an adapted version of https://github.com/batmanlab/HA-GAN/tree/master/evaluation
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
import pytorch_msssim
import lpips
from fid_folder.inception import InceptionV3
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True

parser = ArgumentParser()
parser.add_argument('--path', type=str, default='/data/private/autoPET/medicaldiffusion_results/test_results/ddpm/AutoPET/output_with_segconv_64out/video_results')
parser.add_argument('--real_suffix', type=str, default='eval_600_size_256_resnet50_fold')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--dims', type=int, default=2048)
parser.add_argument('--latent_dim', type=int, default=1024)
parser.add_argument('--basename', type=str, default="256_1024_Alpha_SN_v4plus_4_l1_GN_threshold_600_fold")
parser.add_argument('--fold', type=int)
parser.add_argument('--num_samples', type=int, default=1000)


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

        start_idx = i * args.batch_size
        end_idx = start_idx + pred.shape[0]

        if end_idx > pred_arr.shape[0]:
            end_idx = pred_arr.shape[0]

        pred_arr[start_idx:end_idx] = pred.cpu().numpy()[:end_idx - start_idx]

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
    model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
                                   Flatten())  # (N, 512)
    # ckpt from https://drive.google.com/file/d/1399AsrYpQDi1vq6ciKRQkfknLsQQyigM/view?usp=sharing
    ckpt = torch.load("/data/private/autoPET/medicaldiffusion_results/pretrain/resnet_50.pth")
    ckpt = trim_state_dict_name(ckpt["state_dict"])
    model.load_state_dict(ckpt)  # No conv_seg module in ckpt
    model = nn.DataParallel(model).cuda()
    model.eval()
    print("Feature extractor weights loaded")
    return model


def calculate_fid_real(args, data_loader1, data_loader2):
    """Calculates the FID of two paths"""

    model = get_feature_extractor()

    act1 = get_activations_from_dataloader(model, data_loader1, args)
    act2 = get_activations_from_dataloader(model, data_loader2, args)

    m1, s1 = post_process(act1)
    m2, s2 = post_process(act2)


    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    print('FID: ', fid_value)


def calculate_fid(args, data_loader):
    assert os.path.exists(args.path)

    model = get_feature_extractor()

    act = get_activations_from_dataloader(model, data_loader, args)

    m, s = post_process(act)

    return m, s


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, real=True):

        self.folder_path = folder_path
        self.real = real
        self.head = 'real' if self.real else 'fake'
        self.image_files = [f for f in os.listdir(os.path.join(self.folder_path, self.head)) if f.endswith(".npy")]

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        total_path = os.path.join(self.folder_path, self.head, img_path)
        img = np.load(total_path)
        img = np.squeeze(img, axis=0)
        #img = (img-0.5)*2
        #print(img.shape)
        #(1, 32, 256, 256)
        return img


class ImageFolderDataset_baseline_real(Dataset):
    def __init__(self, folder_path):

        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(self.folder_path) if f.endswith(".npy")]

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        total_path = os.path.join(self.folder_path, img_path)
        img = np.load(total_path).transpose((1, 0, 2, 3))

        return img


class ImageFolderDataset_baseline_fake(Dataset):
    def __init__(self, folder_path):

        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(self.folder_path) if f.endswith(".npy")]

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        total_path = os.path.join(self.folder_path, img_path)
        img = np.load(total_path).transpose((1, 0, 2, 3))
        return img


def compute_metrics_3d(path_real_root, path_fake_root):
    pool1, pool2 = [], []
    pips, ssim, psnr, rmse = [], [], [], []
    loss_fn_alex = lpips.LPIPS(net='vgg')
    loss_fn_alex = loss_fn_alex.to('cuda:0')
    path_list = [i for i in sorted(os.listdir(path_real_root)) if i.endswith(".npy")]
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model_inc = InceptionV3([block_idx])
    model_inc.cuda()
    for item in path_list:
        path_fake = os.path.join(path_fake_root, item)
        path_real = os.path.join(path_real_root, item)
        try:
            input1 = np.load(path_real).transpose((2, 3, 0, 1))
            input1 = np.expand_dims(input1, axis=0)
            input2 = np.load(path_fake).transpose((-1, 0, 1))
            input2 = np.expand_dims(input2, axis=0)
            input2 = np.expand_dims(input2, axis=0)
            #print(input1.shape)
            #print(input2.shape)
            # (256, 256, 1, 32)
            # (256, 256, 32)
            for i in range(input1.shape[2]):
                input3 = torch.tensor(input1[:, :, i, :, :], dtype=torch.float32).to('cuda:0')
                input4 = torch.tensor(input2[:, :, i, :, :], dtype=torch.float32).to('cuda:0')
                # input3 = input3.unsqueeze(0).unsqueeze(0).to('cuda:0')  # (1, 1, 256, 256)
                # input4 = input4.unsqueeze(0).unsqueeze(0).to('cuda:0')

                ssim_value = pytorch_msssim.ssim(input3, input4)
                ssim.append(ssim_value.mean().item())
                # PIPS lpips
                d = loss_fn_alex(input3, input4)
                pips.append(d.mean().item())
                # PSNR, RMSE
                mse = torch.nn.functional.mse_loss(input3, input4)
                max_pixel_value = 1.0
                psnr_value = 10 * torch.log10((max_pixel_value ** 2) / mse)
                rmse_value = torch.sqrt(mse)
                psnr.append(psnr_value.mean().item())
                rmse.append(rmse_value.mean().item())
                input3_rgb = input3.expand(-1, 3, -1, -1)
                input4_rgb = input4.expand(-1, 3, -1, -1)
                pool_real = model_inc(input3_rgb.float())[0][:, :, 0, 0]
                pool1 += [pool_real]
                pool_fake = model_inc(input4_rgb.float())[0][:, :, 0, 0]
                pool2 += [pool_fake]
        except:
            print("wrong data")


    total_samples = len(pips)
    real_pool = torch.cat(pool1, 0)
    mu_real, sigma_real = torch.mean(real_pool, 0), torch_cov(real_pool, rowvar=False)
    fake_pool = torch.cat(pool2, 0)
    mu_fake, sigma_fake = torch.mean(real_pool, 0), torch_cov(fake_pool, rowvar=False)
    fid = numpy_calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake, eps=1e-6)
    avg_pips = sum(pips) / total_samples
    avg_ssim = sum(ssim) / total_samples
    avg_psnr = sum(psnr) / total_samples
    avg_rmse = sum(rmse) / total_samples
    avg_pips = np.array(avg_pips)
    avg_ssim = np.array(avg_ssim)
    avg_psnr = np.array(avg_psnr)
    avg_rmse = np.array(avg_rmse)
    #fid_value = fid_score.calculate_fid_given_paths([path_real_root, path_fake_root], batch_size=50, device='cuda', dims=2048)
    #print(f"FID: {fid_value}")
    print(avg_pips, avg_ssim, avg_psnr, avg_rmse, fid)


def numpy_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        Taken from https://github.com/bioinf-jku/TTUR
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representive data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representive data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1, sigma1, mu2, sigma2 = mu1.detach().cpu().numpy(), sigma1.detach().cpu().numpy(), mu2.detach().cpu().numpy(), sigma2.detach().cpu().numpy()

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
            #print('wat')
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                #print('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        out = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        return out


def torch_cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()




if __name__ == '__main__':
    evaluate_our = True
    if evaluate_our:
        args = parser.parse_args()
        start_time = time.time()
        path = "/data/private/autoPET/medicaldiffusion_results/test_results/ddpm/AutoPET/output_with_segconv_64out/video_results"

        dataset_real = ImageFolderDataset(folder_path=path, real=True)
        print(len(dataset_real))
        data_loader_real = torch.utils.data.DataLoader(dataset_real, batch_size=10, shuffle=False, num_workers=4)
        dataset_fake = ImageFolderDataset(folder_path=path, real=False)
        data_loader_fake = torch.utils.data.DataLoader(dataset_fake, batch_size=10, shuffle=False, num_workers=4)
        # calculate_fid(args, data_loader_real, data_loader_fake)
        m1, s1 = calculate_fid(args, data_loader_real)
        m2, s2 = calculate_fid(args, data_loader_fake)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        print('FID: ', fid_value)
        print("Done. Using", (time.time() - start_time) // 60, "minutes.")
    else:
        args = parser.parse_args()
        start_time = time.time()
        path2 = "/data/private/autoPET/ddim-AutoPET-256-segguided/fake_npy"
        path1 = "/data/private/autoPET/ddim-AutoPET-256-segguided/real_npy"

        dataset_real = ImageFolderDataset_baseline_real(folder_path=path1)
        print(len(dataset_real))
        data_loader_real = torch.utils.data.DataLoader(dataset_real, batch_size=10, shuffle=False, num_workers=4)
        dataset_fake = ImageFolderDataset_baseline_fake(folder_path=path2)
        data_loader_fake = torch.utils.data.DataLoader(dataset_fake, batch_size=10, shuffle=False, num_workers=4)
        # calculate_fid(args, data_loader_real, data_loader_fake)
        m1, s1 = calculate_fid(args, data_loader_real)
        m2, s2 = calculate_fid(args, data_loader_fake)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        print('FID: ', fid_value)
        print("Done. Using", (time.time() - start_time) // 60, "minutes.")


