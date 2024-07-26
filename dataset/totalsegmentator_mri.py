import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchio as tio
import os
import nibabel as nib
import random
from torchvision import transforms as TR


def preprocess_input(opt, data, test=False):
    data['label'] = data['label'].long()
    data['label'] = data['label'].cuda()
    label_map = data['label']
    bs, _, h, w = label_map.size()
    nc = opt.semantic_nc
    if opt.gpu_ids != "-1":
        input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    else:
        input_label = torch.FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)
    if test:
        return data['image'], data['ct_image'], input_semantics
    else:
        return data['image'], input_semantics


class Transform:
    def __init__(self, target_depth=32, label=False):
        self.target_depth = target_depth
        self.label = label

    def normalization(self, data):
        arr_min = data.min()
        arr_max = data.max()
        data_normalized = (data - arr_min) / (arr_max - arr_min)
        data_normalized = data_normalized * 2 - 1
        return data_normalized

    def randomdepthcrop(self, img, seg=None):
        h, w, d = img.shape

        if self.label:
            assert img.shape == seg.shape
            target_d = self.target_depth

            d_start = np.random.randint(2, d - target_d + 1)
            cropped_img = img[:, :, d_start:d_start + target_d]
            cropped_seg = seg[:, :, d_start:d_start + target_d]

            return cropped_img, cropped_seg
        else:
            target_d = self.target_depth

            d_start = np.random.randint(0, d - target_d + 1)
            cropped_img = img[:, :, d_start:d_start + target_d]

            return cropped_img

    def pad_and_concatenate_image(self, array):

        c, a, b = array.shape[0], array.shape[1], array.shape[2]

        max_length = max(a, b)

        if a < max_length:
            pad1 = np.tile(array[:, 0:1, :], (1, (max_length - a) // 2, 1))
            pad2 = np.tile(array[:, -1:, :], (1, (max_length - a) - (max_length - a) // 2, 1))
            padded_array = np.concatenate((pad1, array, pad2), axis=1)
        else:
            padded_array = array

        if b < max_length:
            pad1 = np.tile(padded_array[:, :, 0:1], (1, 1, (max_length - b) // 2))
            pad2 = np.tile(padded_array[:, :, -1:], (1, 1, (max_length - b) - (max_length - b) // 2))
            padded_array = np.concatenate((pad1, padded_array, pad2), axis=2)

        return padded_array

    def __call__(self, img, seg=None):
        if self.label:
            img, seg = self.randomdepthcrop(img, seg)
            final_img = self.normalization(img)

            return final_img, seg
        else:
            img = self.randomdepthcrop(img)
            final_img = self.normalization(img)

            return final_img


PREPROCESSING_TRANSORMS = tio.Compose([
    tio.RescaleIntensity(out_min_max=(-1, 1)),
    tio.CropOrPad(target_shape=(256, 256, 32)),
    tio.Lambda(lambda x: x.float())
])

TRAIN_TRANSFORMS = tio.Compose([
    # tio.RandomAffine(scales=(0.03, 0.03, 0), degrees=(
    # 0, 0, 3), translation=(4, 4, 0)),
    tio.RandomFlip(axes=(1), flip_probability=0.5),
])

normalization = tio.Compose([
    tio.RescaleIntensity(out_min_max=(-1, 1)),
    tio.Lambda(lambda x: x.float())
])

crop = tio.Compose([
    tio.CropOrPad(target_shape=(256, 256, 32)),

])


class TotalSegmentator_mri_Dataset(Dataset):
    """
    root_dir: root_dir: /data/private/autoPET/SynthRad2024_withoutmask/train  for training
                       /data/private/autoPET/SynthRad2024_withoutmask/test  for test and validation

    """

    def __init__(self, root_dir: str, sem_map=False):
        super().__init__()
        self.root_dir = root_dir
        self.sem_map = sem_map
        self.Norm = normalization
        self.Crop = crop
        if self.sem_map:
            self.mr_paths, self.label_paths = self.get_data_files()
        else:
            self.mr_paths = self.get_data_files()

    def get_data_files(self):

        mr_names = [os.path.join(self.root_dir, 'mr', subfolder) for subfolder in
                    os.listdir(os.path.join(self.root_dir, 'mr'))
                    if subfolder.endswith('nii.gz')]
        if self.sem_map:
            label_names, mr_names_ = [], []
            for mr_path in mr_names:
                label_path = mr_path.replace('/mr/', '/label/')
                if os.path.exists(mr_path) and os.path.exists(label_path):
                    mr_names_.append(mr_path)
                    label_names.append(label_path)

            return mr_names_, label_names
        else:

            return mr_names

    def __len__(self):
        return len(self.mr_paths)

    def __getitem__(self, idx: int):
        img = tio.ScalarImage(self.mr_paths[idx])
        img = self.Crop(img)
        img = self.Norm(img)
        if self.sem_map:
            label = tio.ScalarImage(self.label_paths[idx])
            label = self.Crop(label)

            return {'image': img.data.permute(0, -1, 1, 2), 'label': label.data.permute(0, -1, 1, 2)}
        else:
            return {'image': img.permute(0, -1, 1, 2)}


