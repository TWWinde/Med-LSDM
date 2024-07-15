import numpy as np
import torch
from torch.utils.data import Dataset
import os
import nibabel as nib
import random
from torchvision import transforms as TR
import torchio as tio

normalization = tio.Compose([
    tio.RescaleIntensity(out_min_max=(-1, 1)),
    tio.Lambda(lambda x: x.float())
])

crop = tio.Compose([
    tio.CropOrPad(target_shape=(256, 256, 32)),

])

TRAIN_TRANSFORMS = tio.Compose([
    # tio.RandomAffine(scales=(0.03, 0.03, 0), degrees=(
    # 0, 0, 3), translation=(4, 4, 0)),
    tio.RandomFlip(axes=(1), flip_probability=0.5),
])


class SemanticMapDataset(Dataset):
    def __init__(self, root_dir: str, sem_map=False):
        super().__init__()
        self.root_dir = root_dir
        self.sem_map = sem_map
        self.Norm = normalization
        self.Crop = crop

        self.seg_paths = self.get_data_files()

    def get_data_files(self):
        subfolder_names = os.listdir(os.path.join(self.root_dir, 'label'))
        seg_paths = [os.path.join(self.root_dir, 'label', subfolder) for subfolder in subfolder_names
                     if subfolder.endswith('nii.gz')]

        return seg_paths

    def __len__(self):
        return len(self.seg_paths)

    def __getitem__(self, idx: int):
        label = tio.ScalarImage(self.seg_paths[idx])
        label = self.Crop(label)
        label = label.data.permute(0, -1, 1, 2)  # torch.Size([1, 32, 256, 256])
        #random_n = torch.rand(1)
        #if random_n[0] > 0.5:
            #label = np.flip(label, 2)

        return {'image': label}
