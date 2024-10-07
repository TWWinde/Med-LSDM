import random

from torch.utils.data import Dataset
import torchio as tio
import os
from typing import Optional
import argparse


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


class DUKEDataset_unet(Dataset):

    def __init__(self, root_dir: str, sem_map=False, percentage=1.0):
        super().__init__()
        self.root_dir = root_dir
        self.sem_map = sem_map
        self.Norm = normalization
        self.Crop = crop
        self.percentage = percentage

        self.mr_fake_names, self.mr_real_names, self.label_names = self.get_data_files()

    def get_data_files(self):

        mr_real_names = [os.path.join(self.root_dir, "baseline_nifti", subfolder) for subfolder in sorted(os.listdir(os.path.join(self.root_dir,"baseline_nifti")))
                    if subfolder.endswith('nii.gz')]
        mr_fake_names = [os.path.join(self.root_dir, "baseline_nifti", subfolder) for subfolder in sorted(os.listdir(os.path.join(self.root_dir,"baseline_nifti")))
                    if subfolder.endswith('nii.gz')]
        label_names = [os.path.join(self.root_dir, "baseline_seg_nifti", subfolder) for subfolder in sorted(os.listdir(os.path.join(self.root_dir,"final_label")))
                    if subfolder.endswith('nii.gz')]

        return mr_fake_names, mr_real_names, label_names

    def __len__(self):
        return len(self.mr_real_names)

    def __getitem__(self, idx: int):

        mr_real = tio.ScalarImage(self.mr_real_names[idx])
        mr_real = self.Crop(mr_real)
        mr_real = self.Norm(mr_real)
        mr_fake = tio.ScalarImage(self.mr_fake_names[idx])
        mr_fake = self.Crop(mr_fake)
        mr_fake = self.Norm(mr_fake)

        label = tio.ScalarImage(self.label_names[idx])
        label = self.Crop(label)

        return {'mr_real': mr_real.data.permute(0, -1, 1, 2).float(), 'mr_fake': mr_fake.data.permute(0, -1, 1, 2).float(), 'label': label.data.permute(0, -1, 1, 2).float()}
