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


class DUKEDataset(Dataset):

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

        mr_names = [os.path.join(self.root_dir, subfolder) for subfolder in os.listdir(os.path.join(self.root_dir))
                    if subfolder.endswith('nii.gz')]
        if self.sem_map:
            label_names, mr_names_ = [], []
            for mr_path in mr_names:
                label_path = mr_path.replace('mr', 'label')
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
            return {'image': img.data.permute(0, -1, 1, 2)}
