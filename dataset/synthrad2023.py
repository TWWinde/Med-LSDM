import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchio as tio
import os


class RandomDepthCrop:
    def __init__(self, target_depth):
        self.target_depth = target_depth

    def __call__(self, img):
        _, d, h, w = img.shape
        target_d = self.target_depth

        d_start = np.random.randint(0, d - target_d + 1)
        cropped_img = img[:, d_start:d_start + target_d, :, :]

        return cropped_img


PREPROCESSING_TRANSORMS = tio.Compose([
    tio.RescaleIntensity(out_min_max=(-1, 1)),
    RandomDepthCrop(target_depth=32),
    tio.CropOrPad(target_shape=(256, 256, 32)),

    tio.Lambda(lambda x: x.float())
])

TRAIN_TRANSFORMS = tio.Compose([
    # tio.RandomAffine(scales=(0.03, 0.03, 0), degrees=(
    # 0, 0, 3), translation=(4, 4, 0)),
    tio.RandomFlip(axes=(1), flip_probability=0.5),
])


class SynthRAD2023Dataset(Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
        self.root_dir = root_dir
        self.preprocessing = PREPROCESSING_TRANSORMS
        self.transforms = TRAIN_TRANSFORMS
        self.file_paths = self.get_data_files()

    def get_data_files(self):
        subfolder_names = os.listdir(self.root_dir)
        folder_names = [os.path.join(
            self.root_dir, subfolder, 'mr.nii.gz') for subfolder in subfolder_names ] # if subfolder.endswith('.nii.gz')]
        return folder_names

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        img = tio.ScalarImage(self.file_paths[idx])
        img = self.preprocessing(img)
        img = self.transforms(img)
        return {'data': img.data.permute(0, -1, 1, 2)}


if __name__ == '__main__':
    dataset = SynthRAD2023Dataset(root_dir = '')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
