import numpy as np
import torch
from torch.utils.data import Dataset
import os
import nibabel as nib
import random
from torchvision import transforms as TR
import torchio as tio


class Transform:
    def __init__(self, target_depth=32, label=False, size=(256, 256, 32)):
        self.target_depth = target_depth
        self.label = label
        self.size = size
        self.num_classes = 37

    def normalization(self, data):
        arr_min = data.min()
        arr_max = data.max()
        data_normalized = (data - arr_min) / (arr_max - arr_min)
        data_normalized = data_normalized * 2 - 1
        return data_normalized

    def randomdepthcrop(self, img, seg=None):
        h, w, d = img.shape
        target_d = self.target_depth

        if d <= target_d:
            raise ValueError("target depth id bigger than image depth")

        d_start = np.random.randint(2, d - target_d - 1)
        cropped_img = img[:, :, d_start:d_start + target_d]

        if self.label:
            assert img.shape == seg.shape
            cropped_seg = seg[:, :, d_start:d_start + target_d]
            return cropped_img, cropped_seg
        else:
            return cropped_img

    def crop_center(self, img, seg=None):
        new_x, new_y, new_z = self.size
        x, y, z = img.shape
        start_x = x // 2 - new_x // 2
        start_y = y // 2 - new_y // 2
        start_z = z // 2 - new_z // 2

        cropped_img = img[start_x:start_x + new_x, start_y:start_y + new_y, start_z:start_z + new_z]

        if self.label:
            assert img.shape == seg.shape
            cropped_seg = seg[start_x:start_x + new_x, start_y:start_y + new_y, start_z:start_z + new_z]
            return cropped_img, cropped_seg
        else:
            return cropped_img

    def resize_3d(self, img, label=None):
        img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
        resized_img = torch.nn.functional.interpolate(img, size=self.size, mode='trilinear', align_corners=False).squeeze(0).squeeze(0)

        if self.label and label is not None:
            label = torch.tensor(label).unsqueeze(0).unsqueeze(0).float()
            resized_label = torch.nn.functional.interpolate(label, size=self.size, mode='nearest').squeeze(0).squeeze(0)
            return resized_img.numpy(), resized_label.numpy()
        else:
            return resized_img.numpy()

    def one_hot_encode(self, label):
        if self.num_classes is None:
            raise ValueError("have to apply num_classes")

        label = torch.from_numpy(label).long()
        one_hot = torch.nn.functional.one_hot(label, num_classes=self.num_classes)
        return one_hot

    def __call__(self, img, seg=None):
        if self.label:
            img, seg = self.randomdepthcrop(img, seg)
            img, seg = self.crop_center(img, seg)
            if img.shape != self.size:
                img, seg = self.resize_3d(img, seg)
            final_img = self.normalization(img)
            seg = self.one_hot_encode(seg)
            final_img = torch.tensor(final_img).unsqueeze(0).permute(0, -1, 1, 2).float()
            seg = seg.permute(-1, 2, 0, 1).float()   # torch.Size([10, 256, 256, 32, 37])

            return final_img, seg
        else:
            img = self.randomdepthcrop(img)
            img = self.crop_center(img)
            if img.shape != self.size:
                img = self.resize_3d(img)
            final_img = self.normalization(img)

            final_img = torch.from_numpy(final_img).unsqueeze(0).float().permute(0, -1, 1, 2)

            return final_img



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


class AutoPETDataset(Dataset):
    """
    root_dir: root_dir: /data/private/autoPET/autopet_3d/train  for training
                       /data/private/autoPET/autopet_3d/test  for test and validation

    """
    def __init__(self, root_dir: str, sem_map=False):
        super().__init__()
        self.root_dir = root_dir
        self.sem_map = sem_map
        self.Norm = normalization
        self.Crop = crop
        if self.sem_map:
            self.ct_paths, self.label_paths = self.get_data_files()
        else:
            self.ct_paths = self.get_data_files()

    def get_data_files(self):

        ct_names = [os.path.join(self.root_dir, 'ct', subfolder) for subfolder in sorted(os.listdir(os.path.join(self.root_dir, 'ct')))
                    if subfolder.endswith('nii.gz')]
        if self.sem_map:
            label_names, ct_names_ = [], []
            for ct_path in sorted(ct_names):
                label_path = ct_path.replace('ct', 'label')
                if os.path.exists(ct_path) and os.path.exists(label_path):
                    ct_names_.append(ct_path)
                    label_names.append(label_path)

            return ct_names_, label_names
        else:

            return ct_names

    def __len__(self):
        return len(self.ct_paths)

    def __getitem__(self, idx: int):
        img = tio.ScalarImage(self.ct_paths[idx])
        img = self.Crop(img)
        img = self.Norm(img)
        if self.sem_map:
            label = tio.ScalarImage(self.label_paths[idx])
            label = self.Crop(label)

            return {'image': img.data.permute(0, -1, 1, 2), 'label': label.data.permute(0, -1, 1, 2)}
        else:
            return {'image': img.data.permute(0, -1, 1, 2)}


class AutoPETDataset2(Dataset):
    def __init__(self, root_dir: str, sem_map=False):
        super().__init__()
        self.root_dir = root_dir
        self.sem_map = sem_map
        self.Norm = normalization
        self.Crop = crop
        if self.sem_map:
            #self.transform = Transform(target_depth=32, label=True)
            self.ct_paths, self.label_paths = self.get_data_files()
        else:
            #self.transform = Transform(target_depth=32, label=False)
            self.ct_paths = self.get_data_files()

    def get_data_files(self):

        """
        subfolder_names = os.listdir(self.root_dir)
        ct_names = [os.path.join(self.root_dir, subfolder) for subfolder in subfolder_names
                    if subfolder.endswith('0001.nii.gz')]
        if self.sem_map:
            label_names, ct_names_ = [], []
            for ct_path in ct_names:
                label_path = ct_path.replace('0001.nii.gz', '0002.nii.gz')
                if os.path.exists(ct_path) and os.path.exists(label_path):
                    ct_names_.append(ct_path)
                    label_names.append(label_path)

        """
        subfolder_names = sorted(os.listdir(os.path.join(self.root_dir, 'ct')))
        ct_names = [os.path.join(self.root_dir, 'ct', subfolder) for subfolder in subfolder_names
                    if subfolder.endswith('nii.gz')]
        if self.sem_map:
            label_names, ct_names_ = [], []
            for ct_path in sorted(ct_names):
                label_path = ct_path.replace('ct', 'label')
                if os.path.exists(ct_path) and os.path.exists(label_path):
                    ct_names_.append(ct_path)
                    label_names.append(label_path)

            return ct_names_, label_names
        else:

            return ct_names

    def __len__(self):
        return len(self.ct_paths)

    def __getitem__(self, idx: int):
        #img = nib.load(self.ct_paths[idx])
        #img = img.get_fdata()
        img = tio.ScalarImage(self.ct_paths[idx])
        img = self.Crop(img)
        img = self.Norm(img)
        if self.sem_map:
            label = tio.ScalarImage(self.label_paths[idx])
            label = self.Crop(label)

            #label = nib.load(self.label_paths[idx])
            #label = label.get_fdata()
            #img, label = self.transform(img, label)

            #if random.random() < 0.5:
                #img = TR.functional.hflip(img)
                #label = TR.functional.hflip(label)
            #if random.random() < 0.5:
                #angle = random.choice([90, 180, 270])
                #img = TR.functional.rotate(img, angle)
                #label = TR.functional.rotate(label, angle)

            return {'image': img.data.permute(0, -1, 1, 2), 'label': label.data.permute(0, -1, 1, 2)}
        else:
            img = self.transform(img)
            img = torch.from_numpy(img).unsqueeze(0).float().permute(0, -1, 1, 2)

            if random.random() < 0.5:
                img = TR.functional.hflip(img)
            if random.random() > 0.5:
                img = TR.functional.vflip(img)
            if random.random() < 0.5:
                angle = random.choice([90, 180, 270])
                img = TR.functional.rotate(img, angle)

            return {'image': img}




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




class AutoPETDataset1(Dataset):
    def __init__(self, root_dir: str, sem_map=False):
        super().__init__()
        self.root_dir = root_dir
        self.sem_map = sem_map
        if self.sem_map:
            self.transform = Transform(target_depth=32, label=True)
            self.mr_paths, self.label_paths = self.get_data_files()
        else:
            self.transform = Transform(target_depth=32, label=False)
            self.mr_paths = self.get_data_files()

    def get_data_files(self):
        if self.sem_map:
            mr_names, label_names = [], []
            subfolder_names = sorted(os.listdir(self.root_dir))
            for item in subfolder_names:
                mr_path = os.path.join(self.root_dir, item, 'mr.nii.gz')
                label_path = os.path.join(self.root_dir, item, 'label.nii.gz')
                if os.path.exists(mr_path) and os.path.exists(label_path):
                    mr_names.append(mr_path)
                    label_names.append(label_path)

            return mr_names, label_names
        else:
            subfolder_names = os.listdir(self.root_dir)
            mr_names = [os.path.join(self.root_dir, subfolder, 'mr.nii.gz') for subfolder in subfolder_names
                        if os.path.exists(os.path.join(self.root_dir, subfolder, 'mr.nii.gz'))]

            return mr_names

    def __len__(self):
        return len(self.mr_paths)

    def __getitem__(self, idx: int):
        if self.sem_map:
            img = nib.load(self.mr_paths[idx])
            label = nib.load(self.label_paths[idx])
            img = img.get_fdata()
            label = label.get_fdata()
            img, label = self.transform(img, label)
            img = torch.from_numpy(img).float().permute(-1, 0, 1)
            label = torch.from_numpy(label).float().permute(-1, 0, 1)

            return {'image': img, 'label': label}
        else:
            img = nib.load(self.mr_paths[idx])
            img = img.get_fdata()
            img = self.transform(img)
            img = torch.from_numpy(img).float().permute(-1, 0, 1)

            return {'image': img}
