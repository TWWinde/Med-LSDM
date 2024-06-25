import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchio as tio
import os
import nibabel as nib


def preprocess_input(opt, data, test=False):
    data['label'] = data['label'].long()
    if opt.gpu_ids != "-1":
        data['label'] = data['label'].cuda()
        if not test:
            data['image'] = data['image'].cuda()
        else:
            data['image'] = data['image'].cuda()
            data['ct_image'] = data['ct_image'].cuda()
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
        print('datasize', data.shape)
        arr_min = data.min()
        arr_max = data.max()
        data_normalized = (data - arr_min) / (arr_max - arr_min)
        data_normalized = data_normalized * 2 - 1
        return data_normalized

    def randomdepthcrop(self, img, seg=None):
        h, w, d = img.shape

        if self.label:
            print(img.shape)
            print(seg.shape)
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


class SynthRAD2023Dataset1(Dataset):
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


class SynthRAD2023Dataset(Dataset):
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
            subfolder_names = os.listdir(self.root_dir)
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


if __name__ == '__main__':
    #dataset = MRDataset(root_dir='/Users/tangwenwu/Desktop/Master_thesis/data/dataset', sem_map=True)
    dataset = SynthRAD2023Dataset(root_dir='/Users/tangwenwu/Desktop/Master_thesis/data/dataset')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    for i, data in enumerate(dataloader):
        print(f"Batch {i + 1}:")
        #print("Image shape:", data['image'].shape)
        #print("Label shape:", data['label'].shape)
        print("Label shape:", data['data'].shape)

