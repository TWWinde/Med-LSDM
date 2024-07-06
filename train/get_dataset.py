from dataset import MRNetDataset, BRATSDataset, ADNIDataset, DUKEDataset, LIDCDataset, DEFAULTDataset, SynthRAD2023Dataset, AutoPETDataset
from torch.utils.data import WeightedRandomSampler


def get_dataset(cfg):
    if cfg.dataset.name == 'MRNet':
        train_dataset = MRNetDataset(
            root_dir=cfg.dataset.root_dir, task=cfg.dataset.task, plane=cfg.dataset.plane, split='train')
        val_dataset = MRNetDataset(root_dir=cfg.dataset.root_dir,
                                   task=cfg.dataset.task, plane=cfg.dataset.plane, split='valid')
        sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weight, num_samples=len(train_dataset.sample_weight))
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'BRATS':
        train_dataset = BRATSDataset(
            root_dir=cfg.dataset.root_dir, imgtype=cfg.dataset.imgtype, train=True, severity=cfg.dataset.severity, resize=cfg.dataset.resize)
        val_dataset = BRATSDataset(
            root_dir=cfg.dataset.root_dir, imgtype=cfg.dataset.imgtype, train=True, severity=cfg.dataset.severity, resize=cfg.dataset.resize)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'ADNI':
        train_dataset = ADNIDataset(
            root_dir=cfg.dataset.root_dir, augmentation=True)
        val_dataset = ADNIDataset(
            root_dir=cfg.dataset.root_dir, augmentation=True)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'DUKE':
        train_dataset = DUKEDataset(
            root_dir=cfg.dataset.root_dir)
        val_dataset = DUKEDataset(
            root_dir=cfg.dataset.root_dir)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'LIDC':
        train_dataset = LIDCDataset(
            root_dir=cfg.dataset.root_dir, augmentation=True)
        val_dataset = LIDCDataset(
            root_dir=cfg.dataset.root_dir, augmentation=True)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'DEFAULT':
        train_dataset = DEFAULTDataset(
            root_dir=cfg.dataset.root_dir)
        val_dataset = DEFAULTDataset(
            root_dir=cfg.dataset.root_dir)
        sampler = None

    if cfg.dataset.name == 'SynthRAD2023':
        train_dataset = SynthRAD2023Dataset(
            root_dir=cfg.dataset.root_dir)
        val_dataset = SynthRAD2023Dataset(
            root_dir=cfg.dataset.root_dir)
        sampler = None
        return train_dataset, val_dataset, sampler

    if cfg.dataset.name == 'SynthRAD2023_wo_mask':
        train_dataset = SynthRAD2023Dataset(
            root_dir=cfg.dataset.root_dir)
        val_dataset = SynthRAD2023Dataset(
            root_dir=cfg.dataset.root_dir)
        sampler = None
        return train_dataset, val_dataset, sampler

    if cfg.dataset.name == 'AutoPET':
        if cfg.model == 'vq_gan_3d':
            train_dataset = AutoPETDataset(
                root_dir=cfg.dataset.root_dir)
            val_dataset = AutoPETDataset(
                root_dir=cfg.dataset.root_dir)
            sampler = None
            return train_dataset, val_dataset, sampler
        elif cfg.model == 'ddpm':
            train_dataset = AutoPETDataset(
                root_dir=cfg.dataset.root_dir, sem_map=True)
            val_dataset = AutoPETDataset(
                root_dir=cfg.dataset.root_dir, sem_map=True)
            sampler = None
            return train_dataset, val_dataset, sampler






    if cfg.dataset.name == 'AutoPET_wo_artifacts':
        train_dataset = AutoPETDataset(
            root_dir=cfg.dataset.root_dir)
        val_dataset = AutoPETDataset(
            root_dir=cfg.dataset.root_dir)
        sampler = None
        return train_dataset, val_dataset, sampler



    raise ValueError(f'{cfg.dataset.name} Dataset is not available')
