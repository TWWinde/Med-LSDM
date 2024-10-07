import sys
sys.path.append('/misc/no_backups/d1502/medicaldiffusion')
import os
from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from dataset import DUKEDataset
from dataset import DUKEDataset_unet
from u_net.RecursiveUnet3D import UNet3D
from u_net.dice_loss import DC_and_CE_loss
import matplotlib.pyplot as plt


def get_config():
    c = {
        "data_root_dir": "/data/private/autoPET/duke",
        "data_dir": "/data/private/autoPET/duke/final_labeled_mr",
        "test_data_dir": "/data/private/autoPET/duke",
        "split_dir": "/data/private/autoPET/duke/autoPET",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 4,
        "patch_size": (64, 64, 64),
        "n_epochs": 10,
        "learning_rate": 0.00005,
        "plot_freq": 10,
        "image_freq": 50,
        "num_classes": 3,
        "root_dir": "/data/private/autoPET/medicaldiffusion_results/unet",
        "checkpoint_dir": "/data/private/autoPET/medicaldiffusion_results/unet/checkpoint",
        "image_dir": "/data/private/autoPET/medicaldiffusion_results/unet/image",
        "do_load_checkpoint": False,
        "name": "Basic_UNet",
    }
    return c


class UNetExperiment3D:
    def __init__(self, config, percentage=1.0):
        self.config = config
        self.device = torch.device(self.config['device'])
        self.percentage = percentage
        self.name = f'without_pretrain_{self.percentage*100}%_data'
        train_dataset = DUKEDataset(root_dir=self.config['data_dir'], sem_map=True, percentage=self.percentage)
        val_dataset = DUKEDataset(root_dir=self.config['data_dir'], sem_map=True)
        self.train_data_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4)
        self.val_data_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4)
        test_dataset = DUKEDataset_unet(root_dir=self.config['test_data_dir'], sem_map=True)
        self.test_data_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=True,
                                           num_workers=4)

        self.model = UNet3D(num_classes=3, in_channels=1)
        self.model.to(self.device)
        self.checkpoint_dir = os.path.join(self.config['root_dir'], self.name, "checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.image_dir = os.path.join(self.config['root_dir'], self.name, "image_train")
        os.makedirs(self.image_dir, exist_ok=True)
        self.image_dir_test = os.path.join(self.config['root_dir'], self.name, "image_test")
        os.makedirs(self.image_dir_test, exist_ok=True)
        self.loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'smooth_in_nom': True,
                                    'do_bg': False, 'rebalance_weights': None, 'background_weight': 1}, OrderedDict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        if self.config['do_load_checkpoint']:
            self.load_checkpoint(self.checkpoint_dir)

    def save_checkpoint(self, epoch):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))

    def load_checkpoint(self, checkpoint_dir):
        check_path = os.path.join(checkpoint_dir, "checkpoint_epoch_3.pt")
        self.model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "checkpoint_epoch_3.pt")))
        print(f"{check_path}_Checkpoint loaded.")

    def preprocess_input(self, data):

        data = data.long()

        label_map = data
        bs, _, t, h, w = label_map.size()
        nc = self.config['num_classes']
        input_label = torch.FloatTensor(bs, nc, t, h, w).zero_().to(self.device)
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        return input_semantics

    def train(self):
        loss_image = []
        dice_loss_list = []
        ce_loss_list = []
        step = 0
        for epoch in range(self.config['n_epochs']):
            print(f"===== TRAINING - EPOCH {epoch+1} =====")
            self.model.train()
            total_loss = 0.0

            for batch_idx, data_batch in enumerate(self.train_data_loader):
                self.optimizer.zero_grad()
                image = data_batch['image'].float().to(self.device)
                label = data_batch['label'].long().to(self.device)
                target = self.preprocess_input(label)
                #print(data.shape) torch.Size([4, 1, 32, 256, 256]) torch.Size([4, 3, 32, 256, 256])

                pred = self.model(image)

                loss, ce_loss, dc_loss = self.loss(pred, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                pred_save = torch.argmax(pred, dim=1, keepdim=True)
                loss_image.append(loss.item())
                dice_loss_list.append(dc_loss.item())
                ce_loss_list.append(ce_loss.item())
                if batch_idx % self.config['plot_freq'] == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}, CrossEntropy_Loss: {ce_loss.item()}, Dice_Loss: {dc_loss.item()}")

                if batch_idx % self.config['image_freq']*2 == 0:

                    self.plot_loss(loss_image, "Total_Loss")
                    self.plot_loss(dice_loss_list, "Dice_Loss")
                    self.plot_loss(ce_loss_list, "CrossEntropy_Loss")

                if batch_idx % self.config['image_freq'] == 0:
                    self.save_results_slices(image, label, pred_save, batch_idx, self.image_dir)

                step += 1

            avg_loss = total_loss / len(self.train_data_loader)
            print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")

            self.save_checkpoint(epoch)

            self.validate(epoch)

    def save_results_slices(self, image, label, pred_save, batch_idx, save_dir, mode="train"):
        slice_index = 16  # Specify which slice you want to save

        # Path to save images
        path_images = os.path.join(save_dir)
        os.makedirs(path_images, exist_ok=True)
        image_np = image.detach().cpu().numpy()
        label_np = label.detach().cpu().numpy()
        label_np[label_np != 0] = 1
        pred_np = pred_save.detach().cpu().numpy()
        pred_np[pred_np != 0] = 1

        # For image_np
        plt.imshow(image_np[0, 0, slice_index, :, :], cmap='gray')  # Grayscale image
        plt.axis('off')
        plt.savefig(os.path.join(path_images, f'{batch_idx}_{mode}_image_slice_{slice_index}.png'), bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        # For label_np (assuming RGB)
        slice_index = 16  # Specify the slice you want to save
        vmin = min(label_np.min(), pred_np.min())  # Get the minimum value from both images
        vmax = max(label_np.max(), pred_np.max())  # Get the maximum value from both images

        # Path to save images

        # Plot label image with the same colormap and value range
        plt.imshow(label_np[0, 0, slice_index, :, :], cmap='viridis', vmin=vmin,
                   vmax=vmax)  # Grayscale or color image
        plt.axis('off')
        plt.savefig(os.path.join(path_images, f'{batch_idx}_label_slice_{slice_index}.png'),
                    bbox_inches='tight', pad_inches=0)
        plt.close()

        # Plot predicted image with the same colormap and value range
        plt.imshow(pred_np[0, 0, slice_index, :, :], cmap='viridis', vmin=vmin,
                   vmax=vmax)  # Grayscale or color image
        plt.axis('off')
        plt.savefig(os.path.join(path_images, f'{batch_idx}_{mode}_pred_slice_{slice_index}.png'),
                    bbox_inches='tight', pad_inches=0)
        plt.close()

    def validate(self, epoch):
        print("===== VALIDATING =====")
        self.model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for data_batch in self.val_data_loader:
                image = data_batch['image'].float().to(self.device)
                label = data_batch['label'].long().to(self.device)
                target = self.preprocess_input(label)
                # print(data.shape) torch.Size([4, 1, 32, 256, 256]) torch.Size([4, 3, 32, 256, 256])

                pred = self.model(image)

                loss, ce_loss, dc_loss = self.loss(pred, target)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(self.val_data_loader)
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}, CrossEntropy_Loss: {ce_loss.item()}, Dice_Loss: {dc_loss.item()}")

        self.scheduler.step(avg_val_loss)

    def dice_coefficient(self, pred, target, smooth=1.):
        """
        计算Dice系数
        :param pred:  (B, C, X, Y, Z) (B, C, X, Y) (batch, channels, depth, height, width)
        :param target:
        :param smooth:
        :return: Dice
        """
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()

        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()

        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return dice

    def test(self):

        print("===== TESTING =====")
        self.model.eval()
        total_test_loss_real = 0.0
        total_test_loss_fake = 0.0
        total_dice_real = 0.0
        total_dice_fake = 0.0

        batch_idx = 0
        with torch.no_grad():
            for data_batch in self.test_data_loader:
                mr_real = data_batch['mr_real'].float().to(self.device)
                mr_fake = data_batch['mr_fake'].float().to(self.device)
                label = data_batch['label'].long().to(self.device)
                target = self.preprocess_input(label)
                # print(data.shape) torch.Size([4, 1, 32, 256, 256]) torch.Size([4, 3, 32, 256, 256])
                pred_real = self.model(mr_real)
                pred_fake = self.model(mr_fake)

                pred_save_real = torch.argmax(pred_real, dim=1, keepdim=True)
                real_loss, real_ce_loss, real_dc_loss = self.loss(pred_real, target)
                dice_real = self.dice_coefficient(pred_real, target, smooth=1.)
                total_test_loss_real += real_loss.item()
                total_dice_real += dice_real.item()

                pred_save_fake = torch.argmax(pred_fake, dim=1, keepdim=True)
                fake_loss, fake_ce_loss, fake_dc_loss = self.loss(pred_fake, target)
                dice_fake = self.dice_coefficient(pred_fake, target, smooth=1.)
                total_test_loss_fake += fake_loss.item()
                total_dice_fake += dice_fake.item()

                self.save_results_slices(mr_real, label, pred_save_real, batch_idx, self.image_dir_test, mode="real")
                self.save_results_slices(mr_fake, label, pred_save_fake, batch_idx, self.image_dir_test, mode="fake")
                batch_idx += 1
        avg_val_loss_real = total_test_loss_real / len(self.test_data_loader)
        avg_val_loss_fake = total_test_loss_fake / len(self.test_data_loader)
        avg_dice_real = total_dice_real / len(self.test_data_loader)
        avg_dice_fake = total_dice_fake / len(self.test_data_loader)
        print(
            f" Test Loss real: {avg_val_loss_real}, Dice_Loss real: {avg_dice_real}")
        print(
            f" Test Loss fake: {avg_val_loss_fake}, Dice_Loss real: {avg_dice_fake}")
        pass

    def plot_loss(self, loss, name):
        """
        save loss figure after every epoch
        """
        plt.figure()
        plt.plot(range(1, len(loss) + 1), loss, marker='o')
        plt.xlabel('Batch')
        plt.ylabel(f'{name}')
        plt.title(f'{name}')
        plt.grid(True)

        path_images = os.path.join(self.config['root_dir'], self.name, f'{name}.png')
        os.makedirs(self.config['root_dir'], exist_ok=True)
        plt.savefig(path_images)
        plt.close()


if __name__ == '__main__':
    c = get_config()
    experiment = UNetExperiment3D(config=c)
    #experiment.train()
    experiment.test()