#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from u_net.RecursiveUnet3D import UNet3D
from u_net.dice_loss import DC_and_CE_loss
import matplotlib.pyplot as plt


# 获取配置
def get_config():
    c = {
        "data_root_dir": "/data/private/autoPET/duke",
        "data_dir": "/data/private/autoPET/duke/final_labeled_mr",
        "data_test_dir": "/data/private/autoPET/duke/final_label",
        "split_dir": "/data/private/autoPET/duke/autoPET",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 4,
        "patch_size": (64, 64, 64),
        "n_epochs": 10,
        "learning_rate": 0.0002,
        "plot_freq": 10,
        "num_classes": 3,
        "checkpoint_dir": "/data/private/autoPET/medicaldiffusion_results/unet/checkpoint",
        "image_dir": "/data/private/autoPET/medicaldiffusion_results/unet/image",
        "do_load_checkpoint": False,
        "name": "Basic_UNet",
    }
    return c


class UNetExperiment3D:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config['device'])

        train_dataset = DUKEDataset(root_dir=self.config['data_dir'], sem_map=True)
        val_dataset = DUKEDataset(root_dir=self.config['data_dir'], sem_map=True)
        self.train_data_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4)
        self.val_data_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4)

        self.model = UNet3D(num_classes=3, in_channels=1)
        self.model.to(self.device)

        self.loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'smooth_in_nom': True,
                                    'do_bg': False, 'rebalance_weights': None, 'background_weight': 1}, OrderedDict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        if self.config['do_load_checkpoint']:
            self.load_checkpoint(self.config['checkpoint_dir'])

    def save_checkpoint(self, epoch):
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.config['checkpoint_dir'], f"checkpoint_epoch_{epoch}.pt"))

    def load_checkpoint(self, checkpoint_dir):
        self.model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "checkpoint_last.pt")))
        print("Checkpoint loaded.")

    def preprocess_input(self, data):

        # move to GPU and change data types
        data = data.long()

        # create one-hot label map
        label_map = data
        bs, _, t, h, w = label_map.size()
        nc = self.config['num_classes']
        input_label = torch.FloatTensor(bs, nc, t, h, w).zero_().to(self.device)
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        return input_semantics

    def train(self):
        for epoch in range(self.config['n_epochs']):
            print(f"===== TRAINING - EPOCH {epoch+1} =====")
            self.model.train()
            total_loss = 0.0

            for batch_idx, data_batch in enumerate(self.train_data_loader):
                self.optimizer.zero_grad()
                data = data_batch['image'].float().to(self.device)
                target = data_batch['label'].long().to(self.device)
                target = self.preprocess_input(target)
                #print(data.shape) torch.Size([4, 1, 32, 256, 256]) torch.Size([4, 3, 32, 256, 256])

                pred = self.model(data)

                loss = self.loss(pred, target)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if batch_idx % self.config['plot_freq'] == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}")
                    slice_index = 16  # Specify which slice you want to save

                    # Path to save images
                    path_images = os.path.join(self.config['image_dir'])
                    os.makedirs(path_images, exist_ok=True)
                    image_np = data.detach().numpy()
                    label_np = target.detach().numpy()
                    pred_np = pred.detach().numpy()

                    # For image_np
                    plt.imshow(image_np[0, 0, slice_index, :, :], cmap='gray')  # Grayscale image
                    plt.axis('off')
                    plt.savefig(os.path.join(path_images, f'{batch_idx}_image_slice_{slice_index}.png'), bbox_inches='tight',
                                pad_inches=0)
                    plt.close()

                    # For label_np (assuming RGB)
                    plt.imshow(label_np[0, 0, slice_index, :, :])  # Color image, transpose (H, W, C)
                    plt.axis('off')
                    plt.savefig(os.path.join(path_images, f'{batch_idx}_label_slice_{slice_index}.png'), bbox_inches='tight',
                                pad_inches=0)
                    plt.close()

                    plt.imshow(pred_np[0, 0, slice_index, :, :])  # Color image, transpose (H, W, C)
                    plt.axis('off')
                    plt.savefig(os.path.join(path_images, f'{batch_idx}_pred_slice_{slice_index}.png'), bbox_inches='tight',
                                pad_inches=0)
                    plt.close()

            avg_loss = total_loss / len(self.train_data_loader)
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")

            # 保存模型检查点
            self.save_checkpoint(epoch)

            # 验证
            self.validate(epoch)

    def validate(self, epoch):
        print("===== VALIDATING =====")
        self.model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for data_batch in self.val_data_loader:
                data = data_batch['image'].float().to(self.device)
                target = data_batch['label'].long().to(self.device)

                # 前向传播
                pred = self.model(data)

                # 计算验证损失
                loss = self.loss(pred, target.squeeze())
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(self.val_data_loader)
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}")

        # 学习率调度器更新
        self.scheduler.step(avg_val_loss)

    def test(self):
        print("===== TESTING =====")
        # TODO: 实现测试逻辑
        pass


if __name__ == '__main__':
    c = get_config()
    experiment = UNetExperiment3D(config=c)
    experiment.train()