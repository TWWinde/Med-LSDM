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
        "checkpoint_dir": "/path/to/checkpoints",
        "do_load_checkpoint": False,
        "name": "Basic_UNet",
    }
    return c


# 定义UNet训练类
class UNetExperiment3D:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config['device'])

        # 加载数据集
        train_dataset = DUKEDataset(root_dir=self.config['data_dir'], sem_map=True)
        val_dataset = DUKEDataset(root_dir=self.config['data_dir'], sem_map=True)
        self.train_data_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4)
        self.val_data_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4)

        # 初始化模型
        self.model = UNet3D(num_classes=3, in_channels=1)
        self.model.to(self.device)

        # 初始化损失函数
        self.loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'smooth_in_nom': True,
                                    'do_bg': False, 'rebalance_weights': None, 'background_weight': 1}, OrderedDict())

        # 初始化优化器和学习率调度器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        # 加载模型检查点（如果需要）
        if self.config['do_load_checkpoint']:
            self.load_checkpoint(self.config['checkpoint_dir'])

    def save_checkpoint(self, epoch):
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.config['checkpoint_dir'], f"checkpoint_epoch_{epoch}.pt"))

    def load_checkpoint(self, checkpoint_dir):
        self.model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "checkpoint_last.pt")))
        print("Checkpoint loaded.")

    def train(self):
        for epoch in range(self.config['n_epochs']):
            print(f"===== TRAINING - EPOCH {epoch+1} =====")
            self.model.train()
            total_loss = 0.0

            for batch_idx, data_batch in enumerate(self.train_data_loader):
                self.optimizer.zero_grad()
                data = data_batch['image'].float().to(self.device)
                target = data_batch['label'].long().to(self.device)
                print(data.shape)
                print(target.shape)
                # 前向传播
                pred = self.model(data)

                # 计算损失并反向传播
                loss = self.loss(pred, target.squeeze())
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if batch_idx % self.config['plot_freq'] == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}")

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
