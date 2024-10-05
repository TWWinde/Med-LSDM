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

import os
from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from trixi.experiment.pytorchexperiment import PytorchExperiment
from torch.utils.data import DataLoader
from trixi.util import Config
from dataset import DUKEDataset
from u_net.RecursiveUnet3D import UNet3D
from u_net.dice_loss import DC_and_CE_loss


def get_config():
    # Set your own path, if needed.
    data_root_dir = os.path.abspath('/data/private/autoPET/duke')  # The path where the downloaded dataset is stored.

    c = Config(
        update_from_argv=True,  # If set 'True', it allows to update each configuration by a cmd/terminal parameter.

        # Train parameters
        num_classes=1,
        in_channels=1,
        batch_size=8,
        patch_size=64,
        n_epochs=10,
        learning_rate=0.0002,
        fold=0,  # The 'splits.pkl' may contain multiple folds. Here we choose which one we want to use.

        device="cuda",  # 'cuda' is the default CUDA device, you can use also 'cpu'. For more information, see https://pytorch.org/docs/stable/notes/cuda.html

        # Logging parameters
        name='Basic_Unet',
        author='wenwu',  # Author of this project
        plot_freq=10,  # How often should stuff be shown in visdom
        append_rnd_string=False,  # Appends a random string to the experiment name to make it unique.
        start_visdom=True,  # You can either start a visom server manually or have trixi start it for you.

        do_instancenorm=True,  # Defines whether or not the UNet does a instance normalization in the contracting path
        do_load_checkpoint=False,
        checkpoint_dir='',

        # Adapt to your own path, if needed.
        #google_drive_id='1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C',  # This id is used to download the example dataset.
        dataset_name='Duke_breast',
        base_dir=os.path.abspath('/data/private/autoPET/medicaldiffusion_results/unet'),  # Where to log the output of the experiment.

        data_root_dir='/data/private/autoPET/duke',  # The path where the downloaded dataset is stored.
        data_dir=os.path.join(data_root_dir, '/final_labeled_mr'),  # This is where your training and validation data is stored
        data_test_dir=os.path.join(data_root_dir, '/final_label'),  # This is where your test data is stored

        split_dir=os.path.join(data_root_dir, 'autoPET'),  # This is where the 'splits.pkl' file is located, that holds your splits.

        # execute a segmentation process on a specific image using the model
        model_dir=os.path.join(os.path.abspath('/data/private/autoPET/medicaldiffusion_results/unet/u-net_output_experiment'), 'model'),  # the model being used for segmentation
    )

    print(c)
    return c


class UNetExperiment3D(PytorchExperiment):
    """
    The UnetExperiment is inherited from the PytorchExperiment. It implements the basic life cycle for a segmentation task with UNet(https://arxiv.org/abs/1505.04597).
    It is optimized to work with the provided NumpyDataLoader.

    The basic life cycle of a UnetExperiment is the same s PytorchExperiment:

        setup()
        (--> Automatically restore values if a previous checkpoint is given)
        prepare()

        for epoch in n_epochs:
            train()
            validate()
            (--> save current checkpoint)

        end()
    """

    def setup(self):
        # 读取配置文件

        train_dataset = DUKEDataset(
            root_dir=self.config.data_dir, sem_map=True)
        val_dataset = DUKEDataset(
            root_dir=self.config.data_dir, sem_map=True)

        self.train_data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
        self.val_data_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)
        self.test_data_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        self.model = UNet3D(num_classes=3, in_channels=1)

        self.model.to(self.device)

        # We use a combination of DICE-loss and CE-Loss in this example.
        # This proved good in the medical segmentation decathlon.
        self.loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'smooth_in_nom': True,
                                    'do_bg': False, 'rebalance_weights': None, 'background_weight': 1}, OrderedDict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        # If directory for checkpoint is provided, we load it.
        if self.config.do_load_checkpoint:
            if self.config.checkpoint_dir == '':
                print('checkpoint_dir is empty, please provide directory to load checkpoint.')
            else:
                self.load_checkpoint(name=self.config.checkpoint_dir, save_types=("model",))

        self.save_checkpoint(name="checkpoint_start")
        self.elog.print('Experiment set up.')

    def train(self, epoch):
        self.elog.print('=====TRAIN=====')
        self.model.train()

        batch_counter = 0
        for i, data_batch in self.train_data_loader:

            self.optimizer.zero_grad()

            # Shape of data_batch = [1, b, c, w, h]
            # Desired shape = [b, c, w, h]
            # Move data and target to the GPU
            data = data_batch['image'].float().to(self.device)
            target = data_batch['label'].long().to(self.device)

            pred = self.model(data)

            loss = self.loss(pred, target.squeeze())
            # loss = self.ce_loss(pred, target.squeeze())
            loss.backward()
            self.optimizer.step()

            # Some logging and plotting
            if (batch_counter % self.config.plot_freq) == 0:
                self.elog.print('Epoch: %d Loss: %.4f' % (self._epoch_idx, loss))

                self.add_result(value=loss.item(), name='Train_Loss', tag='Loss', counter=epoch + (batch_counter / len(self.train_data_loader)/4))

                self.clog.show_image_grid(data[:,:,30].float(), name="data", normalize=True, scale_each=True, n_iter=epoch)
                self.clog.show_image_grid(target[:,:,30].float(), name="mask", title="Mask", n_iter=epoch)
                self.clog.show_image_grid(torch.argmax(pred.cpu(), dim=1, keepdim=True)[:,:,30], name="unt_argmax", title="Unet", n_iter=epoch)

            batch_counter += 1

    def validate(self, epoch):
        if epoch % 5 != 0:
            return
        self.elog.print('VALIDATE')
        self.model.eval()

        data = None
        loss_list = []

        with torch.no_grad():
            for data_batch in self.val_data_loader:
                data = data_batch['data'][0].float().to(self.device)
                target = data_batch['seg'][0].long().to(self.device)

                pred = self.model(data)

                loss = self.loss(pred, target.squeeze())
                loss_list.append(loss.item())

        assert data is not None, 'data is None. Please check if your dataloader works properly'
        self.scheduler.step(np.mean(loss_list))

        self.elog.print('Epoch: %d Loss: %.4f' % (self._epoch_idx, float(np.mean(loss_list))))

        self.add_result(value=np.mean(loss_list), name='Val_Loss', tag='Loss', counter=epoch+1)

        self.clog.show_image_grid(data[:,:,30].float(), name="data_val", normalize=True, scale_each=True, n_iter=epoch)
        self.clog.show_image_grid(target[:,:,30].float(), name="mask_val", title="Mask", n_iter=epoch)
        self.clog.show_image_grid(torch.argmax(pred.data.cpu()[:,:,30], dim=1, keepdim=True), name="unt_argmax_val", title="Unet", n_iter=epoch)

    def test(self):
        # TODO
        print('TODO: Implement your test() method here')


if __name__ == '__main__':
    c = get_config()
    exp = UNetExperiment3D(config=c, name=c.name, n_epochs=c.n_epochs,
                           seed=42, append_rnd_to_name=c.append_rnd_string, globs=globals())
    exp.run()
    exp.run_test(setup=False)



