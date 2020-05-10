"""
This file defines the core research contribution
"""
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from argparse import ArgumentParser

import pytorch_lightning as pl


class CoolSystem(pl.LightningModule):

    def __init__(self, hparams):
        super(CoolSystem, self).__init__()
        # not the best model...
        self.hparams = hparams
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        return {'loss': F.cross_entropy(y_hat, y)}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(MNIST(self.hparams.data.folder, train=True, download=True, transform=transforms.ToTensor()), batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(self.hparams.data.folder, train=True, download=True, transform=transforms.ToTensor()), batch_size=self.hparams.batch_size)