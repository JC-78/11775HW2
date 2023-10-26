from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import *

from torchmetrics import Accuracy

class Network(torch.nn.Module):
    def __init__(self,input_size,output_size):
        super(Network,self).__init__()
        self.input_size=input_size
        self.output_size=output_size
        layers=[]
        dims=[self.input_size]+[64]
        in_out_dims=list(zip(dims[:-1],dims[1:]))
        for i in range(len(in_out_dims)):
            in_dim,out_dim=in_out_dims[i]
            layers+=self.make_layer(in_dim,out_dim)
        layers+=[nn.Linear(out_dim,self.output_size)]
        self.layers=nn.Sequential(*layers)
        self.initialize_weights()

    def make_layer(self,in_dim,out_dim):
        return [nn.Linear(in_dim,out_dim),
        nn.BatchNorm1d(out_dim),
        nn.GELU(),
        # nn.Dropout(np.random.uniform(0.1,0.6))]
        nn.Dropout(np.random.uniform(0.1,0.5))]


    def forward(self,x):
        # print(x.size)
        x = x.squeeze()
        return self.layers(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            print("Done with weight initialization")

class MlpClassifier(pl.LightningModule):

    def __init__(self, hparams):
        super(MlpClassifier, self).__init__()
        self.save_hyperparameters(hparams)
        # layers = [
        #     # TODO: define model layers here
        #     # Input self.hparams.num_features
        #     # Output self.hparams.num_classes
        # ]
        # layers = [
        #     nn.Linear(self.hparams.num_features, 512),  # Example hidden layer with 512 units
        #     # nn.Linear(512,512),
        #     nn.ReLU(),
        #     nn.Linear(512,512),
        #     nn.ReLU(),
        #     nn.Linear(512,256),
        #     nn.ReLU(),
        #     nn.Linear(256,256),
        #     nn.ReLU(),
        #     nn.Linear(256,128),
        #     nn.Linear(128, self.hparams.num_classes)  # Output layer with num_classes units
        # ]
        self.model=Network(input_size=self.hparams.num_features,output_size=self.hparams.num_classes)
        # self.model = nn.Sequential(*layers)
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log_dict({'val_loss': loss, 'val_acc': acc},
                      on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self(x)
        pred = y_hat.argmax(dim=-1)
        return pred

    def configure_optimizers(self):
        # TODO: define optimizer and optionally learning rate scheduler
        # The simplest form would be `return torch.optim.Adam(...)`
        # For more advanced usages, see https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, factor=self.hparams.scheduler_factor, patience=self.hparams.scheduler_patience
        # )
        optimizer = torch.optim.AdamW(self.model.parameters(),lr=0.001,weight_decay=4.5e-4)
        scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=2,eta_min=0.0001)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_features', type=int)
        parser.add_argument('--num_classes', type=int, default=15)
        parser.add_argument('--learning_rate', type=float, default=0.01)
        parser.add_argument('--scheduler_factor', type=float, default=0.3)
        parser.add_argument('--scheduler_patience', type=int, default=5)
        return parser
