from turtle import color
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models import *
en_keys = ['WaterTemperature', 'PH' ,'dissolved oxygen', 'Conductivity','Turbidity','PermanganateIndex',
        'AmmoniaNitrogen','TP','TN', 'humidity','room temperature','chlorophyll','Algae density']
class SeqModule(pl.LightningModule):
    def __init__(self, features, lGet, lPre, descaler, loss=F.l1_loss, lr=1e-3):
        super().__init__()
        self.net = Seq2Seq(features) 
        self.features = features
        self.lGet = lGet
        self.lPre = lPre
        self.descaler = descaler
        self.loss = loss
        self.lr = lr
        self.keys = en_keys

    def forward(self, x, y, ratio):
        predictions = self.net(x, y, ratio)
        return predictions

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x, y, 0.5)
        loss_value = self.loss(predictions, y)
        self.log('TrainLoss', loss_value)
        return {'loss':loss_value}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x, y, 1)
        loss_value = self.loss(predictions, y)
        self.log('ValLoss', loss_value, batch_size=1)
        if batch_idx == 0:
            self.valPoltter(x, y, predictions)
        return {'valloss': loss_value}

    def valPoltter(self, x, y, pre):
        tensorboard = self.logger.experiment
        # k = np.random.randint(0, x.shape[1])
        k = 0
        x = x[:, k, :].cpu().numpy().squeeze()
        y = y[:, k, :].cpu().numpy().squeeze()
        pre = pre[:, k, :].cpu().numpy().squeeze()

        x = self.descaler(x)
        y = self.descaler(y)
        pre = self.descaler(pre)

        lGet, lPre = x.shape[0], y.shape[0]
        length = lGet + lPre
        xx = np.linspace(0, 1, length)
        fig, axes = plt.subplots(self.features, 1, figsize=(20, 3*self.features), constrained_layout=True)
        for i in range(self.features):
            axes[i].set_title(self.keys[i], fontsize=20)
            
            input = axes[i].plot(xx[:lGet], x[:, i], '-k')
            real = axes[i].plot(xx[lGet:], y[:, i], '--r')
            prediction = axes[i].plot(xx[lGet:], pre[:, i], '--b')
            
            axes[i].legend(handles = [input[0], real[0], prediction[0]],
                            labels = ['data', 'real', 'pre'], fontsize=15)
        tensorboard.add_figure(tag='Validate Figure', figure=fig, global_step=self.current_epoch)
        plt.close(fig)
        return True

    def test_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [lr_scheduler]


class SCIModule(pl.LightningModule):
    def __init__(self, features, lGet, lPre, Tree_levels, hidden_size_rate, descaler, 
                    loss=F.l1_loss, lr=1e-3):
        super().__init__()
        self.net = SCINet(features, lPre, lGet, Tree_levels, hidden_size_rate)
        self.features = features
        self.lGet = lGet
        self.lPre = lPre
        self.descaler = descaler
        self.loss = loss
        self.lr = lr
        self.keys = en_keys

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch):
        x, y = batch
        predictions = self(x)
        loss_value = self.loss(predictions, y)
        self.log('TrainLoss', loss_value)
        return {'loss':loss_value}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss_value = self.loss(predictions, y)
        self.log('ValLoss', loss_value, batch_size=1)
        if batch_idx == 0:
            self.valPoltter(x, y, predictions)
        return {'valloss': loss_value}

    def valPoltter(self, x, y, pre):
        tensorboard = self.logger.experiment
        k = 0
        x = x[k].cpu().numpy().squeeze()
        y = y[k].cpu().numpy().squeeze()
        pre = pre[k].cpu().numpy().squeeze()

        x = self.descaler(x)
        y = self.descaler(y)
        pre = self.descaler(pre)
        real = np.concatenate((x, y), axis=1)

        xx = np.linspace(0, 1, real.shape[1])
        fig, axes = plt.subplots(self.features, 1, figsize=(20, 3*self.features), constrained_layout=True)
        
        for i in range(self.features):
            axes[i].set_title(self.keys[i], fontsize=20)
            
            real_line = axes[i].plot(xx, real[i], '-k')
            pre_line = axes[i].plot(xx[self.lGet:], pre[i], '-r')
            
            axes[i].legend(handles = [real_line[0], pre_line[0]],
                            labels = ['real', 'pre'], fontsize=15)
        tensorboard.add_figure(tag='Validate Figure', figure=fig, global_step=self.current_epoch)
        plt.close(fig)
        return True


    def test_step(self, *args, **kwargs):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [lr_scheduler]



module_names = {'SeqModule': SeqModule, 'SCIModule': SCIModule}
