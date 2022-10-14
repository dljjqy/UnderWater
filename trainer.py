from turtle import color
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models import Seq2Seq

class SeqModule(pl.LightningModule):
    def __init__(self, features, lGet, lPre, writer, loss=F.mse_loss):
        super().__init__()
        self.net = Seq2Seq(features) 
        self.features = features
        self.lGet = lGet
        self.lPre = lPre
        self.loss = loss

    def forward(self, x, y, ratio):
        predictions = self.net(x, y, ratio)
        return predictions

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x, y, 0.5)
        loss_value = self.loss(predictions, y)
        self.log('TrainLoss', loss_value)
        return {'loss', loss_value}


    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x, y, 1)
        loss_value = self.loss(predictions, y)
        self.log('ValLoss', loss_value)
        
        if batch_idx == 1:
            self.valPlotter(x, y, predictions)
        return {'valloss': loss_value}

    def valPlotter(self, x, y, pre):
        tensorboard = self.logger.experiment
        x = x.cpu().numpy().squeeze()
        y = y.cpu().numpy().squeeze()
        pre = pre.cpu().numpy().squeeze()

        lGet, lPre = x.shape[0], y.shape[0]
        length = lGet + lPre
        xx = np.linspace(0, 1, length)
        fig = plt.figure(figsize=(5, 10))
        ax = plt.figure()
        ax.plot(xx[:lGet], x, '-k')
        ax.plot(xx[lGet:], y, '--r')
        ax.plot(xx[lGet:], pre, 'bo')
        tensorboard.add_figure(tag='Validate Figure', figure=fig, global_step=self.current_epoch)
        plt.close(fig)
        return True

    def test_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [lr_scheduler]