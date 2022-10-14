import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

def minmaxScaler(data):
    maxs = np.max(data, axis=0)
    mins = np.min(data, axis=0)
    data = (data - mins) / maxs
    return data, maxs, mins


class WaterDataSet(Dataset):
    def __init__(self, data, lGet, lPre):
        super().__init__()
        self.data = torch.from_numpy(data).to(dtype=torch.float16)
        self.lGet = lGet
        self.lPre = lPre

    def __getitem__(self, idx):
        k = idx * (self.lGet + self.lPre)
        x = self.data[k: k+self.lGet, :]
        y = self.data[k+self.lGet: k+self.lGet+self.lPre, :]
        return x, y
    
    def __len__(self):
        total = self.data.shape[0]
        nums = total // (self.lGet + self.lPre) - 1
        return nums

def my_collate_fn(data):
    '''
    Set the batch dimension as the second not first.
    L x N x F
    '''
    xs, ys = zip(*data)
    xs = torch.stack(xs, dim=1)
    ys = torch.stack(ys, dim=1)
    return (xs, ys)


class WaterDataModule(pl.LightningDataModule):
    def __init__(self, path, lGet=24, lPre=6, train_N=1000, val_N=10, batch_size=10):
        '''
        lGet: How many rows used to predict.
        lPre: How many rows you want to predict.
        train_N: How many groups of data you need for train.
        val_N: How mant groups of data you need for validation.
                One group equals to lGet and lPre rows of data
        '''
        data = pd.read_csv(path, index_col=0).values.copy()
        self.data, self.maxs, self.mins = minmaxScaler(data)
        rows_for_train = (lGet + lPre) * train_N + lPre
        rows_for_val = (lGet + lPre) * val_N + lPre
        self.train_data = self.data[0:rows_for_train, :].copy()
        self.val_data = self.data[rows_for_train: rows_for_train+rows_for_val, :].copy()
        self.batch_size = batch_size
        self.lGet = lGet
        self.lPre = lPre
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_ds = WaterDataSet(self.train_data, self.lGet, self.lPre)
            self.val_ds = WaterDataSet(self.val_data, self.lGet, self.lPre)
        elif stage == 'test':
            pass
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, self.batch_size, shuffle=True, num_workers=6, collate_fn=my_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, 1, shuffle=False, num_workers=6, collate_fn=my_collate_fn)

    def test_dataloader(self):
        pass

if __name__ == '__main__':
    dm = WaterDataModule('./data3.csv')
    dm.setup()
    dl = dm.train_dataloader()
    for data in dl:
        xs, ys = data
        print(xs.shape)
        print(ys.shape)
        break