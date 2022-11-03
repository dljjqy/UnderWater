from inspect import Attribute
from random import triangular
import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

def minmaxScaler(df):
    df_norm = (df - df.min())/(df.max() - df.min())
    return df_norm, df.max().values, df.min().values

def meanstdScaler(df):
    df_norm = (df -df.mean())/ df.std()
    return df_norm, df.mean().values, df.std().values()

def deScaler_meanstd(mean, std, vals):
    denorm_val = vals * std + mean
    return denorm_val

def deScaler_minman(max, min, vals):
    denorm_val = vals * (max - min) + min
    return denorm_val


class WaterDataSet(Dataset):
    def __init__(self, features, data, lGet, lPre):
        super().__init__()
        self.data = torch.from_numpy(data).to(dtype=torch.float32)
        self.lGet = lGet
        self.lPre = lPre
        self.features = features

    def __getitem__(self, idx):
        x = self.data[idx: idx+self.lGet, self.features:]
        y = self.data[idx: idx+self.lPre, :self.features]
        return x, y
    
    def __len__(self):
        return self.data.shape[0] - self.lGet + 1

def lstm_collate_fn(data):
    '''
    L x N x F
    '''
    xs, ys = zip(*data)
    xs = torch.stack(xs, dim=1)
    ys = torch.stack(ys, dim=1)
    return (xs, ys)

def scinet_collate_fn(data):
    '''
    N x F x L
    '''
    xs, ys = zip(*data)
    xs = torch.stack(xs, dim=0)
    ys = torch.stack(ys, dim=0)
    xs = torch.permute(xs, (0, 2, 1))
    ys = torch.transpose(ys, 1, 2)
    return (xs, ys)


class WaterDataModule(pl.LightningDataModule):
    def __init__(self, path, features=3, lGet=24, lPre=6, train_N=3000, val_N=100, 
                batch_size=10, collate_fn=lstm_collate_fn):
        '''
        lGet: How many rows used to predict.
        lPre: How many rows you want to predict.
        train_N: How many groups of data you need for train.
        val_N: How mant groups of data you need for validation.
                One group equals to lGet and lPre rows of data
        '''
        super().__init__()
        df = pd.read_csv(path, index_col=0)
        self.mean, self.std = df.loc['mean'].values, df.loc['var'].values
        self.mean = self.mean[..., np.newaxis]
        self.std = self.std[..., np.newaxis]

        df = df.drop(['mean', 'var'])
        
        valdf = df.values.copy()
        valdfs = df.shift(lGet).values.copy()
        data = np.concatenate((valdf, valdfs), axis=1)
        data = data[~np.isnan(data).any(axis=1)]

        train_end = train_N + lGet - 1
        val_end = train_end + val_N + lGet - 1
        self.train_data = data[0: train_end, :].copy()
        self.val_data = data[train_end: val_end, :].copy()
        
        self.batch_size = batch_size
        self.lGet = lGet
        self.lPre = lPre
        self.features = features
        self.collate = collate_fn
    
    def descaler(self):
        return lambda x:deScaler_meanstd(self.mean, self.std, x)


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_ds = WaterDataSet(self.features, self.train_data, self.lGet, self.lPre)
            self.val_ds = WaterDataSet(self.features, self.val_data, self.lGet, self.lPre)
        elif stage == 'test':
            pass
    
    def train_dataloader(self):
        return DataLoader(self.train_ds,shuffle=True, num_workers=6, collate_fn=self.collate, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, shuffle=False, num_workers=6, collate_fn=self.collate, batch_size=10)

    def test_dataloader(self):
        pass


data_module_names = {'WaterDataModule': WaterDataModule}

if __name__ == '__main__':
    dm = WaterDataModule('./data/luban.csv', features=9, lPre=42, lGet=84, batch_size=1,
            train_N=3000, val_N=1000, collate_fn=scinet_collate_fn)
    dm.setup()
    dl = dm.val_dataloader()
    x, y = dm.val_ds[0]
    print(x[0, :])
    print(x.shape)
    print(y.shape)
    for data in dl:
        x, y = data
        print(x.shape)
        print(y.shape)
        print(x[0, :, 0])
        break
    # print(x[0, :])
    # print(dm.descaler()(x))