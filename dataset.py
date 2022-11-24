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

def deScaler_minmax(max, min, vals):
    denorm_val = vals * (max - min) + min
    return denorm_val


class WaterDataSet(Dataset):
    def __init__(self, data, lGet, lPre):
        super().__init__()
        self.data = torch.from_numpy(data).to(torch.float32)
        self.lGet = lGet
        self.lPre = lPre
        
    def __getitem__(self, idx):
        x = self.data[idx, :self.lGet]
        y = self.data[idx, self.lGet:]
        return x, y
    
    def __len__(self):
        return self.data.shape[0]

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
    def __init__(self, path, lGet=24, lPre=6, batch_size=10, collate_fn = scinet_collate_fn):
        '''
        lGet: How many rows used to predict.
        lPre: How many rows you want to predict.
        '''
        super().__init__()
        self.path = path
        self.lGet = lGet
        self.lPre = lPre
        data = np.load(path)
        describe = pd.read_csv(f'{path[:-4]}_describe.csv', index_col=0)
        # mins = describe.loc['min'].values
        # maxs = describe.loc['max'].values
        # mins, maxs = mins.reshape(1, 1, -1), maxs.reshape(1, 1, -1)
        # self.descaler = lambda x: (x - mins) /(maxs - mins) 
        # self.scaler = lambda x: x * (maxs - mins) + mins

        means = describe.loc['mean'].values.reshape(1, 1, -1)
        stds = describe.loc['std'].values.reshape(1, 1, -1)
        self.descaler = lambda x: x * stds + means
        self.scaler = lambda x: (x - means) / stds
        
        data = self.scaler(data)
        l = data.shape[0]
        idx = np.arange(l)
        np.random.shuffle(idx)
        trainN = l - 200
 
        self.train_data = data[idx[:trainN]]
        self.val_data = data[idx[trainN:]]
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_ds = WaterDataSet(self.train_data, self.lGet, self.lPre)
            self.val_ds = WaterDataSet(self.val_data, self.lGet, self.lPre)
        elif stage == 'test':
            pass
    
    def train_dataloader(self):
        return DataLoader(self.train_ds,shuffle=True, num_workers=6, collate_fn=self.collate_fn, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, shuffle=False, num_workers=0, collate_fn=self.collate_fn ,batch_size=10)

    def test_dataloader(self):
        pass


data_module_names = {'WaterDataModule': WaterDataModule}

if __name__ == '__main__':
    dm = WaterDataModule('./all_data/fujiang_1d/all.npy', 18, 6, 0.95, 1)
    dm.setup()
    dl = dm.train_dataloader()

    
