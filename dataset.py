import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

def minmaxScaler(df):
    df_norm = (df - df.min())/(df.max() - df.min())
    return df_norm, df.max().values, df.min().values

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
    def __init__(self, path, features=3, lGet=24, lPre=6, train_N=3000, val_N=100, batch_size=10):
        '''
        lGet: How many rows used to predict.
        lPre: How many rows you want to predict.
        train_N: How many groups of data you need for train.
        val_N: How mant groups of data you need for validation.
                One group equals to lGet and lPre rows of data
        '''
        super().__init__()
        df, self.maxs, self.mins = minmaxScaler(pd.read_csv(path, index_col=0))
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
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_ds = WaterDataSet(self.features, self.train_data, self.lGet, self.lPre)
            self.val_ds = WaterDataSet(self.features, self.val_data, self.lGet, self.lPre)
        elif stage == 'test':
            pass
    
    def train_dataloader(self):
        return DataLoader(self.train_ds,shuffle=True, num_workers=6, collate_fn=my_collate_fn, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, shuffle=False, num_workers=0, collate_fn=my_collate_fn, batch_size=1)

    def test_dataloader(self):
        pass

if __name__ == '__main__':
    dm = WaterDataModule('./data3.csv')
    dm.setup()
    print(len(dm.val_ds))
    print(len(dm.train_ds))
    dl = dm.train_dataloader()
    i = 0
    for data in dl:
        i += 1
    print(i)