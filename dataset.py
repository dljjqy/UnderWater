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
        self.data, self.maxs, self.mins = minmaxScaler(data)
        self.data = torch.from_numpy(self.data).to(dtype=torch.float16)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return self.data.shape[0]

def my_collate_fn(data):
    '''
    Set the batch dimension as the second not first.
    L x N x F
    '''
    print(len(data))
    print(type(data))
    print(data[0])
    xs, ys = zip(*data)
    xs = torch.stack(xs, dim=1)
    ys = torch.stack(ys, dim=1)
    return (xs, ys)


class WaterDataModule(pl.LightningDataModule):
    def __init__(self, path, lGet, lPre, ratio=0.9, batch_size=10):
        data = pd.read_csv(path, index_col=0).values.copy()
        n, features = data.shape
        train_N = int(n * ratio)
        self.train_data = data[:train_N, :]
        self.val_data = data[train_N:, :]
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_ds = WaterDataSet(self.train_data)
            self.val_ds = WaterDataSet(self.val_data)
        elif stage == 'test':
            pass
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, self.batch_size, shuffle=True, num_workers=0, collate_fn=my_collate_fn)

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
        break