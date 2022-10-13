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
    def __init__(self, data):
        super().__init__()
        self.data, self.maxs, self.mins = minmaxScaler(data)
        self.data = torch.from_numpy(self.data).to(dtype=torch.float16)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return self.data.shape[0]

class WaterDataModule(pl.LightningDataModule):
    def __init__(self, path, ratio=0.8, batch_size=10):
        data = pd.read_csv(path, index_col=0).values.copy()
        n, features = data.shape
        train_N = int(n * ratio)
        self.train_data = data[:train_N, :]
        self.val_data = data[train_N:, :]
        self.batch_size = batch_size
    
    def setup(self, stage):
        if stage == 'fit' or stage is None:
            self.train_ds = WaterDataSet(self.train_data)
            self.val_ds = WaterDataSet(self.val_data)
        elif stage == 'test':
            pass
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, self.batch_size, shuffle=True, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.val_ds, 1, shuffle=False, num_workers=6)

    def test_dataloader(self):
        pass

if __name__ == '__main__':
    ds = WaterDataSet('./data3.csv')
    print(len(ds))
    print(ds.data.shape)
    print(type(ds.data))
    print(ds[0])
