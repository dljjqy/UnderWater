import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


class WaterDataSet(Dataset):
    def __init__(self, path, scaler, labels=None):
        super().__init__()

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass
