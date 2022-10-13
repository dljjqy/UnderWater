import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from scipy.stats import zscore

def main(kwargs):
    # Initilize the Data Module
    dm = kwargs['pl_dataModule']

    # Initilize the model
    pl_model = kwargs['pl_model']
    # Initilize Pytorch lightning trainer
    pl_trainer = pl.Trainer(
        gpus=kwargs['gpus'],
        callbacks=kwargs['check_point'],
        max_epochs=kwargs['max_epochs'],
        precision=kwargs['precision'],
        check_val_every_n_epoch=kwargs['check_val_every_n_epoch'],
        log_every_n_steps=10,
        logger=kwargs['logger'],
    )
    if kwargs['mode'] == 'fit':
        if kwargs['ckpt_path']:
            pl_trainer.fit(
                model=pl_model,
                datamodule=dm,
                ckpt_path=kwargs['ckpt_path'])
        else:
            pl_trainer.fit(
                model=pl_model,
                datamodule=dm)
    if kwargs['mode'] == 'test':
        if kwargs['ckpt_path']:
            pl_trainer.test(
                model=pl_model,
                datamodule=dm,
                ckpt_path=kwargs['ckpt_path'])
        else:
            print("No ckpt_path,CAN NOT USE UNTRAINED MODEL FOR TEST")
            return False

    del dm, pl_model, pl_trainer
    torch.cuda.empty_cache()
    return True

def dataClean(excel_path, save_path ='./data2.csv',header=2, labels=['采集时间', '水温', 'pH', '溶解氧']):
    # Read the original data and copy one.
    # 'Date' is default as the first colume.
    df = pd.read_excel(excel_path, header=2, usecols=labels).copy()
    df.set_index(labels[0], inplace=True)
    
    # Clean the wrong data type and labeled them as 'None'
    for k in labels[1:]:
        df[k].mask(df[k] == '--', None, inplace=True)
    df = df.astype('float64')
    df = df.abs()
    
    # Clean the Outliers
    for k in labels[1:]:
        vals = df[k].values.copy()
        idxs = compute_zscore(df, k, 1.2)
#         idxs = detect_outlier(df, k, 4)
        vals[idxs] = None
        df.loc[:, k] = vals
    
    # Drop rows with nan
    df.dropna(inplace=True)
    
    #  Denoise by moving average
    for k in labels[1:]:
        vals = df[k].values.copy()
        smooth_vals = pd.Series(vals).rolling(window=18).mean()
        df.loc[:, k] = vals
        
    # Sort by the date and remove incomplete data
    ls = []
    df = df.reset_index(drop=False)
    times = pd.to_datetime(arg=df[labels[0]], format='%Y-%m-%d %H:%M:%S')
    for group in df.groupby([times.dt.year, times.dt.month, times.dt.day]):
        if group[1].shape[0] == 6:
            dfn = group[1][::-1].values.copy()
            ls.append(dfn)
    arr = np.concatenate(ls, axis=0)
    new_data = pd.DataFrame(arr, columns=labels)
    new_data.to_csv(save_path, index=False)
    return new_data

def compute_zscore(df, k, threshold=1.5):
    '''
    使用标准差来筛选数据，返回异常数据坐标
    '''
    all_value = df[k].values.copy()
    indices = np.array(list(map(lambda x: not x, np.isnan(all_value))))
    true_value = all_value[indices]
#     print(true_value.mean())
    z_value = zscore(true_value)
    
    all_value[indices] = z_value
    all_value = pd.Series(all_value)
    return all_value.abs() > threshold


def detect_outlier(df, label, rate=4):
    '''
    使用分位数来筛选数据，返回异常数据坐标
    '''
    all_values = df[label].values.copy()
    indices = np.array(list(map(lambda x: not x, np.isnan(all_values))))
    true_values = all_values[indices]
    
    Q1 = np.percentile(true_values, rate)
    Q3 = np.percentile(true_values, 100-rate)    
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    all_values = pd.Series(all_values)
    return (all_values < lower_limit)&(all_values > upper_limit)
