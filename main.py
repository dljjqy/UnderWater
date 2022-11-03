
import torch
import pytorch_lightning as pl
from dataset import *
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F
from trainer import *
def main(kwargs):
    # Initilize the Data Module
    dm = data_module_names[kwargs['DataModule']](**kwargs['data_module_kwargs'])

    kwargs['module_kwargs']['descaler'] = dm.descaler()
    pl_model = module_names[kwargs['Module']](**kwargs['module_kwargs'])
    # Initilize Pytorch lightning trainer
    pl_trainer = pl.Trainer(**kwargs['pl_trainer'])
    
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
    del dm, pl_model, pl_trainer
    torch.cuda.empty_cache()
    return True

kwargs_lstm = {
    'data_path' : './data/wangfeidao.csv',
    'features': 9,
    'lPre' : 42,
    'train_N': 3500,
    'val_N':500,
    'batch_size':50,
    'collate_fn':scinet_collate_fn,
    'loss':F.mse_loss,
    'lr':1e-2,
    
    'gpus': 1,
    'check_point':ModelCheckpoint(monitor= 'ValLoss', mode='min' , every_n_epochs=1, save_top_k=3, save_last=True,),
    'max_epochs': 300,
    'precision': 32,
    'check_val_every_n_epoch': 1,
    'logger':TensorBoardLogger(save_dir='./'),
    'mode': 'fit',
    'ckpt_path': False
}

kwargs_scinet = {
    'Module': 'SCIModule',
    'module_kwargs':{
        'features': 9,
        'lPre': 42,
        'lGet' : 84,
        'Tree_levels':2,
        'hidden_size_rate':2,
        'loss':F.l1_loss,
        'lr':1e-3,},
    
    'DataModule': 'WaterDataModule',
    'data_module_kwargs':{
            'path' : './data/luban.csv',
            'features': 9,
            'lPre' : 42,
            'lGet' : 84,
            'train_N': 4000,
            'val_N':100,
            'batch_size':64,
            'collate_fn':scinet_collate_fn,},
    'pl_trainer':{
        'accelerator': 'gpu',
        'devices':1,
        'callbacks':ModelCheckpoint(monitor= 'TrainLoss', mode='min' , every_n_epochs=1, save_top_k=3, save_last=True,),
        'max_epochs': 300,
        'precision': 32,
        'check_val_every_n_epoch': 1,
        'logger':TensorBoardLogger(save_dir='./'),
    },
    'mode': 'fit',
    'ckpt_path': False
}

if __name__ == '__main__':
    main(kwargs_scinet)