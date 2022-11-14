
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

    kwargs['module_kwargs']['descaler'] = dm.descaler(kwargs['new_axis'])
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
    'DataModule': 'WaterDataModule',
    'data_module_kwargs':{
            'path' : './data/luban.csv',
            'features': 9,
            'lPre' : 6,
            'lGet' : 12,
            'train_N': 9000,
            'val_N':100,
            'batch_size':32,
            'collate_fn':lstm_collate_fn,},
    
    'Module': 'SeqModule',
    'module_kwargs':{
        'features': 9,
        'lPre': 1,
        'lGet' : 12,
        'loss':F.l1_loss,
        'lr':1e-2,},
    
    'pl_trainer':{
        'accelerator': 'gpu',
        'devices':1,
        'callbacks':ModelCheckpoint(monitor= 'TrainLoss', mode='min' , every_n_epochs=1, save_top_k=3, save_last=True,),
        'max_epochs': 150,
        'precision': 32,
        'check_val_every_n_epoch': 1,
        'logger':TensorBoardLogger(save_dir='./Seq/'),
    },
    'mode': 'fit',
    'ckpt_path': False,
    'new_axis':False,
}
kwargs_scinet = {
    'Module': 'SCIModule',
    'module_kwargs':{
        'features': 9,
        'lPre': 6,
        'lGet' : 12,
        'Tree_levels':1,
        'hidden_size_rate':6,
        'loss':F.l1_loss,
        'lr':1e-2,},
    
    'DataModule': 'WaterDataModule',
    'data_module_kwargs':{
            'path' : './data/luban.csv',
            'features': 9,
            'lPre' : 6,
            'lGet' : 12,
            'train_N': 9000,
            'val_N':100,
            'batch_size':32,
            'collate_fn':scinet_collate_fn,},
    'pl_trainer':{
        'accelerator': 'gpu',
        'devices':1,
        'callbacks':ModelCheckpoint(monitor= 'TrainLoss', mode='min' , every_n_epochs=1, save_top_k=3, save_last=True,),
        'max_epochs': 150,
        'precision': 32,
        'check_val_every_n_epoch': 1,
        'logger':TensorBoardLogger(save_dir='./'),
    },
    'mode': 'fit',
    'ckpt_path': False,
    'new_axis':True,

}

if __name__ == '__main__':
    # main(kwargs_lstm)
    main(kwargs_scinet)
