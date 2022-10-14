
import torch
import pytorch_lightning as pl
from dataset import WaterDataModule
from models import Seq2Seq
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from trainer import SeqModule
def main(kwargs):
    # Initilize the Data Module
    dm = kwargs['pl_dataModule']

    # Initilize the model
    pl_model = kwargs['pl_model']
    # Initilize Pytorch lightning trainer
    pl_trainer = pl.Trainer(
        accelerator='gpu',
        devices=kwargs['gpus'],
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

kwargs = {
    'pl_dataModule': WaterDataModule('./data3.csv', lGet=300, lPre=30, 
                        train_N=3500, val_N=50, batch_size=10),
    'pl_model': SeqModule(features=3, lGet=24, lPre=6),
    'gpus': 1,
    'check_point':ModelCheckpoint(monitor= 'ValLoss', mode='min' , every_n_epochs=1, save_top_k=3, save_last=True,),
    'max_epochs': 200,
    'precision': 32,
    'check_val_every_n_epoch': 1,
    'logger':TensorBoardLogger(save_dir='./'),
    'mode': 'fit',
    'ckpt_path': False
}

if __name__ == '__main__':
    main(kwargs)