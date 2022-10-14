
import torch
import pytorch_lightning as pl
from dataset import WaterDataModule
from models import Seq2Seq
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
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

kwargs = {
    'pl_dataModule': WaterDataModule('./data3.csv', lGet=24, lPre=6, 
                        train_N=1000, val_N=10, batch_size=10),
    'pl_model': Seq2Seq(features=3, hidsize=256, Eembsize=128, Dembsize=128, 
                        n_layers=2, dropout=0.5),
    'gpus': 1,
    'check_point':ModelCheckpoint(monitor= 'TrainLoss', mode='min', every_n_train_steps=0,
                                        every_n_epochs=1, train_time_interval=None, save_top_k=3, save_last=True,),
    'max_epochs': 150,
    'precision': 32,
    'check_val_every_n_epoch': 1,
    'logger':TensorBoardLogger('./lightning_logs/', 'test_train'),
    'mode': 'fit',
    'ckpt': False
}

if __name__ == '__main__':
    main(kwargs)