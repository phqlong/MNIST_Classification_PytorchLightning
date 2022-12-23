import torch
import wandb
from argparse import ArgumentParser
from datetime import datetime, timezone, timedelta

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import WandbLogger

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, DeviceStatsMonitor, StochasticWeightAveraging

from data import MNIST_Datamodule
from model import MNIST_ClassificationModel
from callback import LogPredictionsCallback


def parse_args():
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("--conda_env", type=str, default="base")
    parser.add_argument('--debuging', type = bool, default = True)
    parser.add_argument('--wandb_key', type = str, default="")
    
    # add Model & datamodule specific args
    parser = MNIST_ClassificationModel.add_model_specific_args(parser)
    parser = MNIST_Datamodule.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --accelerator --devices --num_nodes ... --fast_dev_run all work in the cli
    # parser = Trainer.add_argparse_args(parser,)

    args = parser.parse_args()
    dict_args = vars(args)

    print(args)
    return args, dict_args


def main():
    seed_everything(2022)
    args, dict_args = parse_args()

    # Init data module & model
    datamodule = MNIST_Datamodule(**dict_args)
    model = MNIST_ClassificationModel(**dict_args)
    print(model)

    # Login to wandb, run in cmd: wandb login
    wandb_logger = WandbLogger(project='MNIST', # group runs in "MNIST" project
                                name=datetime.now(tz=timezone(+timedelta(hours=7))).strftime("%Y%m%d-%Hh%Mm%Ss"),
                                log_model='all') # log all new checkpoints during training
    
    # Setup callbacks fior Trainer
    checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max')
    early_stopping_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")
    log_predictions_callback = LogPredictionsCallback(wandb_logger)

    trainer = Trainer(
        default_root_dir = 'checkpoints/',
        logger=wandb_logger,                    # W&B integration
        callbacks=[log_predictions_callback,    # logging of sample predictions
                    checkpoint_callback],        # our model checkpoint callback
        accelerator="gpu",                      # use GPU
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=10,
        # fast_dev_run=True,
        # limit_train_batches=10, 
        # limit_val_batches=5,
        # profiler="simple",
        # callbacks = [
        #   early_stop_callback,
        #   DeviceStatsMonitor(),
        #   StochasticWeightAveraging(swa_lrs=1e-2)
        # ],
        # devices=1 if torch.cuda.is_available() else None,
        # auto_scale_batch_size = 'binsearch',
        # auto_lr_find=True,
        )

    trainer.fit(model, datamodule=datamodule)
    wandb.finish()

if __name__ == "__main__":
    main()
