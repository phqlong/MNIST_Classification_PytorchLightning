import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms


class MNIST_Datamodule(pl.LightningDataModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MNIST_Datamodule")
        parser.add_argument("--data_path", type=str, default="data")
        parser.add_argument("--num_workers", type=int, default=8)
        return parent_parser

    def __init__(self,
                 task_name: str = "MNIST_Datamodule",
                 batch_size: int = 64,
                 **kwargs         
    ):
        super().__init__()
        self.save_hyperparameters()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        
    def setup(self, stage: str) -> None:
        """
        Called at the beginning of fit (train + validate), validate, test, or predict. 
        This is a good hook when you need to build models dynamically or adjust something about them. 
        This hook is called on every process when using DDP.
        Args:
            stage: either 'fit', 'validate', 'test', or 'predict'
        Examles:
            class LitModel(...):
                def __init__(self):
                    self.l1 = None

                def prepare_data(self):
                    download_data()
                    tokenize()

                    # don't do this
                    self.something = else

                def setup(self, stage):
                    data = load_data(...)
                    self.l1 = nn.Linear(28, data.num_classes)

        """
        if stage in [None, 'fit']:
            dataset = MNIST(root="./data", download=False, transform=self.transform)
            training_set, validation_set = random_split(dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(2022))
            self.train_ds = training_set
            self.val_ds = validation_set


    def prepare_data(self):
        """
        Use this to download and prepare data. 
        Downloading and saving data with multiple processes (distributed settings) will result in corrupted data. 
        Lightning ensures this method is called only within a single process, so you can safely add your downloading logic within.

        Example:
            def prepare_data(self):
                # good
                download_data()
                tokenize()
                etc()

                # bad
                self.split = data_split
                self.some_state = some_other_state()
        """         
        if not os.path.exists(self.hparams.data_path+"/MNIST"):
            dataset = MNIST(root="./data", download=True)
            print(dataset)
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
