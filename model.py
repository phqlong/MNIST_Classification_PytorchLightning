import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torchmetrics.functional import accuracy


class MNIST_ClassificationModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MNIST_ClassificationModel")
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--lr", type=int, default=1e-3)
        return parent_parser

    def __init__(
        self,
        n_classes : int = 10, 
        n_layer_1 : int = 128, 
        n_layer_2 : int = 256, 
        **kwargs
    ):
        super().__init__()
        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

        # MNIST images are (1, 28, 28) (channels, width, height)
        self.model = nn.Sequential(
            nn.Linear(28 * 28, n_layer_1),
            nn.ReLU(),
            nn.Linear(n_layer_1, n_layer_2),
            nn.ReLU(),
            nn.Linear(n_layer_2, n_classes)
        )

        # loss fuction
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs):
        """ For inference: inputs -> outputs """
        batch_size, channels, width, height = inputs.size()

        # Transform size: (b, 1, 28, 28) -> (b, 1*28*28)
        inputs = inputs.view(batch_size, -1)

        # let's do 3 x (linear + relu)
        outputs = self.model(inputs)
        return outputs


    def training_step(self, batch, batch_idx):
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('train_loss', loss)
        self.log('train_accuracy', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('val_loss', loss)
        self.log('val_accuracy', acc)

        # Let's return preds to use it in a custom callback
        return preds

    def test_step(self, batch, batch_idx):
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('test_loss', loss)
        self.log('test_accuracy', acc)
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)
    
    def _get_preds_loss_accuracy(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        acc = accuracy(preds, y, 'multiclass', num_classes=10)
        return preds, loss, acc
