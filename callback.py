import wandb
from pytorch_lightning.callbacks import Callback


class LogPredictionsCallback(Callback):
    """
    Using WandbLogger to log Images, Text and More
    Pytorch Lightning is extensible through its callback system. We can create a custom callback to automatically log sample predictions during validation. `WandbLogger` provides convenient media logging functions:
    * `WandbLogger.log_text` for text data
    * `WandbLogger.log_image` for images
    * `WandbLogger.log_table` for [W&B Tables](https://docs.wandb.ai/guides/data-vis).

    An alternate to `self.log` in the Model class is directly using `wandb.log({dict})` or `trainer.logger.experiment.log({dict})`

    In this case we log the first 20 images in the first batch of the validation dataset along with the predicted and ground truth labels.
    """
    def __init__(self, wandb_logger):
        super().__init__()
        self.wandb_logger = wandb_logger

    def on_validation_batch_end(
        self, 
        trainer, 
        pl_module, 
        outputs,
        batch, 
        batch_idx, 
        dataloader_idx
    ):
        """Called when the validation batch ends."""
 
        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case
        
        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}' for y_i, y_pred in zip(y[:n], outputs[:n])]
            
            # Option 1: log images with `WandbLogger.log_image`
            self.wandb_logger.log_image(key='sample_images', images=images, caption=captions)

            # Option 2: log predictions as a Table
            columns = ['image', 'ground truth', 'prediction']
            data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
            self.wandb_logger.log_table(key='sample_table', columns=columns, data=data)
