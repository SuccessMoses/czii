import lightning.pytorch as pl
from monai.losses import TverskyLoss
from monai.metrics import DiceMetric
import torch

class LiteModel(pl.LightningModule):
    def __init__(self, batch_size, dim_in, chennels, model_name, model, lr: float=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.loss_fn = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True)  # softmax=True for multiclass
        self.metric_fn = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)

        self.train_loss = 0
        self.val_metric = 0
        self.num_train_batch = 0
        self.num_val_batch = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.train_loss += loss
        self.num_train_batch += 1
        torch.cuda.empty_cache()
        return loss

    def on_train_epoch_end(self):
        loss_per_epoch = self.train_loss/self.num_train_batch
        self.log('train_loss', loss_per_epoch, prog_bar=True)
        self.train_loss = 0
        self.num_train_batch = 0
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
