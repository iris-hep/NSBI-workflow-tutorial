import torch
torch.set_float32_matmul_precision("high")
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

class MultiClassLightning(pl.LightningModule):
    def __init__(self,
                n_hidden        = 4,
                n_neurons       = 1000,
                input_dim       = 11,
                learning_rate   = 0.1,
                activation      = "swish",
                num_classes     = 3,
                callback_factor = 0.1,
                callback_patience = 30
                ):
        
        super().__init__()

        self.save_hyperparameters()
        
        self.lr = learning_rate

        # Get activation function
        activations = {
            "swish": nn.SiLU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
        }
        activation_choice = activations.get(activation, nn.SiLU)

        # Build architecture - feed forward MLP
        layers = []
        input_dim_ = input_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(input_dim_, n_neurons))
            layers.append(activation_choice())
            input_dim_ = n_neurons
        
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(input_dim_, num_classes)

    def forward(self,x):
        x = self.mlp(x)
        return self.out(x)

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        y = y.long()
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, reduction='none')

        # batch_size = y.size(0)
        loss = (loss * w).sum() / w.sum()

        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        y = y.long()
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, reduction='none')

        # batch_size = y.size(0)
        loss = (loss * w).sum() / w.sum()

        self.log("val_loss", loss, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):

        optimizer = torch.optim.NAdam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.hparams.callback_factor,
            patience=self.hparams.callback_patience,
            min_lr=1e-9
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",   
                "interval": "epoch",
                "frequency": 1
            }
        }
    