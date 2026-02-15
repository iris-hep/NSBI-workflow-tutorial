import torch
torch.set_float32_matmul_precision("high")
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

class PrintEpochMetrics(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        
        output = f"Epoch {trainer.current_epoch:4d} | "
        
        if "train_loss" in m:
            output += f"train_loss = {m['train_loss'].item():.6f} | "
        if "train_acc" in m:
            output += f"train_acc = {m['train_acc'].item():.4f} | "
        if "val_loss" in m:
            output += f"val_loss = {m['val_loss'].item():.6f} | "
        if "val_acc" in m:
            output += f"val_acc = {m['val_acc'].item():.4f}"
        
        print(output)

class LossHistory(pl.Callback):

    def __init__(self):
        self.train_loss = []
        self.val_loss = []

    def on_train_epoch_end(self, trainer, pl_module):
        v = trainer.callback_metrics.get("train_loss")
        if v is not None:
            self.train_loss.append(v.cpu().item())

    def on_validation_epoch_end(self, trainer, pl_module):
        v = trainer.callback_metrics.get("val_loss")
        if v is not None:
            self.val_loss.append(v.cpu().item())