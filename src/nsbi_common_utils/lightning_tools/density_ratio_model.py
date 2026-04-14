from functools import partial

import torch
torch.set_float32_matmul_precision("medium")
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

class DensityRatioLightning(pl.LightningModule):
    '''
    Pytorch-lighning module for estimation of density ratios
    '''
    def __init__(self,
                n_hidden        = 4,
                n_neurons       = 1000,
                input_dim       = 11,
                learning_rate   = 0.1,
                use_log_loss    = False,
                activation      = "swish", 
                callback_factor = 0.01, 
                callback_patience = 30):
        
        super().__init__()

        self.save_hyperparameters()

        self.lr = learning_rate
        self.use_log_loss = use_log_loss

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

        if use_log_loss:
            self.out = nn.Linear(input_dim_, 1)
            self.from_logits = True
        else:
            self.out = nn.Linear(input_dim_, 1)
            self.from_logits = False

    def forward(self, x):

        x = self.mlp(x)
        x = self.out(x)
        if not self.use_log_loss:
            x = torch.sigmoid(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y, w = batch
        y = y.float().view(-1, 1)
        w = w.float().view(-1, 1)

        s_hat = self(x)
        if self.use_log_loss:
            loss = F.binary_cross_entropy_with_logits(s_hat, y, reduction="none")
        else:
            loss = F.binary_cross_entropy(s_hat, y, reduction="none")

        weighted_loss = (loss * w).sum() / w.sum()
    
        self.log("train_loss", weighted_loss, prog_bar=True, on_step=False, on_epoch=True)
        return weighted_loss
    
    
    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        y = y.float().view(-1, 1)
        w = w.float().view(-1, 1)

        s_hat = self(x)

        if self.use_log_loss:
            loss = F.binary_cross_entropy_with_logits(s_hat, y, reduction="none")
        else:
            loss = F.binary_cross_entropy(s_hat, y, reduction="none")

        loss = (loss * w).sum() / w.sum()

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):

        optimizer = torch.optim.NAdam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.callback_patience,   
            gamma=self.hparams.callback_factor        
        )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=100,
        #     eta_min=1e-11
        # )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
    



class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
    

class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)



class LoRADensityRatioLightning(pl.LightningModule):
    '''
    Pytorch-lighning LoRA module for estimation of density ratios
    '''
    def __init__(self,
                model, # the model to be finetuned
                lora_rank, # the rank of the LoRA finetuning
                lora_alpha=1.0, # the alpha parameter for scaling the LoRA updates
                learning_rate   = 0.1,
                use_log_loss    = False,
                callback_factor = 0.01, 
                callback_patience = 30):
        
        super().__init__()

        self.save_hyperparameters()

        self.lr = learning_rate
        self.use_log_loss = use_log_loss


        # Freeze the original model parameters, only the LoRA layers will be updated during training
        for param in model.parameters():
            param.requires_grad = False 


        assign_lora = partial(LinearWithLoRA, rank=lora_rank, alpha=lora_alpha)

        # Build architecture - feed forward MLP
        layers = []
        last_dim = None

        # MLP
        for layer in model.mlp:
            if isinstance(layer, nn.Linear):
                lora_layer = assign_lora(layer)
                layers.append(lora_layer)
                last_dim = layer.out_features
            else:
                layers.append(layer)
        # Output layer
        if isinstance(model.out, nn.Linear):
            lora_layer = assign_lora(model.out)
            layers.append(lora_layer)
            last_dim = model.out.out_features
        else:
            layers.append(model.out)
            last_dim = model.out.out_features

        self.lora_mlp = torch.compile(nn.Sequential(*layers))

        print(f"Initialized LoRA finetuning with rank {lora_rank} and alpha {lora_alpha}")
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_percentage = 100 * trainable_params / total_params if total_params > 0 else 0
        print(f"Trainable parameters: {trainable_params}")
        print(f"Total parameters: {total_params}")
        print(f"Percentage of trainable parameters: {trainable_percentage:.4f}%")

        if use_log_loss:
            self.out = nn.Linear(last_dim, 1)
            self.from_logits = True
        else:
            self.out = nn.Linear(last_dim, 1)
            self.from_logits = False

    def forward(self, x):

        x = self.lora_mlp(x)
        x = self.out(x)
        if not self.use_log_loss:
            x = torch.sigmoid(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y, w = batch
        y = y.float().view(-1, 1)
        w = w.float().view(-1, 1)

        s_hat = self(x)
        if self.use_log_loss:
            loss = F.binary_cross_entropy_with_logits(s_hat, y, reduction="none")
        else:
            loss = F.binary_cross_entropy(s_hat, y, reduction="none")

        weighted_loss = (loss * w).sum() / w.sum()
    
        self.log("train_loss", weighted_loss, prog_bar=True, on_step=False, on_epoch=True)
        return weighted_loss
    
    
    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        y = y.float().view(-1, 1)
        w = w.float().view(-1, 1)

        s_hat = self(x)

        if self.use_log_loss:
            loss = F.binary_cross_entropy_with_logits(s_hat, y, reduction="none")
        else:
            loss = F.binary_cross_entropy(s_hat, y, reduction="none")

        loss = (loss * w).sum() / w.sum()

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):

        optimizer = torch.optim.NAdam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.callback_patience,   
            gamma=self.hparams.callback_factor        
        )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=100,
        #     eta_min=1e-11
        # )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
