import torch
torch.set_float32_matmul_precision("medium")
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

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
                callback_patience = 30,
                use_cosine_anneal_scheduler = False):
        
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
        self.out = nn.Linear(input_dim_, 1)

        if use_log_loss:
            self.from_logits = True
        else:
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

        if self.hparams.use_cosine_anneal_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=100,
                eta_min=1e-6
            )

        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams.callback_patience,   
                gamma=self.hparams.callback_factor        
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
