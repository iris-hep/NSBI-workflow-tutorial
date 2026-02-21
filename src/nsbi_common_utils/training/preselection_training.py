#import libraries
import os, importlib, sys, shutil
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 
import math
import pickle 

import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import torch
torch.set_float32_matmul_precision("high")
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from torch.utils.data import Subset

from nsbi_common_utils.lightning_tools import MultiClassLightning, DensityRatioLightning, PrintEpochMetrics, LossHistory, WeightedTensorDataset
from nsbi_common_utils.training import save_model, predict_with_onnx, convert_torch_to_onnx, convert_logLR_to_score, load_trained_model


from pathlib import Path
from typing import Union, Dict
from joblib import dump, load

import onnx
import onnxruntime as rt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.compose import ColumnTransformer

from nsbi_common_utils.calibration import HistogramCalibrator, IsotonicCalibrator
from nsbi_common_utils.plotting import plot_loss, plot_all_features, plot_all_features, plot_reweighted, plot_calibration_curve, plot_calibration_curve_ratio, plot_overfit_side_by_side

import logging
_LOG_LEVELS = {
    0: logging.WARNING,  
    1: logging.INFO,    
    2: logging.DEBUG,   
}

logger = logging.getLogger("Training Logs")
logger.propagate = True  


def configure_logging(verbose_level: int = 1):
    """
    Configure the logger
    """
    level = _LOG_LEVELS.get(verbose_level, logging.INFO)
    logger.setLevel(level)

    if not logger.handlers:
        h = logging.StreamHandler()
        h.setLevel(level)
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        h.setFormatter(fmt)
        logger.addHandler(h)


class preselection_network_trainer:
    '''
    A class for training the multi-class classification neural network 
    used to preselect phase space for SBI
    '''
    def __init__(self, dataset, features, features_scaling, 
                    train_labels_column = 'train_labels',
                    weights_normed_column = 'weights_normed'):
        '''
        dataset: dataframe with the multiple classes for training
        num_classes: number of classes corresponding to the number of output nodes of softmax layer
        features: input features to use for training
        features_scaling: subset of input features to standardize before training
        '''
        self.dataset                            = dataset
        self.data_features_training             = dataset[features].copy()
        self.features                           = features
        self.features_scaling                   = features_scaling
        self.num_classes                        = len(np.unique(dataset.train_labels))

        self.train_labels_column                = train_labels_column
        self.weights_normed_column              = weights_normed_column

    # Defining a simple NN training for preselection - no need for "flexibility" here
    def train(self, test_size=0.15, 
                    random_state=42, 
                    path_to_save='', 
                    epochs=20, 
                    batch_size=1024, 
                    hidden_layers=4,
                    neurons=1000,
                    verbose=2, 
                    learning_rate=0.1,
                    validation_split=0.1,
                    activation='swish',
                    num_workers=4):

        '''
        The function will train the preselection NN, assign it to self.model variable, and save the model to user-provided path_to_save directory.

        test_size:      the fraction of dataset to set aside for diagnostics, not used in training and validation of the loss vs epoch curves
        random_state:   random state to use for splitting the train/test dataset before training NN
        epochs:         the number of epochs to train the NNs
        batch_size:     the size of each batch used during gradient optimization
        learning_rate:  the initial learning rate to pass to the optimizer
        '''

        # Split data into training and validation sets (including weights)
        data_train, data_holdout, y_train, y_holdout, weight_train, weight_holdout = train_test_split(self.data_features_training, 
                                                                                                    self.dataset[self.train_labels_column], 
                                                                                                    self.dataset[self.weights_normed_column], 
                                                                                                    test_size=test_size, 
                                                                                                    random_state=random_state, 
                                                                                                    stratify=self.dataset[self.train_labels_column])

        # Standardize the input features
        self.scaler = ColumnTransformer([("scaler", StandardScaler(), self.features_scaling)],remainder='passthrough')
        data_train_scaled = self.scaler.fit_transform(data_train)  # Fit & transform training data
        data_holdout = self.scaler.transform(data_holdout)

        train_ds = WeightedTensorDataset(
                data_train_scaled,
                y_train,
                weight_train
            )

        val_size = int(len(train_ds) * validation_split)
        train_size = len(train_ds) - val_size
        
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

        train_loader = DataLoader(train_ds, 
                                  batch_size=batch_size, 
                                  shuffle=True,
                                  num_workers=num_workers,  
                                pin_memory=True,  
                                persistent_workers=False  
                                )
        
        val_loader   = DataLoader(val_ds, 
                                  batch_size=batch_size, 
                                  shuffle=False, 
                                  num_workers=num_workers,  
                                pin_memory=True,  
                                persistent_workers=False  
                                )
        
        self.model = MultiClassLightning(
                n_hidden=hidden_layers,
                n_neurons=neurons,
                input_dim=len(self.features),
                learning_rate=learning_rate,
                activation=activation,
                num_classes=self.num_classes
            )

        loss_history = LossHistory()

        self.trainer = Trainer(
            accelerator="auto",
            devices="auto",
            max_epochs=epochs,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=30),
                LearningRateMonitor(),
                loss_history,
                PrintEpochMetrics()
            ],
            logger=True,
            enable_checkpointing=False,
            enable_progress_bar=True,
            # precision='32-true'
        )

        self.trainer.fit(self.model, train_loader, val_loader)

        # Save the trained model if user provides with a path
        if path_to_save=='': 
            path_to_save='./'
        else:
            path_to_save      = Path(path_to_save)
            path_to_save.mkdir(parents=True, exist_ok=True)

        path_to_model           = path_to_save / 'model_preselection.onnx'
        path_to_scaler          = path_to_save / 'model_scaler_presel.bin'

        # Save lightning module as ONNX
        save_model(self.model, 
                   torch.randn((1, len(self.features))), 
                   path_to_model, 
                   self.scaler, 
                   path_to_scaler,
                   softmax_output = True)

        # Reassign the model to be in ONNX format
        self.scaler, self.model = load_trained_model(path_to_model, path_to_scaler)


    def assign_trained_model(self, 
                         path_to_models: str) -> None:
        '''
        Method to load the trained model

        path_to_models: path to the directory with saved model and scaler files
        '''

        path_to_saved_scaler        = path_to_models + '/model_scaler_presel.bin'
        path_to_saved_models        = path_to_models + '/model_preselection.onnx'

        self.scaler, self.model     = load_trained_model(path_to_saved_models, path_to_saved_scaler)

    def predict(self, dataset):
        '''
        Method that evaluates density ratios on provided dataset, using self.model

        dataset: the dataset to evaluate trained model on
        '''
        pred                        = predict_with_onnx(dataset[self.features], 
                                                        self.scaler, 
                                                        self.model)
        
        return pred

    