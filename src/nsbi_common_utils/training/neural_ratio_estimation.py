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
from nsbi_common_utils.training.utils import save_model, predict_with_onnx, convert_torch_to_onnx, convert_logLR_to_score, load_trained_model

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

        
class density_ratio_trainer:
    '''
    A class for training the density ratio neural networks for SBI analysis
    '''
    def __init__(self, dataset, 
                      weights, 
                      training_labels, 
                      features, 
                      features_scaling, 
                      sample_name, 
                      output_name, 
                      path_to_figures='',
                      path_to_models='', 
                      path_to_ratios='',
                      use_log_loss=False, 
                      split_using_fold=False,
                      delete_existing_models=False):
        '''
        dataset: the main dataframe containing two classes p_A, p_B for density ratio p_A/p_B estimation
        weights: the weight vector, normalized independently for each class A & B
        training_labels: array of 1s for p_A hypothesis and 0s for p_B hypothesis
        features: training features x in p_A(x)/p_B(x)
        features_scaling: training features to standardize before training
        sample_name: set with strings containing names of A and B
        '''
        self.dataset = dataset
        self.weights = weights
        self.training_labels = training_labels
        self.features = features
        self.features_scaling = features_scaling
        self.sample_name = sample_name
        self.output_name = output_name
        self.use_log_loss = use_log_loss
        self.split_using_fold = split_using_fold

        # Initialize a list of models to train - if no ensemble, this is a 1 member list
        self.model_NN = [None]
        self.scaler = [None]
        
        self.path_to_figures = path_to_figures

        if delete_existing_models:
            if os.path.exists(path_to_figures):
                shutil.rmtree(path_to_figures)
            
            if os.path.exists(path_to_models):
                shutil.rmtree(path_to_models)
                
            if os.path.exists(path_to_ratios):
                shutil.rmtree(path_to_ratios)


        if not os.path.exists(path_to_figures):
                os.makedirs(path_to_figures)
        
        self.path_to_models = path_to_models
        if not os.path.exists(path_to_models):
                os.makedirs(path_to_models)

        self.path_to_ratios=path_to_ratios
        if not os.path.exists(path_to_ratios):
                os.makedirs(path_to_ratios)
        
    def train_ensemble(self, hidden_layers, 
                            neurons, 
                            number_of_epochs, 
                            batch_size,
                            learning_rate, 
                            scalerType, 
                            calibration=False, 
                            type_of_calibration="isotonic", 
                            num_bins_cal = 40, 
                            callback = True, 
                            callback_patience=30, 
                            callback_factor=0.01,
                            activation='swish', 
                            verbose=2, 
                            validation_split=0.1, 
                            holdout_split=0.3, 
                            plot_scaled_features=False, 
                            load_trained_models = False,
                            recalibrate_output=False,
                            summarize_model: bool = False,
                            num_ensemble_members=1,
                            num_workers = 0):
        '''
        Train an ensemble of NNs
        '''
        logger.info(f"starting ensemble training")
        self.num_ensemble_members = num_ensemble_members
        
        # Define an array with random integers for boostrap training
        random_state_arr = np.random.randint(0, 2**32 -1, size=num_ensemble_members)

        self.model_NN           = [None for i in range(num_ensemble_members)]
        self.histogram_calibrator = [None for i in range(num_ensemble_members)]
        self.scaler             = [None for i in range(num_ensemble_members)]
        self.full_data_prediction     = [None for i in range(num_ensemble_members)]

        self.full_data_prediction = np.zeros((num_ensemble_members, len(self.weights)))
        self.train_idx          = [None for i in range(num_ensemble_members)]
        self.holdout_idx        = [None for i in range(num_ensemble_members)]

        # Train ensemble of NNs in series
        for ensemble_index in range(num_ensemble_members):

            if load_trained_models:
                if os.path.exists(f"{self.path_to_models}/model{ensemble_index}.onnx"):
                    logger.info(f"Loading existing model for ensemble member {ensemble_index}")
                    load_trained_models_ensemble_member = True
                else:
                    load_trained_models_ensemble_member = False

            else:
                load_trained_models_ensemble_member = False
            
            # Train ensemble NNs with different train/test split each time (bootstrapping without replacement)
            self.train(hidden_layers, 
                        neurons, 
                        number_of_epochs, 
                        batch_size,
                        learning_rate, 
                        scalerType, 
                        calibration, 
                        type_of_calibration,
                        num_bins_cal, 
                        callback, 
                        callback_patience, 
                        callback_factor,
                        activation, 
                        verbose                 = verbose,
                        rnd_seed                = random_state_arr[ensemble_index], 
                        ensemble_index          = ensemble_index, 
                        validation_split        = validation_split, 
                        holdout_split           = holdout_split, 
                        plot_scaled_features    = plot_scaled_features, 
                        load_trained_models     = load_trained_models_ensemble_member,
                        recalibrate_output      = recalibrate_output,
                        num_workers             = num_workers)
            
        
    def train(self, hidden_layers, 
                    neurons, 
                    number_of_epochs, 
                    batch_size,
                    learning_rate, 
                    scalerType, 
                    calibration=False,
                    type_of_calibration="isotonic", 
                    num_bins_cal = 40, 
                    callback = True, 
                    callback_patience=30, 
                    callback_factor=0.01,
                    activation='swish', 
                    verbose=2, 
                    rnd_seed=2,
                    ensemble_index='', 
                    validation_split=0.1, 
                    holdout_split=0.3, 
                    plot_scaled_features=False, 
                    load_trained_models = False,
                    recalibrate_output=False,
                    num_workers=0):
        '''
        Method that trains the density ratio NNs

        batch_size: the size of each batch used during gradient optimization
        learning_rate: the initial learning rate to pass to the optimizer
        scalerType: option to one of three standardizing options: ['MinMax', 'StandardScaler', 'PowerTransform_Yeo'] 
        holdout_split: the fraction of dataset to set aside for diagnostics, not used in training and validation of the loss vs epoch curves
        epochs: the number of epochs to train the NNs

        calibration: boolean to do Histogram-based calibration of the NN ourput
        num_bins_cal: number of bins used for calibration histogram
        '''

        self.calibration = calibration
        self.calibration_switch = False # Set the switch to false for first evaluation for calibration

        configure_logging(verbose)
        self.verbose = verbose

        if ensemble_index=='':
            self.model_NN              = [None]
            self.scaler                = [None]
            self.histogram_calibrator  = [None]

            self.full_data_prediction  = np.zeros((1, len(self.weights)))
            self.train_idx             = [None]
            self.holdout_idx           = [None]
            self.num_ensemble_members  = 1
            ensemble_index             = 0

        
        if load_trained_models:
            # Load the number of holdout events and random state used for train/test split when using saved models
            holdout_num, rnd_seed = np.load(f"{self.path_to_models}num_events_random_state_train_holdout_split{ensemble_index}.npy")
        else:
            # Get the number of holdout events from the holdout_split fraction
            holdout_num = math.floor(self.dataset.shape[0] * holdout_split)

        
        # HyperParameters for the NN training
        validation_split = validation_split
        self.batch_size = batch_size

        idx_incl = np.arange(len(self.weights))

        # Get the indicies
        self.train_idx[ensemble_index], self.holdout_idx[ensemble_index] = train_test_split(idx_incl, 
                                                                                            test_size=holdout_num, 
                                                                                            random_state=rnd_seed,                                                        
                                                                                            stratify=self.training_labels)

        self.dataset_training   = self.dataset.iloc[self.train_idx[ensemble_index]].copy()
        self.dataset_holdout   = self.dataset.iloc[self.holdout_idx[ensemble_index]].copy()

        # split the original dataset into training and holdout
        data_train_full, data_holdout_full = self.dataset_training.copy(), self.dataset_holdout.copy()
        label_train, label_holdout = self.training_labels[self.train_idx[ensemble_index]].copy(), self.training_labels[self.holdout_idx[ensemble_index]].copy()
        weight_train, weight_holdout = self.weights[self.train_idx[ensemble_index]].copy(), self.weights[self.holdout_idx[ensemble_index]].copy()

        # dataset to be used for training
        data_train, data_holdout = data_train_full[self.features], data_holdout_full[self.features]

        # Load pre-trained models and scaling
        if load_trained_models:

            path_to_saved_scaler        = f"{self.path_to_models}model_scaler{ensemble_index}.bin"
            path_to_saved_model         = f"{self.path_to_models}model{ensemble_index}.onnx"

            logger.info(f"Reading saved models from {self.path_to_models}")
            self.scaler[ensemble_index], self.model_NN[ensemble_index] = load_trained_model(path_to_saved_model, path_to_saved_scaler)
            
        # Else setup a new scaler
        else:

            if (scalerType == 'MinMax'):
                self.scaler[ensemble_index] = ColumnTransformer([("scaler",MinMaxScaler(feature_range=(-1.5,1.5)), self.features_scaling)],remainder='passthrough')
                
            if (scalerType == 'StandardScaler'):
                self.scaler[ensemble_index] = ColumnTransformer([("scaler",StandardScaler(), self.features_scaling)],remainder='passthrough')
                
            if (scalerType == 'PowerTransform_Yeo'):
                self.scaler[ensemble_index] = ColumnTransformer([("scaler",PowerTransformer(method='yeo-johnson', standardize=True), self.features_scaling)],remainder='passthrough')


        scaled_data_train = self.scaler[ensemble_index].fit_transform(data_train)
        scaled_data_train = pd.DataFrame(scaled_data_train, columns=self.features)

        if plot_scaled_features:
            plot_all_features(scaled_data_train, weight_train, label_train)

        scaled_data_holdout = self.scaler[ensemble_index].transform(data_holdout)
        scaled_data_holdout = pd.DataFrame(scaled_data_holdout, columns=self.features)

        # Train the model if not loaded
        if not load_trained_models:

            # Check if the datasets are normalized
            logger.info(f"Sum of weights of class 0: {np.sum(weight_train[label_train==0])}")
            logger.info(f"Sum of weights of class 1: {np.sum(weight_train[label_train==1])}")
    
            logger.info(f"Using {activation} activation function")

            train_ds = WeightedTensorDataset(
                scaled_data_train.values,
                label_train,
                weight_train
            )

            val_size = int(len(train_ds) * validation_split)
            train_size = len(train_ds) - val_size

            train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

            train_loader = DataLoader(train_ds, 
                                        batch_size=batch_size, 
                                        shuffle=True,
                                        num_workers=num_workers,  
                                        pin_memory=False,  
                                        persistent_workers=False)
            
            val_loader   = DataLoader(val_ds, 
                                      batch_size=batch_size, 
                                      shuffle=False,
                                        num_workers=num_workers,  
                                        pin_memory=False,  
                                        persistent_workers=False)

            model = DensityRatioLightning(
                n_hidden=hidden_layers,
                n_neurons=neurons,
                input_dim=len(self.features),
                learning_rate=learning_rate,
                use_log_loss=self.use_log_loss,
                activation=activation,
                callback_factor=callback_factor,
                callback_patience=callback_patience
            )

            loss_history = LossHistory()

            if callback:
                trainer = Trainer(
                    accelerator="auto",
                    devices="auto",
                    max_epochs=number_of_epochs,
                    callbacks=[
                        EarlyStopping(monitor="val_loss", patience=callback_patience),
                        LearningRateMonitor(),
                        loss_history,
                        PrintEpochMetrics()
                    ],
                    logger=True,
                    enable_checkpointing=False,
                    enable_progress_bar=True,
                )
            else:
                trainer = Trainer(
                    accelerator="auto",
                    devices="auto",
                    max_epochs=number_of_epochs,
                    callbacks=[loss_history],
                    logger=True,
                    enable_checkpointing=False
                )

            trainer.fit(model, train_loader, val_loader)

            self.model_NN[ensemble_index] = model
        
            logger.info("Finished Training")
                
            path_to_saved_scaler        = f"{self.path_to_models}model_scaler{ensemble_index}.bin"
            path_to_saved_model         = f"{self.path_to_models}model{ensemble_index}.onnx"

            save_model(self.model_NN[ensemble_index], 
                        torch.randn((1, len(self.features))), 
                        path_to_saved_model,
                        self.scaler[ensemble_index], 
                        path_to_saved_scaler,
                        softmax_output = False)
            
            # Reassign the model to be in ONNX format
            self.scaler[ensemble_index], self.model_NN[ensemble_index] = load_trained_model(path_to_saved_model, 
                                                                                            path_to_saved_scaler)

            # Save metadata
            np.save(f"{self.path_to_models}num_events_random_state_train_holdout_split{ensemble_index}.npy", 
                    np.array([holdout_num, rnd_seed]))
    
            plot_loss(loss_history, path_to_figures=self.path_to_figures)

        
        # Do a first prediction without calibration layers
        train_data_prediction = self.predict_with_model(data_train_full, 
                                                        ensemble_index = ensemble_index, 
                                                        use_log_loss = self.use_log_loss)

        calibration_method = 'direct'
        
        # If calibrating, use the train_data_prediction for building histogram
        if self.calibration:

            self.calibration_switch = True
            path_to_calibrated_object = f"{self.path_to_models}model_calibrated_hist{ensemble_index}.obj"

            if type_of_calibration == "histogram":
                calibration_data_num = train_data_prediction[label_train==1]
                calibration_data_den = train_data_prediction[label_train==0]

                w_num = weight_train[label_train==1]
                w_den = weight_train[label_train==0]

            if not load_trained_models:

                if type_of_calibration == "histogram":
            
                    self.histogram_calibrator[ensemble_index] =  HistogramCalibrator(calibration_data_num, calibration_data_den, w_num, w_den, 
                                                                    nbins=num_bins_cal, method=calibration_method, mode='dynamic')
                
                elif type_of_calibration == "isotonic":
                    self.histogram_calibrator[ensemble_index] =  IsotonicCalibrator(train_data_prediction, label_train, weight_train)
                
                else:
                    raise Exception(f"Type of calibration not recognized - choose between isotonic and histogram")
                
                file_calib = open(path_to_calibrated_object, 'wb') 
    
                pickle.dump(self.histogram_calibrator[ensemble_index], file_calib)
    
            else:
                if not os.path.exists(path_to_calibrated_object) or recalibrate_output:
                    
                    logger.info(f"Calibrating the saved model with {num_bins_cal} bins")
                    
                    if type_of_calibration == "histogram":
                        self.histogram_calibrator[ensemble_index] =  HistogramCalibrator(calibration_data_num, calibration_data_den, w_num, w_den, 
                                                                    nbins=num_bins_cal, method=calibration_method, mode='dynamic')
                    elif type_of_calibration == "isotonic":
                        self.histogram_calibrator[ensemble_index] =  IsotonicCalibrator(train_data_prediction, label_train, weight_train)

                    else:
                        raise Exception(f"Type of calibration not recognized - choose between isotonic and histogram")
                
                    file_calib = open(path_to_calibrated_object, 'wb') 
        
                    pickle.dump(self.histogram_calibrator[ensemble_index], file_calib)
                else:
                
                    file_calib = open(path_to_calibrated_object, 'rb') 
                    self.histogram_calibrator[ensemble_index] = pickle.load(file_calib)
                    logger.info(f"calibrator object loaded = {self.histogram_calibrator}")
            
            self.full_data_prediction[ensemble_index] = self.predict_with_model(self.dataset, 
                                                                                ensemble_index = ensemble_index, 
                                                                                use_log_loss=self.use_log_loss)

        # Else, continue evaluating using the base model
        else:
            self.full_data_prediction[ensemble_index] = self.predict_with_model(self.dataset, 
                                                                                ensemble_index = ensemble_index, 
                                                                                use_log_loss=self.use_log_loss)

        
        # TRAINING inputs
        self.score_den_training = self.full_data_prediction[ensemble_index][self.train_idx[ensemble_index]][self.training_labels[self.train_idx[ensemble_index]]==0]
        self.weight_den_training   = self.weights[self.train_idx[ensemble_index]][self.training_labels[self.train_idx[ensemble_index]]==0]
        self.score_num_training = self.full_data_prediction[ensemble_index][self.train_idx[ensemble_index]][self.training_labels[self.train_idx[ensemble_index]]==1]
        self.weight_num_training   = self.weights[self.train_idx[ensemble_index]][self.training_labels[self.train_idx[ensemble_index]]==1]

        # HOLDOUT inputs
        self.score_den_holdout = self.full_data_prediction[ensemble_index][self.holdout_idx[ensemble_index]][self.training_labels[self.holdout_idx[ensemble_index]]==0]
        self.weight_den_holdout   = self.weights[self.holdout_idx[ensemble_index]][self.training_labels[self.holdout_idx[ensemble_index]]==0]
        self.score_num_holdout = self.full_data_prediction[ensemble_index][self.holdout_idx[ensemble_index]][self.training_labels[self.holdout_idx[ensemble_index]]==1]
        self.weight_num_holdout   = self.weights[self.holdout_idx[ensemble_index]][self.training_labels[self.holdout_idx[ensemble_index]]==1]

        # Some diagnostics to ensure numerical stability - min/max must not be exactly 0 or 1
        min_max_values = [
            (self.sample_name[1], "training", np.amin(self.score_den_training), 
                                              np.amax(self.score_den_training)),
            (self.sample_name[0], "training", np.amin(self.score_num_training), 
                                              np.amax(self.score_num_training)),
            (self.sample_name[1], "holdout", np.amin(self.score_den_holdout), 
                                              np.amax(self.score_den_holdout)),
            (self.sample_name[0], "holdout", np.amin(self.score_num_holdout), 
                                              np.amax(self.score_num_holdout))
        ]
        
        for name, training_holdout_label, min_val, max_val in min_max_values:
            
            if min_val == 0:
                logger.warning(f"{name} {training_holdout_label} data has min score = 0 for ensemble member {ensemble_index}, which may indicate numerical instability!")
            
            if max_val == 1:
                logger.warning(f"{name} {training_holdout_label} data has max score = 1 for ensemble member {ensemble_index}, which may indicate numerical instability!")            


    
    def predict_with_model(self, data, ensemble_index = 0, use_log_loss=False):
        '''
        Method that evaluates density ratios on provided dataset, using self.model

        data: the dataset to evaluate trained model on
        '''

        pred = predict_with_onnx(data[self.features], 
                                self.scaler[ensemble_index],
                                self.model_NN[ensemble_index])

        if use_log_loss:

            pred = convert_logLR_to_score(pred)

        if (self.calibration) & (self.calibration_switch):
    
            pred = self.histogram_calibrator[ensemble_index].cali_pred(pred)
            pred = pred.reshape(pred.shape[0],)
            pred = np.clip(pred, 1e-8, 1.0 - 1e-8)

        return pred
    
    
    def make_overfit_plots(self, ensemble_index=0):
        '''
        Plot predictions for training and holdout to test compatibility
        '''

        plot_overfit_side_by_side(
            self.score_den_training, self.score_den_holdout,
            self.weight_den_training, self.weight_den_holdout,
            self.score_num_training, self.score_num_holdout,
            self.weight_num_training, self.weight_num_holdout,
            nbins=30, plotRange=[0.0, 1.0], holdout_index=0,
            labels=(f'{self.sample_name[1]}', f'{self.sample_name[0]}'),
            path_to_figures=self.path_to_figures
        )

    def make_calib_plots(self, observable='score', nbins=10, ensemble_index=0):
        '''
        Test the probability calibration of NN output

        observable: choose between 'score' for relative probability p_A/p_A+p_B, and 'llr' for log-likelihood ratio log p_A/p_B
        '''
        importlib.reload(sys.modules['nsbi_common_utils.plotting'])
        from nsbi_common_utils.plotting import plot_calibration_curve, plot_calibration_curve_ratio

        if observable=='score':
            # Plot Calibration curves - score function
            plot_calibration_curve(self.score_den_training, 
                                   self.weight_den_training, 
                                   self.score_num_training, 
                                   self.weight_num_training, 
                                   self.score_den_holdout, 
                                   self.weight_den_holdout, 
                                   self.score_num_holdout, 
                                   self.weight_num_holdout, 
                                   self.path_to_figures, 
                                   nbins=nbins, 
                                   label="Calibration Curve - "+str(self.sample_name[0]))

        elif observable=='llr':
            # Plot Calibration curves - nll function
            plot_calibration_curve_ratio(self.score_den_training, 
                                        self.weight_den_training, 
                                        self.score_num_training, 
                                        self.weight_num_training, 
                                        self.score_den_holdout, 
                                        self.weight_den_holdout, 
                                        self.score_num_holdout, 
                                        self.weight_num_holdout, 
                                        self.path_to_figures, 
                                        nbins=nbins, 
                                        label="Calibration Curve - "+str(self.sample_name[0]))

        else:
            raise Exception("observable not recognized - choose between score and llr options")

    def make_reweighted_plots(self, variables, scale, num_bins, ensemble_index=0):
        '''
        Test the quality of the NN predicted density ratios using a reweighting check p_A/p_B * p_B ~ p_A

        variables: list of variables to plot
        scale: linear or log y-axis scales
        num_bins: number of bins in the reweighting diagnostic plot
        '''
        importlib.reload(sys.modules['nsbi_common_utils.plotting'])
        from nsbi_common_utils.plotting import plot_reweighted

        plot_reweighted(
            self.dataset_training, self.score_den_training, self.weight_den_training, self.score_num_training, self.weight_num_training,
            self.dataset_holdout, self.score_den_holdout, self.weight_den_holdout, self.score_num_holdout, self.weight_num_holdout,
            variables=variables, num=num_bins, sample_name=self.sample_name,
            scale=scale, path_to_figures=self.path_to_figures,
            label_left='Training Data Diagnostic', label_right='Holdout Data Diagnostic'
        )

    def make_reweighted_plots_old(self, variables, scale, num_bins, ensemble_index = 0):
        '''
        Test the quality of the NN predicted density ratios using a reweighting check p_A/p_B * p_B ~ p_A

        variables: list of variables to plot
        scale: linear or log y-axis scales
        num_bins: number of bins in the reweighting diagnostic plot
        '''
        importlib.reload(sys.modules['nsbi_common_utils.plotting'])
        from nsbi_common_utils.plotting import plot_reweighted

        plot_reweighted(self.dataset_training, 
                        self.score_den_training, 
                        self.weight_den_training, 
                        self.score_num_training, 
                        self.weight_num_training,
                        variables=variables, 
                        num=num_bins,
                        sample_name=self.sample_name, scale=scale,  
                        path_to_figures=self.path_to_figures, label='Training Data Diagnostic')

        plot_reweighted(self.dataset_holdout, 
                        self.score_den_holdout, 
                        self.weight_den_holdout,
                        self.score_num_holdout, 
                        self.weight_num_holdout,
                        variables=variables, 
                        num=num_bins,
                        sample_name=self.sample_name, scale=scale, 
                        path_to_figures=self.path_to_figures, label='Holdout Data Diagnostic')

    def test_normalization(self):
        '''
        Test if \int p_A/p_B x p_B ~ 1
        '''
        # Normalized reference (denominator) hypothesis
        weight_ref = self.weights[self.training_labels==0].copy()

        ratio_rwt = np.zeros((self.num_ensemble_members, weight_ref.shape[0]))

        for ensemble_index in range(self.num_ensemble_members):
            
            # Calculate p_A/p_B for B hypothesis events
            score_rwt = self.predict_with_model(self.dataset[self.features], 
                                                ensemble_index=ensemble_index, 
                                                use_log_loss=self.use_log_loss)[self.training_labels==0]
            ratio_rwt[ensemble_index] = score_rwt / ( 1.0 - score_rwt )
    
            # Calculate \sum p_A/p_B x p_B
            logger.info(f"\n\n\nThe sum of PDFs in ensemble member {ensemble_index} is {np.sum(ratio_rwt[ensemble_index] * weight_ref)}\n\n")

        ratio_rwt_aggregate = np.mean(ratio_rwt, axis=0)
        
        logger.info(f"The sum of PDFs using the whole ensemble is {np.sum(ratio_rwt_aggregate * weight_ref)}\n\n\n")
        

    def evaluate_and_save_ratios(self, dataset, aggregation_type = 'mean_ratio'):
        '''
        Evaluate with self.model on the input dataset, and save to self.path_to_ratios

        aggregation_type: choose an option on how to aggregate the ensemble models - 'median_ratio', 'mean_ratio', 'median_score', 'mean_score'
        '''

        logger.info(f"Evaluating density ratios")
        score_pred = np.ones((self.num_ensemble_members, dataset.shape[0]))
        ratio_pred = np.ones((self.num_ensemble_members, dataset.shape[0]))
        log_ratio_pred = np.ones((self.num_ensemble_members, dataset.shape[0]))

        for ensemble_index in range(self.num_ensemble_members):
            score_pred[ensemble_index] = self.predict_with_model(dataset[self.features], 
                                                                 use_log_loss=self.use_log_loss, 
                                                                 ensemble_index=ensemble_index)
            
            ratio_pred[ensemble_index] = score_pred[ensemble_index] / (1.0 - score_pred[ensemble_index])
            log_ratio_pred[ensemble_index] = np.log(score_pred[ensemble_index] / (1.0 - score_pred[ensemble_index]))

        if aggregation_type == 'median_ratio':
            ratio_ensemble = np.median(ratio_pred, axis=0)
            
        elif aggregation_type == 'mean_ratio':
            ratio_ensemble = np.mean(ratio_pred, axis=0)
            
        elif aggregation_type == 'median_score':
            score_aggregate = np.median(score_pred, axis=0)
            ratio_ensemble = score_aggregate / (1.0 - score_aggregate)
            
        elif aggregation_type == 'mean_score':
            score_aggregate = np.mean(score_pred, axis=0)
            ratio_ensemble = score_aggregate / (1.0 - score_aggregate)

        else:
            raise Exception("aggregation_type not recognized, please choose between median_ratio, mean_ratio, median_score or mean_score")

        saved_ratio_path = f"{self.path_to_ratios}ratio_{self.sample_name[0]}.npy"
        np.save(saved_ratio_path, ratio_ensemble)

        return saved_ratio_path

