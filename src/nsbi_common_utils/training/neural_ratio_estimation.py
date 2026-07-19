#import libraries
import os, importlib, sys, shutil, gc
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 
import math
import pickle 

import warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except (ImportError, TypeError):
    pass

import torch
torch.set_float32_matmul_precision("medium")
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch.utils.data import Subset

import nsbi_common_utils

from pathlib import Path
from typing import Union, Dict
from joblib import dump, load

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
                      use_log_loss=False, 
                      split_using_fold=False,
                      delete_existing_models=False):
        """
        Initialise the density ratio trainer.

        This class trains one or more neural networks to estimate the density ratio :math:`p_A(x) / p_B(x)` between two fixed hypothesis A and B using a binary-classification approach. The classifier output score is then converted to a density ratio via :math:`r = s / (1 - s)`.

        Parameters
        ----------
        dataset : pandas.DataFrame
            The full dataset containing events from both hypotheses A and B. Rows must be aligned with ``weights`` and ``training_labels``.

        weights : numpy.ndarray, shape (n_events,)
            Per-event weights. Weights should be normalised **independently** within each class so that each class sums to the same total weight before being passed in. Mis-normalised weights will bias the ratio.

        training_labels : numpy.ndarray of int, shape (n_events,)
            Binary class labels: ``1`` for hypothesis A (numerator) and ``0`` for hypothesis B (denominator).

        features : list of str
            Column names in ``dataset`` to use as input features to the neural network.

        features_scaling : list of str
            Subset of ``features`` that will be passed through the chosen scaler. Features not listed here are passed through unchanged (``remainder='passthrough'`` in the ``ColumnTransformer``).

        sample_name : list of str, length 2
            Human-readable labels for the two hypotheses, e.g. ``['signal', 'background']``. Index 0 is A (numerator), index 1 is B (denominator). Used in plot titles and saved file names.

        output_name : str
            A tag used to identify outputs produced by this trainer instance.

        path_to_figures : str, optional
            Directory where diagnostic plots are written. Created automatically if it does not exist. Defaults to the current working directory.

        path_to_models : str, optional
            Directory where trained ONNX model files and scalers are saved. Created automatically if it does not exist.

        use_log_loss : bool, optional
            If ``True``, the network is trained with a log-likelihood-ratio loss instead of binary cross-entropy, and the raw output is treated as :math:`\\log(p_A / p_B)` before conversion to a probability score. Default ``False``.

        split_using_fold : bool, optional
            Reserved for future k-fold splitting support. Not yet implemented. Default ``False``.

        delete_existing_models : bool, optional
            If ``True``, the directories ``path_to_figures`` and ``path_to_models`` are deleted and recreated before training, ensuring a clean run. Use with caution — this is irreversible. Default ``False``.

        Notes
        -----
        * This class trains a single density-ratio network per instance. Ensemble training is handled externally by submitting multiple independent jobs and passing ``ensemble_index`` to distinguish saved outputs.
        * Weight normalisation is the caller's responsibility. A common convention is to normalise each class so that :math:`\\sum_{i \\in A} w_i = \\sum_{j \\in B} w_j = 1`.
        * ``features_scaling`` is typically identical to ``features``, but can be a strict subset if some features are already on a suitable scale (e.g. boolean flags).
        """
        self.dataset = dataset
        self.weights = weights
        self.training_labels = training_labels
        self.features = features
        self.features_scaling = features_scaling
        self.sample_name = sample_name
        self.output_name = output_name
        self.use_log_loss = use_log_loss
        self.split_using_fold = split_using_fold

        self.path_to_figures = path_to_figures

        if delete_existing_models:
            if os.path.exists(path_to_figures):
                shutil.rmtree(path_to_figures)
            
            if os.path.exists(path_to_models):
                shutil.rmtree(path_to_models)

        if not os.path.exists(path_to_figures):
                os.makedirs(path_to_figures)
        
        self.path_to_models = path_to_models
        if not os.path.exists(path_to_models):
                os.makedirs(path_to_models)

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
                    rnd_seed=None,
                    ensemble_index=None, 
                    validation_split=0.1, 
                    holdout_split=0.3, 
                    plot_scaled_features=False, 
                    load_trained_models = False,
                    recalibrate_output=False,
                    num_workers=0):
        """
        Train a density-ratio neural network.

        This is the core training routine called internally by :meth:`train_ensemble`. It can also be called directly when no ensemble averaging is required. After training, the model is exported to ONNX format and reloaded for inference so that subsequent calls to :meth:`predict_with_model` are backend-agnostic.

        Parameters
        ----------
        hidden_layers : int
            Number of hidden layers in the neural network.

        neurons : int
            Number of neurons per hidden layer.

        number_of_epochs : int
            Maximum number of training epochs.

        batch_size : int
            Mini-batch size for gradient optimisation.

        learning_rate : float
            Initial learning rate for the Adam optimiser.

        scalerType : str
            Feature scaling method. See :meth:`train_ensemble` for accepted values.

        calibration : bool, optional
            Apply post-hoc probability calibration. Default ``False``.

        type_of_calibration : str, optional
            Calibration algorithm. One of ``'isotonic'`` or ``'histogram'``. Default ``'isotonic'``.

        num_bins_cal : int, optional
            Bins for histogram calibration. Default ``40``.

        callback : bool, optional
            Enable early stopping and learning-rate monitoring callbacks. Default ``True``.

        callback_patience : int, optional
            Early stopping patience in epochs. Default ``30``.

        callback_factor : float, optional
            Learning-rate reduction factor at plateau. Default ``0.01``.

        activation : str, optional
            Hidden-layer activation function. Default ``'swish'``.

        verbose : int, optional
            Logging verbosity (0=warnings, 1=info, 2=debug). Default ``2``.

        rnd_seed : int or None, optional
            Random seed for the train/holdout split. If ``None`` (default), a random seed is drawn from a uniform integer distribution and saved to disk alongside the model so the same split can be recovered when ``load_trained_models=True``.

        ensemble_index : int or None, optional
            Integer suffix appended to saved model, scaler, calibrator and metadata filenames (e.g. ``model0.onnx``, ``model_scaler0.bin``). Pass the ensemble index here to avoid filename collisions when multiple members are trained in parallel. When ``None``, no suffix is appended (files saved as ``model.onnx`` etc.).

        validation_split : float, optional
            Fraction of training data used for validation loss. Default ``0.1``.

        holdout_split : float, optional
            Fraction of total data reserved for holdout diagnostics. Default ``0.3``.

        plot_scaled_features : bool, optional
            Plot feature distributions after scaling. Default ``False``.

        load_trained_models : bool, optional
            Load a previously saved model instead of training. Default ``False``.

        recalibrate_output : bool, optional
            Force recalibration even when a saved calibrator exists. Default ``False``.

        num_workers : int, optional
            DataLoader worker processes. Default ``0``.

        Raises
        ------
        Exception
            If ``type_of_calibration`` is not ``'isotonic'`` or ``'histogram'``.

        Warns
        -----
        UserWarning
            Logs a warning if the minimum predicted score for any class equals ``0`` or the maximum equals ``1``, which may indicate numerical saturation and unreliable ratio estimates.

        Notes
        -----
        * The holdout set is stratified by ``training_labels`` to ensure class balance is preserved.
        * When ``load_trained_models=True``, the ``holdout_num`` and ``rnd_seed`` are recovered from a saved ``.npy`` file so that the same holdout split is used, guaranteeing consistency between the saved model and any subsequent diagnostic plots.
        * A loss-vs-epoch plot is automatically saved to ``path_to_figures`` after each successful training run.
        """
        self.calibration = calibration
        self.calibration_switch = False # Set the switch to false for first evaluation for calibration

        if rnd_seed is None:
            rnd_seed = np.random.randint(0, 2**32 -1, size=None)

        configure_logging(verbose)
        self.verbose = verbose

        if ensemble_index is None:
            ensemble_index_label = ''
        else:
            ensemble_index_label=str(ensemble_index)

        if load_trained_models:
            path_to_saved_scaler = f"{self.path_to_models}model_scaler{ensemble_index_label}.bin"
            path_to_saved_model  = f"{self.path_to_models}model{ensemble_index_label}.onnx"
            path_to_saved_state  = f"{self.path_to_models}num_events_random_state_train_holdout_split{ensemble_index_label}.npy"

            missing = [p for p in [path_to_saved_scaler, path_to_saved_model, path_to_saved_state] if not os.path.exists(p)]
            if missing:
                logger.warning(f"load_trained_models=True but the following files were not found, retraining:\n" + "\n".join(missing))
                load_trained_models = False

        if load_trained_models:
            # Load the number of holdout events and random state used for train/test split when using saved models
            holdout_num, rnd_seed = np.load(path_to_saved_state)
        else:
            # Get the number of holdout events from the holdout_split fraction
            holdout_num = math.floor(self.dataset.shape[0] * holdout_split)

        
        # HyperParameters for the NN training
        validation_split = validation_split
        self.batch_size = batch_size

        idx_incl = np.arange(len(self.weights))

        # Get the indicies
        self.train_idx, self.holdout_idx = train_test_split(idx_incl, 
                                                                test_size=holdout_num, 
                                                                random_state=rnd_seed,                                                        
                                                                stratify=self.training_labels)

        self.dataset_training   = self.dataset.iloc[self.train_idx].copy()
        self.dataset_holdout   = self.dataset.iloc[self.holdout_idx].copy()

        # split the original dataset into training and holdout
        label_train, label_holdout = self.training_labels[self.train_idx].copy(), self.training_labels[self.holdout_idx].copy()
        weight_train, weight_holdout = self.weights[self.train_idx].copy(), self.weights[self.holdout_idx].copy()

        # dataset to be used for training
        data_train, data_holdout = self.dataset_training[self.features].copy(), self.dataset_holdout[self.features].copy()

        # Load pre-trained models and scaling
        if load_trained_models:

            logger.info(f"Reading saved models from {self.path_to_models}")
            self.scaler, self.model_NN = nsbi_common_utils.training.utils.load_trained_model(path_to_saved_model, path_to_saved_scaler)
            
            scaled_data_train = self.scaler.transform(data_train)    
            scaled_data_train = pd.DataFrame(scaled_data_train, columns=self.features)

        # Else setup a new scaler
        else:

            if (scalerType == 'MinMax'):
                self.scaler = ColumnTransformer([("scaler",MinMaxScaler(feature_range=(-1.5,1.5)), self.features_scaling)],remainder='passthrough')
                
            if (scalerType == 'StandardScaler'):
                self.scaler = ColumnTransformer([("scaler",StandardScaler(), self.features_scaling)],remainder='passthrough')
                
            if (scalerType == 'PowerTransform_Yeo'):
                self.scaler = ColumnTransformer([("scaler",PowerTransformer(method='yeo-johnson', standardize=True), self.features_scaling)],remainder='passthrough')


            scaled_data_train = self.scaler.fit_transform(data_train)    
            scaled_data_train = pd.DataFrame(scaled_data_train, columns=self.features)

        if plot_scaled_features:
            plot_all_features(scaled_data_train, weight_train, label_train)

        # Train the model if not loaded
        if not load_trained_models:

            # Check if the datasets are normalized
            logger.info(f"Sum of weights of class 0: {np.sum(weight_train[label_train==0])}")
            logger.info(f"Sum of weights of class 1: {np.sum(weight_train[label_train==1])}")
    
            logger.info(f"Using {activation} activation function")

            train_ds = nsbi_common_utils.lightning_tools.WeightedTensorDataset(
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
                                        pin_memory=True,  
                                        persistent_workers=False)
            
            val_loader   = DataLoader(val_ds, 
                                      batch_size=batch_size, 
                                      shuffle=False,
                                        num_workers=num_workers,  
                                        pin_memory=True,  
                                        persistent_workers=False)

            # Example use of training API
            model = nsbi_common_utils.lightning_tools.DensityRatioLightning(
                n_hidden=hidden_layers,
                n_neurons=neurons,
                input_dim=len(self.features),
                learning_rate=learning_rate,
                use_log_loss=self.use_log_loss,
                activation=activation,
                callback_factor=callback_factor,
                callback_patience=callback_patience
            )

            loss_history = nsbi_common_utils.lightning_tools.LossHistory()

            checkpoint_callback = ModelCheckpoint(
                                                    monitor="val_loss",          
                                                    dirpath="checkpoints/",      
                                                    filename="best-{epoch:03d}-{val_loss:.6f}"+f"_{self.sample_name[0]}vs{self.sample_name[1]}",
                                                    save_top_k=1,                
                                                    mode="min",                  
                                                    save_last=False,             
                                                )

            if callback:
                trainer = Trainer(
                    accelerator="auto",
                    devices="auto",
                    max_epochs=number_of_epochs,
                    callbacks=[
                        EarlyStopping(monitor="val_loss", patience=callback_patience),
                        LearningRateMonitor(),
                        checkpoint_callback,
                        loss_history,
                        nsbi_common_utils.lightning_tools.PrintEpochMetrics()
                    ],
                    logger=True,
                    enable_checkpointing=True,
                    enable_progress_bar=False,
                )
            else:
                trainer = Trainer(
                    accelerator="auto",
                    devices="auto",
                    max_epochs=number_of_epochs,
                    callbacks=[loss_history, checkpoint_callback],
                    logger=True,
                    enable_checkpointing=False,
                    enable_progress_bar=False
                )

            trainer.fit(model, train_loader, val_loader)

            device = next(model.parameters()).device

            # Memory cleaning
            trainer = None
            train_loader = None
            val_loader = None
            gc.collect()
            torch.cuda.empty_cache()

            best_model = nsbi_common_utils.lightning_tools.DensityRatioLightning.load_from_checkpoint(
                checkpoint_callback.best_model_path,
                map_location=device
                )

            self.model_NN = best_model
        
            logger.info("Finished Training")
                
            path_to_saved_scaler        = f"{self.path_to_models}model_scaler{ensemble_index_label}.bin"
            path_to_saved_model         = f"{self.path_to_models}model{ensemble_index_label}.onnx"

            nsbi_common_utils.training.utils.save_model(self.model_NN, 
                        torch.randn((1, len(self.features))), 
                        path_to_saved_model,
                        self.scaler, 
                        path_to_saved_scaler,
                        softmax_output = False)
            
            # Reassign the model to be in ONNX format
            self.scaler, self.model_NN = nsbi_common_utils.training.utils.load_trained_model(path_to_saved_model, 
                                                                                            path_to_saved_scaler)

            # Save metadata
            np.save(f"{self.path_to_models}num_events_random_state_train_holdout_split{ensemble_index_label}.npy", 
                    np.array([holdout_num, rnd_seed]))
    
            plot_loss(loss_history, path_to_figures=self.path_to_figures)

        
        # Do a first prediction without calibration layers
        train_data_prediction = nsbi_common_utils.training.utils.predict_with_model(self.dataset_training[self.features], 
                                                                        scaler = self.scaler, 
                                                                        model = self.model_NN,
                                                                        calibration_model = None,
                                                                        use_log_loss = self.use_log_loss)
        
        gc.collect()
        torch.cuda.empty_cache()

        calibration_method = 'direct'
        
        # If calibrating, use the train_data_prediction for building histogram
        if self.calibration:

            self.calibration_switch = True
            path_to_calibrated_object = f"{self.path_to_models}model_calibrated_hist{ensemble_index_label}.obj"

            if type_of_calibration == "histogram":
                calibration_data_num = train_data_prediction[label_train==1]
                calibration_data_den = train_data_prediction[label_train==0]

                w_num = weight_train[label_train==1]
                w_den = weight_train[label_train==0]

            if not load_trained_models:

                if type_of_calibration == "histogram":
            
                    self.histogram_calibrator =  HistogramCalibrator(calibration_data_num, calibration_data_den, w_num, w_den, 
                                                                    nbins=num_bins_cal, method=calibration_method, mode='dynamic')
                
                elif type_of_calibration == "isotonic":
                    self.histogram_calibrator =  IsotonicCalibrator(train_data_prediction, label_train, weight_train)
                
                else:
                    raise Exception(f"Type of calibration not recognized - choose between isotonic and histogram")
                
                file_calib = open(path_to_calibrated_object, 'wb') 
    
                pickle.dump(self.histogram_calibrator, file_calib)
    
            else:
                if not os.path.exists(path_to_calibrated_object) or recalibrate_output:
                    
                    logger.info(f"Calibrating the saved model with {num_bins_cal} bins")
                    
                    if type_of_calibration == "histogram":
                        self.histogram_calibrator =  HistogramCalibrator(calibration_data_num, calibration_data_den, w_num, w_den, 
                                                                    nbins=num_bins_cal, method=calibration_method, mode='dynamic')
                    elif type_of_calibration == "isotonic":
                        self.histogram_calibrator =  IsotonicCalibrator(train_data_prediction, label_train, weight_train)

                    else:
                        raise Exception(f"Type of calibration not recognized - choose between isotonic and histogram")
                
                    file_calib = open(path_to_calibrated_object, 'wb') 
        
                    pickle.dump(self.histogram_calibrator, file_calib)
                else:
                
                    file_calib = open(path_to_calibrated_object, 'rb') 
                    self.histogram_calibrator = pickle.load(file_calib)
                    logger.info(f"calibrator object loaded = {self.histogram_calibrator}")
            
            self.full_data_prediction = nsbi_common_utils.training.utils.predict_with_model(self.dataset[self.features], 
                                                                                scaler = self.scaler, 
                                                                                model = self.model_NN,
                                                                                calibration_model = self.histogram_calibrator,
                                                                                use_log_loss = self.use_log_loss)

        # Else, continue evaluating using the base model
        else:
            self.histogram_calibrator = None
            self.full_data_prediction = nsbi_common_utils.training.utils.predict_with_model(self.dataset[self.features], 
                                                                                    scaler = self.scaler, 
                                                                                    model = self.model_NN,
                                                                                    calibration_model = None,
                                                                                    use_log_loss = self.use_log_loss)

        
        # TRAINING inputs
        self.score_den_training     = self.full_data_prediction[self.train_idx][self.training_labels[self.train_idx]==0]
        self.weight_den_training    = self.weights[self.train_idx][self.training_labels[self.train_idx]==0]
        self.score_num_training     = self.full_data_prediction[self.train_idx][self.training_labels[self.train_idx]==1]
        self.weight_num_training    = self.weights[self.train_idx][self.training_labels[self.train_idx]==1]

        # HOLDOUT inputs
        self.score_den_holdout      = self.full_data_prediction[self.holdout_idx][self.training_labels[self.holdout_idx]==0]
        self.weight_den_holdout     = self.weights[self.holdout_idx][self.training_labels[self.holdout_idx]==0]
        self.score_num_holdout      = self.full_data_prediction[self.holdout_idx][self.training_labels[self.holdout_idx]==1]
        self.weight_num_holdout     = self.weights[self.holdout_idx][self.training_labels[self.holdout_idx]==1]

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
                logger.warning(f"{name} {training_holdout_label} data has min score = 0, which may indicate numerical instability!")
            
            if max_val == 1:
                logger.warning(f"{name} {training_holdout_label} data has max score = 1, which may indicate numerical instability!")            
    
    
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
            path_to_figures=self.path_to_figures, ensemble_index = ensemble_index
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
                                   label="Calibration Curve - "+str(self.sample_name[0]), ensemble_index = ensemble_index)

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
                                        label="Calibration Curve - "+str(self.sample_name[0]), ensemble_index=ensemble_index)

        else:
            raise Exception("observable not recognized - choose between score and llr options")

    def make_reweighted_plots(self, variables, scale, num_bins, ensemble_index=0):
        '''
        Test the quality of the NN predicted density ratios using a reweighting check p_A/p_B * p_B ~ p_A

        variables: list of variables to plot
        scale: linear or log y-axis scales
        num_bins: number of bins in the reweighting diagnostic plot
        '''

        plot_reweighted(
            self.dataset_training, self.score_den_training, self.weight_den_training, self.score_num_training, self.weight_num_training,
            self.dataset_holdout, self.score_den_holdout, self.weight_den_holdout, self.score_num_holdout, self.weight_num_holdout,
            variables=variables, num=num_bins, sample_name=self.sample_name,
            scale=scale, path_to_figures=self.path_to_figures,
            label_left='Training Data Diagnostic', label_right='Holdout Data Diagnostic', ensemble_index = ensemble_index
        )


    def test_normalization(self):
        '''
        Test the normalisation condition :math:`\\int (p_A / p_B) \\, p_B \\, dx \\approx 1`.

        A value close to 1 indicates the density ratio is correctly normalised.
        Significant deviation suggests training instability or weight mis-normalisation.
        '''

        # Normalized reference (denominator) hypothesis
        weight_ref = self.weights[self.training_labels==0].copy()

        # Calculate p_A/p_B for B hypothesis events
        score_rwt = nsbi_common_utils.training.utils.predict_with_model(self.dataset[self.features], 
                                                            scaler = self.scaler, 
                                                            model = self.model_NN,
                                                            calibration_model = self.histogram_calibrator,
                                                            use_log_loss = self.use_log_loss)[self.training_labels==0]

        ratio_rwt = score_rwt / ( 1.0 - score_rwt )

        logger.info(f"\n\n\nThe sum of PDFs is {np.sum(ratio_rwt * weight_ref)}\n\n")
