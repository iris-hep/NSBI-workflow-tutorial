"""
training_NN_paras.py
Configuration file for Neural Network training parameters.
"""

# Default Flag: Set to False to train new models, True to load existing ones.
# This can be overridden by the command line argument --train in the main script.
USE_SAVED_MODELS = True


doCalibration = False
RECALIBRATE = False
num_bins_cal = 500
scaling_type = 'MinMax'
batch_size = 512
validation_split = 0.1
holdout_split = 0.25

num_epochs = 100
callback_patience = 30

num_layers = 4
num_neurons_per_layer = 1000

num_ensemble_members = 10

verbose_level = 1

PLOT_SCALED_FEATURES = False

CALIBRATION_TYPE = "histogram"


training_settings = {

    'htautau': {
        'hidden_layers':        num_layers,
        'neurons':              num_neurons_per_layer,
        'number_of_epochs':     num_epochs,
        'batch_size':           batch_size,
        'learning_rate':        0.1,
        'scalerType':           scaling_type,
        'calibration':          doCalibration,
        'num_bins_cal':         num_bins_cal,
        'callback':             True,
        'callback_patience':    callback_patience,
        'callback_factor':      0.01,
        'validation_split':     validation_split,
        'holdout_split':        holdout_split,
        'verbose':              verbose_level,
        'plot_scaled_features': PLOT_SCALED_FEATURES,
        'load_trained_models':  USE_SAVED_MODELS,
        'recalibrate_output'   : RECALIBRATE,
        'type_of_calibration'  : CALIBRATION_TYPE,
        'num_ensemble_members': num_ensemble_members,
        'summarize_model': True
    },
    
    'ttbar': {
        'hidden_layers':        num_layers,
        'neurons':              num_neurons_per_layer,
        'number_of_epochs':     num_epochs,
        'batch_size':           batch_size,
        'learning_rate':        0.1,
        'scalerType':           scaling_type,
        'calibration':          doCalibration,
        'num_bins_cal':         num_bins_cal,
        'callback':             True,
        'callback_patience':    callback_patience,
        'callback_factor':      0.01,
        'validation_split':     validation_split,
        'holdout_split':        holdout_split,
        'verbose':              verbose_level,
        'plot_scaled_features': PLOT_SCALED_FEATURES,
        'load_trained_models':  USE_SAVED_MODELS,
        'recalibrate_output'   : RECALIBRATE,
        'type_of_calibration'  : CALIBRATION_TYPE,
        'num_ensemble_members': num_ensemble_members,
        'summarize_model': True
    },
    
    'ztautau': {
        'hidden_layers':        num_layers,
        'neurons':              num_neurons_per_layer,
        'number_of_epochs':     num_epochs,
        'batch_size':           batch_size,
        'learning_rate':        0.1,
        'scalerType':           scaling_type,
        'calibration':          doCalibration,
        'num_bins_cal':         num_bins_cal,
        'callback':             True,
        'callback_patience':    callback_patience,
        'callback_factor':      0.01,
        'validation_split':     validation_split,
        'holdout_split':        holdout_split,
        'verbose':              verbose_level,
        'plot_scaled_features': PLOT_SCALED_FEATURES,
        'load_trained_models':  USE_SAVED_MODELS,
        'recalibrate_output'   : RECALIBRATE,
        'type_of_calibration'  : CALIBRATION_TYPE,
        'num_ensemble_members': num_ensemble_members,
        'summarize_model': True
    }
}