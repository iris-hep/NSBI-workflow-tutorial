import os, sys, importlib
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.optimizers import Nadam
import mplhep as hep
import pickle
import matplotlib.pyplot as plt
import yaml

sys.path.append('../src')
import nsbi_common_utils
from nsbi_common_utils import plotting, training, \
inference, datasets, configuration, workspace_builder, model
from nsbi_common_utils.inference import inference, plot_NLL_scans

import glob
import numpy as np

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


import logging
import warnings
# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

hep.style.use(hep.style.ATLAS)

def parse_args():
    parser = argparse.ArgumentParser(description="Download and process HiggsML data for analysis.")
    
    parser.add_argument(
        "--config_hist", 
        type=str, 
        default="./config_hist.yml",
        help="Configuration file path for histogram settings."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="./config.yml",
        help=" "
    )
    parser.add_argument(
        "--measurement2fit", 
        type=str, 
        default="higgs_tautau_signal_strength",
        help="Measurement to fit in the analysis."
    )
    
    return parser.parse_args()

def build_workspace(cfg_path):
    """Builds the analysis workspace using nsbi_common_utils.workspace_builder."""
    logger.info(f"Building workspace from config: {cfg_path}")
    ws_builder = nsbi_common_utils.workspace_builder.WorkspaceBuilder(config_path = cfg_path)
    ws_builder.build()
    return ws_builder


def inference_object(workspace_hist, workspace_nsbi, measurement2fit):
    """Creates an inference object for parameter fitting."""
    model_obj_hist = nsbi_common_utils.model.Model(workspace = workspace_hist, 
                                          measurement_to_fit = measurement2fit)
    list_parameters, initial_parameter_values = model_obj_hist.get_model_parameters()     
    num_unconstrained_params = model_obj_hist.num_unconstrained_param   

    model_obj_nsbi = nsbi_common_utils.model.Model(workspace = workspace_nsbi, 
                                            measurement_to_fit = measurement2fit)
    
    inference_obj_nsbi = nsbi_common_utils.inference.inference(model_nll = model_obj_nsbi.model, 
                                                    initial_values = initial_parameter_values, 
                                                    list_parameters = list_parameters,
                                                    num_unconstrained_params = num_unconstrained_params)

    logger.info("Fit results nsbi:\n", inference_obj_nsbi.perform_fit())

    inference_obj_hist = nsbi_common_utils.inference.inference(model_nll = model_obj_hist.model, 
                                                    initial_values = initial_parameter_values, 
                                                    list_parameters = list_parameters,
                                                    num_unconstrained_params = num_unconstrained_params)
    logger.info("Fit results hist:\n", inference_obj_hist.perform_fit())

    return inference_obj_nsbi, inference_obj_hist

def main():
    args = parse_args()

    # Execution Flow
    try:
        
        workspace_hist = build_workspace(args.config_hist)
        workspace_nsbi = build_workspace(args.config)
        logger.info("The workspace nsbi:\n", workspace_nsbi)

        inference_obj_nsbi, inference_obj_hist = inference_object(workspace_hist, workspace_nsbi, 
                                                                  args.measurement2fit)
        
        logger.info("Data loading workflow completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
