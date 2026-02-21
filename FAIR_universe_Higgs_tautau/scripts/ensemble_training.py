import os
import sys
import argparse
import warnings
import logging
import numpy as np
import tensorflow as tf
import yaml
import mplhep as hep

sys.path.append('../src')
import nsbi_common_utils
from nsbi_common_utils import datasets, configuration
from nsbi_common_utils.training import density_ratio_trainer

hep.style.use(hep.style.ATLAS)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

def parse_args():
    parser = argparse.ArgumentParser(description="Neural Likelihood Ratio Estimation Trainer")
    parser.add_argument('--config', type=str, default='config.pipeline.yaml', 
                        help='Path to the YAML configuration file')
    parser.add_argument('--ensemble_size', type=int, default=None,
                        help='Size of the ensemble.')
    parser.add_argument('--process', type=str, default=None,
                        help='Basis point process to train (e.g. htautau, ztautau, ttbar).')
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info("Starting ensemble training.")

    logger.info(f"Loading configuration from {args.config}")
    config_workflow = load_config(args.config)["neural_likelihood_ratio_estimation"]

    nsbi_config_path = config_workflow["nsbi_config"]
    logger.info(f"Initializing NSBI ConfigManager from: {nsbi_config_path}")
    config_nsbi = nsbi_common_utils.configuration.ConfigManager(file_path_string=nsbi_config_path)