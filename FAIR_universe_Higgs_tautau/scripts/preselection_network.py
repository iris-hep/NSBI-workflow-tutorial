import os
import sys
import argparse
import logging
import numpy as np
import warnings
import tensorflow as tf

sys.path.append('../')

# Import custom modules
import nsbi_common_utils
from nsbi_common_utils import training, datasets, configuration
from nsbi_common_utils.training import preselection_network_trainer
from utils import calculate_preselection_observable

tf.config.optimizer.set_jit(False)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Preselection Neural Network Training & Inference")
    parser.add_argument('--config', type=str, default='./config.yml', 
                        help='Path to the YAML configuration file')
    parser.add_argument('--train', action='store_true', 
                        help='Force training of a new preselection model (overrides loading existing model)')
    parser.add_argument('--model-path', type=str, default='./saved_datasets/preselection_model/', 
                        help='Path to save/load the preselection model')
    
    args = parser.parse_args()

    logger.info(f"Loading configuration from {args.config}")
    config = nsbi_common_utils.configuration.ConfigManager(file_path_string=args.config)
    
    features, features_scaling = config.get_training_features()
    logger.info(f"Training features loaded: {features}")

    logger.info("Initializing Datasets...")
    datasets_helper = nsbi_common_utils.datasets.datasets(
        config_path=args.config,
        branches_to_load=features
    )

    train_label_sample_dict = {
        "htautau": 0,
        "ttbar": 1,
        "ztautau": 2
    }

    dataset_incl_dict = datasets_helper.load_datasets_from_config(load_systematics=True)
    
    dataset_incl_nominal = dataset_incl_dict["Nominal"].copy()

    logger.info("Merging nominal samples for training...")
    dataset_incl_nominal_training = datasets_helper.merge_dataframe_dict_for_training(
        dataset_incl_nominal, 
        train_label_sample_dict,
        samples_to_merge=dataset_incl_nominal.keys()
    )

    PATH_PRESEL_MODEL = args.model_path
    FORCE_TRAIN = args.train
    
    # Initialize Trainer
    preselectionTraining = preselection_network_trainer(
        dataset_incl_nominal_training, 
        features, 
        features_scaling
    )

    model_exists = os.path.exists(PATH_PRESEL_MODEL)
    
    if FORCE_TRAIN or not model_exists:
        logger.info(f"Starting Training (Force Train={FORCE_TRAIN}, Model Exists={model_exists})...")
        
        preselectionTraining.train(
            test_size=0.2, 
            random_state=42, 
            path_to_save=PATH_PRESEL_MODEL,
            batch_size=1024,
            epochs=50, 
            learning_rate=0.1
        )
        logger.info(f"Training complete. Model saved to {PATH_PRESEL_MODEL}")
        
    else:
        logger.info(f"Loading existing model from {PATH_PRESEL_MODEL}...")
        preselectionTraining.assign_trained_model(PATH_PRESEL_MODEL)

    for region_name, dataset_sample_dict in dataset_incl_dict.items():
        logger.debug(f"Processing region: {region_name}")
        
        for sample_name, dataset in dataset_sample_dict.items():
            
            # Get predictions (softmax outputs)
            pred_NN_incl = preselectionTraining.predict(dataset)
            
            presel_score = calculate_preselection_observable(
                pred_NN_incl, 
                train_label_sample_dict, 
                signal_processes       = ['htautau'], 
                background_processes   = ['ttbar', 'ztautau'], 
                pre_factor_dict        = {'htautau': 1.0, 'ttbar': 1.0, 'ztautau': 1.0}
            )

            dataset_incl_dict[region_name][sample_name]['presel_score'] = presel_score

    logger.info("Saving datasets with new 'presel_score' branch...")
    
    datasets_helper.add_appended_branches(['presel_score'])
    
    datasets_helper.save_datasets(dataset_incl_dict, save_systematics=True)

    logger.info("Preselection workflow completed successfully.")

if __name__ == "__main__":
    main()