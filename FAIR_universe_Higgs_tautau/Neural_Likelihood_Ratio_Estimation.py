import os
import sys
import argparse
import warnings
import logging
import numpy as np
import tensorflow as tf

sys.path.append('../')

# Load the package and modules for training and plotting
import nsbi_common_utils
from nsbi_common_utils import datasets, configuration
from nsbi_common_utils.training import density_ratio_trainer

# Import extracted training parameters
import training_NN_paras as nn_params

import mplhep as hep
hep.style.use(hep.style.ATLAS)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    
    parser = argparse.ArgumentParser(description="Neural Likelihood Ratio Estimation Trainer")
    parser.add_argument('--config', type=str, default='./config.yml', 
                        help='Path to the YAML configuration file')
    parser.add_argument('--train', action='store_true', 
                        help='Force training of new models (overrides training_NN_paras.py USE_SAVED_MODELS)')
    parser.add_argument('--savedDataPath', type=str, default='./saved_datasets/', 
                        help='path to the saved dataset')
    parser.add_argument('--use-log-loss', action='store_true', 
                        help='Use log loss instead of default loss')
    parser.add_argument('--delete-existing-models', action='store_true',
                        help='Delete existing models before training')
    args = parser.parse_args()
 

    logger.info("Starting Neural Likelihood Ratio Estimation workflow.")

    logger.info(f"Loading configuration from {args.config}")
    config = nsbi_common_utils.configuration.ConfigManager(file_path_string=args.config)

    features, features_scaling = config.get_training_features()
    logger.info(f"Training features: {features}")

    logger.info("Initializing Datasets...")
    branches_to_load = features + ['presel_score']
    
    Datasets = nsbi_common_utils.datasets.datasets(
        config_path=args.config,
        branches_to_load=branches_to_load
    )

    logger.info("Loading datasets from config...")
    dataset_incl_dict = Datasets.load_datasets_from_config(load_systematics=False)
    dataset_incl_nominal = dataset_incl_dict["Nominal"].copy()
    dataset_SR_nominal = Datasets.filter_region_dataset(dataset_incl_nominal, region="SR")

    PATH_TO_SAVED_DATA = args.savedDataPath
    if not PATH_TO_SAVED_DATA.endswith('/'):
        PATH_TO_SAVED_DATA += '/'
    TRAINING_OUTPUT_PATH = f'{PATH_TO_SAVED_DATA}output_training_nominal/'
    
    basis_processes = config.get_basis_samples()
    ref_processes = config.get_reference_samples()
    logger.info(f"Basis processes: {basis_processes}")
    logger.info(f"Reference processes: {ref_processes}")


    NN_training_mix_model = {}
    use_log_loss = args.use_log_loss
    DELETE_EXISTING_MODELS = args.delete_existing_models

    if DELETE_EXISTING_MODELS:
        logger.warning("DELETE_EXISTING_MODELS is True. Old models in the output directory will be removed.")

    path_to_ratios = {}
    path_to_figures = {}
    path_to_models = {}

    logger.info("Preparing datasets and initializing trainers...")
    for process_type in basis_processes:
        dataset_mix_model = Datasets.prepare_basis_training_dataset(
            dataset_SR_nominal, [process_type], dataset_SR_nominal, ref_processes
        )

        output_name = f'{process_type}'
        output_dir = f'{TRAINING_OUTPUT_PATH}general_output_{process_type}'
        path_to_ratios[process_type] = f'{TRAINING_OUTPUT_PATH}output_ratios_{process_type}/'
        path_to_figures[process_type] = f'{TRAINING_OUTPUT_PATH}output_figures_{process_type}/'
        path_to_models[process_type] = f'{TRAINING_OUTPUT_PATH}output_model_params_{process_type}/'

        NN_training_mix_model[process_type] = density_ratio_trainer(
            dataset_mix_model,
            dataset_mix_model['weights_normed'].to_numpy(),
            dataset_mix_model['train_labels'].to_numpy(),
            features,
            features_scaling,
            [process_type, 'ref'],
            output_dir, output_name,
            path_to_figures=path_to_figures[process_type],
            path_to_ratios=path_to_ratios[process_type],
            path_to_models=path_to_models[process_type],
            use_log_loss=use_log_loss,
            delete_existing_models=DELETE_EXISTING_MODELS
        )
        
        del dataset_mix_model

    num_gpus = len(tf.config.list_physical_devices('GPU'))
    logger.info(f"Num GPUs Available: {num_gpus}")

    if num_gpus == 0:
        logger.warning("No GPUs found. Training might be slow.")

    
    force_train = args.train
    
    for count, process_type in enumerate(basis_processes):
        logger.info(f"Processing {process_type}...")
        
        settings = nn_params.training_settings[process_type].copy()
        
        # Override load_trained_models if --train argument is present
        if force_train:
            logger.info("Force training mode enabled via CLI.")
            settings['load_trained_models'] = False
        else:
            logger.info(f"Using USE_SAVED_MODELS={settings['load_trained_models']} from config.")
        
        logger.info(f"Starting training/loading for {process_type}")
        NN_training_mix_model[process_type].train_ensemble(**settings)
        
        logger.info(f"Testing normalization for {process_type}...")
        NN_training_mix_model[process_type].test_normalization()

    logger.info("Training/Loading complete.")

    logger.info("Merging dataframes for final evaluation.")
    dataset_combined_SR = Datasets.merge_dataframe_dict_for_training(
        dataset_SR_nominal, None, samples_to_merge=["htautau", "ztautau", "ttbar"]
    )

    path_to_saved_ratios = {}
    
    for process_type in basis_processes:
        logger.info(f"Evaluating the density ratios p_c/p_ref for the full dataset and ")
        logger.info(f"saveing for the inference step for {process_type}...")
        path_to_saved_ratios[process_type] = NN_training_mix_model[process_type].evaluate_and_save_ratios(
            dataset_combined_SR, 
            aggregation_type='median_score'
        )
    
    logger.info(f"Ratios saved to: {path_to_saved_ratios}")

    path_to_save_root = f"{PATH_TO_SAVED_DATA}/dataset_Asimov_SR.root"
    logger.info(f"Saving Asimov_SR dataset to {path_to_save_root}...")
    nsbi_common_utils.datasets.save_dataframe_as_root(
        dataset_combined_SR, 
        path_to_save=path_to_save_root,
        tree_name="nominal"
    )

    # Save Asimov weights
    path_to_save_weights = f"{PATH_TO_SAVED_DATA}/asimov_weights.npy"
    logger.info(f"Saving Asimov_weights to {path_to_save_weights}...")
    np.save(f"{path_to_save_weights}", dataset_combined_SR["weights"].to_numpy())

    logger.info("Workflow completed successfully.")

if __name__ == "__main__":
    main()