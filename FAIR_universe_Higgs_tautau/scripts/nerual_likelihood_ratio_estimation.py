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
    parser.add_argument('--train', action='store_true', 
                        help='Force training of new models (overrides config settings)')
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info("Starting Neural Likelihood Ratio Estimation workflow.")

    logger.info(f"Loading configuration from {args.config}")
    config_workflow = load_config(args.config)["neural_likelihood_ratio_estimation"]
    
    nsbi_config_path = config_workflow["nsbi_config"]
    logger.info(f"Initializing NSBI ConfigManager from: {nsbi_config_path}")
    config_nsbi = nsbi_common_utils.configuration.ConfigManager(file_path_string=nsbi_config_path)

    features, features_scaling = config_nsbi.get_training_features()
    logger.info(f"Training features loaded: {len(features)} features")

    logger.info("Initializing Datasets...")
    branches_to_load = features + ['presel_score']
    
    Datasets = nsbi_common_utils.datasets.datasets(
        config_path=nsbi_config_path,
        branches_to_load=branches_to_load
    )

    logger.info("Loading datasets from config...")
    dataset_incl_dict = Datasets.load_datasets_from_config(load_systematics=False)
    dataset_incl_nominal = dataset_incl_dict["Nominal"].copy()
    dataset_SR_nominal = Datasets.filter_region_dataset(dataset_incl_nominal, region="SR")

    path_to_saved_data = config_workflow["saved_data_path"]
    if not path_to_saved_data.endswith('/'):
        path_to_saved_data += '/'
        
    training_output_dir_name = config_workflow["output_training_dir"]
    training_output_path = os.path.join(path_to_saved_data, training_output_dir_name)
    if not training_output_path.endswith('/'):
        training_output_path += '/'
        
    logger.info(f"Training output path: {training_output_path}")
    
    basis_processes = config_nsbi.get_basis_samples()
    ref_processes = config_nsbi.get_reference_samples()
    logger.info(f"Basis processes: {basis_processes}")
    logger.info(f"Reference processes: {ref_processes}")

    NN_training_mix_model = {}
    use_log_loss = config_workflow["use_log_loss"]
    delete_existing = config_workflow["delete_existing_models"]

    if delete_existing:
        logger.warning("DELETE_EXISTING_MODELS is True. Old models will be removed.")

    path_to_ratios = {}
    path_to_figures = {}
    path_to_models = {}

    logger.info("Preparing datasets and initializing trainers...")
    for process_type in basis_processes:
        dataset_mix_model = Datasets.prepare_basis_training_dataset(
            dataset_SR_nominal, [process_type], dataset_SR_nominal, ref_processes
        )

        output_name = f'{process_type}'
        output_dir = os.path.join(training_output_path, f'general_output_{process_type}')
        
        path_to_ratios[process_type] = os.path.join(training_output_path, f'output_ratios_{process_type}/')
        path_to_figures[process_type] = os.path.join(training_output_path, f'output_figures_{process_type}/')
        path_to_models[process_type] = os.path.join(training_output_path, f'output_model_params_{process_type}/')

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
            delete_existing_models=delete_existing
        )
        
        del dataset_mix_model

    num_gpus = len(tf.config.list_physical_devices('GPU'))
    logger.info(f"Num GPUs Available: {num_gpus}")
    if num_gpus == 0:
        logger.warning("No GPUs found. Training might be slow.")

    force_train = args.train or config_workflow["force_train"]
    
    training_settings = config_workflow["training_settings"]

    for process_type in basis_processes:
        logger.info(f"Processing {process_type}...")
        
        if process_type not in training_settings:
            logger.error(f"Settings for process '{process_type}' not found in 'density_ratio_estimation.training_settings'.")
            raise KeyError(f"Missing config for {process_type}")

        settings = training_settings[process_type].copy()
        
        if force_train:
            logger.info(f"Force training enabled. Setting load_trained_models=False for {process_type}.")
            settings['load_trained_models'] = False
        else:
            logger.info(f"Using load_trained_models={settings['load_trained_models']} from config for {process_type}.")
        
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
        logger.info(f"Evaluating density ratios for {process_type}...")
        path_to_saved_ratios[process_type] = NN_training_mix_model[process_type].evaluate_and_save_ratios(
            dataset_combined_SR, 
            aggregation_type='median_score'
        )
    
    logger.info(f"Ratios saved to: {path_to_saved_ratios}")

    path_to_save_root = os.path.join(path_to_saved_data, "dataset_Asimov_SR.root")
    logger.info(f"Saving Asimov_SR dataset to {path_to_save_root}...")
    nsbi_common_utils.datasets.save_dataframe_as_root(
        dataset_combined_SR, 
        path_to_save=path_to_save_root,
        tree_name="nominal"
    )

    path_to_save_weights = os.path.join(path_to_saved_data, "asimov_weights.npy")
    logger.info(f"Saving Asimov_weights to {path_to_save_weights}...")
    np.save(path_to_save_weights, dataset_combined_SR["weights"].to_numpy())

    logger.info("Workflow completed successfully.")

if __name__ == "__main__":
    main()