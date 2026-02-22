import os
import sys
import argparse
import logging
import numpy as np
import yaml
import warnings
import random 
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
    parser = argparse.ArgumentParser(description="Systematic Uncertainty Estimation")
    parser.add_argument('--config', type=str, default='config.pipeline.yaml', 
                        help='Path to the main pipeline configuration file')
    parser.add_argument('--train', action='store_true', 
                        help='Force training of Neural Network for systematics')
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info("Starting Systematic Uncertainty Estimation workflow.")

    logger.info(f"Loading configuration from {args.config}")
    config_workflow = load_config(args.config)["systematic_uncertainty"]
    
    nsbi_config_path = config_workflow["nsbi_fit_config"]
    logger.info(f"Initializing NSBI ConfigManager from: {nsbi_config_path}")
    config_nsbi = nsbi_common_utils.configuration.ConfigManager(file_path_string=nsbi_config_path)
    
    features, features_scaling = config_nsbi.get_training_features()
    logger.info(f"Training features loaded: {len(features)}")

    logger.info("Initializing Datasets...")
    branches_to_load = features + ['presel_score']
    
    datasets_helper = nsbi_common_utils.datasets.datasets(
        config_path=nsbi_config_path,
        branches_to_load=branches_to_load
    )
    
    logger.info("Loading full datasets (Nominal + Systematics)...")
    dataset_incl_dict = datasets_helper.load_datasets_from_config(load_systematics=True)
    
    region = config_workflow["filter_region"]
    logger.info(f"Filtering datasets for region: {region}")
    dataset_SR_dict = datasets_helper.filter_region_by_type(dataset_incl_dict, region = region)
    
    path_to_saved_data = config_workflow["saved_data_path"]
    if not path_to_saved_data.endswith('/'):
        path_to_saved_data += '/'
    
    if not os.path.exists(path_to_saved_data):
        os.makedirs(path_to_saved_data)


    # =================================================================================
    # PART 2: SYSTEMATIC TRAINING 
    # =================================================================================
    
    should_train = config_workflow["force_train"]
    
    if should_train:
        logger.info("--- Starting Part 2: Neural Network Training for Systematics ---")
        
        # Load training settings directly from pipeline config
        sys_training_params = config_workflow["training_settings"].copy()
        
        # Force training override
        if should_train:
            logger.info("Force training enabled. Setting load_trained_models=False.")
            sys_training_params['load_trained_models'] = False

        NN_training_syst_process  = {}
        path_to_ratios            = {}
        path_to_figures           = {}
        path_to_models            = {}

        for process in config_nsbi.get_basis_samples():
            
            NN_training_syst_process[process] = {}
            path_to_ratios[process]           = {}
            path_to_figures[process]          = {}
            path_to_models[process]           = {}
            
            # Iterate through systematics defined in NSBI config
            for dict_syst in config_nsbi.config["Systematics"]:
                syst = dict_syst["Name"]
                NN_training_syst_process[process][syst] = {}
                path_to_ratios[process][syst]           = {}
                path_to_figures[process][syst]          = {}
                path_to_models[process][syst]           = {}

                for direction in ["Up", "Dn"]:
                    samples_to_train = config_nsbi.get_samples_in_syst_for_training(syst, direction)
                    
                    if (process not in samples_to_train):
                        # Clean up if process not affected by this systematic
                        if not NN_training_syst_process[process][syst]:
                             del NN_training_syst_process[process][syst]
                             del path_to_ratios[process][syst]
                             del path_to_figures[process][syst]
                             del path_to_models[process][syst]
                        continue

                    syst_key_name = f"{syst}_{direction}"
                    if syst_key_name not in dataset_SR_dict: continue

                    logger.info(f"Initializing Trainer: {process} vs {syst_key_name}")

                    # Prepare Dataset: Ratio of Systematic / Nominal 
                    dataset_syst_nom = datasets_helper.prepare_basis_training_dataset(
                        dataset_SR_dict[syst_key_name], 
                        [process], 
                        dataset_SR_dict["Nominal"], 
                        [process]
                    )

                    # Get the path where trained models will be saved
                    training_output_dir_name = config_workflow["output_training_dir"]
                    training_output_path = os.path.join(path_to_saved_data, training_output_dir_name)
                    if not training_output_path.endswith('/'):
                        training_output_path += '/'

                    logger.info(f"Training output path: {training_output_path}")
                    
                    output_name = f'{process}_{syst}_{direction}'
                    
                    path_to_ratios[process][syst][direction]    = os.path.join(training_output_path, f'output_ratios_{output_name}/')
                    path_to_figures[process][syst][direction]   = os.path.join(training_output_path, f'output_figures_{output_name}/')
                    path_to_models[process][syst][direction]    = os.path.join(training_output_path, f'output_model_params_{output_name}/')
                    
                    NN_training_syst_process[process][syst][direction] = nsbi_common_utils.training.density_ratio_trainer(
                        dataset_syst_nom, 
                        dataset_syst_nom['weights_normed'].to_numpy(),
                        dataset_syst_nom['train_labels'].to_numpy(),
                        features, 
                        features_scaling,
                        [syst+'_'+direction, process], 
                        output_name, 
                        path_to_figures=path_to_figures[process][syst][direction],
                        path_to_ratios=path_to_ratios[process][syst][direction], 
                        path_to_models=path_to_models[process][syst][direction],
                        delete_existing_models=False
                    )

        logger.info("Executing Training Loop...")

        for process, process_dict in NN_training_syst_process.items():
            for syst, syst_dict in process_dict.items():
                for direction in syst_dict.keys():
                    
                    logger.info(f"Training: {process} | {syst} | {direction}")
                    NN_training_syst_process[process][syst][direction].train_ensemble(**sys_training_params)
    
    logger.info("Systematic workflow completed.")

if __name__ == "__main__":
    main()