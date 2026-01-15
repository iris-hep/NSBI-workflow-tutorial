import os
import sys
import argparse
import logging
import numpy as np
import yaml
import warnings
import random 
import tensorflow as tf

# Ensure the parent directory is in the path for imports
sys.path.append('../')

# Import custom modules
import nsbi_common_utils
from nsbi_common_utils import datasets, configuration
from nsbi_common_utils.training import density_ratio_trainer

import mplhep as hep
hep.style.use(hep.style.ATLAS)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_training_settings(config_path):
    """
    Helper function to load training settings from the config_training_NN_parms.yml file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Training config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    return data['systematic_uncertainty_training']

def main():
    parser = argparse.ArgumentParser(description="Systematic Uncertainty Estimation")
    parser.add_argument('--config', type=str, default='./config.yml', 
                        help='Path to the main dataset configuration file')
    parser.add_argument('--training-config', type=str, default='./config_training_NN_parms.yml', 
                        help='Path to the training parameter YAML file')
    parser.add_argument('--savedDataPath', type=str, default='./saved_datasets/', 
                        help='Path to save the generated ROOT files')
    parser.add_argument('--region', type=str, default='SR', 
                        help='Region to filter (e.g., SR)')
    parser.add_argument('--train', action='store_true', 
                        help='Run the Neural Network training for systematics')
    
    args = parser.parse_args()

    PATH_TO_SAVED_DATA = args.savedDataPath
    if not PATH_TO_SAVED_DATA.endswith('/'):
        PATH_TO_SAVED_DATA += '/'
    
    if not os.path.exists(PATH_TO_SAVED_DATA):
        os.makedirs(PATH_TO_SAVED_DATA)

    logger.info(f"Loading configuration from {args.config}")
    config = nsbi_common_utils.configuration.ConfigManager(file_path_string=args.config)
    
    features, features_scaling = config.get_training_features()

    logger.info("Initializing Datasets...")
    branches_to_load = features + ['presel_score']
    
    Datasets = nsbi_common_utils.datasets.datasets(
        config_path=args.config,
        branches_to_load=branches_to_load
    )
    
    logger.info("Loading full datasets (Nominal + Systematics)...")
    dataset_incl_dict = Datasets.load_datasets_from_config(load_systematics=True)
    
    region = args.region
    logger.info(f"Filtering datasets for region: {region}")
    dataset_SR_dict = Datasets.filter_region_dataset(dataset_incl_dict, region=region)
    
    # =================================================================================
    # PART 1: SAVE ROOT FILES 
    # =================================================================================
    logger.info("--- Starting Part 1: Saving ROOT files for variations ---")
    
    samples_to_merge = ["htautau", "ztautau", "ttbar"]

    for variation_name, sample_dataset in dataset_SR_dict.items():
        if variation_name == 'Nominal': continue

        # Simple check to avoid re-saving if desired, or just overwrite
        try:
            dataset_SR = Datasets.merge_dataframe_dict_for_training(
                sample_dataset, None, samples_to_merge=samples_to_merge
            )
            filename = f"dataset_{variation_name}_{region}.root"
            path_to_save = f"{PATH_TO_SAVED_DATA}{filename}"
            
            logger.info(f"Saving {filename}...")
            nsbi_common_utils.datasets.save_dataframe_as_root(
                dataset_SR, path_to_save=path_to_save, tree_name="nominal"
            )
        except Exception as e:
            logger.error(f"Failed to save {variation_name}: {e}")

    # =================================================================================
    # PART 2: SYSTEMATIC TRAINING 
    # =================================================================================
    if args.train:
        logger.info("--- Starting Part 2: Neural Network Training for Systematics ---")
        
        logger.info(f"Loading systematic training parameters from {args.training_config}")
        sys_training_params = load_training_settings(args.training_config)
        
        NN_training_syst_process  = {}
        path_to_ratios            = {}
        path_to_figures           = {}
        path_to_models            = {}

        rnd_seed_traintestsplit = random.randint(0, 2**32 - 1)
        
        for process in config.get_basis_samples():
            
            NN_training_syst_process[process] = {}
            path_to_ratios[process]           = {}
            path_to_figures[process]          = {}
            path_to_models[process]           = {}
            
            for dict_syst in config.config["Systematics"]:
                syst = dict_syst["Name"]
                NN_training_syst_process[process][syst] = {}
                path_to_ratios[process][syst]           = {}
                path_to_figures[process][syst]          = {}
                path_to_models[process][syst]           = {}

                for direction in ["Up", "Dn"]:
                    samples_to_train = config.get_samples_in_syst_for_training(syst, direction)
                    
                    if (process not in samples_to_train):
                        # Cleanup empty dicts
                        if not NN_training_syst_process[process][syst]:
                             del NN_training_syst_process[process][syst]
                             del path_to_ratios[process][syst]
                             del path_to_figures[process][syst]
                             del path_to_models[process][syst]
                        continue

                    syst_key_name = syst + '_' + direction
                    if syst_key_name not in dataset_SR_dict: continue

                    logger.info(f"Initializing: {process} vs {syst_key_name}")

                    # Prepare Dataset: Ratio of Systematic / Nominal
                    dataset_syst_nom = Datasets.prepare_basis_training_dataset(
                        dataset_SR_dict[syst_key_name], 
                        [process], 
                        dataset_SR_dict["Nominal"], 
                        [process]
                    )

                    top_path = f'{PATH_TO_SAVED_DATA}output_training_systematics/'
                    output_name = f'{process}_{syst}_{direction}'
                    output_dir = f'{top_path}general_output_{output_name}'
                    
                    path_to_ratios[process][syst][direction]    = f'{top_path}output_ratios_{output_name}/'
                    path_to_figures[process][syst][direction]   = f'{top_path}output_figures_{output_name}/'
                    path_to_models[process][syst][direction]    = f'{top_path}output_model_params_{output_name}/'
                    
                    NN_training_syst_process[process][syst][direction] = density_ratio_trainer(
                        dataset_syst_nom, 
                        dataset_syst_nom['weights_normed'].to_numpy(),
                        dataset_syst_nom['train_labels'].to_numpy(),
                        features, 
                        features_scaling,
                        [syst+'_'+direction, process], 
                        output_dir, output_name, 
                        path_to_figures=path_to_figures[process][syst][direction],
                        path_to_ratios=path_to_ratios[process][syst][direction], 
                        path_to_models=path_to_models[process][syst][direction],
                        delete_existing_models=False
                    )

        logger.info("Executing Training Loop...")
        
        # If --train is passed, we FORCE training, ignoring 'load_trained_models' in YAML if needed
        # sys_training_params['load_trained_models'] = False 

        for process, process_dict in NN_training_syst_process.items():
            for syst, syst_dict in process_dict.items():
                for direction in syst_dict.keys():
                    
                    logger.info(f"Training: {process} | {syst} | {direction}")
                    
                    NN_training_syst_process[process][syst][direction].train_ensemble(**sys_training_params)    
                    
                    # NN_training_syst_process[process][syst][direction].test_normalization()

        logger.info("Evaluating Ratios on Asimov Dataset...")
        path_to_load = f"{PATH_TO_SAVED_DATA}dataset_Asimov_SR.root"
        
        dataset_Asimov_SR = nsbi_common_utils.datasets.load_dataframe_from_root(
            path_to_load=path_to_load, tree_name="nominal", branches_to_load=branches_to_load
        )
        
        ensemble_aggregation_type = 'mean_ratio'
        path_to_saved_ratios_eval = {} # Renamed to avoid confusion

        for process in config.get_basis_samples():
            path_to_saved_ratios_eval[process] = {}
            for dict_syst in config.config["Systematics"]:
                syst = dict_syst["Name"]
                if (process not in dict_syst["Samples"]) or (dict_syst["Type"] != "NormPlusShape"): continue
                
                path_to_saved_ratios_eval[process][syst] = {}
                for direction in ["Up", "Dn"]:
                            
                            logger.info(f"Evaluating {process} {syst} {direction}")
                            path_to_saved_ratios_eval[process][syst][direction] = \
                                NN_training_syst_process[process][syst][direction].evaluate_and_save_ratios(
                                    dataset_Asimov_SR, aggregation_type=ensemble_aggregation_type
                                )
    
    logger.info("Systematic workflow completed.")

if __name__ == "__main__":
    main()