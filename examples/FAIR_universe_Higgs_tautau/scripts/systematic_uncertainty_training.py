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
    parser.add_argument('--process', type=str, default=None,
                        help='Single process to train (e.g. ttbar). If not set, trains all.')
    parser.add_argument('--systematic', type=str, default=None,
                        help='Single systematic to train (e.g. JES). If not set, trains all.')
    parser.add_argument('--direction', type=str, default=None, choices=['Up', 'Dn'],
                        help='Single direction to train. If not set, trains both.')
    parser.add_argument('--ensemble_index', type=int, default=None,
                        help='Ensemble member index. If not set, trains without ensemble suffix.')
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
    
    # Load training settings directly from pipeline config
    sys_training_params = config_workflow["training_settings"].copy()

    force_train = config_workflow["force_train"]

    if force_train:
        logger.info(f"Force training enabled. Setting load_trained_models=False.")
        sys_training_params['load_trained_models'] = False
    else:
        logger.info(f"Using load_trained_models={sys_training_params['load_trained_models']} from config.")
    

    ensemble_index = args.ensemble_index
    if ensemble_index is not None:
        sys_training_params['ensemble_index'] = int(ensemble_index)
        ensemble_index_label = str(ensemble_index)
    else:
        sys_training_params['ensemble_index'] = None
        ensemble_index_label = ''

    NN_training_syst_process  = {}
    path_to_figures           = {}
    path_to_models            = {}

    basis_samples = config_nsbi.get_basis_samples()
    if args.process:
        basis_samples = [p for p in basis_samples if p == args.process]

    for process in basis_samples:

        NN_training_syst_process[process] = {}
        path_to_figures[process]          = {}
        path_to_models[process]           = {}

        # Iterate through systematics defined in NSBI config
        for dict_syst in config_nsbi.config["Systematics"]:

            if (process not in dict_syst["Samples"]) or (dict_syst["Type"] != "NormPlusShape"): continue

            syst = dict_syst["Name"]
            if args.systematic and syst != args.systematic: continue

            NN_training_syst_process[process][syst] = {}
            path_to_figures[process][syst]          = {}
            path_to_models[process][syst]           = {}

            directions = ["Up", "Dn"]
            if args.direction:
                directions = [args.direction]

            for direction in directions:
                samples_to_train = config_nsbi.get_samples_in_syst_for_training(syst, direction)
                
                if (process not in samples_to_train):
                    # Clean up if process not affected by this systematic
                    if not NN_training_syst_process[process][syst]:
                            del NN_training_syst_process[process][syst]
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
                
                output_name = f'{process}_{syst}_{direction}{ensemble_index_label}'

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
                    path_to_models=path_to_models[process][syst][direction],
                    delete_existing_models=False
                )

                logger.info(f"Training: {process} | {syst} | {direction}")
                NN_training_syst_process[process][syst][direction].train(**sys_training_params)

                NN_training_syst_process[process][syst][direction].test_normalization()
                NN_training_syst_process[process][syst][direction].make_overfit_plots()
        
                num_bins_cal = 50
                NN_training_syst_process[process][syst][direction].make_calib_plots(nbins=num_bins_cal, observable='score')
                # NN_training_mix_model[process_type].make_calib_plots(nbins=num_bins_cal, observable='llr')
        
                variables_to_plot=['log_DER_pt_h'] # The 1D variable for reweighting closure
                yscale_type='log'
                num_bins_plotting=21
                NN_training_syst_process[process][syst][direction].make_reweighted_plots(variables_to_plot, yscale_type, num_bins_plotting)

    logger.info("Systematic workflow completed.")

if __name__ == "__main__":
    main()
