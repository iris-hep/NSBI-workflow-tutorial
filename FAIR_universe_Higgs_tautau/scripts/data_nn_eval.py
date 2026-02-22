import os
import sys
import argparse
import warnings
import logging
import numpy as np
import yaml
import mplhep as hep
import pickle

import nsbi_common_utils

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
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info("Starting neural network evaluation.")

    logger.info(f"Loading configuration from {args.config}")
    # Load the workflow parameters
    config_workflow_nominal         = load_config(args.config)["neural_likelihood_ratio_estimation"]
    config_workflow_systematics     = load_config(args.config)["systematic_uncertainty"]

    nsbi_fit_config_path = config_workflow_nominal["nsbi_fit_config"]
    logger.info(f"Initializing NSBI ConfigManager from: {nsbi_fit_config_path}")
    fit_config_nsbi = nsbi_common_utils.configuration.ConfigManager(file_path_string=nsbi_fit_config_path)

    # Load the training features defined in fit configuration file -- this can also be passed separately if just using training APIs
    features, features_scaling = fit_config_nsbi.get_training_features()
    logger.info(f"Training features loaded: {len(features)} features")

    logger.info("Initializing datasets...")
    branches_to_load = features + ['presel_score'] # Can be defined independently of config when using just the APIs
    
    # datasets library helps with preparation of data, reads metadata from fit configuration file
    datasets_helper = nsbi_common_utils.datasets.datasets(
        config_path=nsbi_fit_config_path,
        branches_to_load=branches_to_load
    )

    logger.info("Loading datasets from paths defined in fit config...")
    dataset_incl_dict = datasets_helper.load_datasets_from_config(load_systematics=False)

    # The loaded dataframe is a dictionary, with "Nominal" key referring to the nominal dataset
    dataset_incl_nominal = dataset_incl_dict["Nominal"].copy()

    # Get the signal region events to be used for SBI fit
    region = config_workflow_nominal["filter_region"]
    dataset_SR_nominal = datasets_helper.filter_region_dataset(dataset_incl_nominal, region=region)

    # Get the anchor/basis points used to build the full statistical model
    basis_processes = fit_config_nsbi.get_basis_samples()
    logger.info(f"Basis processes: {basis_processes}")

    logger.info("Merging dataframes for final evaluation.")
    dataset_Asimov_SR = datasets_helper.merge_dataframe_dict_for_training(
        dataset_SR_nominal, None, samples_to_merge=basis_processes
    )

    # Save Asimov weights for inference
    weight_save_path = fit_config_nsbi.get_channel_asimov_weight_path(channel_name=region) # Get path to asimov weights from fit config
    np.save(weight_save_path, dataset_Asimov_SR.weights.to_numpy()) # save the weights

    # Get the path where intermediate data from the workflow is saved
    path_to_saved_data = config_workflow_nominal["saved_data_path"]
    if not path_to_saved_data.endswith('/'):
        path_to_saved_data += '/'
    
    # Get the path where trained models were saved
    training_input_dir_name = config_workflow_nominal["output_training_dir"]
    trained_models_path = os.path.join(path_to_saved_data, training_input_dir_name)
    if not trained_models_path.endswith('/'):
        trained_models_path += '/'
        
    logger.info(f"Trained models path: {trained_models_path}")

    # TODO: add support for use_log_loss
    use_log_loss = config_workflow_nominal["use_log_loss"]

    ensemble_members    = config_workflow_nominal.get("num_ensemble_members", 1)
    aggregation_type    = config_workflow_nominal.get("ensemble_aggregation_type", "median_score")

    logger.info("Evaluating and saving nominal density ratios on Asimov dataset")
    for process_type in basis_processes:

        path_to_saving_evaluated_ratios         = os.path.join(trained_models_path, f'output_ratios_{process_type}/')
        path_to_trained_models                  = os.path.join(trained_models_path, f'output_model_params_{process_type}/')

        score_pred = np.ones((ensemble_members, dataset_Asimov_SR.shape[0]))
        ratio_pred = np.ones((ensemble_members, dataset_Asimov_SR.shape[0]))

        for ensemble_index in range(ensemble_members):

            path_to_saved_scaler        = f"{path_to_trained_models}model_scaler{ensemble_index}.bin"
            path_to_saved_model         = f"{path_to_trained_models}model{ensemble_index}.onnx"

            logger.info(f"Reading saved models from {path_to_trained_models}")
            scaler, model_NN                = nsbi_common_utils.training.load_trained_model(path_to_saved_model, path_to_saved_scaler)
            score_pred[ensemble_index]      = nsbi_common_utils.training.predict_with_onnx(dataset_Asimov_SR[features], scaler, model_NN, batch_size = 10_000)
            ratio_pred[ensemble_index]      = nsbi_common_utils.training.convert_score_to_ratio(score_pred[ensemble_index])    


        if aggregation_type == 'median_ratio':
            ratio_ensemble = np.median(ratio_pred, axis=0)
            
        elif aggregation_type == 'mean_ratio':
            ratio_ensemble = np.mean(ratio_pred, axis=0)
            
        elif aggregation_type == 'median_score':
            score_aggregate = np.median(score_pred, axis=0)
            ratio_ensemble = score_aggregate / (1.0 - score_aggregate)
            
        elif aggregation_type == 'mean_score':
            score_aggregate = np.mean(score_pred, axis=0)
            ratio_ensemble = score_aggregate / (1.0 - score_aggregate)
    
        else:
            raise Exception("aggregation_type not recognized, please choose between median_ratio, mean_ratio, median_score or mean_score")

        saved_ratio_path = f"{path_to_saving_evaluated_ratios}ratio_{process_type}.npy"
        np.save(saved_ratio_path, ratio_ensemble)

        logger.info(f"Nominal density ratios for {process_type} basis point saved to: {saved_ratio_path}")

    logger.info("All nominal ratios evaluated on Asimov and saved.")

    logger.info("Running evaluation on Asimov with systematic variation networks...")

    # Get the path where trained models were saved
    training_input_dir_name = config_workflow_systematics["output_training_dir"]
    trained_models_path = os.path.join(path_to_saved_data, training_input_dir_name)
    if not trained_models_path.endswith('/'):
        trained_models_path += '/'
        
    logger.info(f"Trained models path: {trained_models_path}")

    calibration_flag        = config_workflow_systematics["training_settings"].get("calibration", False)

    for process_type in basis_processes:

        ensemble_index = 0 #TODO: support ensemble evaluations for systematics too

        for dict_syst in fit_config_nsbi.config["Systematics"]:

            # Only evaluate norm+shape systematics where the process_type is involved
            if (process_type not in dict_syst["Samples"]) or (dict_syst["Type"] != "NormPlusShape"): continue

            syst = dict_syst["Name"]

            for direction in ["Up", "Dn"]:

                output_name = f'{process_type}_{syst}_{direction}'
                        
                path_to_saving_evaluated_ratios     = os.path.join(trained_models_path, f'output_ratios_{output_name}/')
                path_to_trained_models              = os.path.join(trained_models_path, f'output_model_params_{output_name}/')

                path_to_saved_scaler        = f"{path_to_trained_models}model_scaler{ensemble_index}.bin"
                path_to_saved_model         = f"{path_to_trained_models}model{ensemble_index}.onnx"

                logger.info(f"Evaluating and Saving Ratios for {process_type} {syst} {direction}")

                logger.info(f"Reading saved models from {path_to_trained_models}")
                scaler, model_NN                = nsbi_common_utils.training.load_trained_model(path_to_saved_model, path_to_saved_scaler)

                path_to_calibrator_model    = None
                if calibration_flag:
                    path_to_calibrator_model         = f"{path_to_trained_models}model_calibrated_hist{ensemble_index}.obj"
                    if not os.path.exists(path_to_calibrator_model):
                        logger.warning(f"No calibration model found with name {path_to_calibrator_model}")
                        calibration_model = None
                    else:
                        file_calibration = open(path_to_calibrator_model, 'rb') 
                        calibration_model = pickle.load(file_calibration)

                score_pred      = nsbi_common_utils.training.predict_with_onnx(dataset_Asimov_SR[features], 
                                                                                               scaler, model_NN, 
                                                                                               calibration_model = calibration_model, 
                                                                                               batch_size = 10_000)
                ratio_pred      = nsbi_common_utils.training.convert_score_to_ratio(score_pred)    

                saved_ratio_path = f"{path_to_saving_evaluated_ratios}ratio_{process_type}.npy"

                np.save(saved_ratio_path, ratio_pred)

                logger.info(f"Systematic density ratios for {syst}_{direction} affecting the {process_type} basis point saved to: {saved_ratio_path}")

                logger.info("All systematic density ratios evaluated on Asimov and saved.")

if __name__ == "__main__":
    main()
