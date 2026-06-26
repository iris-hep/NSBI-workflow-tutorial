import os
import sys, pathlib
from pathlib import Path
import argparse
import warnings
import logging
import numpy as np
import uproot
import yaml
import mplhep as hep
import pickle

import nsbi_common_utils

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    h = logging.StreamHandler(sys.stdout) 
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(h)
    logger.propagate = False 

hep.style.use(hep.style.ATLAS)

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Neural Likelihood Ratio Estimation Trainer")
    parser.add_argument('--config', type=str, default='config.pipeline.yaml', 
                        help='Path to the YAML configuration file')
    return parser.parse_args()

def _aggregate_ensemble(ratio_pred, loaded_indices, aggregation_type):
    """Aggregate ensemble predictions (density ratios) into a single ratio array via mean or median across ensemble members. Uses nan-skipping variants so a single pathological member (e.g. a log-loss training that saturated float32 in test_normalization and emits NaN/Inf for some events) does not poison the aggregate; median is also robust to up to ~half the members producing Inf."""
    if not loaded_indices:
        raise RuntimeError("No ensemble members were loaded — cannot aggregate.")
    if aggregation_type == 'median':
        return np.nanmedian(ratio_pred[loaded_indices], axis=0)
    elif aggregation_type == 'mean':
        return np.nanmean(ratio_pred[loaded_indices], axis=0)
    else:
        raise ValueError(f"Unknown aggregation_type: {aggregation_type}; expected 'mean' or 'median'")

def main():
    print(f"Running Main2")

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

    # Check if fold_index exists in the data (assigned during data_preprocessing)
    _sample0 = fit_config_nsbi.config["Samples"][0]
    with uproot.open(f"{_sample0['SamplePath']}:{_sample0['Tree']}") as _tree:
        if "fold_index" in _tree.keys() and "fold_index" not in branches_to_load:
            branches_to_load.append("fold_index")

    # datasets library helps with preparation of data, reads metadata from fit configuration file
    datasets_helper = nsbi_common_utils.datasets.datasets(
        config_path=nsbi_fit_config_path,
        branches_to_load=branches_to_load
    )

    logger.info("Loading datasets from paths defined in fit config...")
    print("Loading datasets from paths defined in fit config...")

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
    use_log_loss = config_workflow_nominal.get("use_log_loss", False)

    ensemble_members    = config_workflow_nominal.get("num_ensemble_members_evaluation", 1)
    print(f"ensemble_members = {ensemble_members}")

    aggregation_type    = config_workflow_nominal.get("ensemble_aggregation_type", "mean")
    print(f"aggregation_type = {aggregation_type}")

    # K-fold settings — infer from the data
    has_folds = "fold_index" in dataset_Asimov_SR.columns
    num_folds = int(dataset_Asimov_SR["fold_index"].max()) + 1 if has_folds else 1
    use_kfold = num_folds > 1
    if use_kfold:
        logger.info(f"K-fold evaluation mode with {num_folds} folds")

    logger.info("Evaluating and saving nominal density ratios on Asimov dataset")
    for process_type in basis_processes:

        path_to_saving_evaluated_ratios = os.path.join(trained_models_path, f'output_ratios_{process_type}/')

        if use_kfold:
            # Per-fold evaluation: each fold's models predict on their held-out events
            # We need fold_index from the nominal samples to know which events belong to which fold
            ratio_per_event = np.ones(dataset_Asimov_SR.shape[0])
            event_order = dataset_Asimov_SR.index.to_numpy()

            for fold_idx in range(num_folds):
                # Get the held-out event mask for this fold
                fold_mask = dataset_Asimov_SR["fold_index"].to_numpy() == fold_idx
                fold_data = dataset_Asimov_SR[fold_mask]
                n_fold_events = fold_data.shape[0]
                logger.info(f"Fold {fold_idx}: {n_fold_events} eval events for {process_type}")

                ratio_pred = np.ones((ensemble_members, n_fold_events))
                loaded_indices = []

                for ensemble_index in range(ensemble_members):
                    fold_label = f'_fold{fold_idx}'
                    path_to_trained_models = os.path.join(
                        trained_models_path, f'output_model_params_{process_type}{fold_label}{ensemble_index}/')
                    path_to_saved_scaler = f"{path_to_trained_models}model_scaler{ensemble_index}.bin"
                    path_to_saved_model  = f"{path_to_trained_models}model{ensemble_index}.onnx"

                    model_file = Path(path_to_saved_model)
                    if not model_file.is_file():
                        logger.warning(f"No model for fold {fold_idx} ensemble {ensemble_index} for {process_type}")
                        continue

                    scaler, model_NN = nsbi_common_utils.training.load_trained_model(path_to_saved_model, path_to_saved_scaler)
                    ratio_pred[ensemble_index] = nsbi_common_utils.training.predict_with_model(fold_data[features], scaler, model_NN, use_log_loss=use_log_loss)
                    loaded_indices.append(ensemble_index)

                # Aggregate ensemble for this fold
                fold_ratio = _aggregate_ensemble(ratio_pred, loaded_indices, aggregation_type)
                ratio_per_event[fold_mask] = fold_ratio

            ratio_ensemble = ratio_per_event

        else:
            # Original non-kfold path
            ratio_pred = np.ones((ensemble_members, dataset_Asimov_SR.shape[0]))
            loaded_indices = []

            for ensemble_index in range(ensemble_members):
                path_to_trained_models = os.path.join(trained_models_path, f'output_model_params_{process_type}{ensemble_index}/')
                path_to_saved_scaler = f"{path_to_trained_models}model_scaler{ensemble_index}.bin"
                path_to_saved_model  = f"{path_to_trained_models}model{ensemble_index}.onnx"

                model_file = Path(path_to_saved_model)
                if not model_file.is_file():
                    print(f"No model exists for ensemble index {ensemble_index} for process {process_type}")
                    continue
                logger.info(f"Reading saved models from {path_to_saved_model}")

                scaler, model_NN = nsbi_common_utils.training.load_trained_model(path_to_saved_model, path_to_saved_scaler)
                ratio_pred[ensemble_index] = nsbi_common_utils.training.predict_with_model(dataset_Asimov_SR[features], scaler, model_NN, use_log_loss = use_log_loss)
                loaded_indices.append(ensemble_index)

            ratio_ensemble = _aggregate_ensemble(ratio_pred, loaded_indices, aggregation_type)

        saved_ratio_path = f"{path_to_saving_evaluated_ratios}ratio_{process_type}.npy"
        os.makedirs(path_to_saving_evaluated_ratios, exist_ok=True)
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
    syst_ensemble_members   = config_workflow_systematics.get("num_ensemble_members_evaluation", 1)
    syst_aggregation_type   = config_workflow_systematics.get("ensemble_aggregation_type", "mean")

    use_log_loss = config_workflow_systematics.get("use_log_loss", False)
    # K-fold settings for systematics
    # Systematics use the same fold assignment as nominal
    syst_num_folds = num_folds
    syst_use_kfold = use_kfold

    for process_type in basis_processes:

        for dict_syst in fit_config_nsbi.config["Systematics"]:

            # Only evaluate norm+shape systematics where the process_type is involved
            if (process_type not in dict_syst["Samples"]) or (dict_syst["Type"] != "NormPlusShape"): continue

            syst = dict_syst["Name"]

            for direction in ["Up", "Dn"]:

                output_name_base = f'{process_type}_{syst}_{direction}'
                path_to_saving_evaluated_ratios = os.path.join(trained_models_path, f'output_ratios_{output_name_base}/')

                if syst_use_kfold:
                    ratio_per_event = np.ones(dataset_Asimov_SR.shape[0])

                    for fold_idx in range(syst_num_folds):
                        fold_mask = dataset_Asimov_SR["fold_index"].to_numpy() == fold_idx
                        fold_data = dataset_Asimov_SR[fold_mask]
                        n_fold = fold_data.shape[0]

                        ratio_pred_all = np.ones((syst_ensemble_members, n_fold))
                        loaded_indices = []

                        for ensemble_index in range(syst_ensemble_members):
                            fold_label = f'_fold{fold_idx}'
                            output_name = f'{output_name_base}{fold_label}{ensemble_index}'
                            path_to_trained_models = os.path.join(trained_models_path, f'output_model_params_{output_name}/')

                            path_to_saved_scaler = f"{path_to_trained_models}model_scaler{ensemble_index}.bin"
                            path_to_saved_model  = f"{path_to_trained_models}model{ensemble_index}.onnx"

                            model_file = Path(path_to_saved_model)
                            if not model_file.is_file():
                                logger.warning(f"No model for fold {fold_idx} ensemble {ensemble_index} for {output_name_base}")
                                continue

                            scaler, model_NN = nsbi_common_utils.training.load_trained_model(path_to_saved_model, path_to_saved_scaler)

                            calibration_model = None
                            if calibration_flag:
                                path_to_calibrator_model = f"{path_to_trained_models}model_calibrated_hist{ensemble_index}.obj"
                                if os.path.exists(path_to_calibrator_model):
                                    with open(path_to_calibrator_model, 'rb') as file_calibration:
                                        calibration_model = pickle.load(file_calibration)

                            ratio_pred_all[ensemble_index] = nsbi_common_utils.training.predict_with_model(
                                fold_data[features], scaler, model_NN, calibration_model=calibration_model, use_log_loss = use_log_loss)
                            loaded_indices.append(ensemble_index)

                        if not loaded_indices:
                            logger.error(f"No ensemble members for fold {fold_idx} {output_name_base}, skipping fold")
                            continue

                        ratio_per_event[fold_mask] = _aggregate_ensemble(
                            ratio_pred_all, loaded_indices, syst_aggregation_type)

                    ratio_ensemble = ratio_per_event

                else:
                    # Original non-kfold path
                    ratio_pred_all = np.ones((syst_ensemble_members, dataset_Asimov_SR.shape[0]))
                    loaded_indices = []

                    for ensemble_index in range(syst_ensemble_members):
                        output_name = f'{output_name_base}{ensemble_index}'
                        path_to_trained_models = os.path.join(trained_models_path, f'output_model_params_{output_name}/')

                        path_to_saved_scaler = f"{path_to_trained_models}model_scaler{ensemble_index}.bin"
                        path_to_saved_model  = f"{path_to_trained_models}model{ensemble_index}.onnx"

                        model_file = Path(path_to_saved_model)
                        if not model_file.is_file():
                            logger.warning(f"No model exists for ensemble index {ensemble_index} for {output_name_base}")
                            continue

                        scaler, model_NN = nsbi_common_utils.training.load_trained_model(path_to_saved_model, path_to_saved_scaler)

                        calibration_model = None
                        if calibration_flag:
                            path_to_calibrator_model = f"{path_to_trained_models}model_calibrated_hist{ensemble_index}.obj"
                            if os.path.exists(path_to_calibrator_model):
                                with open(path_to_calibrator_model, 'rb') as file_calibration:
                                    calibration_model = pickle.load(file_calibration)

                        ratio_pred_all[ensemble_index] = nsbi_common_utils.training.predict_with_model(
                            dataset_Asimov_SR[features], scaler, model_NN, calibration_model=calibration_model)
                        loaded_indices.append(ensemble_index)

                    if not loaded_indices:
                        logger.error(f"No ensemble members loaded for {output_name_base}, skipping")
                        continue

                    ratio_ensemble = _aggregate_ensemble(ratio_pred_all, loaded_indices, syst_aggregation_type)

                saved_ratio_path = f"{path_to_saving_evaluated_ratios}ratio_{process_type}.npy"
                os.makedirs(path_to_saving_evaluated_ratios, exist_ok=True)
                np.save(saved_ratio_path, ratio_ensemble)

                logger.info(f"Systematic density ratios for {syst}_{direction} ({len(loaded_indices)} ensemble members) "
                            f"affecting {process_type} saved to: {saved_ratio_path}")

    logger.info("All systematic density ratios evaluated on Asimov and saved.")

if __name__ == "__main__":
    print(f"Running Main")
    main()
