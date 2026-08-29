import os
import sys
import argparse
import warnings
import logging
import numpy as np
import uproot
import yaml

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
    parser.add_argument('--ensemble_index', type=int, default=None,
                        help='Ensemble member index.')
    parser.add_argument('--process', type=str, default=None,
                        help='Basis point process to train (e.g. htautau, ztautau, ttbar).')
    parser.add_argument('--fold_index', type=int, default=None,
                        help='K-fold index: train on all folds except this one.')
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info("Starting Neural Likelihood Ratio Estimation workflow.")

    logger.info(f"Loading configuration from {args.config}")
    # Load the workflow parameters
    config_workflow = load_config(args.config)["neural_likelihood_ratio_estimation"]
   
    # Load the fit configuration
    nsbi_fit_config_path = config_workflow["nsbi_fit_config"]
    logger.info(f"Initializing NSBI ConfigManager from: {nsbi_fit_config_path}")
    fit_config_nsbi = nsbi_common_utils.configuration.ConfigManager(file_path_string=nsbi_fit_config_path)

    # Load the training features defined in fit configuration file -- this can also be passed separately if just using training APIs
    features, features_scaling = fit_config_nsbi.get_training_features()
    logger.info(f"Training features loaded: {len(features)} features")

    logger.info("Initializing datasets")
    branches_to_load = features + ['presel_score'] # Can be defined independently of config when using just the APIs

    # Check if fold_index exists in the data (assigned during data_preprocessing)
    # If it does, load it so we can do k-fold splitting
    fit_config_nsbi = nsbi_common_utils.configuration.ConfigManager(file_path_string=nsbi_fit_config_path)
    _sample0 = fit_config_nsbi.config["Samples"][0]
    with uproot.open(f"{_sample0['SamplePath']}:{_sample0['Tree']}") as _tree:
        if "fold_index" in _tree.keys() and "fold_index" not in branches_to_load:
            branches_to_load.append("fold_index")

    # datasets library helps with preparation of data, reads metadata from fit configuration file
    datasets_helper = nsbi_common_utils.datasets.datasets(
        config_path=nsbi_fit_config_path,
        branches_to_load=branches_to_load
    )

    logger.info("Loading datasets from paths defined in fit config")
    dataset_incl_dict = datasets_helper.load_datasets_from_config(load_systematics=False)

    # The loaded dataframe is a dictionary, with "Nominal" key referring to the nominal dataset
    dataset_incl_nominal = dataset_incl_dict["Nominal"].copy()

    # Get the signal region events to be used for SBI fit
    region = config_workflow["filter_region"]
    dataset_SR_nominal = datasets_helper.filter_region_dataset(dataset_incl_nominal, region=region)

    # Get the path where intermediate data from the workflow is saved
    path_to_saved_data = config_workflow["saved_data_path"]
    if not path_to_saved_data.endswith('/'):
        path_to_saved_data += '/'
    
    # Get the path where trained models will be saved
    training_output_dir_name = config_workflow["output_training_dir"]
    training_output_path = os.path.join(path_to_saved_data, training_output_dir_name)
    if not training_output_path.endswith('/'):
        training_output_path += '/'
        
    logger.info(f"Training output path: {training_output_path}")
    
    # Get the anchor/basis points used to build the full statistical model
    basis_processes = fit_config_nsbi.get_basis_samples()
    logger.info(f"Basis processes: {basis_processes}")

    # Reference mixture for prepare_basis_training_dataset; see its docstring for the spec. Skip this and pass your own reference DataFrame to density_ratio_trainer if you don't want the helper.
    reference_priors_raw = config_workflow.get("reference_priors", None)
    if not reference_priors_raw:
        raise KeyError(
            "config.pipeline.yaml: `neural_likelihood_ratio_estimation.reference_priors` is required because this script builds the reference via "
            "datasets.prepare_basis_training_dataset. If you'd rather supply your own reference dataset, bypass that helper and pass your prepared "
            "DataFrame + weights + labels directly to density_ratio_trainer."
        )

    reference_priors = {}
    excluded_samples = []
    for _name, _spec in reference_priors_raw.items():
        if _spec == 0 or _spec is False:
            excluded_samples.append(_name)
            continue
        reference_priors[_name] = _spec
    if not reference_priors:
        raise ValueError(f"reference_priors has no active samples (all entries are 0/False): {reference_priors_raw}")
    if excluded_samples:
        logger.info(f"Reference samples explicitly excluded (spec=0): {excluded_samples}")

    ref_processes = list(reference_priors.keys())
    logger.info(f"Reference processes: {ref_processes}")
    logger.info(f"Resolved reference_priors: {reference_priors}")

    NN_training_mix_model = {}
    use_log_loss = config_workflow["use_log_loss"]

    # Start afresh? Set delete_existing_models=True
    delete_existing = config_workflow["delete_existing_models"]

    if delete_existing:
        logger.warning("delete_existing_models is True. Old models will be removed.")

    path_to_figures = {}
    path_to_models = {}

    process_type_input = args.process
    if process_type_input is not None:
        logger.info(f"Only training process type {process_type_input}")
    else:
        logger.info(f"Train all processes")
    logger.info("Preparing datasets and initializing trainers")

    for process_type in basis_processes:
        if process_type_input is not None:
            if process_type_input != process_type:
                continue

        # Get training hyperparameters
        training_settings = config_workflow["training_settings"]

        if process_type not in training_settings:
            raise KeyError(f"Missing config for {process_type}")

        settings = training_settings[process_type].copy()

        # Flag that forces the retraining of density ratios
        force_train = config_workflow["force_train"]

        if force_train:
            logger.info(f"Force training enabled. Setting load_trained_models=False for {process_type}.")
            settings['load_trained_models'] = False
        else:
            logger.info(f"Using load_trained_models={settings['load_trained_models']} from config for {process_type}.")

        ensemble_index = args.ensemble_index
        if ensemble_index is not None:
            settings["ensemble_index"] = int(ensemble_index)
            ensemble_index_label = str(ensemble_index)
        else:
            settings["ensemble_index"] = ensemble_index
            ensemble_index_label = ''

        # K-fold: if fold_index is given, train on all folds except this one
        # num_folds is inferred from the data (set during data_preprocessing)
        fold_index = args.fold_index
        first_sample = next(iter(dataset_SR_nominal.values()))
        has_folds = "fold_index" in first_sample.columns
        num_folds = int(first_sample["fold_index"].max()) + 1 if has_folds else 1
        fold_label = f'_fold{fold_index}' if fold_index is not None else ''

        if fold_index is not None and num_folds > 1:
            logger.info(f"K-fold mode: training on all folds except fold {fold_index} (of {num_folds})")
            dataset_SR_train = nsbi_common_utils.datasets.datasets.split_by_fold(
                dataset_SR_nominal, fold_index, num_folds, mode="train"
            )
        else:
            dataset_SR_train = dataset_SR_nominal

        print(f"ensemble index is {ensemble_index_label}, fold: {fold_label}")

        # Prepare dataset to be passed to training
        dataset_mix_model = datasets_helper.prepare_basis_training_dataset(
            dataset_SR_train,
            [process_type],
            dataset_SR_train,
            ref_processes,
            denominatorisreferencehypothesis=False,
            reference_priors=reference_priors,
        )

        output_name = f'{process_type}'

        path_to_figures[process_type] = os.path.join(training_output_path, f'output_figures_{process_type}{fold_label}{ensemble_index_label}/')
        path_to_models[process_type] = os.path.join(training_output_path, f'output_model_params_{process_type}{fold_label}{ensemble_index_label}/')
        
        # setup the training of density ratios using density_ratio_trainer API
        NN_training_mix_model[process_type] = nsbi_common_utils.training.density_ratio_trainer(
                                                                                                dataset                 = dataset_mix_model,    # dataframe containing all the relevant features for training
                                                                                                weights                 = dataset_mix_model['weights_normed'].to_numpy(),
                                                                                                training_labels         = dataset_mix_model['train_labels'].to_numpy(),
                                                                                                features                = features,
                                                                                                features_scaling        = features_scaling,
                                                                                                sample_name             = [process_type, 'ref'],
                                                                                                output_name             = output_name,
                                                                                                path_to_figures         = path_to_figures[process_type],
                                                                                                path_to_models          = path_to_models[process_type],
                                                                                                use_log_loss            = use_log_loss,
                                                                                                delete_existing_models  = delete_existing
                                                                                            )
        
        del dataset_mix_model

        logger.info(f"Processing {process_type}")
        
        logger.info(f"Starting training/loading for {process_type}")

        NN_training_mix_model[process_type].train(**settings)

        logger.info(f"Testing normalization for {process_type}")
        NN_training_mix_model[process_type].test_normalization()
        NN_training_mix_model[process_type].make_overfit_plots(ensemble_index = ensemble_index_label)

        num_bins_cal = 50
        NN_training_mix_model[process_type].make_calib_plots(nbins=num_bins_cal, observable='score', ensemble_index = ensemble_index_label)
        # NN_training_mix_model[process_type].make_calib_plots(nbins=num_bins_cal, observable='llr')

        variables_to_plot=['log_DER_pt_h'] # The 1D variable for reweighting closure
        yscale_type='log'
        num_bins_plotting=21
        NN_training_mix_model[process_type].make_reweighted_plots(variables_to_plot, yscale_type, num_bins_plotting, ensemble_index = ensemble_index_label)

    logger.info("Training/Loading complete.")

    logger.info("Workflow completed successfully.")

if __name__ == "__main__":

    import mplhep as hep
    import nsbi_common_utils

    hep.style.use(hep.style.ATLAS)
    
    main()
