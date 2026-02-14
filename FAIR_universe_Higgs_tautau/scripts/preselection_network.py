import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import warnings
import yaml
import matplotlib.pyplot as plt
import mplhep as hep
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from utils import calculate_preselection_observable

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

hep.style.use(hep.style.ATLAS)

sys.path.append('../src')
import nsbi_common_utils
from nsbi_common_utils import training, datasets, configuration
from nsbi_common_utils.training import preselection_network_trainer

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Preselection Neural Network Training & Inference")
    parser.add_argument('--config', type=str, default='config.pipeline.yaml', 
                        help='Path to the YAML configuration file')
    return parser.parse_args()


def plot_score_distribution(dataset_dict, output_dir):
    """
    Generates the distribution of preselection scores for various processes.
    Matches the logic in '3_preselection_network.ipynb'.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created plot directory: {output_dir}")

    presel_scores = []
    weights_list = []
    
    for sample_name, sample_dataset in dataset_dict["Nominal"].items():
        if 'presel_score' in sample_dataset.columns:
            presel_scores.append(sample_dataset['presel_score'].values)
            weights_list.append(sample_dataset['weights'].values)
    
    if not presel_scores:
        print("No preselection scores found to plot.")
        return

    all_scores = np.concatenate(presel_scores)
    min_pred = np.min(all_scores)
    max_pred = np.max(all_scores)
    
    bins = np.linspace(min_pred, max_pred, num=50)
    
    plt.figure(figsize=(8, 6))
    
    for sample_name, sample_dataset in dataset_dict["Nominal"].items():
        if 'presel_score' not in sample_dataset.columns:
            continue
            
        scores = sample_dataset['presel_score']
        weights = sample_dataset['weights']
        
        hist, _ = np.histogram(scores, weights=weights, bins=bins)
        hist_err, _ = np.histogram(scores, weights=weights**2, bins=bins)
        
        # Plot using mplhep
        hep.histplot(hist, bins=bins, 
                     alpha=0.6, label=sample_name, 
                     density=True, linewidth=2.0, 
                     yerr=np.sqrt(hist_err)) 

    plt.xlabel("Preselection Score", size=18)
    plt.ylabel("Density", size=18)
    plt.legend()
    plt.yscale('log')
    
  
    save_path = os.path.join(output_dir, "score_distribution_notebook_style.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Score distribution plot saved to {save_path}")


def main():
    args = parse_args()
    logger.info(f"Loading configuration from {args.config}")
    config_workflow = load_config(args.config)["preselection_network"]

    config_train        = config_workflow["training"]
    config_preselection = config_workflow["preselection_observable"]

    
    fit_config_path = config_workflow["fit_config_path"]
    logger.info(f"Initializing ConfigManager from: {fit_config_path}")
    
    fit_config = nsbi_common_utils.configuration.ConfigManager(file_path_string = fit_config_path)
    
    features, features_scaling = fit_config.get_training_features()
    logger.info(f"Training features loaded from NSBI config: {len(features)} features")
    
    label_dict = config_train["labels"]
    
    logger.info(f"Initializing Datasets...")
    datasets_helper = nsbi_common_utils.datasets.datasets(
        config_path=fit_config_path,
        branches_to_load=features
    )

    dataset_incl_dict = datasets_helper.load_datasets_from_config(load_systematics=True)
    dataset_incl_nominal = dataset_incl_dict["Nominal"].copy()

    logger.info("Merging nominal samples for training...")
    dataset_incl_nominal_training = datasets_helper.merge_dataframe_dict_for_training(
        dataset_incl_nominal, 
        label_dict,
        samples_to_merge=dataset_incl_nominal.keys()
    )

    preselectionTraining = preselection_network_trainer(
        dataset_incl_nominal_training, 
        features, 
        features_scaling
    )

    model_path = config_workflow["model_path"]
    force_train = config_workflow["force_train"]
    load_trained_models = config_workflow["load_trained_models"]
    
    if force_train or not load_trained_models:
        logger.info(f"Starting Training")
        preselectionTraining.train(
            test_size=config_train["test_size"], 
            random_state=config_train["random_state"], 
            path_to_save=model_path,
            batch_size=config_train["batch_size"],
            epochs=config_train["epochs"], 
            learning_rate=config_train["learning_rate"]
        )
        logger.info(f"Training complete. Model saved to {model_path}")
    else:
        logger.info(f"Using load_trained_models={load_trained_models} from config.")
        preselectionTraining.assign_trained_model(model_path)

    logger.info("Running inference")

    for region_name, dataset_sample_dict in dataset_incl_dict.items():
        for sample_name, dataset in dataset_sample_dict.items():
            
            pred_NN_incl = preselectionTraining.predict(dataset)
            
            presel_score = calculate_preselection_observable(
                pred_NN_incl, 
                samples_list        =   label_dict,
                signal_processes    =   config_preselection["signal_processes"], 
                background_processes=   config_preselection["background_processes"],
                pre_factor_dict     =   config_preselection["pre_factor_dict"]
            )

            dataset_incl_dict[region_name][sample_name]['presel_score'] = presel_score

    
    logger.info("Saving datasets with new 'presel_score' branch...")
    datasets_helper.add_appended_branches(['presel_score'])
    datasets_helper.save_dataset_to_ntuple(dataset_incl_dict, save_systematics=True)

    logger.info("Saving the 'Preselection Score' plots...")
    plot_score_distribution(
        dataset_dict=dataset_incl_dict,
        output_dir=config_workflow["output"]["plots_dir"]
    )

    logger.info("Preselection workflow completed successfully.")

if __name__ == "__main__":
    main()
