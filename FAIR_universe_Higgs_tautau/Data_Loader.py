import os
import argparse
import logging
import warnings
import numpy as np
import pandas as pd
import mplhep as hep
import uproot
from HiggsML.datasets import download_dataset
from HiggsML.systematics import systematics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# # Suppress warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

hep.style.use(hep.style.ATLAS)

def parse_args():
    parser = argparse.ArgumentParser(description="Download and process HiggsML data for analysis.")
    
    parser.add_argument(
        "--url", 
        type=str, 
        default="https://zenodo.org/records/15131565/files/FAIR_Universe_HiggsML_data.zip",
        help="URL to the dataset zip file."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./saved_datasets/",
        help="Directory to save the processed ROOT files."
    )
    parser.add_argument(
        "--train-size", 
        type=float, 
        default=0.35,
        help="Fraction of dataset to use for training."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for sampling."
    )
    
    return parser.parse_args()

def download_and_load(url, train_size):
    """Downloads the dataset and loads the training set."""
    logger.info(f"Downloading dataset from {url}...")
    data = download_dataset(url)
    
    logger.info(f"Loading training set with size fraction: {train_size}")
    data.load_train_set(train_size=train_size)
    
    df_training_full = data.get_train_set()
    del data # Clean up memory
    return df_training_full

def process_data(df, list_of_processes, seed):
    """Filters specific processes and balances the dataset."""
    
    all_labels = df["detailed_labels"].unique()
    process_to_exclude = list(set(all_labels) - set(list_of_processes))
    logger.info(f"Excluding processes: {process_to_exclude}")

    # Filter dataframe
    mask_process_exclusion = ~np.isin(df["detailed_labels"], process_to_exclude)
    df_filtered = df[mask_process_exclusion].copy()
    
    counts = df_filtered["detailed_labels"].value_counts()
    logger.info(f"Counts before balancing:\n{counts}")

    # Trim the dataset, so all processes have equal entries

    # Here the notebook implemented the the number of ttbar events (lowest)
    min_process = counts.idxmin()
    n_min = counts.min()
    logger.info(f"Balancing to minimum process count ({min_process}): {n_min}")

    df_list = []
    for _, df_process in df_filtered.groupby('detailed_labels'):
        weight_sum_orig = df_process.weights.sum()

        df_sampled = df_process.sample(n=n_min, random_state=seed)

        df_sampled['weights'] *= weight_sum_orig / df_sampled['weights'].sum()

        df_list.append(df_sampled)
        del df_sampled

    df_balanced = pd.concat(df_list).reset_index(drop=True)
    return df_balanced

def apply_systematics(df, syst_settings):
    """Generates variations of the dataset based on systematics."""
    dataset_dict = {}

    logger.info("Generating nominal dataset...")
    dataset_dict['nominal'] = systematics(
        data_set=df,
        dopostprocess=False
    )

    for sample_name, syst_args in syst_settings.items():
        logger.info(f"Generating systematic variation: {sample_name}")
        dataset_dict[sample_name] = systematics(
            data_set=df, 
            dopostprocess=False, 
            **syst_args
        )
    
    return dataset_dict

def save_root_files(dataset_dict, output_dir, processes, selections):
    """Saves the datasets to ROOT files applying selections."""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    for sample, df in dataset_dict.items():
        output_path = os.path.join(output_dir, f"dataset_{sample}.root")
        logger.info(f"Writing {output_path}...")

        with uproot.recreate(output_path) as ntuple:
            for process in processes:
                
                df_process = df[df["detailed_labels"] == process].copy()
                
                
                df_process = df_process.query(selections).copy()

                
                columns_to_keep = list(set(df_process.columns.tolist()) - {"detailed_labels"})
                
                
                arrays = {col: df_process[col].to_numpy() for col in columns_to_keep}

                if arrays:
                    ntuple[f"tree_{process}"] = arrays
                else:
                    logger.warning(f"No events found for {process} in {sample} after selection.")

def main():
    args = parse_args()
    
    list_of_processes_to_model = ["htautau", "ztautau", "ttbar"]
    
    syst_settings = {
        'TES_up': {'tes': 1.02, 'seed': args.seed},
        'TES_dn': {'tes': 0.98, 'seed': args.seed},
        'JES_up': {'jes': 1.02, 'seed': args.seed},
        'JES_dn': {'jes': 0.98, 'seed': args.seed}
    }

    # Some common analysis selections to remove low-stats regions
    selections = (
        "DER_mass_transverse_met_lep <= 250.0 and "
        "DER_mass_vis <= 500.0 and "
        "DER_sum_pt <= 1000 and "
        "DER_pt_tot <= 250 and "
        "DER_deltar_had_lep <= 4.5 and "
        "DER_pt_h <= 400 and "
        "DER_pt_ratio_lep_had <= 9.0"
    )

    # Execution Flow
    try:
        
        df_training_full = download_and_load(args.url, args.train_size)

        df_training = process_data(df_training_full, list_of_processes_to_model, args.seed)
        del df_training_full

        
        dataset_dict = apply_systematics(df_training, syst_settings)

        
        save_root_files(dataset_dict, args.output_dir, list_of_processes_to_model, selections)
        
        logger.info("Data loading workflow completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
