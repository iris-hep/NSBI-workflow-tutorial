import os
import argparse
import logging
import numpy as np
import pandas as pd
import uproot
from HiggsML.datasets import download_dataset
from HiggsML.systematics import systematics

import yaml


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and process HiggsML data for analysis.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.pipeline.yaml",
        help="Configuration file path.",
    )
    return parser.parse_args()


def download_and_load(url: str, train_size: float) -> pd.DataFrame:
    """Download the dataset and return the training set as a DataFrame."""
    logger.info(f"Downloading dataset from {url}...")
    data = download_dataset(url)

    logger.info(f"Loading training set with size fraction: {train_size}")
    data.load_train_set(train_size=train_size)

    df = data.get_train_set()
    del data
    return df


def process_data(df: pd.DataFrame, list_of_processes: list, seed: int) -> pd.DataFrame:
    """
    (Completely optimal utility)
    Filter to the requested processes and balance by downsampling to the smallest class.
    """

    all_labels = df["detailed_labels"].unique()
    processes_to_exclude = list(set(all_labels) - set(list_of_processes))
    logger.info(f"Excluding processes: {processes_to_exclude}")

    df_filtered = df[~np.isin(df["detailed_labels"], processes_to_exclude)].copy()

    counts = df_filtered["detailed_labels"].value_counts()
    logger.info(f"Counts before balancing:\n{counts}")

    # Downsample every process to the size of the smallest one ,
    # rescaling weights so the total weight sum is preserved per process.
    min_process = counts.idxmin()
    n_min = counts.min()
    logger.info(f"Balancing to minimum process count ('{min_process}'): {n_min}")

    df_list = []
    for _, df_process in df_filtered.groupby("detailed_labels"):
        weight_sum_orig = df_process["weights"].sum() 
        df_sampled = df_process.sample(n=n_min, random_state=seed)
        df_sampled = df_sampled.copy()                  
        df_sampled["weights"] *= weight_sum_orig / df_sampled["weights"].sum()
        df_list.append(df_sampled)

    return pd.concat(df_list).reset_index(drop=True)


def apply_systematics(df: pd.DataFrame, syst_settings: dict) -> dict:
    """Generate the nominal dataset and all systematic variations."""
    dataset_dict = {}

    logger.info("Generating nominal dataset...")
    dataset_dict["nominal"] = systematics(data_set=df, dopostprocess=False)

    for sample_name, syst_args in syst_settings.items():
        logger.info(f"Generating systematic variation: {sample_name}")
        dataset_dict[sample_name] = systematics(data_set=df, dopostprocess=False, **syst_args)

    return dataset_dict


def save_root_files(
    dataset_dict: dict,
    output_dir: str,
    processes: list,
    selections: str,
) -> None:
    """
    Save each (sample, process) combination to its own ROOT file.
    """
    os.makedirs(output_dir, exist_ok=True)  
    logger.info(f"Output directory: {output_dir}")

    for sample, df in dataset_dict.items():
        output_path = os.path.join(output_dir, f"dataset_{sample}.root")
        logger.info(f"Writing {output_path}...")

        with uproot.recreate(output_path) as root_file:

            for process in processes:
                df_process = df[df["detailed_labels"] == process].copy()
                df_process = df_process.query(selections)

                # Drop the label column — not needed in the output tree
                columns_to_keep = [c for c in df_process.columns if c != "detailed_labels"]
                arrays = {col: df_process[col].to_numpy() for col in columns_to_keep}

                if arrays and len(df_process) > 0:
                    tree_name = f"tree_{process}"      
                    root_file[tree_name] = arrays
                    logger.info(f"  Wrote tree '{tree_name}' with {len(df_process)} events.")
                else:
                    logger.warning(
                        f"  No events for process '{process}' in sample '{sample}' "
                        f"after selection — tree not written."
                    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)["data_loader"]

    list_of_processes = config["data"]["processes"]
    selections = config["preprocess"]["selections"]
    seed = config["preprocess"]["seed"]

    # Inject the shared seed into each systematic's config
    syst_settings = config["systematics"]
    for syst_cfg in syst_settings.values():
        syst_cfg["seed"] = seed

    try:
        df_full = download_and_load(config["data"]["url"], config["data"]["train_size"])
        df_balanced = process_data(df_full, list_of_processes, seed)
        del df_full

        dataset_dict = apply_systematics(df_balanced, syst_settings)

        save_root_files(
            dataset_dict,
            config["output"]["dir"],
            list_of_processes,
            selections,
        )

        logger.info("Data loading workflow completed successfully.")

    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()