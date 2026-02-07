import os, sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import yaml
import logging
import warnings

# Add source path for utils
sys.path.append('../src')
import nsbi_common_utils
from nsbi_common_utils import datasets

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

hep.style.use(hep.style.ATLAS)

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Process HiggsML data features.")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.pipeline.yaml",
        help="Path to configuration file."
    )
    
    return parser.parse_args()

def ds_helper(cfg_path, branches):
    '''
    Uses nsbi_common_utils.datasets to load data.
    '''
    datasets_helper = nsbi_common_utils.datasets.datasets(
        config_path=cfg_path,
        branches_to_load=branches
    )
    return datasets_helper

def process_data(df, input_features_by_jet, branches):
    """Filters specific processes and balances the dataset."""
    
    median_feature = {}

    # 1. Calculate Medians from Nominal samples
    if "Nominal" in df:
        nominal_data = df["Nominal"]
    else:
        nominal_data = df.get("Nominal", df)

    for sample, sample_dataset in nominal_data.items(): 
        median_feature[sample] = {}

        for nJets, feat_list in input_features_by_jet.items():
            for feature in feat_list:
                # Calculate median for valid jet counts
                vals = sample_dataset.loc[sample_dataset['PRI_n_jets'] >= nJets, feature]
                if len(vals) > 0:
                    median_feature[sample][feature] = np.median(vals)
                else:
                    median_feature[sample][feature] = 0.0

    logger.info(f"Extracting additional branches from the engineered features") 
    branches_to_add = []

    # 2. Apply Engineering to all datasets (Systematics/Regions)
    for region, sample_datasets in df.items():

        for sample, sample_dataset in sample_datasets.items():   
            
            # --- Categorical Jet Masks ---
            sample_dataset['njet_0'] = (sample_dataset['PRI_n_jets'] == 0).astype(int)
            sample_dataset['njet_1'] = (sample_dataset['PRI_n_jets'] == 1).astype(int)
            sample_dataset['njet_2'] = (sample_dataset['PRI_n_jets'] >= 2).astype(int)

            for m in ['njet_0', 'njet_1', 'njet_2']:
                if m not in branches_to_add: branches_to_add.append(m)

            # --- Per-Jet Masks and Imputation ---
            for i, feat_list in input_features_by_jet.items():
                # Create mask
                mask_col = f'jet{i}_mask'
                sample_dataset[mask_col] = (sample_dataset['PRI_n_jets'] >= i).astype(float)

                if mask_col not in branches_to_add: branches_to_add.append(mask_col)

                # Impute
                for feat in feat_list:
                    # Use median from Nominal sample if available, else 0
                    med_val = median_feature.get(sample, {}).get(feat, 0)
                    sample_dataset[feat] = sample_dataset[feat].where(
                        sample_dataset['PRI_n_jets'] >= i, med_val
                    )

            # --- Log Transformations ---
            for feat in branches:
                if feat not in sample_dataset.columns: continue

                kin = sample_dataset[feat].to_numpy()
                if len(kin) == 0: continue

                if (np.amin(kin) > 0.0) and (np.amax(kin) > 100):
                    log_feat = 'log_' + feat
                    sample_dataset[log_feat] = np.log(kin + 10.0)

                    if log_feat not in branches_to_add:
                        branches_to_add.append(log_feat)

            df[region][sample] = sample_dataset
    
    return df, branches_to_add


def main():
    args = parse_args()
    
    cfg_full = load_config(args.config)["data_preprocessing"]
        
    feats = cfg_full["features"]

    input_features_noJets = feats["no_jets"]
    input_features_1Jets  = feats["one_jet"]
    input_features_2Jets  = feats["two_jets"]
    input_features_nJets  = feats["n_jets"]

    branches_to_load = (input_features_noJets + 
                        input_features_1Jets + 
                        input_features_2Jets + 
                        input_features_nJets)
    
    input_features_by_jet = {
        1 : input_features_1Jets, 
        2 : input_features_2Jets
    }

    try:
        logger.info(f"Loading and converting the dataset to Pandas DataFrame for processing...")
        
        datasets_helper = ds_helper(cfg_full['config_path'], branches_to_load)
        
        datasets_all = datasets_helper.load_datasets_from_config(load_systematics=True)

        datasets_all, add_branches = process_data(
            datasets_all, 
            input_features_by_jet, 
            branches=branches_to_load
        )
        
        logger.info(f"Adding additional branches to the DataFrame: {len(add_branches)} new features")
        datasets_helper.add_appended_branches(add_branches)

        datasets_helper.save_datasets(
            datasets_all, 
            save_systematics=True
        )
        
        logger.info("Data Preprocessing workflow completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()