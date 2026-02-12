import os
import sys
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


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process HiggsML data features.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.pipeline.yaml",
        help="Path to configuration file."
    )
    return parser.parse_args()


def process_data(df: dict, input_features_by_jet: dict, branches: list) -> tuple:
    """
    Apply feature engineering to all samples in all regions.
    
    Returns:
        (modified_df_dict, list_of_new_branch_names)
    """
    median_feature = {}

    # 1. Calculate medians from nominal samples
    nominal_data = df["Nominal"]

    logger.info("Computing medians from Nominal samples for imputation...")
    for sample, sample_dataset in nominal_data.items(): 
        median_feature[sample] = {}
        for n_jets, feat_list in input_features_by_jet.items():
            for feature in feat_list:
                # Calculate median for events with n_jets >= threshold
                vals = sample_dataset.loc[sample_dataset['PRI_n_jets'] >= n_jets, feature]
                median_feature[sample][feature] = np.median(vals) if len(vals) > 0 else 0.0

    logger.info("Applying feature engineering to all regions and samples...")
    branches_to_add = []

    # 2. Apply engineering to all datasets (systematics/regions)
    for region, sample_datasets in df.items():
        for sample, sample_dataset in sample_datasets.items():
            
            df_modified = sample_dataset.copy()
            
            # --- Categorical jet masks ---
            df_modified['njet_0'] = (df_modified['PRI_n_jets'] == 0).astype(int)
            df_modified['njet_1'] = (df_modified['PRI_n_jets'] == 1).astype(int)
            df_modified['njet_2'] = (df_modified['PRI_n_jets'] >= 2).astype(int)

            for mask_name in ['njet_0', 'njet_1', 'njet_2']:
                if mask_name not in branches_to_add:
                    branches_to_add.append(mask_name)

            # --- Per-jet masks and imputation ---
            for n_jets, feat_list in input_features_by_jet.items():
                mask_col = f'jet{n_jets}_mask'
                df_modified[mask_col] = (df_modified['PRI_n_jets'] >= n_jets).astype(float)
                
                if mask_col not in branches_to_add:
                    branches_to_add.append(mask_col)

                # Impute missing features for events below the jet threshold
                for feat in feat_list:
                    med_val = median_feature.get(sample, {}).get(feat, 0.0)
                    df_modified[feat] = df_modified[feat].where(
                        df_modified['PRI_n_jets'] >= n_jets, 
                        med_val
                    )

            # --- Log transformations ---
            for feat in branches:
                if feat not in df_modified.columns:
                    continue

                kin = df_modified[feat].to_numpy()
                if len(kin) == 0:
                    continue

                # Only apply log if all values are positive and range is large
                if (np.amin(kin) > 0.0) and (np.amax(kin) > 100.0):
                    log_feat = f'log_{feat}'
                    df_modified[log_feat] = np.log(kin + 10.0)
                    
                    if log_feat not in branches_to_add:
                        branches_to_add.append(log_feat)

            df[region][sample] = df_modified.copy()

    return df, branches_to_add


def main() -> None:
    args = parse_args()
    config = load_config(args.config)["data_preprocessing"]

    features = config["features"]

     # Specify branches to load from the ROOT ntuples
    input_features_noJets = ['PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_had_pt', 'PRI_had_eta',
        'PRI_had_phi', 'PRI_met', 'PRI_met_phi', 'DER_mass_transverse_met_lep',
        'DER_mass_vis', 'DER_pt_h', 'DER_deltar_had_lep', 'DER_pt_tot', 'DER_sum_pt',
        'DER_pt_ratio_lep_had', 'DER_met_phi_centrality']
    
    for feat in input_features_noJets:
        if feat not in features:
            input_features_noJets.remove(feat)

    input_features_1Jets = ['PRI_jet_leading_pt', 'PRI_jet_leading_eta',
        'PRI_jet_leading_phi',
        'PRI_jet_all_pt']
    
    for feat in input_features_1Jets:
        if feat not in features:
            input_features_1Jets.remove(feat)

    input_features_2Jets = ['PRI_jet_subleading_pt',
        'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
        'DER_prodeta_jet_jet',
        'DER_lep_eta_centrality']
    
    for feat in input_features_2Jets:
        if feat not in features:
            input_features_2Jets.remove(feat)

    input_features_nJets = ['PRI_n_jets']

    for feat in input_features_nJets:
        if feat not in features:
            input_features_nJets.remove(feat)

    branches_to_load = input_features_noJets \
                        + input_features_1Jets \
                        + input_features_2Jets \
                        + input_features_nJets
    

    input_features_by_jet = {
        1: input_features_1Jets, 
        2: input_features_2Jets
    }

    try:
        logger.info("Loading dataset into Pandas DataFrames...")
        logger.info(f"DEBUG: Config path = {config['config_path']}")
        datasets_helper = nsbi_common_utils.datasets.datasets(
                                                                config_path=config['config_path'],
                                                                branches_to_load=branches_to_load
                                                            )
        datasets_all = datasets_helper.load_datasets_from_config(load_systematics = True)

        logger.info(f"DEBUG: First dataset load keys = {datasets_all}")

        logger.info("Applying feature engineering...")
        datasets_all, new_branches = process_data(
            datasets_all, 
            input_features_by_jet, 
            branches=branches_to_load
        )

        logger.info(f"DEBUG: post-process dataset load keys = {datasets_all}")

        logger.info(f"Adding {len(new_branches)} new engineered features to output schema.")
        datasets_helper.add_appended_branches(new_branches)

        logger.info("Saving processed datasets...")
        datasets_helper.save_datasets(datasets_all, save_systematics=True)

        logger.info("Data preprocessing workflow completed successfully.")

    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()