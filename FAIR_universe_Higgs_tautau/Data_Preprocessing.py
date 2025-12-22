import os, sys, importlib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import yaml
import uproot

from utils import plot_kinematic_features

sys.path.append('../src')
import nsbi_common_utils
from nsbi_common_utils import configuration
from nsbi_common_utils import datasets

import logging
import warnings
# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

hep.style.use(hep.style.ATLAS)

def parse_args():
    parser = argparse.ArgumentParser(description="Download and process HiggsML data for analysis.")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default='./config.yml',
        help="config file path"
    )
    
    
    return parser.parse_args()

def ds_helper(cfg, branches):
    '''
    Uses nsbi_common_utils.datasets to load data.
    '''
    datasets_helper = nsbi_common_utils.datasets.datasets(config_path = cfg,
                                                branches_to_load = branches)
    return datasets_helper

def process_data(df, input_features_by_jet, branches):
    """Filters specific processes and balances the dataset."""
    
    median_feature = {}

    for sample, sample_dataset in df["Nominal"].items(): 

        median_feature[sample] = {}

        for nJets, feat_list in input_features_by_jet.items():
            for feature in feat_list:

                median_feature[sample][feature] = np.median(sample_dataset.loc[sample_dataset['PRI_n_jets'] >= nJets, feature])

    logger.info(f"extracting additional branches from the engineered features") 
    branches_to_add = []

    for region, sample_datasets in df.items():

        for sample, sample_dataset in sample_datasets.items():   
            
            sample_dataset['njet_0'] = (sample_dataset['PRI_n_jets'] == 0).astype(int)
            sample_dataset['njet_1'] = (sample_dataset['PRI_n_jets'] == 1).astype(int)
            sample_dataset['njet_2'] = (sample_dataset['PRI_n_jets'] >= 2).astype(int)

            branches_to_add += ['njet_0', 'njet_1', 'njet_2']

            for i, feat_list in input_features_by_jet.items():
                mask_i = (sample_dataset['PRI_n_jets'] >= i).astype(float)
                sample_dataset[f'jet{i}_mask'] = mask_i

                branches_to_add += [f'jet{i}_mask']

                for feat in feat_list:
                    sample_dataset[feat] = sample_dataset[feat].where(sample_dataset['PRI_n_jets'] >= i, median_feature[sample][feat])

            for feat in branches.copy():

                kin = sample_dataset[feat].to_numpy()
                
                if (np.amin(kin) > 0.0) and (np.amax(kin)>100):
                    log_feat = 'log_'+feat
                    sample_dataset[log_feat] = np.log(kin+10.0)

                    if log_feat not in branches_to_add:
                        branches_to_add  += [log_feat]

            df[region][sample] = sample_dataset
    
    return df, branches_to_add


def main():
    args = parse_args()
    
    # Specify branches to load from the ROOT ntuples
    input_features_noJets = ['PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_had_pt', 'PRI_had_eta',
        'PRI_had_phi', 'PRI_met', 'PRI_met_phi', 'DER_mass_transverse_met_lep',
        'DER_mass_vis', 'DER_pt_h', 'DER_deltar_had_lep', 'DER_pt_tot', 'DER_sum_pt',
        'DER_pt_ratio_lep_had', 'DER_met_phi_centrality']

    input_features_1Jets = ['PRI_jet_leading_pt', 'PRI_jet_leading_eta',
        'PRI_jet_leading_phi',
        'PRI_jet_all_pt']

    input_features_2Jets = ['PRI_jet_subleading_pt',
        'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
        'DER_prodeta_jet_jet',
        'DER_lep_eta_centrality']

    input_features_nJets = ['PRI_n_jets']

    branches_to_load = input_features_noJets \
                        + input_features_1Jets \
                        + input_features_2Jets \
                        + input_features_nJets
    input_features_by_jet = {
        1   :   input_features_1Jets, 
        2   :   input_features_2Jets
    }

    # Execution Flow
    try:
        logger.info(f"Loading and converting the dataset to Pandas DataFrame for processing...")
        datasets_helper = ds_helper(args.config, branches_to_load)
        datasets_all = datasets_helper.load_datasets_from_config(load_systematics = True)

        datasets_all, add_branches = process_data(datasets_all, input_features_by_jet, 
                                                     branches=branches_to_load)
        
        logger.info(f"adding additional branches to the DataFrame")
        datasets_helper.add_appended_branches(add_branches)

        datasets_helper.save_datasets(datasets_all, 
                        save_systematics = True)
        
        
        logger.info("Data Preprocessing workflow completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
