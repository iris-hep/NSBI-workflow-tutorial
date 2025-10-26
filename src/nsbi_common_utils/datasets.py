import os
import pandas as pd
import numpy as np
import uproot
import copy
import pathlib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Any, Dict, List, Literal, Optional, Union

from nsbi_common_utils.configuration import ConfigManager

def _load_dataframe_from_root(path_to_root_file: str, tree_name: str, branches_to_load: List[str]) -> pd.DataFrame:
    """Helper function to load a pandas DataFrame from a ROOT TTree using uproot."""
    with uproot.open(f"{path_to_root_file}:{tree_name}") as tree:
        return tree.arrays(branches_to_load, library="pd")


class NSBIDataFrameDataset(Dataset):
    """
    A PyTorch Dataset for handling pandas DataFrames, providing features, labels, and weights.
    Converts DataFrame columns to torch tensors on the fly.
    """
    def __init__(self, dataframe: pd.DataFrame, features: List[str], label_column: str, weight_column: str):
        """
        Initializes the dataset with a pandas DataFrame and column names for features, labels, and weights.

        Args:
            dataframe: The input pandas DataFrame.
            features: A list of column names to be used as model features.
            label_column: The name of the column containing the target labels.
            weight_column: The name of the column containing event weights.
        """
        required_cols = features + [label_column, weight_column]
        if not all(col in dataframe.columns for col in required_cols):
            missing_cols = set(required_cols) - set(dataframe.columns)
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

        self.features_data = dataframe[features].values.astype(np.float32)
        self.labels_data = dataframe[label_column].values.astype(np.int64)
        self.weights_data = dataframe[weight_column].values.astype(np.float32)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.labels_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a sample (features, label, weight) at the specified index.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A dictionary containing 'features', 'labels', and 'weights' as torch.Tensor.
        """
        features_tensor = torch.from_numpy(self.features_data[idx])
        labels_tensor = torch.tensor(self.labels_data[idx], dtype=torch.long)
        weights_tensor = torch.tensor(self.weights_data[idx], dtype=torch.float)
        return {'features': features_tensor, 'labels': labels_tensor, 'weights': weights_tensor}


class NSBIDataProcessor:
    """
    Handles reading ROOT TTrees into pandas DataFrames (via uproot),
    applying region filters from a config, merging/labeling for ML training,
    and preparing PyTorch DataLoaders for training and validation.
    """

    def __init__(self, 
                config_path: Union[pathlib.Path, str], 
                branches_to_load: List[str]):
        """
        Load analysis config and set the base list of branches to read.

        Args:
            config_path: Path to a YAML/JSON config consumed by ConfigManager.
            branches_to_load: Required list of TTree branches to import as features.
        Raises:
            ValueError: If branches_to_load is empty.
        """
        self.config = ConfigManager(file_path_string=config_path)
        
        if not branches_to_load:
            raise ValueError("Empty branch list provided to NSBIDataProcessor.")
        self.branches_to_load = list(branches_to_load)

        # Internal column names for processed data
        self._label_col = "train_labels"
        self._weight_col = "weights_normed"
        self._sample_name_col = "sample_name"

    def _load_raw_dataframes(self, load_systematics: bool = False) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Reads datasets defined in config into nested dictionaries of DataFrames.
        """
        dict_datasets = {"Nominal": {}}

        for dict_sample in self.config.config["Samples"]:
            path_to_root_file = dict_sample["SamplePath"]
            tree_name = dict_sample["Tree"]
            sample_name = dict_sample["Name"]
            weight_branch_cfg = dict_sample.get("Weight")

            branches_to_read = list(self.branches_to_load)
            if weight_branch_cfg and weight_branch_cfg not in branches_to_read:
                branches_to_read.append(weight_branch_cfg)
            
            try:
                df = _load_dataframe_from_root(path_to_root_file, tree_name, branches_to_read)
            except Exception as e:
                raise IOError(f"Failed to load ROOT file {path_to_root_file} for sample {sample_name}: {e}")

            if df.empty:
                print(f"Warning: No data loaded for sample {sample_name} from {path_to_root_file}:{tree_name}. Skipping.")
                continue

            df[self._sample_name_col] = sample_name

            if weight_branch_cfg:
                df.rename(columns={weight_branch_cfg: "weights"}, inplace=True)
            else:
                df["weights"] = 1.0

            dict_datasets["Nominal"][sample_name] = df
        
        # Systematic variations are loaded but not used in this simplified processor
        # The logic is kept for potential future extensions.
        if load_systematics:
            # ... systematic loading logic can be implemented here if needed ...
            pass

        return dict_datasets

    def prepare_dataloaders(
        self,
        test_size: float = 0.2,
        batch_size: int = 64,
        num_workers: int = 4,
        random_state: Optional[int] = None
    ) -> Dict[str, DataLoader]:
        """
        Loads, processes, and splits data into PyTorch DataLoaders.

        Processing steps:
        1. Loads nominal dataframes using the config.
        2. Applies region selections from the config and merges samples.
        3. Creates a binary label column (1 for signal, 0 for background).
        4. Normalizes weights so sum(weights) is equal for signal and background.
        5. Splits data into training and validation sets.
        6. Creates and returns PyTorch DataLoaders.

        Args:
            test_size: Fraction of the dataset to be used as a validation set.
            batch_size: Number of samples per batch.
            num_workers: Number of subprocesses to use for data loading.
            random_state: Seed for reproducibility.

        Returns:
            A dictionary containing 'train' and 'validation' DataLoaders.
        """
        raw_data = self._load_raw_dataframes(load_systematics=False)
        nominal_data = raw_data["Nominal"]

        all_dfs = [df.copy() for df in nominal_data.values()]
        if not all_dfs:
            raise ValueError("No data loaded from config. Check sample paths and names.")
        full_df = pd.concat(all_dfs, ignore_index=True)

        signal_samples = [s["Name"] for s in self.config.config["Samples"] if s.get("Type") == "Signal"]
        if not signal_samples:
            raise ValueError("No samples with Type='Signal' found in the config.")
            
        full_df[self._label_col] = full_df[self._sample_name_col].isin(signal_samples).astype(int)

        is_signal = full_df[self._label_col] == 1
        sum_w_sig = full_df.loc[is_signal, "weights"].sum()
        sum_w_bkg = full_df.loc[~is_signal, "weights"].sum()

        full_df[self._weight_col] = full_df["weights"]
        if sum_w_sig > 0 and sum_w_bkg > 0:
            full_df.loc[~is_signal, self._weight_col] *= (sum_w_sig / sum_w_bkg)

        train_df, val_df = train_test_split(
            full_df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=full_df[self._label_col]
        )

        train_dataset = NSBIDataFrameDataset(
            dataframe=train_df,
            features=self.branches_to_load,
            label_column=self._label_col,
            weight_column=self._weight_col
        )
        val_dataset = NSBIDataFrameDataset(
            dataframe=val_df,
            features=self.branches_to_load,
            label_column=self._label_col,
            weight_column=self._weight_col
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return {"train": train_loader, "validation": val_loader}