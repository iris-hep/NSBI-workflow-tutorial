import os
from collections import defaultdict
from typing import Dict, List, Optional, Union
import uproot
import numpy as np
import pandas as pd


class datasets:
    """
    Utility class for loading and saving HiggsML datasets from/to ROOT files.
    
    Handles:
    - Loading nominal samples and systematic variations from ROOT files
    - Applying feature engineering and adding new branches
    - Saving modified DataFrames back to ROOT files without losing trees
    """

    def __init__(self, config_path: str, branches_to_load: List[str]):
        """
        Initialize the datasets helper.

        Args:
            config_path: Path to YAML config defining samples and systematics.
            branches_to_load: List of branch/column names to read from ROOT files.
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.branches_to_load = branches_to_load
        self.branches_all = branches_to_load.copy()

    def _load_config(self, path: str) -> dict:
        """Load YAML configuration file."""
        import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def add_appended_branches(self, new_branches: List[str]) -> None:
        """
        Register additional branches (e.g., engineered features) to be saved.

        Args:
            new_branches: List of new column names created during preprocessing.
        """
        for branch in new_branches:
            if branch not in self.branches_all:
                self.branches_all.append(branch)

    def load_datasets_from_config(self, load_systematics: bool = False) -> Dict:
        """
        Load datasets according to the config structure.

        Returns:
            Nested dict: {region: {sample_name: DataFrame}}
            - region: "Nominal" or systematic variation names (e.g., "JES_Up")
            - sample_name: process name (e.g., "htautau", "ztautau")

        Args:
            load_systematics: If True, also load systematic variation samples.
        """
        dict_datasets = {}

        # 1. Load nominal samples
        dict_datasets["Nominal"] = {}
        for sample_dict in self.config["Samples"]:
            sample_name = sample_dict["Name"]
            file_path = sample_dict["SamplePath"]
            tree_name = sample_dict["Tree"]
            
            # Determine which branches to load (include weight branch if specified)
            weight_branch = sample_dict.get("Weight")
            branches = self.branches_to_load.copy()
            if weight_branch and weight_branch not in branches:
                branches.append(weight_branch)

            df = self._load_dataframe_from_root(file_path, tree_name, branches)
            
            df["sample_name"] = sample_name
            if weight_branch:
                df = df.rename(columns={weight_branch: "weights"})
            else:
                df["weights"] = 1.0
            
            dict_datasets["Nominal"][sample_name] = df

        # 2. Load systematic variations (if requested)
        if load_systematics:
            systematics_list = self.config.get("Systematics", [])
            for syst_dict in systematics_list:
                syst_name = syst_dict["Name"]
                syst_type = syst_dict["Type"]

                if syst_type == "NormPlusShape":
                    for direction in ["Up", "Dn"]:
                        region_key = f"{syst_name}_{direction}"
                        dict_datasets[region_key] = {}

                        for sample_dict in syst_dict.get(direction, []):
                            sample_name = sample_dict["SampleName"]
                            file_path = sample_dict["Path"]
                            tree_name = sample_dict["Tree"]
                            
                            # Include weight branch if specified
                            weight_branch = sample_dict.get("Weight")
                            branches = self.branches_to_load.copy()
                            if weight_branch and weight_branch not in branches:
                                branches.append(weight_branch)

                            df = self._load_dataframe_from_root(file_path, tree_name, branches)
                            
                            df["sample_name"] = sample_name
                            if weight_branch:
                                df = df.rename(columns={weight_branch: "weights"})
                            else:
                                df["weights"] = 1.0
                            
                            dict_datasets[region_key][sample_name] = df

        return dict_datasets

    def _load_dataframe_from_root(
        self, 
        file_path: str, 
        tree_name: str, 
        branches: List[str]
    ) -> pd.DataFrame:
        """
        Load a TTree from a ROOT file into a pandas DataFrame.

        Args:
            file_path: Path to the ROOT file.
            tree_name: Name of the TTree to read.
            branches: List of branch names to load.

        Returns:
            DataFrame with the requested branches as columns.

        Raises:
            FileNotFoundError: If the ROOT file doesn't exist.
            KeyError: If the tree is not found in the file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ROOT file not found: {file_path}")

        try:
            with uproot.open(f"{file_path}:{tree_name}") as tree:
                try:
                    arrays = tree.arrays(branches, library="pd")
                    if isinstance(arrays, pd.DataFrame):
                        return arrays
                except (ValueError, TypeError):
                    pass  
                
                arrays_dict = tree.arrays(branches, library="np")
                return pd.DataFrame(arrays_dict)
                
        except uproot.exceptions.KeyInFileError as e:
            # Provide helpful error message with available trees
            with uproot.open(file_path) as f:
                available = [k.split(";")[0] for k in f.keys() if "TTree" in str(f.classname_of(k))]
            raise KeyError(
                f"Tree '{tree_name}' not found in {file_path}. "
                f"Available trees: {available}"
            ) from e
        except Exception as e:
            # If we can't convert to numpy (jagged branches), that's a data structure problem
            raise ValueError(
                f"Could not load tree '{tree_name}' from {file_path} as a flat DataFrame. "
                f"The tree may contain jagged (variable-length) branches. "
                f"Original error: {e}"
            ) from e

    def save_dataset_to_ntuple(
        self, 
        dict_datasets: Dict, 
        save_systematics: bool = False
    ) -> None:
        """
        Write DataFrames back to ROOT files, preserving other existing trees.

        Args:
            dict_datasets: Nested dict from load_datasets_from_config().
            save_systematics: If True, also save systematic variation samples.
        """
        # Ensure "weights" and "sample_name" are in branches_all
        # These columns are added during load, so they must be saved
        for col in ["weights", "sample_name"]:
            if col not in self.branches_all:
                self.branches_all.append(col)

        # 1. Save nominal samples
        self._save_region_datasets(
            dict_datasets["Nominal"], 
            self.config["Samples"]
        )

        # 2. Save systematic variations (if requested)
        if save_systematics:
            systematics_list = self.config.get("Systematics", [])
            for syst_dict in systematics_list:
                syst_name = syst_dict["Name"]
                syst_type = syst_dict["Type"]

                if syst_type == "NormPlusShape":
                    for direction in ["Up", "Dn"]:
                        region_key = f"{syst_name}_{direction}"
                        if region_key not in dict_datasets:
                            continue

                        sample_config = syst_dict.get(direction, [])
                        self._save_region_datasets(
                            dict_datasets[region_key], 
                            sample_config
                        )

    def _save_region_datasets(
        self, 
        region_data: Dict[str, pd.DataFrame], 
        sample_config_list: List[dict]
    ) -> None:
        """
        Save all samples in a region (nominal or a systematic variation).

        Groups samples by their target file path to avoid overwrites.

        Args:
            region_data: Dict mapping sample_name -> DataFrame.
            sample_config_list: List of sample config dicts with keys:
                - 'Name' or 'SampleName': process name
                - 'SamplePath' or 'Path': ROOT file path
                - 'Tree': tree name
        """
        # Group samples by file path
        file_to_trees: Dict[str, Dict[str, pd.DataFrame]] = defaultdict(dict)

        for sample_dict in sample_config_list:
            # Handle both Nominal config (Name, SamplePath) and Syst config (SampleName, Path)
            sample_name = sample_dict.get("Name") or sample_dict.get("SampleName")
            file_path = sample_dict.get("SamplePath") or sample_dict.get("Path")
            tree_name = sample_dict["Tree"]

            if sample_name not in region_data:
                continue

            # Filter to only the branches we want to save
            df = region_data[sample_name]
            available_branches = [b for b in self.branches_all if b in df.columns]
            df_filtered = df[available_branches]

            file_to_trees[file_path][tree_name] = df_filtered

        # Write each file atomically with all its trees
        for file_path, trees_dict in file_to_trees.items():
            self._write_file_with_trees(file_path, trees_dict)

    def _write_file_with_trees(
        self, 
        file_path: str, 
        new_trees_dict: Dict[str, pd.DataFrame]
    ) -> None:
        """
        Write/update a ROOT file with multiple trees in one atomic operation.

        Preserves any existing trees not in new_trees_dict, and overwrites
        trees whose names match keys in new_trees_dict.

        Args:
            file_path: Path to the ROOT file (created if it doesn't exist).
            new_trees_dict: Dict mapping tree_name -> DataFrame.
        """
        tmp_path = file_path + ".tmp"
        trees_to_write = {}

        # 1. If file exists, copy over all trees except those being replaced
        if os.path.exists(file_path):
            try:
                with uproot.open(file_path) as fin:
                    for key, classname in fin.classnames().items():
                        tree_name = key.split(";")[0]  # strip cycle number
                        if tree_name in new_trees_dict:
                            continue  # will be replaced by new version
                        if classname == "TTree":
                            trees_to_write[tree_name] = fin[tree_name].arrays(library="ak")
            except Exception as e:
                print(f"Warning: Could not read existing trees from {file_path}: {e}")
                # Continue anyway — we'll just write the new trees

        # 2. Add the new/updated trees
        trees_to_write.update(new_trees_dict)

        # 3. Write everything to a temp file, then atomically replace
        with uproot.recreate(tmp_path) as fout:
            for tree_name, data in trees_to_write.items():
                fout[tree_name] = data

        os.replace(tmp_path, file_path)
        print(f"✓ Saved {len(new_trees_dict)} tree(s) to {file_path}: {list(new_trees_dict.keys())}")


    def merge_dataframe_dict_for_training(self, 
                                        dataset_dict, 
                                        label_sample_dict: Union[dict[str, int], None] = None,
                                        samples_to_merge = []):
        """
        Concatenate selected samples; optionally add normalized weights + labels. 
        The returned sample is ready for training.
        Args:
            dataset_dict: dict[sample_name] -> DataFrame.
            label_sample_dict: Optional mapping of sample_name -> class id.
            samples_to_merge: List of sample names to include.
        Returns:
            pd.DataFrame: merged (and optionally labeled/normalized) dataset.
        Raises:
            Exception: If samples_to_merge is empty.
        """
        if len(samples_to_merge) == 0:
            raise Exception

        list_dataframes = []
        for sample_name, dataset in dataset_dict.items():
            if sample_name not in samples_to_merge: continue
            list_dataframes.append(dataset)

        dataset = pd.concat(list_dataframes)

        if label_sample_dict is not None:

            dataset = self._add_normalised_weights_and_train_label_class(dataset, 
                                                                        label_sample_dict)

        return dataset

    def _add_normalised_weights_and_train_label_class(self,
                                                    dataset, 
                                                    label_sample_dict: dict[str, int]):
        """
        Add per-class normalized weights and integer training labels.

        Process:
            - 'train_labels' set per sample_name using label_sample_dict.
            - 'weights_normed' scaled so each class sums to 1.0.

        Args:
            dataset: Input DataFrame with 'sample_name' and 'weights'.
            label_sample_dict: Mapping sample_name -> class id.
        Returns:
            pd.DataFrame with 'train_labels' and 'weights_normed' columns.
        """
        dataset['weights_normed']       = dataset['weights'].to_numpy()
        dataset['train_labels']         = -999

        for sample_name, label in label_sample_dict.items():

            mask_sample_name                                     = np.isin(dataset["sample_name"], [sample_name])

            dataset.loc[mask_sample_name, "train_labels"]        = label

        train_labels_unique = np.unique(dataset.train_labels)

        for train_label in train_labels_unique:

            mask_train_label                                     = np.isin(dataset["train_labels"], [train_label])

            total_train_weight                                   = dataset.loc[mask_train_label, "weights"].sum()

            dataset.loc[mask_train_label, "weights_normed"]      = dataset.loc[mask_train_label, "weights_normed"] / total_train_weight

        return dataset
    
    def prepare_basis_training_dataset(self, dataset_numerator, processes_numerator, dataset_denominator, processes_denominator):

        ref_train_label_sample_dict = {**{ref: 0 for ref in processes_denominator}}

        dataset_ref     = self.merge_dataframe_dict_for_training(dataset_denominator, 
                                                                  ref_train_label_sample_dict, 
                                                                  samples_to_merge = processes_denominator)
        
        numerator_train_label_sample_dict = {**{numerator: 1 for numerator in processes_numerator}}
        
        dataset_num = self.merge_dataframe_dict_for_training(dataset_numerator, 
                                                            numerator_train_label_sample_dict, 
                                                            samples_to_merge = processes_numerator)
        
        dataset_mix_model = pd.concat([dataset_num, dataset_ref])

        return dataset_mix_model


    def merge_dataframe_dict_for_training(self, 
                                        dataset_dict, 
                                        label_sample_dict: Union[dict[str, int], None] = None,
                                        samples_to_merge = []):
        """
        Concatenate selected samples; optionally add normalized weights + labels. 
        The returned sample is ready for training.
        Args:
            dataset_dict: dict[sample_name] -> DataFrame.
            label_sample_dict: Optional mapping of sample_name -> class id.
            samples_to_merge: List of sample names to include.
        Returns:
            pd.DataFrame: merged (and optionally labeled/normalized) dataset.
        Raises:
            Exception: If samples_to_merge is empty.
        """
        if len(samples_to_merge) == 0:
            raise Exception

        list_dataframes = []
        for sample_name, dataset in dataset_dict.items():
            if sample_name not in samples_to_merge: continue
            list_dataframes.append(dataset)

        dataset = pd.concat(list_dataframes)

        if label_sample_dict is not None:

            dataset = self._add_normalised_weights_and_train_label_class(dataset, 
                                                                        label_sample_dict)

        return dataset
    
    def _add_normalised_weights_and_train_label_class(self,
                                                    dataset, 
                                                    label_sample_dict: dict[str, int]):
        """
        Add per-class normalized weights and integer training labels.

        Process:
            - 'train_labels' set per sample_name using label_sample_dict.
            - 'weights_normed' scaled so each class sums to 1.0.

        Args:
            dataset: Input DataFrame with 'sample_name' and 'weights'.
            label_sample_dict: Mapping sample_name -> class id.
        Returns:
            pd.DataFrame with 'train_labels' and 'weights_normed' columns.
        """
        dataset['weights_normed']       = dataset['weights'].to_numpy()
        dataset['train_labels']         = -999

        for sample_name, label in label_sample_dict.items():

            mask_sample_name                                     = np.isin(dataset["sample_name"], [sample_name])

            dataset.loc[mask_sample_name, "train_labels"]        = label

            train_labels_unique = np.unique(dataset.train_labels)

        for train_label in train_labels_unique:

            mask_train_label                                     = np.isin(dataset["train_labels"], [train_label])

            total_train_weight                                   = dataset.loc[mask_train_label, "weights"].sum()

            dataset.loc[mask_train_label, "weights_normed"]      = dataset.loc[mask_train_label, "weights_normed"] / total_train_weight

        return dataset
    
    
    def prepare_basis_training_dataset(self, dataset_numerator, processes_numerator, dataset_denominator, processes_denominator):

        ref_train_label_sample_dict = {**{ref: 0 for ref in processes_denominator}}

        dataset_ref     = self.merge_dataframe_dict_for_training(dataset_denominator, 
                                                                  ref_train_label_sample_dict, 
                                                                  samples_to_merge = processes_denominator)
        
        numerator_train_label_sample_dict = {**{numerator: 1 for numerator in processes_numerator}}
        
        dataset_num = self.merge_dataframe_dict_for_training(dataset_numerator, 
                                                            numerator_train_label_sample_dict, 
                                                            samples_to_merge = processes_numerator)
        
        dataset_mix_model = pd.concat([dataset_num, dataset_ref])

        return dataset_mix_model


def save_dataframe_as_root(dataset        : pd.DataFrame,
                           path_to_save   : str,
                           tree_name      : str) -> None:
    """
    Utility: create/overwrite a ROOT file with a single TTree from a DataFrame.

    Args:
        dataset: DataFrame to serialize (all columns become branches).
        path_to_save: Target ROOT file path.
        tree_name: Name of the TTree to create.
    """
    with uproot.recreate(f"{path_to_save}") as ntuple:

        arrays = {col: dataset[col].to_numpy() for col in dataset.columns}

        ntuple[tree_name] = arrays
