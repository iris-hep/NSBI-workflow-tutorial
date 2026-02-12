"""
Drop-in replacement for nsbi_common_utils.datasets.datasets class.

Key fix: _save_dataset_to_ntuple replaced with batched file writing to prevent
tree overwrites when multiple samples target the same ROOT file.
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional
import uproot
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

            dict_datasets["Nominal"][sample_name] = self._load_dataframe_from_root(
                file_path, tree_name, self.branches_to_load
            )

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

                            dict_datasets[region_key][sample_name] = self._load_dataframe_from_root(
                                file_path, tree_name, self.branches_to_load
                            )

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
                # First, try library="pd" which should work for flat branches
                try:
                    arrays = tree.arrays(branches, library="pd")
                    # Verify it's actually a DataFrame (not Awkward masquerading as one)
                    if isinstance(arrays, pd.DataFrame):
                        return arrays
                except (ValueError, TypeError):
                    pass  # Fall through to numpy conversion
                
                # If library="pd" failed or returned Awkward, use numpy conversion
                # This works for flat (non-jagged) branches and explicitly constructs a DataFrame
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

    def save_datasets(
        self, 
        dict_datasets: Dict, 
        save_systematics: bool = False
    ) -> None:
        """
        Write DataFrames back to ROOT files, preserving other existing trees.

        BUG FIX: The original implementation called _save_dataset_to_ntuple()
        once per sample, which caused each invocation to overwrite trees saved
        by the previous call (because uproot.open() reads the original file state,
        not the in-progress updated one).

        The fix: batch all samples destined for the same file, then write them
        all in one atomic operation.

        Args:
            dict_datasets: Nested dict from load_datasets_from_config().
            save_systematics: If True, also save systematic variation samples.
        """
        # Ensure 'weights' is in the branch list
        if "weights" not in self.branches_all:
            self.branches_all.append("weights")

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

    # -------------------------------------------------------------------------
    # Legacy method for backwards compatibility (now deprecated in favor of
    # the batched _save_region_datasets approach)
    # -------------------------------------------------------------------------

    def _save_dataset_to_ntuple(
        self, 
        dataset: pd.DataFrame, 
        path_to_root_file: str, 
        tree_name: str
    ) -> None:
        """
        DEPRECATED: This method has been replaced by _write_file_with_trees.
        
        Left here for backwards compatibility, but calls to this should be
        replaced with the batched save logic in save_datasets().

        Args:
            dataset: DataFrame to write.
            path_to_root_file: ROOT file path.
            tree_name: Tree name to write/overwrite.
        """
        print(
            f"Warning: _save_dataset_to_ntuple is deprecated. "
            f"Use save_datasets() instead for proper multi-tree handling."
        )
        # Filter to available branches
        available_branches = [b for b in self.branches_all if b in dataset.columns]
        df_filtered = dataset[available_branches]

        # Write using the fixed logic
        self._write_file_with_trees(path_to_root_file, {tree_name: df_filtered})