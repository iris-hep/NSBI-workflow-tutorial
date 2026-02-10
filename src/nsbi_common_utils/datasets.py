import os
import pandas as pd
import numpy as np
import uproot
import copy
import pathlib
import shutil
import tempfile
import awkward as ak
from typing import Any, Dict, List, Literal, Optional, Union

from nsbi_common_utils.configuration import ConfigManager

class datasets:

    """Lightweight helper for reading ROOT TTrees into pandas DataFrames (via uproot),
    applying region filters from a config, merging/labeling for ML training, and
    writing updated trees back to ROOT files."""

    def __init__(self, 
                config_path: Union[pathlib.Path, str], 
                branches_to_load: List):
        """Load analysis config and set the base list of branches to read.

        Args:
            config_path: Path to a YAML/JSON config consumed by ConfigManager.
            branches_to_load: Required list of TTree branches to import.
        Raises:
            Exception: If branches_to_load is empty.
        """
        self.config              = ConfigManager(file_path_string = config_path)
        
        if len(branches_to_load) == 0:
            raise Exception(f"Empty branch list.")
        self.branches_to_load           = list(branches_to_load)
        self.branches_all               = list(self.branches_to_load)

    def load_datasets_from_config(self,
                                load_systematics = False):
        """Read datasets defined in config into nested dictionaries of DataFrames.

        Structure:
            {
              "Nominal": {sample_name: pd.DataFrame, ...},
              "<Syst>_Up": {...}, "<Syst>_Dn": {...}  # if requested
            }

        Notes:
            - Adds a 'sample_name' column and ensures a 'weights' column
              (renaming per config 'Weight' when present, else defaults to 1.0).
        Args:
            load_systematics: If True, also load 'NormPlusShape' systematics.
        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: datasets by type (nominal vs systematics) then sample.
        """
        dict_datasets = {}
        dict_datasets["Nominal"] = {}

        for dict_sample in self.config.config["Samples"]:

            weight_branch       = [dict_sample["Weight"]] if "Weight" in dict_sample.keys() else []

            path_to_root_file   = dict_sample["SamplePath"]
            tree_name           = dict_sample["Tree"]
            sample_name         = dict_sample["Name"]
            branches_to_load    = list(self.branches_to_load)
            if weight_branch[0] not in branches_to_load:
                branches_to_load += weight_branch
                
            dict_datasets["Nominal"][sample_name] = load_dataframe_from_root(path_to_root_file, 
                                                                            tree_name, 
                                                                            branches_to_load)

            dict_datasets["Nominal"][sample_name]["sample_name"] = sample_name

            if "Weight" in dict_sample.keys():
                dict_datasets["Nominal"][sample_name] = dict_datasets["Nominal"][sample_name].rename(columns={dict_sample['Weight']: "weights"})
            else:
                dict_datasets["Nominal"][sample_name]["weights"] = 1.0

        if load_systematics:
            systematics_dict_list = self.config.config.get("Systematics", [{}])
            for dict_syst in systematics_dict_list:
                syst_name = dict_syst["Name"]
                syst_type = dict_syst["Type"]
                if syst_type == "NormPlusShape":
                    for direction in ["Up", "Dn"]:
                        syst_name_var        = syst_name + "_" + direction
                        dict_datasets[syst_name_var] = {}
                        for dict_sample in dict_syst[direction]:
                            path_to_root_file   = dict_sample["Path"]
                            sample_name         = dict_sample["SampleName"]
                            tree_name           = dict_sample["Tree"]
                            weight_branch       = [dict_sample["Weight"]] if "Weight" in dict_sample.keys() else []
                            branches_to_load    = list(self.branches_to_load)
                            if weight_branch[0] not in branches_to_load:
                                branches_to_load += weight_branch
                            dict_datasets[syst_name_var][sample_name] = load_dataframe_from_root(path_to_root_file, 
                                                                                                tree_name, 
                                                                                                branches_to_load)

                            dict_datasets[syst_name_var][sample_name]["sample_name"] = sample_name

                            if "Weight" in dict_sample.keys():
                                dict_datasets[syst_name_var][sample_name] = dict_datasets[syst_name_var][sample_name].rename(columns={dict_sample['Weight']: "weights"})
                            else:
                                dict_datasets[syst_name_var][sample_name]["weights"] = 1.0

        return dict_datasets

    def add_appended_branches(self, 
                              branches: List):
        """
        Declare additional, derived branches to carry through on save.

        Args:
            branches: New branch names to append to the saved schema.
        """
        self.branches_all           = self.branches_to_load + branches

    def save_datasets(self,
                    dict_datasets,
                    save_systematics = False):
        """
        Write DataFrames back into their ROOT files, preserving other TTrees.

        Args:
            dict_datasets: Nested dict from load_datasets_from_config().
            save_systematics: If True, also write available syst variations.
        """
        for dict_sample in self.config.config["Samples"]:

            path_to_root_file   = dict_sample["SamplePath"]
            tree_name           = dict_sample["Tree"]
            sample_name         = dict_sample["Name"]
            self._save_dataset_to_ntuple(dict_datasets["Nominal"][sample_name], 
                                path_to_root_file, 
                                tree_name)

        if save_systematics:
            systematics_dict_list = self.config.config.get("Systematics", [{}])
            for dict_syst in systematics_dict_list:

                syst_name = dict_syst["Name"]
                syst_type = dict_syst["Type"]
                if syst_type == "NormPlusShape":
                    for direction in ["Up", "Dn"]:
                        syst_name_var        = syst_name + "_" + direction
                        if syst_name_var not in dict_datasets.keys(): continue
                        for dict_sample in dict_syst[direction]:
                            path_to_root_file   = dict_sample["Path"]
                            sample_name         = dict_sample["SampleName"]

                            if sample_name not in dict_datasets[syst_name_var].keys(): continue

                            tree_name           = dict_sample["Tree"]
                            self._save_dataset_to_ntuple(dict_datasets[syst_name_var][sample_name],
                                                        path_to_root_file, 
                                                        tree_name)


    def _save_dataset_to_ntuple(self, dataset, path_to_root_file: str, tree_name: str):
        """
        Mofify a specific TTree with DataFrame contents.

        Behavior:
            - Keeps other trees intact by copying them over.
            - Ensures 'weights' exists in the saved branch list.

        Args:
            dataset: DataFrame to write (columns = branches).
            path_to_root_file: Destination ROOT file.
            tree_name: Name of the tree to overwrite.
        """
        if "weights" not in dataset.columns:
            dataset = dataset.assign(weights=1.0)

        new_cols = {col: np.asarray(dataset[col]) for col in dataset.columns}

        # Temp file to avoid curroption
        directory = os.path.dirname(os.path.abspath(path_to_root_file))
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".root")
        os.close(fd)

        try:
            with uproot.open(path_to_root_file) as fin, uproot.recreate(tmp_path) as fout:
                # Copy other TTrees (as you already do)
                for key, classname in fin.classnames().items():
                    k = key.split(";")[0]
                    if k == tree_name:
                        continue
                    if classname == "TTree":
                        fout[k] = fin[k].arrays(library="ak")

                # --- Replace the target tree, but preserve its OTHER BRANCHES ---
                old_tree = fin[tree_name]
                old_arrays = old_tree.arrays(library="ak")  # all branches in that tree

                # Validate event count didn't change (important when merging)
                n_old = old_tree.num_entries
                if len(dataset) != n_old:
                    raise ValueError(
                        f"Refusing to overwrite {path_to_root_file}:{tree_name} "
                        f"because row count changed ({len(dataset)} vs {n_old}). "
                        "Write to a new file/tree instead."
                    )

                merged = {name: old_arrays[name] for name in old_arrays.fields}
                merged.update(new_cols)  # overwrite/add modified columns

                # Write merged tree
                fout[tree_name] = merged

            # Validate tmp file before replacing
            with uproot.open(tmp_path) as chk:
                if tree_name not in chk:
                    raise RuntimeError("Write failed: tree missing in tmp output.")
                if chk[tree_name].num_entries != len(dataset):
                    raise RuntimeError("Write failed: entry count mismatch in tmp output.")

            # Atomic replace
            os.replace(tmp_path, path_to_root_file)

        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            finally:
                raise

    def _save_dataset_to_ntuple(self,
                                dataset, 
                                path_to_root_file: str, 
                                tree_name: str):
        """
        Replace a specific TTree with DataFrame contents.

        Behavior:
            - Keeps other trees intact by copying them over.
            - Ensures 'weights' exists in the saved branch list.

        Args:
            dataset: DataFrame to write (columns = branches).
            path_to_root_file: Destination ROOT file.
            tree_name: Name of the tree to overwrite.
        """
        if "weights" not in self.branches_all:
            self.branches_all =  self.branches_all + ["weights"]
        dataset = dataset[self.branches_all]

        tmp_path = path_to_root_file + ".tmp"

        with uproot.open(path_to_root_file) as fin, uproot.recreate(tmp_path) as fout:
            for _tree_name, classname in fin.classnames().items():
                _tree_name = _tree_name.split(";")[0]
                if _tree_name == tree_name:
                    continue
                if classname == "TTree":
                    arrs = fin[_tree_name].arrays(library="ak")
                    fout[_tree_name] = arrs

            fout[tree_name] = dataset

        os.replace(tmp_path, path_to_root_file)

    def filter_region_by_type(self,
                             dataset: Dict[str, Dict[str, pd.DataFrame]],
                             region: str) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Apply region filters

        Args:
            dataset: dict[type_name][sample_name] -> DataFrame.
            region: Config channel name to query against.
        Returns:
            Same structure with rows filtered per region expression.
        """
        for type_name, type_dict in dataset.items():
            dataset[type_name] = self.filter_region_dataset(type_dict, region = region)

        return dataset

    def filter_region_dataset(self,
                              dataset: Dict[str, pd.DataFrame],
                              region: str) -> Dict[str, pd.DataFrame]:
        """
        Apply config-defined query string to each sample's DataFrame.

        Args:
            dataset: dict[sample_name] -> DataFrame.
            region: Config channel name to look up filters for.
        Returns:
            New dict with filtered DataFrames (copy).
        """
        region_filters = self.config.get_channel_filters(channel_name = region)
        for sample_name, sample_dataframe in dataset.items():
            dataset[sample_name] = sample_dataframe.query(region_filters).copy()
        return dataset

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
    

def load_dataframe_from_root(path_to_load: str,
                             tree_name: str,
                             branches_to_load: list = None) -> pd.DataFrame:
    """
    Utility: read selected branches from a ROOT TTree into a DataFrame.

    Args:
        path_to_load: Source ROOT file path.
        tree_name: TTree name inside the file.
        branches_to_load: Branch names to read. If empty or None, load all branches.
    Returns:
        pd.DataFrame containing the requested branches.
    """
    with uproot.open(f"{path_to_load}:{tree_name}") as tree:
        # If no branches are specified, load all
        if not branches_to_load:
            branches_to_load = tree.keys()
        dataframe = tree.arrays(branches_to_load, library="pd")

    return dataframe

def load_dataframe_from_root(path_to_load: str,
                             tree_name: str,
                             branches_to_load: list = None) -> pd.DataFrame:
    """
    Utility: read selected branches from a ROOT TTree into a DataFrame.

    Args:
        path_to_load: Source ROOT file path.
        tree_name: TTree name inside the file.
        branches_to_load: Branch names to read. If empty or None, load all branches.
    Returns:
        pd.DataFrame containing the requested branches.
    """
    with uproot.open(f"{path_to_load}:{tree_name}") as tree:
        if not branches_to_load:
            branches_to_load = tree.keys()

        out = tree.arrays(branches_to_load, library="pd")

    # Case A: uproot returned multiple DataFrames (unrelated jagged collections)
    if isinstance(out, tuple):
        # Pick the *event-level* frame if present; otherwise raise a clear error.
        event_level = [df for df in out if getattr(df.index, "nlevels", 1) == 1]
        if len(event_level) == 1:
            return event_level[0]

        raise TypeError(
            "uproot returned multiple DataFrames (tuple). "
            "This usually means you requested jagged branches with different multiplicities. "
            "Either: (1) drop/aggregate jagged branches to event-level, or (2) switch to Awkward workflow."
        )

    # Case B: already a DataFrame
    if isinstance(out, pd.DataFrame):
        return out

    # Case C: awkward array/record returned -> convert or raise
    if isinstance(out, (ak.Array, ak.Record)):
        # This yields a DataFrame with a (possibly multi-)index.
        return ak.to_dataframe(out)

    raise TypeError(f"Unexpected type from uproot: {type(out)}")



