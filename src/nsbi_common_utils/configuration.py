"""Provides utilities to handle the nsbi_common_utils configuration."""

import json
import logging
import pathlib
import pkgutil
from typing import Any, Literal, Optional, Union

import yaml


log = logging.getLogger(__name__)

class ConfigError(Exception):
    """Raised for config load/validate/write errors."""

class ConfigManager:

    def __init__(self, 
                file_path_string    : Union[str, pathlib.Path],
                initial_template    : Optional[dict[str, Any]] = None,
                create_if_missing   : bool = False):

        self.path       = pathlib.Path(file_path_string)
        self.config : dict[str, Any]

        if not self.path.exists():
            if create_if_missing:
                self.config         = initial_template or {"Regions": []}
                self.save()
                log.info(f"Created new config file at {self.path}")
            else:
                raise ConfigError(f"Config file does not exist at {self.path}")
        else:
            self.config             = self.load(self.path)

        self.validate(self.config)

    def get_sample_index_unbinned_regions(self,
                                          region_name: str,
                                          sample_name: str) -> Union[int, None]:

        idx = self._index_of_region(channel_name = region_name)

        if idx is None:
            log.info(f"Region {channel_name} not found in the config.")

        list_dict_trained_models: list[dict[str, Any]] = self.config["Regions"][idx]["TrainedModels"]
        for count, sample_model in enumerate(list_dict_trained_models):
            if sample_model.get("SampleName") == sample_name:
                return count
        return None

    def get_syst_index_unbinned_regions(self,
                                          region_name: str,
                                          sample_name: str,
                                         syst_name: str) -> Union[int, None]:

        idx = self._index_of_region(channel_name = region_name)

        if idx is None:
            log.info(f"Region {channel_name} not found in the config.")

        list_dict_trained_models: list[dict[str, Any]] = self.config["Regions"][idx]["TrainedModels"]

        idx_sample = self.get_sample_index_unbinned_regions(region_name, sample_name)

        list_dict_systematics: list[dict[str, Any]] = self.config["Regions"][idx]["TrainedModels"][idx_sample]["Systematics"]
        
        for count, syst_model in enumerate(list_dict_systematics):
            if syst_model.get("SystName") == syst_name:
                return count
        return None

        
    def get_training_features(self):
        return self.config['TrainingFeatures'], self.config['TrainingFeaturesToStandardize']

    def get_analysis_region_cuts(self):
        regions_list = list(self.config["Regions"])
        region_names            = []
        region_selections       = []
        for region_dict in regions_list:
            region_name         = region_dict["Name"]
            region_selection    = region_dict["Filter"]
            region_selections.append(region_selection)
            region_names.append(region_name)
        return region_names, region_selections
    
    def add_channel(
        self,
        channel_name: str,
        filter: str,
        observable: str,
        binning: Optional[list] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Adds (or overwrites) a "Regions" entry in the config file.

        Args:
            channel_name: new region name (must be unique unless overwrite=True)
            filter: string with filter conditions to be applied. e.g "jet_pt >= 200.0"
            observable: name of the obervable to bin in the region. None if unbinned SBI used in the region.
            binning: list with bin boundaries. None if unbinned SBI used in the region
            overwrite: allow replacing an existing region of same name
        """
        regions: list[dict[str, Any]] = self.config["Regions"]

        existing_idx = self._index_of_region(channel_name)
        new_region = {
            "Name": channel_name,
            "Filter": preselections,
            "Variable": observable,
            "Binning": binning,
        }

        if existing_idx is not None:
            if not overwrite:
                raise ConfigError(f"Region '{channel_name}' already exists. Use overwrite=True.")
            regions[existing_idx] = new_region
        else:
            regions.append(new_region)

        self.validate(self.config)

    def remove_channel(self, channel_name: str) -> bool:
        """Remove a specific region from the config, using channel name"""
        idx = self._index_of_region(channel_name)
        if idx is None:
            log.info(f"Region {channel_name} not found in the config. Nothing to do.")
            return False
        del self.config["Regions"][idx]
        self.validate(self.config)
        return True

    def list_channels(self) -> list[str]:
        """list region names in the config file."""
        return [region.get("Name") for region in self.config.get("Regions", [])]

    def load(self, file_path_string) -> dict[str, Any]:
        """Loads, validates, and returns a config file from the provided path.

        Args:
            file_path_string (Union[str, pathlib.Path]): path to config file

        Returns:
            dict[str, Any]: nsbi_common_utils configuration
        """
        file_path = pathlib.Path(file_path_string)
        log.info(f"Opening config file {file_path}")
        try:
            config = yaml.safe_load(file_path.read_text())
        except Exception as e:
            raise ConfigError(f"YAML parse error: {e}") from e
        config = yaml.safe_load(file_path.read_text())
        self.validate(config)
        return config

    def save(self) -> None:
        """Atomically writes current config to `self.path`."""
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            serialized = yaml.safe_dump(self.config, sort_keys=False, allow_unicode=True)
            tmp_path.write_text(serialized, encoding="utf-8")
            tmp_path.replace(self.path) 
            log.info("Wrote config to %s", self.path)
        except Exception as e:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            raise ConfigError(f"Failed to save config: {e}") from e

    def validate(self, config: dict[str, Any]) -> bool:
        """Returns True if the config file is validated, otherwise raises exceptions.

        Checks that the config satisfies the json schema, and performs additional checks to
        validate the config further.

        Args:
            config (dict[str, Any]): nsbi_common_utils configuration

        Raises:
            NotImplementedError: when more than one data sample is found
            ValueError: when region / sample / normfactor / systematic names are not unique

        Returns:
            bool: whether the validation was successful
        """

        # TBA
        
        # if no issues are found
        return True

    def get_basis_samples(self):
        basis_samples = []
        samples: list[dict[str, Any]] = self.config["Samples"]
        for count, sample in enumerate(samples):
            if sample.get("UseAsBasis", False):
                basis_samples.append(sample.get("Name"))

        return basis_samples

    def get_reference_samples(self):
        reference_samples = []
        samples: list[dict[str, Any]] = self.config["Samples"]
        for count, sample in enumerate(samples):
            if sample.get("UseAsReference", False):
                reference_samples.append(sample.get("Name"))

        return reference_samples

    def get_all_samples(self):
        basis_samples = []
        samples: list[dict[str, Any]] = self.config["Samples"]
        for count, sample in enumerate(samples):
            basis_samples.append(sample.get("Name"))
        return basis_samples

    def get_samples_in_syst_for_training(self, systematic_name: str, direction: str) -> list:
        systematics_list: list[dict[str, Any]] = self.config["Systematics"]
        for systematics_dict in systematics_list:
            if (systematics_dict["Type"] != "NormPlusShape") or (systematics_dict["Name"] != systematic_name):
                continue
            else:
                dir_list_dict : list[dict[str, Any]] = systematics_dict.get(direction, [])
                if len(dir_list_dict) == 0:
                    return []
                sample_list = []
                for sample_dict in dir_list_dict:
                    sample_name = sample_dict["SampleName"]
                    sample_list.append(sample_name)
                return sample_list

        
    def get_channel_filters(self, channel_name: str) -> str:
        
        idx = self._index_of_region(channel_name = channel_name)

        if idx is None:
            log.info(f"Region {channel_name} not found in the config.")

        filter_string = self.config["Regions"][idx]["Filter"]

        return filter_string
    
    def get_channel_asimov_weight_path(self, channel_name: str) -> str:
        
        idx = self._index_of_region(channel_name = channel_name)

        if idx is None:
            log.info(f"Region {channel_name} not found in the config.")

        asimov_weight_path = self.config["Regions"][idx]["AsimovWeights"]

        return asimov_weight_path

    def _index_of_region(self, channel_name: str) -> Optional[int]:
        regions: list[dict[str, Any]] = self.config["Regions"]
        for count, region in enumerate(regions):
            if region.get("Name") == channel_name:
                return count
        return None

    


