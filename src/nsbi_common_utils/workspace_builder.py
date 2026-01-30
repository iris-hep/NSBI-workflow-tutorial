import json, math
import numpy as np
import pandas as pd
from collections.abc import Callable as CABC
import pathlib
from typing import Union, Dict, Any, List
import nsbi_common_utils
from nsbi_common_utils import configuration, datasets
import logging
logging.basicConfig(filename="workspace.log",
                    encoding="utf-8",
                    filemode="w",
                    format="{asctime} - {levelname} - {message}",
                    style="{",
                    datefmt="%Y-%m-%d %H:%M",
                    level=logging.WARNING,
                    )


class WorkspaceBuilder:
    """Collects functionality to build a workspace"""

    def __init__(self, config_path: Union[pathlib.Path, str]) -> None:
        """Creates a workspace corresponding to configuration file"""
        self.config_path = config_path
        self.config = nsbi_common_utils.configuration.ConfigManager(file_path_string = config_path)
        self.config_dict = self.config.config
        self.ParametersToFit = self.config_dict["General"]["Measurement"].get("ParametersToFit", None)
        self._check_ParametersToFit()

    def _check_ParametersToFit(self):
        poi = self.config_dict["General"]["Measurement"].get("POI", "")
        if (self.ParametersToFit) and (poi not in self.ParametersToFit):
            logging.warning(f'The POI {poi} is not included in the ParametersToFit. Adding the POI {poi} to ParametersToFit list.')
            self.ParametersToFit.append(poi)
        elif self.ParametersToFit is None:
            logging.warning('No ParametersToFit specified in config. All parameters will be included in the fitting.')
        


    def normfactor_modifiers(self, 
                             region_name: str, 
                             sample_name: str) -> list[dict[str, Any]]:
        '''
        returns the modifier list of all normfactors affecting a sample in a region
        '''
        list_dict_norm_factors = self.config.config.get("NormFactors", [])
        modifiers = []
        for norm_factor_dict in list_dict_norm_factors:
            norm_factor_name = norm_factor_dict["Name"]
            norm_factor_data = norm_factor_dict.get("Data", None)
            regions_affected = norm_factor_dict.get("Region", None)
            if regions_affected is not None:
                if region_name not in regions_affected:
                    continue
            samples_affected = norm_factor_dict.get("Samples", None)
            if samples_affected is not None:
                if sample_name not in samples_affected:
                    continue
                else:
                    modifiers.append({"name": norm_factor_name, 
                                      "data": norm_factor_data, 
                                      "type": "normfactor"})
        return modifiers

    def normplusshape_modifiers(self, 
                                dataset           : pd.DataFrame, 
                                region            : dict[str, Any], 
                                sample            : dict[str, Any], 
                                systematic_dict   : dict[str, Any],
                                nominal_data      : np.array,
                                type_of_fit: str) -> list[dict[str, Any]]:

        syst_name                      = systematic_dict["Name"]
        
        channel_name    = region["Name"]
        sample_name     = sample["Name"]
        sample_path     = sample["SamplePath"]
        region_variable = region["Variable"]
        region_binning  = region["Binning"]

        variation_data = {}

        for direction in ["Up", "Dn"]:

            key_syst            = syst_name + '_' + direction
            
            weights             = dataset[key_syst][sample_name]["weights"].to_numpy()
            
            feature_var         = np.clip(dataset[key_syst][sample_name][region_variable],
                                                          np.amin(region_binning), np.amax(region_binning))
            
            syst_var_data, _       = np.histogram(feature_var, weights = weights, bins = region_binning)

            variation_data[direction] = syst_var_data / nominal_data

        if type_of_fit == "binned":
        
            modifiers = [{"name": syst_name,
                          "type": "normplusshape",
                          "data": {"hi_data": list(variation_data["Up"]),
                                   "lo_data": list(variation_data["Dn"])}}]
        elif type_of_fit == "unbinned":

            idx      = self.config.get_sample_index_unbinned_regions(channel_name, sample_name)
            syst_idx = self.config.get_syst_index_unbinned_regions(channel_name, sample_name, syst_name)

            variation_ratio_up         = region["TrainedModels"][idx]["Systematics"][syst_idx].get("RatiosUp", None)
            variation_ratio_dn         = region["TrainedModels"][idx]["Systematics"][syst_idx].get("RatiosDn", None)
            modifiers = [{"name": syst_name,
                          "type": "normplusshape",
                          "data": {"hi_data": list(variation_data["Up"]),
                                   "hi_ratio": variation_ratio_up,
                                   "lo_data": list(variation_data["Dn"]),
                                   "lo_ratio": variation_ratio_dn}
                         }]

        return modifiers

    def sys_modifiers(self, dataset: pd.DataFrame, 
                      region: dict[str, Any], 
                      sample: dict[str, Any],
                      nominal_data: np.array, 
                      type_of_fit: str = "binned") -> list[dict[str, Any]]:

        sample_name = sample["Name"]
        modifiers = []
        for systematic_dict in self.config_dict.get("Systematics", []):

            regions_affected = systematic_dict.get("Regions", None)
            if regions_affected is not None:
                if region_name not in regions_affected:
                    continue
            samples_affected = systematic_dict.get("Samples", None)
            if samples_affected is not None:
                if sample_name not in samples_affected:
                    continue
                else:
                    if systematic_dict["Type"] == "NormPlusShape":
                        modifiers += self.normplusshape_modifiers(
                            dataset, region, sample, systematic_dict, nominal_data, type_of_fit
                        )
                    else:
                        raise NotImplementedError(
                            "not supporting other systematic types yet"
                        )
        return modifiers
        

    def channels(self) -> List[Dict[str, Any]]:
        """Returns the channel information: yields/density ratio models per sample and modifiers.

        Returns:
            List[Dict[str, Any]]: channels for workspace
        """
        channels = []
        for region in self.config_dict["Regions"]:
            channel = {}
            channel_name = region["Name"]
            channel_type = region["Type"]
            channel.update({"name": channel_name,
                            "type": channel_type})
            type_of_fit  = channel_type

            if type_of_fit == "unbinned":
                region_weights: str = region.get("AsimovWeights", None)
                if region_weights is not None:
                    channel.update({"weights" : region_weights})
                
            region_binning      = region.get("Binning", None)
            region_variable     = region.get("Variable", None)
                
            region_filters      = region["Filter"]
            if region_variable is None:
                region_variable = self.config.get_training_features()[0][0] # Bin any random variable in a single bin for total event yield calculation
                region["Variable"] =  region_variable
                
            branches_to_load    = [region_variable] 
            if region_variable != 'presel_score':
                branches_to_load += ['presel_score'] # Hard coded for now - TODO
                
            samples = []
            for sample_dict in self.config_dict["Samples"]:

                current_sample = {}
                
                sample_name     = sample_dict["Name"]
                current_sample.update({"name": sample_name})
                
                sample_path     = sample_dict["SamplePath"]
                branches_to_load_sample  = branches_to_load.copy()

                datasets            = nsbi_common_utils.datasets.datasets(self.config_path,
                                                                    branches_to_load =  branches_to_load_sample)
                datasets_incl       = datasets.load_datasets_from_config(load_systematics = True)
                dataset_region_dict = datasets.filter_region_by_type(datasets_incl, 
                                                                     region = channel_name)

                dataset_nominal_sample = dataset_region_dict["Nominal"][sample_name].copy()
                
                if region_binning is None:
                    feature_arr_tmp = dataset_nominal_sample[region_variable]
                    region_binning = np.linspace(np.amin(feature_arr_tmp), np.amax(feature_arr_tmp), num=2) # Dummy binning for a single event yield calculation in unbinned region
                    region["Binning"] =  region_binning
                    
                feature_var         = np.clip(dataset_nominal_sample[region_variable],
                                              np.amin(region_binning), np.amax(region_binning))
                
                weights = dataset_region_dict["Nominal"][sample_name]["weights"].to_numpy()
                    
                sample_data, _       = np.histogram(feature_var, weights = weights, bins = region_binning)

                current_sample.update({"data": list(sample_data)})

                if type_of_fit == "unbinned":

                    idx = self.config.get_sample_index_unbinned_regions(channel_name, sample_name)
                    
                    nominal_ratios         = region["TrainedModels"][idx]["Nominal"].get("Ratios", None)
                    if nominal_ratios is None:
                        # Load the model and evaluate ratios - TODO
                        nominal_model         = region["TrainedModels"][idx]["Nominal"].get("Models", None)

                    current_sample.update({"ratios": nominal_ratios})
                    # current_sample.update({"weights": weights})

                modifiers = []

                # modifiers can have region and sample dependence, which is checked
                # check if normfactors affect sample in region, add modifiers as needed
                nf_modifier_list = self.normfactor_modifiers(channel_name, sample_name)

                modifiers += nf_modifier_list

                # check if systematics affect sample in region, add modifiers as needed
                sys_modifier_list = self.sys_modifiers(dataset_region_dict, region, sample_dict, sample_data, type_of_fit = type_of_fit)
                modifiers += sys_modifier_list

                current_sample.update({"modifiers": modifiers})  

                samples.append(current_sample)
                    
                
            channel.update({"samples": samples})
            channels.append(channel)

            
        return channels

    def measurements(self):
        
        measurements = []
        measurement = {}
        measurement.update({"name": self.config_dict["General"]["Measurement"]['Name']})
        config_dict = {}

        # get the norm factor initial values / bounds / constant setting
        parameters_list = []
        for nf in self.config_dict.get("NormFactors", []):
            nf_name = nf["Name"]  # every NormFactor has a name
            init = nf.get("Nominal", None)
            bounds = nf.get("Bounds", None)

            parameter = {"name": nf_name}
            if init is not None:
                parameter.update({"inits": [init]})
            if bounds is not None:
                parameter.update({"bounds": [bounds]})

            parameters_list.append(parameter)

        for sys in self.config_dict.get("Systematics", []):
            sys_name = sys["Name"]
            init = sys.get("Nominal", None)
            bounds = sys.get("Bounds", None)

            parameter = {"name": sys_name}
            if init is not None:
                parameter.update({"inits": [init]})
            if bounds is not None:
                parameter.update({"bounds": [bounds]})
            parameters_list.append(parameter)
        
        if self.ParametersToFit:
            parameters_list = [p for p in parameters_list if p["name"] in self.ParametersToFit]

                
        parameters = {"parameters": parameters_list}
        config_dict.update(parameters)
        config_dict.update({"poi": self.config_dict["General"]["Measurement"].get("POI", "")})
        measurement.update({"config": config_dict})
        measurements.append(measurement)
        return measurements


    def build(self) -> Dict[str, Any]:
        """
        Constructs a workspace.

        Returns:
            Dict[str, Any]
        """
        ws: Dict[str, Any] = {}  # the workspace

        # channels
        channels = self.channels()
        ws.update({"channels": channels})

        # measurements
        measurements = self.measurements()
        ws.update({"measurements": measurements})

        # # build observations
        # observations = self.observations()
        # ws.update({"observations": observations})

        # workspace version
        ws.update({"version": "1.0.0"})

        return ws


    def dump_workspace(self, ws: dict, outpath: str = "workspace.json"):
        with open(outpath, "w") as f:
            json.dump(ws, f, indent=2)
        print(f"Wrote {outpath}")
