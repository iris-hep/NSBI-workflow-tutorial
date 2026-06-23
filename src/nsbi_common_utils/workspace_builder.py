import json, math, re
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
    """Build a pyhf-like JSON workspace from a YAML configuration file.

    Reads sample definitions, region definitions, normalization factors, and systematic uncertainties from a YAML config (via :class:`~nsbi_common_utils.configuration.ConfigManager`), loads the corresponding ROOT datasets and trained density-ratio models, and assembles them into a workspace dictionary that can be consumed by :class:`~nsbi_common_utils.models.sbi_parametric_model`.

    Parameters
    ----------
    config_path : pathlib.Path or str
        Path to the YAML configuration file that defines samples, regions, normalization factors, and systematics. See :doc:`/basics/fit_config` for the expected format.

    See Also
    --------
    nsbi_common_utils.models.sbi_parametric_model : Consumes the workspace dictionary produced by :meth:`build`.

    Examples
    --------
    >>> builder = WorkspaceBuilder("config_fit_nsbi.yml")
    >>> ws = builder.build()
    >>> builder.dump_workspace(ws, "workspace.json")
    """

    def __init__(self, config_path: Union[pathlib.Path, str]) -> None:
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
            
    def vandermonde_modifiers(self, region_name: str, sample_name: str) -> list[dict[str, Any]]:
        """Return normfactor modifiers that affect a given sample in a region.

        Iterates over all ``VandermondeFactors`` in the configuration and keeps only those whose ``Region`` and ``Samples`` lists include the requested region/sample (or are unset, meaning they apply everywhere).

        Parameters
        ----------
        region_name : str
            Name of the region (channel) to filter on.
        sample_name : str
            Name of the physics sample to filter on.

        Returns
        -------
        list of dict
            Each dict has keys ``"name"``, ``"data"``, and
            ``"type": "vandermonde"``.
        """
        list_dict_vandermondes = self.config.config.get("VandermondeFactors", [])
        modifiers = []
        for vandermonde_dict in list_dict_vandermondes:
            vandermonde_name = vandermonde_dict["Name"]
            vandermonde_data = vandermonde_dict.get("Data", None)
            vandermonde_basis = vandermonde_dict.get("Basis", None)
            vandermonde_coeffs = compute_vandermonde_coeffs(vandermonde_basis)

            regions_affected = vandermonde_dict.get("Region", None)
            if regions_affected is not None:
                if region_name not in regions_affected:
                    continue
            samples_affected = vandermonde_dict.get("Samples", None)
            if samples_affected is not None:
                if sample_name not in samples_affected:
                    continue
                else:
                    modifiers.append(
                        {
                            "name": vandermonde_name,
                            "data": vandermonde_data,
                            "coeff": vandermonde_coeffs[sample_name],
                            "type": "vandermonde",
                        }
                    )
        return modifiers

    def normfactor_modifiers(self,
                             region_name: str,
                             sample_name: str) -> list[dict[str, Any]]:
        """Return normfactor modifiers that affect a given sample in a region.

        Iterates over all ``NormFactors`` in the configuration and keeps only those whose ``Region`` and ``Samples`` lists include the requested region/sample (or are unset, meaning they apply everywhere).

        Parameters
        ----------
        region_name : str
            Name of the region (channel) to filter on.
        sample_name : str
            Name of the physics sample to filter on.

        Returns
        -------
        list of dict
            Each dict has keys ``"name"``, ``"data"``, and
            ``"type": "normfactor"``.
        """
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
        """Build a NormPlusShape modifier for one systematic on one sample.

        Histograms the up/down systematic variations, divides by the nominal to obtain variation ratios, and (for unbinned channels) attaches paths to the pre-computed density-ratio arrays.

        Parameters
        ----------
        dataset : dict of dict of DataFrame
            Nested dict keyed by ``"<syst>_Up"``/``"<syst>_Dn"`` then
            sample name, as returned by :meth:`datasets.filter_region_by_type`.
        region : dict
            Region (channel) configuration dictionary.  Must contain ``"Name"``, ``"Variable"``, and ``"Binning"`` keys.
        sample : dict
            Sample configuration dictionary with at least ``"Name"`` and ``"SamplePath"`` keys.
        systematic_dict : dict
            Single systematic entry from the YAML config (has ``"Name"``).
        nominal_data : np.ndarray
            Nominal histogram bin counts used to normalise the variations.
        type_of_fit : ``"binned"`` or ``"unbinned"``
            Determines whether density-ratio file paths are included.

        Returns
        -------
        list of dict
            A single-element list containing the modifier dictionary with keys ``"name"``, ``"type": "normplusshape"``, and ``"data"``.
        """
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
        """Collect all systematic modifiers for a sample in a region.

        Loops over every systematic in the configuration, checks region/sample applicability, and delegates to :meth:`normplusshape_modifiers` for ``NormPlusShape`` types.

        Parameters
        ----------
        dataset : dict
            Dataset dictionary as returned by
            :meth:`datasets.filter_region_by_type`.
        region : dict
            Region configuration dictionary.
        sample : dict
            Sample configuration dictionary.
        nominal_data : np.ndarray
            Nominal histogram used for ratio computation.
        type_of_fit : str, optional
            ``"binned"`` (default) or ``"unbinned"``.

        Returns
        -------
        list of dict
            Modifier dictionaries for all applicable systematics.

        Raises
        ------
        NotImplementedError
            If a systematic type other than ``NormPlusShape`` is encountered.
        """
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
        """Build the ``"channels"`` list for the workspace.

        For every region in the configuration, loads datasets from ROOT files, computes nominal histograms, attaches density-ratio file paths (for unbinned regions), and collects all applicable normfactor and systematic modifiers per sample.

        Returns
        -------
        list of dict
            Each dict represents one channel with keys ``"name"``, ``"type"`` (``"binned"``/``"unbinned"``), ``"samples"``, and optionally ``"weights"`` (path to Asimov weight file for unbinned channels).
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

            # Extract variable names used in the Filter expression
            # so the dataset loader reads the columns needed for df.query()
            filter_variables = [tok for tok in re.split(r'[<>=!&|()\s]+', region_filters)
                                if tok and not tok.replace('.','',1).lstrip('-').isdigit()]

            if region_variable is None:
                # For unbinned regions with no explicit Variable, use the first
                # filter variable for the dummy single-bin yield histogram
                region_variable = filter_variables[0]
                region["Variable"] = region_variable

            branches_to_load = [region_variable]
            for v in filter_variables:
                if v not in branches_to_load:
                    branches_to_load.append(v)

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
                        nominal_model = region["TrainedModels"][idx]["Nominal"].get("Models", None)

                    current_sample.update({"ratios": nominal_ratios})
                    # current_sample.update({"weights": weights})

                modifiers = []

                # modifiers can have region and sample dependence, which is checked
                # check if normfactors affect sample in region, add modifiers as needed
                nf_modifier_list = self.normfactor_modifiers(channel_name, sample_name)

                modifiers += nf_modifier_list

                # check if vandermonde factors affect sample in region, add modifiers as needed
                vandermonde_modifier_list = self.vandermonde_modifiers(
                    channel_name, sample_name
                )

                modifiers += vandermonde_modifier_list

                # check if systematics affect sample in region, add modifiers as needed
                sys_modifier_list = self.sys_modifiers(dataset_region_dict, region, sample_dict, sample_data, type_of_fit = type_of_fit)
                modifiers += sys_modifier_list

                current_sample.update({"modifiers": modifiers})

                samples.append(current_sample)


            channel.update({"samples": samples})
            channels.append(channel)


        return channels

    def measurements(self) -> List[Dict[str, Any]]:
        """Build the ``"measurements"`` list for the workspace.

        Extracts parameter initial values and bounds from the ``NormFactors`` ,``VandermondeFactors`` and ``Systematics`` sections of the config, filters to the ``ParametersToFit`` subset (if specified), and records the parameter of interest (POI).

        Returns
        -------
        list of dict
            A single-element list.  The dict contains ``"name"`` and ``"config"`` (with ``"parameters"`` and ``"poi"`` keys).
        """
        measurements = []
        measurement = {}
        measurement.update({"name": self.config_dict["General"]["Measurement"]["Name"]})
        errordef = self.config_dict["General"]["Measurement"].get("ErrorDef", "LIKELIHOOD")
        measurement.update({"errordef": errordef})
        config_dict = {}

        # get the norm factor initial values / bounds / constant setting
        parameters_list = []
        for lf in self.config_dict.get("VandermondeFactors", []):
            lf_name = lf["Name"]  # every VandermondeFactors has a name
            init = lf.get("Nominal", None)
            bounds = lf.get("Bounds", None)

            parameter = {"name": lf_name}
            if init is not None:
                parameter.update({"inits": [init]})
            if bounds is not None:
                parameter.update({"bounds": [bounds]})

            parameters_list.append(parameter)

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
        config_dict.update(
            {"poi": self.config_dict["General"]["Measurement"].get("POI", "")}
        )
        config_dict.update(
            {"errordef" : self.config_dict["General"]["Measurement"].get("ErrorDef", "LIKELIHOOD")}
        )
        measurement.update({"config": config_dict})
        measurements.append(measurement)
        return measurements


    def build(self) -> Dict[str, Any]:
        """Construct the full workspace dictionary.

        Calls :meth:`channels` and :meth:`measurements` and combines them with a version tag into the top-level workspace dict.

        Returns
        -------
        dict
            Workspace with keys ``"channels"``, ``"measurements"``, and ``"version"``.  Ready to be passed to :class:`~nsbi_common_utils.models.sbi_parametric_model` or serialised via :meth:`dump_workspace`.
        """
        ws: Dict[str, Any] = {}  # the workspace

        # channels
        channels = self.channels()
        ws.update({"channels": channels})

        # measurements
        measurements = self.measurements()
        ws.update({"measurements": measurements})

        # build observations
        observations = self.observations()
        ws.update({"observations": observations})

        # workspace version
        ws.update({"version": "1.0.0"})

        return ws

    def observations(self) -> List[Dict[str, Any]]:
        """Build the ``"observations"`` list for the workspace.

        Loads observed data from ROOT files as specified in the ``Observations`` section of the config, applies region filters, and histograms to produce observed counts per channel.

        Returns
        -------
        list of dict
            Each dict has keys ``"name"`` and ``"data"`` (list of observed counts per bin).
        """
        observations = []
        for region in self.config_dict["Regions"]:
            region_name = region["Name"]
            
            print(f"Processing observation for region {region_name}...")
                        
            region_binning = region.get("Binning", None)
            region_variable = region.get("Variable", None)
            region_filters = region["Filter"]
            filter_variables = [
                tok
                for tok in re.split(r"[<>=!&|()\s]+", region_filters)
                if tok and not tok.replace(".", "", 1).lstrip("-").isdigit()
            ]
            branches_to_load = [region_variable]
            for v in filter_variables:
                if v not in branches_to_load:
                    branches_to_load.append(v)

            dataset_obs = nsbi_common_utils.datasets.datasets(
                self.config_path, branches_to_load=branches_to_load
            )

            dataframe_obs = dataset_obs.load_observations_from_config()
            region_filters = dataset_obs.config_helper.get_channel_filters(channel_name=region_name)

            dataset_obs_region = dataframe_obs.query(region_filters).copy()
            
            if region_binning is None:
                feature_arr_tmp = dataset_obs_region[region_variable]
                region_binning = np.linspace(
                    np.amin(feature_arr_tmp), np.amax(feature_arr_tmp), num=2
                )  # Dummy binning for a single event yield calculation in unbinned region
                region["Binning"] = region_binning

            feature_var = np.clip(
                dataset_obs_region[region_variable],
                np.amin(region_binning),
                np.amax(region_binning),
            )

            weights = dataset_obs_region["weights"].to_numpy()

            sample_data, _ = np.histogram(
                feature_var, weights=weights, bins=region_binning
            )

            observation = {
                "name": region_name,
                "data": list(sample_data),
            }
            observations.append(observation)

        return observations

    def dump_workspace(self, ws: dict, outpath: str = "workspace.json"):
        """
        Serialise a workspace dictionary to a JSON file.

        Parameters
        ----------
        ws : dict
            Workspace dictionary returned by :meth:`build`.
        outpath : str, optional
            Output file path. Defaults to ``"workspace.json"``.
        """

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(outpath, "w") as f:
            json.dump(ws, f, indent=2, cls=NumpyEncoder)
        print(f"Wrote {outpath}")

    @staticmethod
    def load_workspace(path: str) -> dict:
        """Load a workspace dictionary from a JSON file.

        Parameters
        ----------
        path : str
            Path to the JSON workspace file written by :meth:`dump_workspace`.

        Returns
        -------
        dict
            The deserialised workspace dictionary.
        """
        with open(path) as f:
            return json.load(f)


def compute_vandermonde_coeffs(points_dict):
    stat_coeff = {}

    points = list(points_dict.keys())
    n_points = len(points)
    basis_degree = n_points - 1

    # Build Vandermonde matrix: [x^(basis_degree), x^(basis_degree-1), ..., x, 1]
    A = np.array([[x**(basis_degree - j) for j in range(basis_degree + 1)] for x in points])
    A_inv = np.linalg.inv(A)

    for i, x in enumerate(points):
        key = points_dict[x]
        stat_coeff[key] = [A_inv[j][i] for j in range(basis_degree + 1)]

    return stat_coeff