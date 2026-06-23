import numpy as np
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.tree_util import tree_map
from functools import partial
from typing import Dict, Union, Any, Optional

class sbi_parametric_model:
    """
    Statistical model for semi-parametric Simulation-Based Inference (SBI).

    Defines parameterized expected yields, density ratios, and the negative log-likelihood (NLL) passed to fitting algorithms. Supports both binned and unbinned channels with systematic uncertainties handled via polynomial interpolation / exponential extrapolation (HistFactory strategy 5).

    Two JIT-compiled entry points are built:

    * :meth:`model` — evaluates the negative log-likelihood function for fit.
    * :meth:`model_grad` — evaluates the negative log-likelihood gradient via JAX reverse-mode autodiff.

    Parameters
    ----------
    workspace : dict
        A workspace dictionary following the pyhf-like JSON schema. Must contain ``"measurements"`` and ``"channels"`` keys. Channels may be tagged with ``"type": "binned"`` or ``"type": "unbinned"``.
    measurement_to_fit : str
        Name of the measurement block inside ``workspace["measurements"]`` to use. Selects the parameter of interest (POI) and the list of parameters to fit.

    Attributes
    ----------
    list_parameters : list of str
        Ordered parameter names: POI first, then unconstrained norm factors, then constrained nuisance parameters.
    initial_parameter_values : jnp.ndarray
        Starting values for every parameter, in the same order as ``list_parameters``.
    num_unconstrained_param : int
        Number of leading parameters that are unconstrained (POI + free norm factors).
    expected_hist : jnp.ndarray Binned expected yields evaluated at the initial parameter values.

    See Also
    --------
    nsbi_common_utils.inference.inference : Fits and scans this model.
    """
    def __init__(self,
                 workspace: Dict[Any, Any],
                 measurement_to_fit: str):

        self.workspace                                  = workspace
        self.measurements_dict: list[Dict[str, Any]]    = workspace["measurements"]
        for measurement in self.measurements_dict:
            measurement_name = measurement.get("name")
            if measurement_name == measurement_to_fit:
                self.measurement_name                   = measurement_name
                self.poi                                = measurement["config"]["poi"]
                self.measurement_param_dict             = measurement["config"]["parameters"]
                self.errordef                           = measurement["config"]["errordef"]
                break
        self.param_names = [p['name'] for p in self.workspace['measurements'][0]['config']['parameters']]
        
        self.parameters_in_measurement, \
            self.initial_values_dict                    = self._get_parameters_to_fit()
        
        self.channels_binned                            = self._get_channel_list(type_of_fit="binned")
        self.channels_unbinned                          = self._get_channel_list(type_of_fit="unbinned")
        self.all_channels                               = self.channels_binned + self.channels_unbinned

        self.all_samples                                = self._get_samples_list()
        
        sorting_order                                   = {"normfactor": 0, "vandermonde": 1, "normplusshape": 2} 
        self.list_parameters, \
            self.list_parameters_types, \
                self.num_unconstrained_param            = self._get_parameters(sorting_order)

        self.list_syst_normplusshape                    = self._get_list_syst_for_interp()
        
        self.list_normfactors, \
            self.norm_sample_map                        = self._get_norm_factors() 

        self.list_vandermonde_coeffs, \
            self.vandermonde_coeffs_map, \
                self._max_vandermonde_coeffs               = self._get_vandermonde_coefficients()

        self.has_normplusshape                          = len(self.list_syst_normplusshape) > 0

        self.initial_parameter_values                   = self._get_param_vec_initial()

        self.index_normparam_map                        = self._make_map_index_norm()
        self.index_vandermondeparam_map                   = self._make_map_index_vandermorde()

        self.yield_array_dict, _                        = self._get_nominal_expected_arrays( type_of_fit = "binned" )
        
        self.unbinned_total_dict, \
            self.ratios_array_dict                      = self._get_nominal_expected_arrays( type_of_fit = "unbinned" )

        self.combined_var_up_binned, \
            self.combined_var_dn_binned                 = self._get_systematic_data( type_of_fit="binned" )
        
        self.combined_var_up_unbinned, \
            self.combined_var_dn_unbinned, \
                self.combined_tot_up_unbinned, \
                    self.combined_tot_dn_unbinned       = self._get_systematic_data( type_of_fit="unbinned" )
        
        self.weight_arrays_unbinned                     = self._get_asimov_weights_array()

        self.observed_array                             = self._get_observed_arrays()
        self._finalize_to_device()

        self.expected_hist                              = self._get_expected_hist(param_vec = self.initial_parameter_values)
        self.expected_rate_unbinned                     = self._get_expected_rate_unbinned(param_vec = self.initial_parameter_values)

        # Stack per-process dicts into arrays for vectorized NLL
        self._build_stacked_data()
        self._jit_nll, self._jit_val_and_grad           = self._build_jit_functions()

    def get_model_parameters(self):
        """
        Return parameter names and initial values for fitting.

        The returned order matches the convention expected by :class:`~nsbi_common_utils.inference.inference`: POI at index 0, followed by unconstrained norm factors, then constrained nuisance parameters.

        Returns
        -------
        list_parameters : list of str
            Ordered parameter names.
        initial_parameter_values : jnp.ndarray, shape (n_params,)
            Starting values aligned with ``list_parameters``.
        """
        return self.list_parameters, self.initial_parameter_values
        
    def _get_expected_hist(self, param_vec):
        """
        Optimized function for NLL computations
        """                
        param_vec_interpolation = param_vec[ self.num_unconstrained_param : ]
        norm_modifiers          = {}
        hist_vars_binned        = {}

        norm_modifiers     = self._calculate_norm_variations(param_vec)

        for process in self.all_samples:
            
            if self.has_normplusshape:

                hist_vars_binned[process]   = _calculate_combined_var(  param_vec_interpolation, 
                                                                            self.combined_var_up_binned[process],
                                                                            self.combined_var_dn_binned[process]    )

            else:
                hist_vars_binned[process] = jnp.ones_like( self.yield_array_dict[process] )

        data_expected = self._calculate_parameterized_yields(  self.yield_array_dict, 
                                                                        hist_vars_binned, 
                                                                        norm_modifiers )

        return data_expected

    def _get_expected_rate_unbinned(self, param_vec):
        """Compute the total expected rate in unbinned channels at ``param_vec``. Used as Asimov observed rate."""
        param_vec_interpolation = param_vec[ self.num_unconstrained_param : ]
        norm_modifiers = self._calculate_norm_variations(param_vec)

        hist_vars_unbinned = {}
        for process in self.all_samples:
            if self.has_normplusshape:
                hist_vars_unbinned[process] = _calculate_combined_var(  param_vec_interpolation,
                                                                        self.combined_tot_up_unbinned[process],
                                                                        self.combined_tot_dn_unbinned[process]    )
            else:
                hist_vars_unbinned[process] = jnp.ones_like( self.unbinned_total_dict[process] )

        return self._calculate_parameterized_yields(self.unbinned_total_dict, hist_vars_unbinned, norm_modifiers)

    def _make_map_index_norm(self):
        """
        Maps the index of parameter in the parameter vector to norm factor
        """
        dict_index_normfactor = {}
        for normfactor in self.list_normfactors:
            index = self.list_parameters.index( normfactor )
            dict_index_normfactor[normfactor] = index
        return dict_index_normfactor

    def _make_map_index_vandermorde(self):
        """
        Maps the index of parameter in the parameter vector to norm factor
        """
        dict_index_vandermondefactor = {}
        for vandermondefactor in self.list_vandermonde_coeffs:
            index = self.list_parameters.index( vandermondefactor )
            dict_index_vandermondefactor[vandermondefactor] = index
        return dict_index_vandermondefactor

    def _get_param_vec_initial(self):
        initial_values_vec                     = np.ones((len(self.list_parameters),)) 
        for count, parameter in enumerate(self.list_parameters):
            initial_values_vec[count]           = self.initial_values_dict[parameter]
        return jnp.asarray(initial_values_vec)

    def _get_norm_factors(self) -> Union[list, Dict[str, list]]:
        """Assume same normfactor across channels for now (TO-DO: Add support for normfactor per channel)"""
        dict_sample_normfactors         = {sample_name: [] for sample_name in self.all_samples}
        list_all_norm_factors           = []
        for channel in self.all_channels[:1]:
            channel_index = self._index_of_region(channel_name=channel)
            for sample in self.all_samples:
                sample_index = self._index_of_sample(channel_name=channel, sample_name=sample)
                modifier_list = self.workspace["channels"][channel_index]["samples"][sample_index]["modifiers"]
                for modifier in modifier_list:
                    if modifier["type"] == "normfactor":
                        modifier_name = modifier["name"]
                        if modifier_name not in list_all_norm_factors               : list_all_norm_factors.append(modifier_name)
                        if modifier_name not in dict_sample_normfactors[sample]     : dict_sample_normfactors[sample].append(modifier_name)

        list_all_norm_factors = [p for p in list_all_norm_factors if p in self.param_names]
        dict_sample_normfactors = {key: val for key, val in dict_sample_normfactors.items()
                                    if any(p in self.param_names for p in val)
                                }

        return list_all_norm_factors, dict_sample_normfactors

    def _get_vandermonde_coefficients(self) -> Dict[str, list[float]]:
        """Assume same vandermonde coefficients across channels for now 
        """
        dict_sample_vandermonde_factors        = {sample_name: {} for sample_name in self.all_samples}
        list_all_vandermonde_factors           = []
        max_vandermonde_coeffs                 = 0
        for channel in self.all_channels[:1]:
            channel_index = self._index_of_region(channel_name=channel)
            for sample in self.all_samples:
                sample_index = self._index_of_sample(channel_name=channel, sample_name=sample)
                modifier_list = self.workspace["channels"][channel_index]["samples"][sample_index]["modifiers"]
                for modifier in modifier_list:
                    if modifier["type"] == "vandermonde":
                        modifier_name = modifier["name"]
                        modiefier_coeffs = modifier.get("coeff", [1,0]) # Default to linear variation if no coefficients provided
                        max_vandermonde_coeffs = max(max_vandermonde_coeffs, len(modiefier_coeffs))
                        if modifier_name not in list_all_vandermonde_factors               : list_all_vandermonde_factors.append(modifier_name)
                        if modifier_name not in dict_sample_vandermonde_factors[sample]     : dict_sample_vandermonde_factors[sample][modifier_name] = modiefier_coeffs

        list_all_vandermonde_factors = [p for p in list_all_vandermonde_factors if p in self.param_names]
        dict_sample_vandermonde_factors = {key: val for key, val in dict_sample_vandermonde_factors.items()
                                    if any(p in self.param_names for p in val)
                                }
 
        return list_all_vandermonde_factors, dict_sample_vandermonde_factors, max_vandermonde_coeffs

    def _get_parameters_to_fit(self) -> tuple[list[str], dict[str, float]]:
        """
        Outputs a list of parameters specified by the user for fitting in the workspace
        """
        parameters_to_fit        = []
        initial_value_params     = {}
        for parameters in self.measurement_param_dict:
            parameter_name                              = parameters["name"]
            parameter_init                              = parameters["inits"][0]
            parameters_to_fit.append(parameter_name)
            initial_value_params[parameter_name]        = parameter_init

        return parameters_to_fit, initial_value_params

    def _get_list_syst_for_interp(self):
        """Get the list of subset of systematics that need interpolation."""
        mask_normplusshape  = (np.array(self.list_parameters_types) == "normplusshape")
        list_normplusshape  = np.array(self.list_parameters)[mask_normplusshape].tolist()
        return list_normplusshape

    def _get_channel_list(self, 
                          type_of_fit: Union[str, None] = None) -> list:
        """Get the channel list to be used in the measurement"""
        list_channels = []
        channels: list[Dict[str, Any]] = self.workspace["channels"]
        for channel_dict in channels:
            if type_of_fit is not None:
                if channel_dict.get("type") != type_of_fit: continue 
            list_channels.append(channel_dict.get("name"))
        return list_channels
    
    def _get_samples_list(self):
        """Get the sample list from the first channel"""
        list_samples = []
        channels: list[Dict[str, Any]] = self.workspace["channels"]
        for channel_dict in channels:
            samples: list[Dict[str, Any]] = channel_dict["samples"]
            for sample_dict in samples:
                list_samples.append(sample_dict.get("name"))
            break
        return list_samples
    
    def _get_asimov_weights_array(self):
        """
        Get the Asimov weight vector for fitting
        """
        weight_array = np.array([])
        for channel in self.channels_unbinned:
            channel_index       = self._index_of_region(channel)
            weights             = np.load(self.workspace["channels"][channel_index]["weights"])
            weight_array        = np.append(weight_array, weights)
        return weight_array
    
    def _get_parameters(self, sorting_order):
        """Get a list of all parameters."""
        list_param_names = []
        list_param_types = []
        channels: list[Dict[str, Any]] = self.workspace["channels"]
        for channel_dict in channels:
            samples: list[Dict[str, Any]] = channel_dict["samples"]
            for sample_dict in samples:
                modifiers_list: list[Dict[str, Any]] = sample_dict["modifiers"]
                for modifier in modifiers_list:
                    modifier_name = modifier.get("name")
                    if modifier_name not in self.parameters_in_measurement: continue
                    modifier_type = modifier.get("type")
                    if modifier_name not in list_param_names:
                        list_param_names.append(modifier_name)
                        list_param_types.append(modifier_type)

        indices = np.argsort([sorting_order.get(param_type, 999) for param_type in list_param_types])

        list_param_names = [list_param_names[i] for i in indices]
        list_param_types = [list_param_types[i] for i in indices]

        index_poi       = list_param_names.index(self.poi)
        if index_poi    != 0:
            poi_name = list_param_names.pop(index_poi)
            poi_type = list_param_types.pop(index_poi)

            list_param_names.insert(0, poi_name)
            list_param_types.insert(0, poi_type)

        num_unconstrained_params = 0
        for poi_type_ in list_param_types:
            if poi_type_ != "normfactor" and poi_type_ != "vandermonde":
                break
            num_unconstrained_params += 1

        return list_param_names, list_param_types, num_unconstrained_params
    
    def _calculate_parameterized_yields(self, hist_yields, hist_vars, norm_modifiers):

        nu_tot = 0.0
        
        for process in self.all_samples:
            # This will not work in the general case where model is non-linear in POI, needs modifications (TO-DO)
            nu_tot += norm_modifiers[process] * hist_yields[process] * hist_vars[process]

        return nu_tot
    
    def _calculate_parameterized_ratios(self, nu_nominal, nu_vars, 
                                        ratios, ratio_vars, norm_modifiers):

        dnu_dx = jnp.zeros_like(self.weight_arrays_unbinned) # To-do: Generalize to any dataset, not just nominal

        for process in self.all_samples:
            # jax.debug.print("norm_modifiers variations is {x1}", x1 = norm_modifiers[process])
            dnu_dx += norm_modifiers[process] * nu_vars[process] * nu_nominal[process] * ratios[process] * ratio_vars[process]
            
        return jnp.log( dnu_dx )
    
    def model(self, param_array: Union[np.array, jnp.array, list[float]]):
        """
        High-level API that returns the full negative log-likelihood for a parameter point.

        Computes the combined NLL in all channels defined by the input workspace - unbinned SBI and binned Control and Signal regions. This callable is the function to be passed to :class:`~nsbi_common_utils.inference.inference` as ``model_nll``.

        Parameters
        ----------
        param_array : array-like, shape (n_params,)
            Parameter values in the order defined by :meth:`get_model_parameters`.

        Returns
        -------
        nll : jnp.ndarray, scalar
            The negative log-likelihood value (scalar).
        """
        param_array = jnp.asarray(param_array)
        return self._jit_nll(param_array, self._model_data)

    def model_grad(self, param_array: Union[np.array, jnp.array, list[float]]):
        """
        Return the gradient of the NLL with respect to all parameters.

        Uses JAX reverse-mode autodiff (JIT-compiled). Suitable as the
        ``grad`` argument to :class:`iminuit.Minuit`.

        Parameters
        ----------
        param_array : array-like, shape (n_params,)

        Returns
        -------
        grad : np.ndarray, shape (n_params,)
        """
        param_array = jnp.asarray(param_array)
        _, g = self._jit_val_and_grad(param_array, self._model_data)
        return np.asarray(g)
    
    
    def _get_nominal_expected_arrays(self, type_of_fit:str):
        """
        Get an array of expected event yields or ratios
        """
        data_expected   = {sample_name : np.array([]) for sample_name in self.all_samples}
        ratio_expected  = {sample_name : np.array([]) for sample_name in self.all_samples}

        if type_of_fit == "binned":
            channels_list       =   self.channels_binned
        elif type_of_fit == "unbinned": 
            channels_list       =   self.channels_unbinned 

        for sample_name in self.all_samples:

            for channel_name in channels_list:

                channel_index           = self._index_of_region(channel_name = channel_name)
                sample_index            = self._index_of_sample(channel_name = channel_name,
                                                                    sample_name  = sample_name)
            
                if type_of_fit == "binned":
                    sample_data             = np.array(self.workspace["channels"][channel_index]["samples"][sample_index]["data"])
                    sample_ratio            = np.array([])
                elif type_of_fit == "unbinned":
                    sample_data             = np.array(self.workspace["channels"][channel_index]["samples"][sample_index]["data"])
                    sample_ratio            = np.load(self.workspace["channels"][channel_index]["samples"][sample_index]["ratios"])

                data_expected[sample_name]  =   np.append(data_expected[sample_name], sample_data)
                ratio_expected[sample_name] =   np.append(ratio_expected[sample_name], sample_ratio)

        return data_expected, ratio_expected

    def _get_observed_arrays(self):
        """
        Get an array of observed event yields
        """
        data_list = []

        for channel_name in self.channels_binned:
            channel_index = self._index_of_region(channel_name=channel_name)
            channel_data = np.array(self.workspace["observations"][channel_index]["data"])
            
            data_list.append(channel_data)
            combined_length = sum(len(arr) for arr in data_list)

        if data_list:
            data_observed = np.concatenate(data_list)
        else:
            data_observed = np.array([])
        
        return data_observed

    def _calculate_norm_variations(self, param_vec):
        norm_var = {sample_name: 1.0 for sample_name in self.all_samples}
        for sample, params_sample in self.norm_sample_map.items():  
            # params_sample: list[str]
            for param in params_sample:
                index_param             = self.index_normparam_map[param]
                norm_var[sample]       *= param_vec[index_param]
        
        for sample , params_sample in self.vandermonde_coeffs_map.items():
            # params_sample is a dict of param name to list of vandermonde coefficients
            for param, coeffs in params_sample.items():
                index_param             = self.index_vandermondeparam_map[param]
                norm_var[sample]       *= np.polyval(coeffs, param_vec[index_param])
                
        return norm_var
    
    def _get_systematic_data(self, type_of_fit: str) -> Dict[str, jnp.ndarray]:
        """
        Builds a rectangular array with (N_syst, N_datapoints) dimensions, where N_datapoints is the number of bins in binned channels and number of events in unbinned channels. 
        Concatenates all binned or all unbinned channels into one big array for array-based computations.

        type_of_fit -> choose if building array for "unbinned" channels or "binned"
        """
        if type_of_fit == "binned":
            base_array_for_size     = self.yield_array_dict[self.all_samples[0]]
            channel_list            = self.channels_binned
        elif type_of_fit == "unbinned":            
            base_array_for_size     = self.ratios_array_dict[self.all_samples[0]]
            base_tot_for_size       = self.unbinned_total_dict[self.all_samples[0]]
            channel_list            = self.channels_unbinned

        combined_var_up             = {sample_name: np.ones((len(self.list_syst_normplusshape), 
                                                             len(base_array_for_size))) for sample_name in self.all_samples}
        combined_var_dn             = {sample_name: np.ones((len(self.list_syst_normplusshape), 
                                                             len(base_array_for_size))) for sample_name in self.all_samples}

        if type_of_fit == "unbinned":
            combined_tot_up             = {sample_name: np.ones((len(self.list_syst_normplusshape), 
                                                                 len(base_tot_for_size))) for sample_name in self.all_samples}
            combined_tot_dn             = {sample_name: np.ones((len(self.list_syst_normplusshape), 
                                                                 len(base_tot_for_size))) for sample_name in self.all_samples}

        for sample_name in self.all_samples:

            for count, systematic_name in enumerate(self.list_syst_normplusshape):

                var_up_array_syst = np.array([])
                var_dn_array_syst = np.array([])

                if type_of_fit == "unbinned":
                    var_up_tot_syst = np.array([])
                    var_dn_tot_syst = np.array([])

                for channel_name in channel_list:

                    channel_index           = self._index_of_region(channel_name = channel_name)
                    sample_index            = self._index_of_sample(channel_name = channel_name,
                                                                    sample_name  = sample_name)
                    
                    modifier_index          = self._index_of_modifiers(channel_name    = channel_name,
                                                                        sample_name     = sample_name,
                                                                        systematic_name = systematic_name)
                    modifier_dict           = self.workspace["channels"][channel_index]["samples"][sample_index]["modifiers"][modifier_index]

                    if type_of_fit == "binned":
                        var_array_up_channel = modifier_dict["data"]["hi_data"]
                        var_array_dn_channel = modifier_dict["data"]["lo_data"]

                    elif type_of_fit == "unbinned":
                        
                        var_array_up_channel = np.load(modifier_dict["data"]["hi_ratio"])
                        var_total_up_channel = modifier_dict["data"]["hi_data"]

                        var_array_dn_channel = np.load(modifier_dict["data"]["lo_ratio"])
                        var_total_dn_channel = modifier_dict["data"]["lo_data"]

                        var_up_tot_syst       = np.append(var_up_tot_syst, var_total_up_channel)
                        var_dn_tot_syst       = np.append(var_dn_tot_syst, var_total_dn_channel)

                    var_up_array_syst       = np.append(var_up_array_syst, var_array_up_channel)
                    var_dn_array_syst       = np.append(var_dn_array_syst, var_array_dn_channel)

                combined_var_up[sample_name][count] = var_up_array_syst
                combined_var_dn[sample_name][count] = var_dn_array_syst

                if type_of_fit == "unbinned":
                    
                    combined_tot_up[sample_name][count] = var_up_tot_syst
                    combined_tot_dn[sample_name][count] = var_dn_tot_syst

        if type_of_fit == "unbinned":
            return combined_var_up, combined_var_dn, combined_tot_up, combined_tot_dn
            
        return combined_var_up, combined_var_dn

    def _finalize_to_device(self):
        # convert to JAX arrays for JIT compiled function
        self.yield_array_dict           = tree_map(jnp.asarray, self.yield_array_dict)
        self.unbinned_total_dict        = tree_map(jnp.asarray, self.unbinned_total_dict)
        self.ratios_array_dict          = tree_map(jnp.asarray, self.ratios_array_dict)

        self.combined_var_up_unbinned   = tree_map(jnp.asarray, self.combined_var_up_unbinned)
        self.combined_var_dn_unbinned   = tree_map(jnp.asarray, self.combined_var_dn_unbinned)

        self.combined_tot_up_unbinned   = tree_map(jnp.asarray, self.combined_tot_up_unbinned)
        self.combined_tot_dn_unbinned   = tree_map(jnp.asarray, self.combined_tot_dn_unbinned)

        self.combined_var_up_binned     = tree_map(jnp.asarray, self.combined_var_up_binned)
        self.combined_var_dn_binned     = tree_map(jnp.asarray, self.combined_var_dn_binned)

        self.weight_arrays_unbinned     = jnp.asarray(self.weight_arrays_unbinned)
        
    def _build_stacked_data(self):
        """Stack per-process dicts into arrays and bundle into a single pytree for JIT."""
        samples = self.all_samples

        # Stack nominal data: (n_samples, n_datapoints)
        yield_stacked           = jnp.stack([self.yield_array_dict[s] for s in samples])
        unbinned_total_stacked  = jnp.stack([self.unbinned_total_dict[s] for s in samples])
        ratios_stacked          = jnp.stack([self.ratios_array_dict[s] for s in samples])

        # Stack systematic variations: (n_samples, n_syst, n_datapoints)
        var_up_binned_stacked   = jnp.stack([self.combined_var_up_binned[s] for s in samples])
        var_dn_binned_stacked   = jnp.stack([self.combined_var_dn_binned[s] for s in samples])
        var_up_unbinned_stacked = jnp.stack([self.combined_var_up_unbinned[s] for s in samples])
        var_dn_unbinned_stacked = jnp.stack([self.combined_var_dn_unbinned[s] for s in samples])
        tot_up_unbinned_stacked = jnp.stack([self.combined_tot_up_unbinned[s] for s in samples])
        tot_dn_unbinned_stacked = jnp.stack([self.combined_tot_dn_unbinned[s] for s in samples])

        # Norm-factor mask: (n_samples, n_params) — True where param j is a normfactor for sample i.  prod(where(mask, param_vec, 1)) gives the per-sample multiplicative modifier.
        n_samples = len(samples)
        n_params  = len(self.list_parameters)
        norm_matrix = np.zeros((n_samples, n_params), dtype=bool)
        coeffs_matrix = np.zeros((n_samples, n_params, self._max_vandermonde_coeffs), dtype=float) 
        if self._max_vandermonde_coeffs > 0:
            coeffs_matrix[:, :, -1] = 1.0
                
        for i, sample in enumerate(samples):
            if sample in self.norm_sample_map:
                for nf_name in self.norm_sample_map[sample]:
                    j = self.index_normparam_map[nf_name]
                    norm_matrix[i, j] = True

            if sample in self.vandermonde_coeffs_map:
                for lf_name, coeffs in self.vandermonde_coeffs_map[sample].items():
                    j = self.index_vandermondeparam_map[lf_name]
                    
                    c_len = len(coeffs)
                    coeffs_matrix[i, j, -c_len:] = coeffs

        # Bundle everything into a dict pytree passed as a *dynamic* argument
        # to the JIT-compiled NLL so that arrays are traced as abstract inputs
        # (no constant-folding / memory blow-up).
              
        self._model_data = {
            'yield':            yield_stacked,
            'unbinned_total':   unbinned_total_stacked,
            'ratios':           ratios_stacked,
            'var_up_binned':    var_up_binned_stacked,
            'var_dn_binned':    var_dn_binned_stacked,
            'var_up_unbinned':  var_up_unbinned_stacked,
            'var_dn_unbinned':  var_dn_unbinned_stacked,
            'tot_up_unbinned':  tot_up_unbinned_stacked,
            'tot_dn_unbinned':  tot_dn_unbinned_stacked,
            'norm_matrix':      jnp.array(norm_matrix),
            'coeffs_matrix':    jnp.array(coeffs_matrix),
            'observed_hist':    self.observed_array,
            'observed_rate':    jnp.sum(self.weight_arrays_unbinned),
            'weights':          self.weight_arrays_unbinned,
        }

    def _build_jit_functions(self):
        """
        Create JIT-compiled NLL and value-and-grad functions.
        """
        num_unc  = self.num_unconstrained_param
        has_syst = self.has_normplusshape
        if self.errordef == "LIKELIHOOD":
            scale = 1
        else :
            scale = 2

        _batched_var = jax.vmap(_calculate_combined_var, in_axes=(None, 0, 0))
        
        
        def _nll_pure(param_vec, data):
            param_syst = param_vec[num_unc:]
            
            # jax.debug.print("param_vec is {y}", y=param_vec)

            norm_mods = jnp.prod(
                jnp.where(data['norm_matrix'], param_vec[None, :], 1.0), axis=1
            )
            
            if self._max_vandermonde_coeffs > 0:
            
                def compute_norm_mods(coeffs_sample, param_vec):
                    
                    # 1. Map polyval across each parameter's coefficient row
                    poly_vals = jax.vmap(jnp.polyval)(coeffs_sample, param_vec)
                    
                    # 2. Multiply the results of all polynomials for this sample
                    return jnp.prod(poly_vals)

                # Apply across all samples
                final_norm_mods = jax.vmap(compute_norm_mods, in_axes=(0, None))(data['coeffs_matrix'], param_vec)
                
                norm_mods = norm_mods * final_norm_mods
            
            # --- Systematic variations (one vmapped call per category) ---
            if has_syst:
                hist_vars_binned = _batched_var(param_syst,
                                           data['var_up_binned'],
                                           data['var_dn_binned'])
                hist_vars_unbinned = _batched_var(param_syst,
                                           data['tot_up_unbinned'],
                                           data['tot_dn_unbinned'])
                ratio_vars  = _batched_var(param_syst,
                                           data['var_up_unbinned'],
                                           data['var_dn_unbinned'])
            else:
                hist_vars_binned = jnp.ones_like(data['yield'])
                hist_vars_unbinned = jnp.ones_like(data['unbinned_total'])
                ratio_vars  = jnp.ones_like(data['ratios'])

            nu_binned  = jnp.sum(norm_mods[:, None] * data['yield'] * hist_vars_binned,
                                 axis=0)
                        
            llr_binned = -scale * jnp.sum(
                data['observed_hist'] * jnp.log(nu_binned) - nu_binned
            )

            nu_unbinned = jnp.sum(
                norm_mods[:, None] * data['unbinned_total'] * hist_vars_unbinned,
                axis=0
            )
            llr_rate = -scale * jnp.sum(
                data['observed_rate'] * jnp.log(nu_unbinned) - nu_unbinned
            )

            dnu_dx = jnp.sum(
                norm_mods[:, None]
                * hist_vars_unbinned
                * data['unbinned_total']
                * data['ratios']
                * ratio_vars,
                axis=0
            )
            llr_pe = jnp.log(dnu_dx) - jnp.log(nu_unbinned)

            llr_constraints = jnp.sum(param_syst ** 2)
            
            llr_unbinned = - scale * jnp.sum(data['weights'] * llr_pe, axis=0)
            
            llr_total = llr_binned + llr_rate + llr_unbinned + llr_constraints

            # jax.debug.print("nu_binned is {y}", y=nu_binned)
            # jax.debug.print("observed_hist is {y}", y=data['observed_hist'])            
            # jax.debug.print("llr_binned is {y}", y=llr_binned)
            # jax.debug.print("llr_rate is {y}", y=llr_rate)
            # jax.debug.print("llr_unbinned is {y}", y=llr_unbinned)
            # jax.debug.print("llr_constraints is {y}", y=llr_constraints)
            # jax.debug.print("llr_total is {y}", y=llr_total)
            
            return (llr_total)

        jit_nll          = jax.jit(_nll_pure)
        jit_val_and_grad = jax.jit(jax.value_and_grad(_nll_pure, argnums=0))

        return jit_nll, jit_val_and_grad

    def _index_of_modifiers(self,
                          channel_name: str,
                          sample_name: str,
                          systematic_name: str) -> Optional[int]:
        """
        Get the index associated with a systematic, in a specific sample of a particular channel
        """
        channel_index                       = self._index_of_region(channel_name)
        sample_index                        = self._index_of_sample(channel_name, sample_name)
        modifiers: list[dict[str, Any]]     = self.workspace["channels"][channel_index]["samples"][sample_index]["modifiers"]
        for count, modifier in enumerate(modifiers):
            if modifier.get("name") == systematic_name:
                return count
        return None

    def _index_of_sample(self, 
                          channel_name: str,
                          sample_name: str) -> Optional[int]:
        """
        Get the index associated with a sample, in a particular channel
        """
        channel_index = self._index_of_region(channel_name)
        samples: list[dict[str, Any]] = self.workspace["channels"][channel_index]["samples"]
        for count, sample in enumerate(samples):
            if sample.get("name") == sample_name:
                return count
        return None

    def _index_of_region(self, channel_name: str) -> Optional[int]:
        """
        Get the index associated with a particular channel in the workspace
        """
        channels: list[dict[str, Any]] = self.workspace["channels"]
        for count, channel in enumerate(channels):
            if channel.get("name") == channel_name:
                return count
        return None

    
@jax.jit
def _poly_interp(tuple_input):
    """
    Sixth-order polynomial interpolation for systematic variations.

    Implements the HistFactory "strategy 5" interpolation used when
    :math:`|\\alpha| \\le 1`. Smoothly connects the upward and downward
    variation multipliers using a degree-6 polynomial in the nuisance
    parameter :math:`\\alpha`.

    Parameters
    ----------
    tuple_input : tuple of (jnp.ndarray, jnp.ndarray, jnp.ndarray)
        ``(alpha, pow_up, pow_down)`` where *alpha* is the nuisance
        parameter value, *pow_up* the upward variation ratio, and
        *pow_down* the downward variation ratio.

    Returns
    -------
    variation : jnp.ndarray
        Multiplicative correction to apply to the nominal prediction.
    """
    alpha, pow_up, pow_down = tuple_input
    
    logHi         = jnp.log(pow_up)
    logLo         = jnp.log(pow_down)
    pow_up_log    = jnp.multiply(pow_up, logHi)
    pow_down_log  = -jnp.multiply(pow_down, logLo)
    pow_up_log2   =  jnp.multiply(pow_up_log, logHi)
    pow_down_log2 = -jnp.multiply(pow_down_log, logLo)

    S0 = (pow_up + pow_down) / 2.0
    A0 = (pow_up - pow_down) / 2.0
    S1 = (pow_up_log  + pow_down_log) / 2.0
    A1 = (pow_up_log  - pow_down_log) / 2.0
    S2 = (pow_up_log2 + pow_down_log2) / 2.0
    A2 = (pow_up_log2 - pow_down_log2) / 2.0

    a1 = ( 15 * A0 -  7 * S1 + A2)      / 8.0
    a2 = (-24 + 24 * S0 -  9 * A1 + S2) / 8.0
    a3 = ( -5 * A0 +  5 * S1 - A2)      / 4.0
    a4 = ( 12 - 12 * S0 +  7 * A1 - S2) / 4.0
    a5 = (  3 * A0 -  3 * S1 + A2)      / 8.0
    a6 = ( -8 +  8 * S0 -  5 * A1 + S2) / 8.0

    return alpha * (a1 + alpha * ( a2 + alpha * ( a3 + alpha * ( a4 + alpha * ( a5 + alpha * a6 ) ) ) ) )

@jax.jit
def _exp_extrap(tuple_input):
    """
    Exponential extrapolation for systematic variations.

    Used when :math:`|\\alpha| > 1` (outside the interpolation region).
    Extrapolates the variation as a power law in :math:`\\alpha`.

    Parameters
    ----------
    tuple_input : tuple of (jnp.ndarray, jnp.ndarray, jnp.ndarray)
        ``(alpha, varUp, varDown)`` where *alpha* is the nuisance
        parameter value, *varUp* the upward variation ratio, and
        *varDown* the downward variation ratio.

    Returns
    -------
    variation : jnp.ndarray
        Multiplicative correction (minus 1) to apply to the nominal
        prediction.

    See Also
    --------
    _poly_interp : Polynomial interpolation for :math:`|\\alpha| \\le 1`.
    """
    alpha, varUp, varDown = tuple_input

    return jnp.where(alpha>1.0, (varUp)**alpha, (varDown)**(-alpha)) - 1.0

@jax.jit
def _calculate_combined_var(param_vec, combined_var_up, combined_var_down):
    """
    Compute the net multiplicative effect of all systematic variations.

    Sequentially applies each nuisance parameter's variation using
    :func:`_poly_interp` (for :math:`|\\alpha| \\le 1`) or
    :func:`_exp_extrap` (for :math:`|\\alpha| > 1`) via ``jax.lax.scan``.

    Parameters
    ----------
    param_vec : jnp.ndarray, shape (n_syst,)
        Values of the constrained nuisance parameters.
    combined_var_up : jnp.ndarray, shape (n_syst, n_datapoints)
        Upward variation ratios for each systematic and data point.
    combined_var_down : jnp.ndarray, shape (n_syst, n_datapoints)
        Downward variation ratios for each systematic and data point.

    Returns
    -------
    combined_var : jnp.ndarray, shape (n_datapoints,)
        Net multiplicative variation factor across all systematics.
    """

    def calculate_variations(carry, param_val):
        
        param, combined_var_up_NP, combined_var_down_NP = param_val
        
        combined_var_array_alpha = carry
    
        # Strategy 5 of RooFit:
        combined_var_array_alpha += combined_var_array_alpha * jax.lax.cond(jnp.abs(param)<=1.0, 
                                                                            _poly_interp, 
                                                                            _exp_extrap, 
                                                                            (param, combined_var_up_NP, combined_var_down_NP))            
        return combined_var_array_alpha, None

    # Prepare loop_tuple for jax.lax.scan 
    loop_tuple = (param_vec, combined_var_up, combined_var_down)

    # Loop over systematic variations to calculate net effect
    combined_var_array, _ = jax.lax.scan(calculate_variations, jnp.ones_like(combined_var_up[0]), loop_tuple)

    return combined_var_array

