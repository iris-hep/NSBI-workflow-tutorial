import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from typing import Dict, Union, Any, Optional

# Set default tensor type for PyTorch to float64 to match JAX's default (jax_enable_x64)
# This is crucial for numerical stability and consistency in scientific/physics computations.
torch.set_default_dtype(torch.float64)


class NSBILightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for High Energy Physics statistical model fitting.
    This module encapsulates the model definition, parameter handling, and likelihood
    computations (Negative Log-Likelihood) for both binned and unbinned data.

    It replaces the original JAX-based Model, adapting its functionality to PyTorch
    and PyTorch Lightning's paradigm.
    """

    def __init__(self, workspace: Dict[Any, Any], measurement_to_fit: str):
        super().__init__()
        # Store hyperparameters for checkpointing and reproducibility
        self.save_hyperparameters()

        self.workspace = workspace
        self.measurement_to_fit = measurement_to_fit

        # --- Initialize model configuration from workspace ---
        self.measurements_dict: list[Dict[str, Any]] = self.workspace["measurements"]
        self.measurement_name: Optional[str] = None
        self.poi: Optional[str] = None
        self.measurement_param_dict: Optional[list[Dict[str, Any]]] = None

        for measurement in self.measurements_dict:
            measurement_name = measurement.get("name")
            if measurement_name == self.measurement_to_fit:
                self.measurement_name = measurement_name
                self.poi = measurement["config"]["poi"]
                self.measurement_param_dict = measurement["config"]["parameters"]
                break
        if self.measurement_name is None:
            raise ValueError(f"Measurement '{measurement_to_fit}' not found in workspace.")

        self.parameters_in_measurement, \
            self.initial_values_dict = self._get_parameters_to_fit()

        self.channels_binned = self._get_channel_list(type_of_fit="binned")
        self.channels_unbinned = self._get_channel_list(type_of_fit="unbinned")
        self.all_channels = self.channels_binned + self.channels_unbinned
        self.all_samples = self._get_samples_list()

        # Define sorting order for parameters: normfactor types first, then normplusshape
        sorting_order = {"normfactor": 0, "normplusshape": 1}
        self.list_parameters, \
            self.list_parameters_types, \
                self.num_unconstrained_param = self._get_parameters(sorting_order)

        self.list_syst_normplusshape = self._get_list_syst_for_interp()
        self.list_normfactors, \
            self.norm_sample_map = self._get_norm_factors()

        self.has_normplusshape = len(self.list_syst_normplusshape) > 0

        self.index_normparam_map = self._make_map_index_norm()

        # Initialize trainable nuisance parameters as nn.Parameter
        initial_parameter_values = self._get_param_vec_initial()
        self.nuisance_parameters = nn.Parameter(torch.tensor(initial_parameter_values))

        # --- Load and store static precomputed data as torch.Tensors ---
        # These tensors are part of the model's state but are not trainable.
        # They are assigned as attributes, and Lightning will handle moving them to the correct device.

        # Nominal expected yields and ratios
        np_yield_array_binned, _ = self._get_nominal_expected_arrays(type_of_fit="binned")
        self.yield_array_binned_dict = {k: torch.tensor(v) for k, v in np_yield_array_binned.items()}

        np_unbinned_total, np_ratios_array = self._get_nominal_expected_arrays(type_of_fit="unbinned")
        self.unbinned_total_dict = {k: torch.tensor(v) for k, v in np_unbinned_total.items()}
        self.ratios_array_dict = {k: torch.tensor(v) for k, v in np_ratios_array.items()}

        # Systematic variations for binned channels
        np_var_up_binned, np_var_dn_binned = self._get_systematic_data(type_of_fit="binned")
        self.combined_var_up_binned_dict = {k: torch.tensor(v) for k, v in np_var_up_binned.items()}
        self.combined_var_dn_binned_dict = {k: torch.tensor(v) for k, v in np_var_dn_binned.items()}

        # Systematic variations for unbinned channels
        np_var_up_unbinned, np_var_dn_unbinned, np_tot_up_unbinned, np_tot_dn_unbinned = self._get_systematic_data(type_of_fit="unbinned")
        self.combined_var_up_unbinned_dict = {k: torch.tensor(v) for k, v in np_var_up_unbinned.items()}
        self.combined_var_dn_unbinned_dict = {k: torch.tensor(v) for k, v in np_var_dn_unbinned.items()}
        self.combined_tot_up_unbinned_dict = {k: torch.tensor(v) for k, v in np_tot_up_unbinned.items()}
        self.combined_tot_dn_unbinned_dict = {k: torch.tensor(v) for k, v in np_tot_dn_unbinned.items()}

        # Asimov weights for unbinned likelihood (observed data equivalent)
        # Registering as a buffer is the idiomatic way for non-parameter tensors.
        self.register_buffer("weight_arrays_unbinned", torch.tensor(self._get_asimov_weights_array()))

        # The workspace is typically no longer needed after initialization, but kept for methods below
        # if they still need to query it (e.g., _index_of_region).

    def _index_of_region(self, channel_name: str) -> int:
        """Helper to find the index of a channel in the workspace."""
        for i, channel in enumerate(self.workspace["channels"]):
            if channel["name"] == channel_name:
                return i
        raise ValueError(f"Channel '{channel_name}' not found in workspace.")

    def _index_of_sample(self, channel_name: str, sample_name: str) -> int:
        """Helper to find the index of a sample within a specific channel."""
        channel_idx = self._index_of_region(channel_name)
        for i, sample in enumerate(self.workspace["channels"][channel_idx]["samples"]):
            if sample["name"] == sample_name:
                return i
        raise ValueError(f"Sample '{sample_name}' not found in channel '{channel_name}'.")

    def _calculate_combined_var(self, param_vec_interpolation: torch.Tensor,
                                combined_var_up_dict: Dict[str, torch.Tensor],
                                combined_var_dn_dict: Dict[str, torch.Tensor],
                                process_name: str) -> torch.Tensor:
        """
        Calculates the combined systematic variation factor for a given process and parameters.
        This function interpolates between nominal, up, and down variations based on the
        values of shape nuisance parameters using a piecewise linear function. The variations
        are treated as multiplicative factors.

        Args:
            param_vec_interpolation: A 1D tensor containing values of shape nuisance parameters.
            combined_var_up_dict: A dictionary where keys are process names and values are
                                  (num_shape_systs, num_bins_or_events) tensors for up variations.
            combined_var_dn_dict: Same structure as `combined_var_up_dict` for down variations.
            process_name: The name of the process (e.g., 'ttbar', 'diboson').

        Returns:
            A 1D tensor (num_bins_or_events,) representing the combined shape factor for the process.
            Returns a tensor of ones if the process has no shape systematics.
        """
        # If there are no shape systematics for this model, or if this specific process
        # has no shape systematics, return a multiplicative factor of 1.0.
        if not self.has_normplusshape or process_name not in combined_var_up_dict:
            nominal_yields = self.yield_array_binned_dict[process_name]
            return torch.ones_like(nominal_yields)

        process_var_up = combined_var_up_dict[process_name]
        process_var_dn = combined_var_dn_dict[process_name]

        # Reshape nuisance parameters for broadcasting: (num_systs,) -> (num_systs, 1)
        alphas = param_vec_interpolation.unsqueeze(1)

        # Calculate the interpolated variations using piecewise linear interpolation.
        # The stored variations are factors (e.g., up_yield/nominal_yield).
        # The formula for a single systematic is:
        # factor = 1 + alpha * (var_up - 1)         if alpha >= 0
        # factor = 1 + alpha * (1 - var_dn)         if alpha < 0
        term_up = alphas * (process_var_up - 1.0)
        term_dn = alphas * (1.0 - process_var_dn)  # alpha is negative here

        # Select between up and down terms based on the sign of alpha
        interpolation_terms = torch.where(alphas >= 0, term_up, term_dn)

        # The total modification is the product of (1 + term) over all systematics,
        # combining their multiplicative effects.
        combined_factor = (1.0 + interpolation_terms).prod(dim=0)

        return combined_factor