Writing a Fit Configuration
===========================

The fit configuration YAML file is the central input to the entire SBI workflow. It defines the measurement, samples, systematic uncertainties, analysis regions, training features, and pointers to trained density-ratio models. The :class:`~nsbi_common_utils.configuration.ConfigManager` reads this file and exposes its contents through a set of accessor methods that are consumed at every stage of the pipeline:

- **Data preparation** — the ``datasets`` module reads sample paths, ROOT tree names, weight branches, and systematic variation files from the config to load and organise the data.
- **Preselection training** — training features and feature scaling lists are retrieved via ``ConfigManager.get_training_features()``.
- **Density-ratio training** — basis processes (``get_basis_samples()``), the reference hypothesis (``get_reference_samples()``), training features, and region filters (``get_channel_filters()``) are all read from the config.
- **Systematic uncertainty training** — the ``Systematics`` block and ``get_samples_in_syst_for_training()`` determine which processes are affected by each variation and where the varied ROOT files are located.
- **Evaluation** — Asimov weight paths (``get_channel_asimov_weight_path()``) and trained model metadata come from the config.
- **Workspace building** — the :class:`~nsbi_common_utils.workspace_builder.WorkspaceBuilder` assembles the full statistical model from all of the above, producing a workspace dictionary that the statistical model consumes.

This page is the canonical reference for the YAML format. For the Python API, see :doc:`/api/workspace_builder`. For a hands-on walkthrough, see :doc:`model_building_example`.

Measurement and parameters
--------------------------

The ``General`` block names the measurement and declares which parameters enter the fit:

.. code-block:: yaml

   General:
     Measurement:
       Name: higgs_tautau_signal_strength
       POI: mu_htautau
       ParametersToFit:
         - mu_htautau
         - mu_ztautau
         - mu_ttbar
         - TES
         - JES

- **POI** — the parameter of interest (signal strength).
- **ParametersToFit** — only these parameters appear in the likelihood. Omitting a parameter here effectively fixes it at its nominal value.

Samples
-------

Each row in ``Samples`` is a physics process read from a ROOT file:

.. code-block:: yaml

   Samples:
     - Name: htautau
       Tree: tree_htautau
       SamplePath: ./saved_datasets/dataset_nominal.root
       Weight: weights
       UseAsReference: True
       UseAsBasis: True

- **UseAsBasis** — this process gets its own density-ratio network and normalization factor.
- **UseAsReference** — the denominator process in the density ratio :math:`r(x) = p / p_{\text{ref}}`. (Optional, users can choose to pass their own reference hypothesis when just using APIs from the toolkit).

Normalization factors
---------------------

Free multiplicative parameters (signal strengths, background norms):

.. code-block:: yaml

   NormFactors:
     - Name: mu_htautau
       Samples: htautau
       Nominal: 1
       Bounds: [0, 10]

These are **unconstrained** in the fit — no Gaussian penalty term.

Systematic uncertainties
------------------------

Shape + normalization uncertainties point to the up/down varied ROOT files:

.. code-block:: yaml

   Systematics:
     - Name: JES
       Type: NormPlusShape
       Nominal: 0
       Samples: [htautau, ztautau, ttbar]
       Up:
         - SampleName: htautau
           Path: ./saved_datasets/dataset_JES_up.root
           Tree: tree_htautau
           Weight: weights
       Dn:
         - SampleName: htautau
           Path: ./saved_datasets/dataset_JES_dn.root
           Tree: tree_htautau
           Weight: weights

- **Nominal: 0** — the nuisance parameter starts at zero (the Gaussian constraint is centred here).
- The workspace builder computes variation ratios (varied / nominal) automatically from the histograms.

Analysis regions
----------------

Regions define event selections and whether the channel is binned or unbinned:

.. code-block:: yaml

   Regions:
     # Binned control region
     - Name: CR
       Filter: presel_score < -1.0
       Variable: presel_score
       Type: binned
       Binning: [-8.75, -4.0, -2.5, -1.0]

     # Unbinned SBI signal region
     - Name: SR
       Filter: presel_score >= -1.0 & presel_score <= 4.5
       Type: unbinned
       AsimovWeights: ./saved_datasets/asimov_weights.npy
       TrainedModels:
         - SampleName: htautau
           Nominal:
             Ratios: ./saved_datasets/.../ratio_htautau.npy
           Systematics:
             - SystName: JES
               RatiosUp: ./saved_datasets/.../ratio_htautau.npy
               RatiosDn: ./saved_datasets/.../ratio_htautau.npy

For **unbinned** regions, ``TrainedModels`` points to the pre-evaluated density-ratio ``.npy`` arrays produced by the evaluation pipeline step.
For **binned** regions, the workspace builder histograms the data automatically using ``Variable`` and ``Binning``.

Training features
-----------------

Which branches from the ROOT file are used as NN inputs:

.. code-block:: yaml

   TrainingFeatures:
     - DER_mass_transverse_met_lep
     - log_DER_mass_vis
     - log_DER_pt_h

   TrainingFeaturesToStandardize:
     - DER_mass_transverse_met_lep
     - log_DER_mass_vis

Features listed in ``TrainingFeaturesToStandardize`` are z-scored before being passed to the network. Features not listed are passed through unchanged.
