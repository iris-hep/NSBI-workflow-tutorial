Model Building Example
======================

The ``examples/model_building_example/`` directory provides a **standalone** walkthrough of workspace construction, model building, and profile-likelihood fitting — without requiring the full training pipeline. It is the recommended starting point for understanding how the fitting stage works.

Prerequisites
-------------

You need four ingredients, all of which are included in the example directory (via symlinks to the main dataset):

1. A **fit configuration** YAML file (``config_fit_nsbi.yml`` or ``config_fit_histogram.yml``).
2. **Pre-computed density-ratio** ``.npy`` files (nominal + systematic variations) — produced by the training pipeline or any external source.
3. **Asimov weights** (or real-data weights) for the unbinned region.
4. **ROOT files** containing the MC samples used by binned channels.

.. note::

   The ``saved_datasets/`` directory uses **symlinks** into the parent ``FAIR_universe_Higgs_tautau/saved_datasets/`` directory to avoid duplicating large files. After a fresh ``git clone``, make sure to run ``git lfs pull`` so that the LFS-tracked ROOT and NumPy files are downloaded, not just pointer stubs.

Directory layout
----------------

.. code-block:: text

   model_building_example/
     config_fit_histogram.yml          # Binned-only fit configuration
     config_fit_nsbi.yml               # Combined binned + unbinned (NSBI) fit
     1_workspace_building.ipynb        # Notebook: build workspaces from config
     2_parameter_fitting.ipynb         # Notebook: fit and profile scan
     saved_datasets/
       asimov_weights.npy              # Per-event weights (unbinned region)
       dataset_nominal.root            # Nominal MC (binned channels)
       dataset_JES_up.root             # Systematic variation ROOT files
       dataset_JES_dn.root
       dataset_TES_up.root
       dataset_TES_dn.root
       output_training_nominal/
         output_ratios_<sample>/
           ratio_<sample>.npy          # Nominal density ratios
       output_training_systematics/
         output_ratios_<sample>_<syst>_<dir>/
           ratio_<sample>.npy          # Systematic density ratios

Quick start
-----------

.. code-block:: python

   from nsbi_common_utils import workspace_builder, models, inference

   # 1. Build a workspace from the YAML config
   ws = workspace_builder.WorkspaceBuilder(
       config_path="config_fit_nsbi.yml"
   ).build()

   # 2. Initialise the statistical model (JAX-compiled NLL)
   model = models.sbi_parametric_model(
       workspace=ws, measurement_to_fit="my_measurement"
   )

   # 3. Fit
   params, init_vals = model.get_model_parameters()
   fitter = inference.inference(
       model_nll=model.model,
       model_grad=model.model_grad,
       initial_values=init_vals,
       list_parameters=params,
       num_unconstrained_params=model.num_unconstrained_param,
   )
   fitter.perform_fit()

Step 1 — Build the workspace
-----------------------------

The :class:`~nsbi_common_utils.workspace_builder.WorkspaceBuilder` reads the YAML config, loads ROOT datasets and density-ratio arrays, and assembles a JSON-serialisable workspace dictionary:

.. code-block:: python

   builder = workspace_builder.WorkspaceBuilder(config_path="config_fit_nsbi.yml")
   ws = builder.build()

   # Optionally persist to disk so you can skip this step next time
   builder.dump_workspace(ws, "workspace_nsbi.json")

   # Re-load later without re-reading ROOT files
   ws = workspace_builder.WorkspaceBuilder.load_workspace("workspace_nsbi.json")

See :doc:`/api/workspace_builder` for the full API.

Step 2 — Initialise the model
-------------------------------

:class:`~nsbi_common_utils.models.sbi_parametric_model` parses the workspace, stacks all histogram yields and density-ratio arrays onto the JAX device, and compiles a JIT-optimised negative log-likelihood function:

.. code-block:: python

   model = models.sbi_parametric_model(
       workspace=ws,
       measurement_to_fit="NSBI_measurement",
   )

   # Inspect the parameter ordering and starting values
   param_names, init_values = model.get_model_parameters()

The compiled NLL is exposed as ``model.model(param_array)`` and its analytical gradient as ``model.model_grad(param_array)``.

See :doc:`/api/model` for the full API.

Step 3 — Fit and profile scan
-------------------------------

:class:`~nsbi_common_utils.inference.inference` wraps ``iminuit`` to perform the minimisation and profile-likelihood scans:

.. code-block:: python

   fitter = inference.inference(
       model_nll=model.model,
       model_grad=model.model_grad,
       initial_values=init_vals,
       list_parameters=param_names,
       num_unconstrained_params=model.num_unconstrained_param,
   )

   # Global fit
   fitter.perform_fit()

   # Profile likelihood scan of the POI
   pts, nll, pts_stat, nll_stat = fitter.perform_profile_scan(
       parameter_name="mu_htautau",
       bound_range=(0, 3),
       size=50,
       doStatOnly=True,
   )

See :doc:`/api/inference` for the full API.

Fit configuration
-----------------

The YAML config defines five sections consumed by the workspace builder. See :doc:`fit_config` for the full specification; the key points are summarised here.

**Measurement** — which parameters to fit and the parameter of interest (POI).

**Samples** — physics processes (signal, backgrounds) with paths to ROOT files and tree names.

**NormFactors** — free normalisation parameters (one per sample or shared).

**Systematics** — nuisance parameters with paths to up/down ROOT variation files. Currently only ``NormPlusShape`` is supported.

**Regions** — analysis regions tagged as ``binned`` (control/signal regions built from histograms) or ``unbinned`` (signal region using density ratios). Unbinned regions reference the trained model outputs:

.. code-block:: yaml

   Regions:
   - Name: SR
     Type: unbinned
     AsimovWeights: ./saved_datasets/asimov_weights.npy
     TrainedModels:
       - SampleName: htautau
         Nominal:
           Ratios: ./saved_datasets/output_training_nominal/output_ratios_htautau/ratio_htautau.npy
         Systematics:
           - SystName: JES
             RatiosUp: .../output_ratios_htautau_JES_Up/ratio_htautau.npy
             RatiosDn: .../output_ratios_htautau_JES_Dn/ratio_htautau.npy

Notebooks
---------

The example ships with two Jupyter notebooks:

- **1_workspace_building.ipynb** — walks through workspace construction for both the histogram-only and NSBI configurations, and serialises the workspaces to JSON.
- **2_parameter_fitting.ipynb** — loads the workspaces, initialises both models, performs global fits (with JAX autodiff gradients via ``model_grad``), runs profile-likelihood scans, and plots an NSBI-vs-histogram sensitivity comparison.
