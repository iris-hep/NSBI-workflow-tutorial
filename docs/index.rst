Toolkit for Simulation-Based Inference
=======================================

``nsbi-common-utils`` provides building blocks for **Simulation-Based Inference (SBI)** analysis tailored to the statistical models used at the ATLAS and CMS experiments. It implements a semi-parametric approach to SBI in which the statistical model combines non-parametric density-ratio estimates with parametric HistFactory-style systematic uncertainties. The toolkit is modular, offering APIs for dataset preparation, density-ratio estimation, model building, and profiled-likelihood-ratio fitting.

Why Simulation-Based Inference?
-------------------------------

Traditional LHC analyses compress high-dimensional collision data into summary observables, which can discard information sensitive to the parameters of interest. SBI uses neural networks to approximate the likelihood ratio directly from the full feature space — no hand-crafted summaries required. This toolkit makes SBI practical for LHC-scale analyses by embedding neural density-ratio estimates inside a HistFactory-style statistical model, combining unbinned SBI regions with traditional binned template regions in a single profiled-likelihood fit.

See the :doc:`basics/overview` for a detailed introduction, or the method and measurement papers: `arXiv:2412.01600 <https://arxiv.org/abs/2412.01600>`_, `arXiv:2412.01548 <https://arxiv.org/abs/2412.01548>`_.

Key features
------------

- **Training pipeline** — density-ratio neural network training with PyTorch Lightning, ensemble support, and calibration and reweighting diagnostics.
- **Statistical models** — pyhf-like workspace specification supporting both binned (template-based) and unbinned (SBI-style) analysis regions with HistFactory-style systematic uncertainties, compiled to JAX for fast NLL evaluation and automatic differentiation.
- **Inference engine** — profile-likelihood fits and NLL scans via iminuit with analytic gradients, plus plotting utilities.
- **Configuration-driven** — a single YAML :doc:`fit configuration <basics/fit_config>` defines samples, features, systematics, and regions, and is consumed across all stages of the pipeline via :class:`~nsbi_common_utils.configuration.ConfigManager`.
- **Workflow integration** — HTCondor/DAGMan job descriptions for large-scale cluster submission, with end-to-end example pipelines.

Getting started
---------------

Install the package with pip:

.. code-block:: bash

   pip install 'nsbi-common-utils @ git+https://github.com/iris-hep/NSBI-workflow-tutorial.git'

Or, for a full environment with all dependencies (recommended):

.. code-block:: bash

   pixi install -e nsbi-env      # CPU-only
   pixi install -e nsbi-env-gpu  # with CUDA support


Statistical Model Building with SBI
-----------------------------------

Once installed, building a statistical model and running a fit takes just a few lines:

.. code-block:: python

   from nsbi_common_utils import workspace_builder, models, inference

   # Build workspace from the fit configuration (also used for data loading, training, etc.)
   ws = workspace_builder.WorkspaceBuilder("config_fit.yml").build()

   # Create the statistical model (JAX-compiled NLL)
   model = models.sbi_parametric_model(workspace=ws, measurement_to_fit="my_meas")
   params, init_vals = model.get_model_parameters()

   # Fit and profile
   fitter = inference.inference(
       model_nll=model.model,
       list_parameters=params,
       initial_parameter_values=init_vals,
   )
   fitter.perform_fit()

For a hands-on walkthrough, see the :doc:`basics/model_building_example`.

.. toctree::
   :maxdepth: 2
   :caption: Basics

   basics/overview
   basics/density_ratio_training
   basics/preselection_training
   basics/fit_config
   basics/workflow
   basics/model_building_example

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/training
   api/workspace_builder
   api/model
   api/inference
