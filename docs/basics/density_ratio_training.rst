Density Ratio Training
======================

Density ratio estimation is the core machine-learning step in the SBI workflow. The goal is to learn the ratio :math:`p_A(x) / p_B(x)` between two hypotheses directly from simulated data, without estimating either density individually. This is done by training a binary classifier to distinguish events drawn from each hypothesis — the classifier score is then converted to a density ratio.


How it works
------------

The ``density_ratio_trainer`` provides an end-to-end interface for training density ratio networks. Given a dataset containing events from both hypotheses (with per-event weights and binary labels), the trainer handles feature scaling, network training, optional post-hoc calibration, and a suite of diagnostic checks.

A typical training call looks like:

.. code-block:: python

   from nsbi_common_utils.training import density_ratio_trainer

   trainer = density_ratio_trainer(
       dataset=df,
       weights=weights,
       training_labels=labels,
       features=feature_list,
       features_scaling=feature_list,
       sample_name=["htautau", "ztautau"],
       output_name="htautau_vs_ztautau",
       path_to_figures="plots/",
       path_to_models="models/",
   )

   trainer.train(
       hidden_layers=3,
       neurons=64,
       number_of_epochs=200,
       batch_size=1024,
       learning_rate=1e-3,
       scalerType="StandardScaler",
       ensemble_index=0,
   )

The trained model is automatically exported to ONNX format for portable, backend-agnostic inference.


Using the fit configuration
---------------------------

In practice, many of the inputs to the trainer — training features, which processes to train, and which process serves as the reference hypothesis — are read from the :doc:`fit configuration file <fit_config>` via :class:`~nsbi_common_utils.configuration.ConfigManager`:

.. code-block:: python

   from nsbi_common_utils import configuration, datasets

   config = configuration.ConfigManager(file_path_string="config_fit.yml")

   # Training features and which to standardise
   features, features_scaling = config.get_training_features()

   # Which processes get their own density ratio network
   basis_samples = config.get_basis_samples()        # e.g. ["htautau", "ztautau"]

   # The denominator process in the density ratio
   reference_samples = config.get_reference_samples() # e.g. ["ztautau"]

   # Load data from ROOT files defined in the config
   datasets_helper = datasets.datasets(config_path="config_fit.yml", branches_to_load=features)
   dataset_dict = datasets_helper.load_datasets_from_config(load_systematics=False)

These values can also be passed manually if you are using the training APIs independently of the configuration system.


Data requirements
-----------------

The trainer expects a single DataFrame with events from both hypotheses, along with:

- **Weights** — per-event weights, normalised independently per class so each class contributes equally.
- **Labels** — ``1`` for hypothesis A (numerator) and ``0`` for hypothesis B (denominator).

The data is automatically split into training, validation, and holdout sets. The random seed and split metadata are saved to disk for reproducibility.


Feature scaling
---------------

Three scaling strategies are available via the ``scalerType`` parameter: ``"StandardScaler"``, ``"MinMax"``, and ``"PowerTransform_Yeo"``. The ``features_scaling`` argument controls which features are scaled — features not listed pass through unchanged.


Ensemble training
-----------------

To reduce variance in the learned density ratios, multiple independent networks can be trained by passing different ``ensemble_index`` values. Each ensemble member saves its own model, scaler, and metadata with an index suffix. On a cluster, ensemble members are trained in parallel via HTCondor/DAGMan.


Calibration
-----------

Raw classifier outputs may not be perfectly calibrated probabilities. The trainer supports optional post-hoc calibration using either isotonic regression or histogram-based methods. When enabled, the calibrator is saved alongside the model and applied automatically at inference time.


Diagnostics
-----------

After training, several built-in diagnostic methods help validate the quality of the learned density ratios:

- **Overtraining check** (``make_overfit_plots``) — compares score distributions between training and holdout data.
- **Calibration curve** (``make_calib_plots``) — verifies that predicted scores match true class fractions.
- **Reweighting check** (``make_reweighted_plots``) — the key closure test: reweighting hypothesis B by the learned ratio should reproduce hypothesis A.
- **Normalisation test** (``test_normalization``) — checks that :math:`\int r(x) \, p_B(x) \, dx \approx 1`.


Extending with custom models
----------------------------

The training infrastructure is not limited to the built-in ``DensityRatioLightning`` and ``MultiClassLightning`` modules. The Lightning modules, trainer classes, and utility functions (ONNX export, batched inference, calibration) are designed as independent, composable components.

To add a new model type — for example a direct density estimator based on normalising flows — you would:

1. Write a new ``pl.LightningModule`` subclass that defines the architecture, loss, and optimiser. It should expose a ``mlp`` and ``out`` attribute if you want to reuse the ONNX export utilities (``save_model``, ``convert_torch_to_onnx``) directly, or you can handle export separately.
2. Use the existing utility functions (``save_model``, ``load_trained_model``, ``predict_with_onnx``) for serialisation and inference — these work with any ONNX-compatible model.
3. Optionally write a new trainer class following the same pattern as ``density_ratio_trainer`` or ``preselection_network_trainer`` to handle data splitting, scaling, and diagnostics.

The shared utilities in ``nsbi_common_utils.training.utils`` and the callbacks/datasets in ``nsbi_common_utils.lightning_tools`` are reusable across any model type.


Where it fits in the pipeline
-----------------------------

Density ratio training happens after data preprocessing and preselection (Stages 2/2b), and before model evaluation and workspace construction (Stage 3b). The trained models produce per-event density ratio arrays that are assembled into the statistical model by the workspace builder.
