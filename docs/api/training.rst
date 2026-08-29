Training API
============

Density-ratio estimation
------------------------

.. autoclass:: nsbi_common_utils.training.neural_ratio_estimation.density_ratio_trainer
   :members: train, make_overfit_plots, make_calib_plots, make_reweighted_plots, test_normalization

Dataset preparation
-------------------

The training set for each basis process is assembled by :meth:`~nsbi_common_utils.datasets.datasets.prepare_basis_training_dataset`. Its ``reference_priors`` argument determines which samples make up :math:`p_{\text{ref}}` and how each is weighted (auto-yield, numeric, ``{cap: M}``, or exclude); the docstring below documents the full spec.

.. note::

   Using this class is **not required**. The ``datasets`` helper (and the ``reference_priors`` mechanism it exposes) is a convenience for the example pipeline's config-driven workflow. The training APIs — :class:`~nsbi_common_utils.training.neural_ratio_estimation.density_ratio_trainer` and the lower-level utilities — accept a plain ``pandas.DataFrame`` with the features, a numpy array of per-event weights, and a numpy array of class labels (``1`` for the numerator process, ``0`` for the reference). You can build that DataFrame however you like — your own reference mixture, your own re-weighting scheme, a synthetic toy reference — and hand it straight to the trainer. The helper below is one way to assemble those inputs; it is not a hard dependency.

.. autoclass:: nsbi_common_utils.datasets.datasets
   :members: prepare_basis_training_dataset, load_datasets_from_config, save_dataset_to_ntuple, filter_region_dataset, add_appended_branches, split_by_fold

Preselection network
--------------------

.. autoclass:: nsbi_common_utils.training.preselection_training.preselection_network_trainer
   :members: train, assign_trained_model, predict

Utility functions
-----------------

.. autofunction:: nsbi_common_utils.training.utils.save_model

.. autofunction:: nsbi_common_utils.training.utils.load_trained_model

.. autofunction:: nsbi_common_utils.training.utils.predict_with_model

.. autofunction:: nsbi_common_utils.training.utils.predict_with_onnx

.. autofunction:: nsbi_common_utils.training.utils.convert_torch_to_onnx

.. autofunction:: nsbi_common_utils.training.utils.convert_logLR_to_score

.. autofunction:: nsbi_common_utils.training.utils.convert_score_to_ratio