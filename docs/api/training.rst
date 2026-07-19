Training API
============

Density-ratio estimation
------------------------

.. autoclass:: nsbi_common_utils.training.neural_ratio_estimation.density_ratio_trainer
   :members: train, make_overfit_plots, make_calib_plots, make_reweighted_plots, test_normalization

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