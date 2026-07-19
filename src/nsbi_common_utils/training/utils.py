#import libraries
import os, importlib, sys, shutil
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 
import math
import pickle 

import warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except (ImportError, AssertionError, TypeError):
    pass

import tempfile

import torch
torch.set_float32_matmul_precision("medium")
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from torch.utils.data import Subset

from pathlib import Path
from typing import Union, Dict
from joblib import dump, load

import onnx
from joblib import load
 
def save_model(lightning_model, 
               input_sample,
               path_to_save_model: Union[str, Path], 
               scaler_instance, 
               path_to_save_scaler: Union[str, Path],
               softmax_output: bool = False) -> None:
    """
    Export a trained PyTorch Lightning model to ONNX format and save the feature scaler to disk.

    Parameters
    ----------
    lightning_model : DensityRatioLightning
        Trained PyTorch Lightning model instance. Must be in eval mode or will be set to eval mode internally.

    input_sample : torch.Tensor, shape (1, n_features)
        A representative input tensor used to trace the model graph during ONNX export. Values do not affect the exported weights — only the shape matters. Typically ``torch.randn((1, len(features)))``.

    path_to_save_model : str or Path
        Destination path for the exported ``.onnx`` file.

    scaler_instance : sklearn transformer
        Fitted scaler object (e.g. ``ColumnTransformer`` wrapping ``StandardScaler``) to be serialised alongside the model so that the same preprocessing is applied at inference time.

    path_to_save_scaler : str or Path
        Destination path for the serialised scaler ``.bin`` file.

    softmax_output : bool, optional
        If ``True``, wraps the model with a softmax layer before export so that the ONNX output is a probability vector rather than raw logits. Set to ``False`` (default) for density-ratio training, where the raw sigmoid output is used directly.

    Notes
    -----
    * The scaler is serialised with ``joblib.dump`` using compression level 3.
    * ONNX export uses opset version 17 with dynamic batch size axes, so the exported model accepts any batch size at inference.
    * When ``softmax_output=True``, the wrapper accesses ``model.mlp`` and ``model.out`` directly — these attribute names must exist on the Lightning model.
    """
    lightning_model.eval()

    class ModelWithSoftmax(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            # Get logits from the model
            x = self.model.mlp(x)
            logits = self.model.out(x)
            # Apply softmax output
            return F.softmax(logits, dim=1)
        
    if softmax_output:
        lightning_model_export = ModelWithSoftmax(lightning_model)
        print("Exporting ONNX model with softmax output (probabilities)")
    else:
        lightning_model_export = lightning_model
    
    torch.onnx.export(
        lightning_model_export,
        input_sample,
        str(path_to_save_model),
        export_params=True,
        opset_version=17,
        input_names=['features'], 
        output_names=['output'],   
        dynamic_axes={
            'features': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    dump(scaler_instance, str(path_to_save_scaler), compress=True)

def load_trained_model(path_to_saved_model: Union[Path, str], 
                        path_to_saved_scaler: Union[Path, str]):
    """
    Load a previously saved ONNX model and its associated feature scaler.

    Parameters
    ----------
    path_to_saved_model : str or Path
        Path to the ``.onnx`` model file produced by :func:`save_model`.

    path_to_saved_scaler : str or Path
        Path to the ``.bin`` scaler file produced by :func:`save_model`.

    Returns
    -------
    scaler : sklearn transformer
        The deserialised scaler object. Call ``scaler.transform(data)`` to preprocess new data consistently with the training pipeline.

    model : onnx.ModelProto
        The loaded ONNX model graph. Pass this directly to :func:`predict_with_onnx` or :func:`predict_with_model`, which will create an ``onnxruntime.InferenceSession`` internally on first call.

    Notes
    -----
    * The returned ``model`` is an ``onnx.ModelProto``, not an ``onnxruntime.InferenceSession``. The session is created lazily inside :func:`predict_with_onnx` to avoid holding GPU/CPU resources when the model is not actively being used.
    """
    # Load scaler
    scaler          = load(str(path_to_saved_scaler))

    # Load ONNX model
    model           = onnx.load(str(path_to_saved_model))

    return scaler, model

def predict_with_model(data, scaler, model, calibration_model=None, use_log_loss=False):
    """
    Evaluate the trained density-ratio model on an input dataset.

    Applies feature scaling, runs ONNX inference, optionally converts from log-likelihood-ratio space to a probability score, and optionally applies the calibration layer.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset to evaluate on.

    scaler : sklearn transformer
        Fitted scaler with a ``.transform()`` method. Applied to ``dataset`` before inference. Must be the same scaler saved alongside the model via :func:`save_model`.

    model : onnx.ModelProto or onnxruntime.InferenceSession
        The ONNX model to run inference with. If a ``ModelProto`` is passed, an ``InferenceSession`` is created internally. If an ``InferenceSession`` is passed, it is used directly.

    calibration_model :
        Calibration model with ``cali_pred`` method.

    use_log_loss : bool, optional
        If ``True``, the raw model output is interpreted as :math:`\\log(p_A / p_B)` and converted to a probability score via :math:`s = \\sigma(\\log r) = 1 / (1 + r^{-1})` before returning. Must match the ``use_log_loss`` setting used during training. Default ``False``.

    Returns
    -------
    numpy.ndarray, shape (n_events,)
        Predicted scores in the range ``(0, 1)``, where values close to ``1`` indicate high probability of belonging to hypothesis A (numerator) and values close to ``0`` indicate hypothesis B (denominator). If calibration is enabled, the output is additionally clipped to ``[1e-8, 1 - 1e-8]`` for numerical safety.

    Notes
    -----
    * To obtain the density ratio :math:`r = p_A / p_B` from the returned score :math:`s`, use :math:`r = s / (1 - s)`.
    """

    pred = predict_with_onnx(data, scaler, model)

    if use_log_loss:
        pred = convert_logLR_to_score(pred)

    if calibration_model is not None:
        pred = calibration_model.cali_pred(pred)
        pred = np.clip(pred.reshape(-1), 1e-9, 1.0 - 1e-9)

    return pred

def predict_with_onnx(dataset, 
                    scaler, 
                    model, 
                    batch_size = 10_000,
                    softmax_output: bool = False):
    """
    Run batched ONNX inference on a dataset.

    Scales the input features, runs inference through the ONNX runtime in fixed-size batches to avoid memory exhaustion on large datasets, and optionally applies a calibration model to the raw outputs.

    Parameters
    ----------
    dataset : pandas.DataFrame or numpy.ndarray
        Input data. Must contain the feature columns in the same order used during training. Additional columns are ignored if a DataFrame is passed, provided the scaler was fitted with named columns.

    scaler : sklearn transformer
        Fitted scaler with a ``.transform()`` method. Applied to ``dataset`` before inference. Must be the same scaler saved alongside the model via :func:`save_model`.

    model : onnx.ModelProto or onnxruntime.InferenceSession
        The ONNX model to run inference with. If a ``ModelProto`` is passed, an ``InferenceSession`` is created internally. If an ``InferenceSession`` is passed, it is used directly.

    batch_size : int, optional
        Number of events processed per inference call. Reduce this if GPU memory is limited. Default ``10_000``.

    softmax_output : bool, optional
        If ``False`` (default), the output array is flattened to shape ``(n_events,)``. If ``True``, the 2D output ``(n_events, n_classes)`` is preserved, as returned by a model exported with softmax.

    Returns
    -------
    preds : numpy.ndarray
        - Shape ``(n_events,)`` when ``softmax_output=False``.
        - Shape ``(n_events, n_classes)`` when ``softmax_output=True``.

        Dtype is ``float32``.

    Raises
    ------
    TypeError
        If ``model`` is neither an ``onnx.ModelProto`` nor an ``onnxruntime.InferenceSession``.

    Notes
    -----
    * The ONNX session is configured with ``intra_op_num_threads=1`` and ``inter_op_num_threads=1``. This is intentional for HTCondor jobs where CPU resources are explicitly requested — unconstrained threading can cause resource contention across concurrent jobs on the same node.
    * CUDA execution is attempted first; the runtime falls back to CPU automatically if no compatible GPU is available.
    """
    import onnxruntime as rt

    sess_opts = rt.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1

    if isinstance(model, onnx.ModelProto):
        model = rt.InferenceSession(model.SerializeToString(), 
                                    sess_options = sess_opts,
                                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    
    elif isinstance(model, rt.InferenceSession):
        model = model
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

    scaled_dataset  = scaler.transform(dataset)
    n_samples       = len(scaled_dataset)
    
    input_name      = model.get_inputs()[0].name
    output_name     = model.get_outputs()[0].name
    
    first_batch     = scaled_dataset[:min(batch_size, n_samples)]
    first_pred      = model.run([output_name], {input_name: first_batch})[0]
    
    if len(first_pred.shape) > 1:
        output_shape = (n_samples, first_pred.shape[1])
    else:
        output_shape = (n_samples,)
    
    preds = np.empty(output_shape, dtype=np.float32)
    preds[:len(first_batch)] = first_pred
    
    # Process remaining batches
    for i in range(batch_size, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch = scaled_dataset[i:end_idx]
        preds[i:end_idx] = model.run([output_name], {input_name: batch})[0]

    if not softmax_output:
        preds = preds.reshape(preds.shape[0],)
    
    return preds

def convert_torch_to_onnx(lightning_model, input_dim, opset=17):
    """
    Convert a trained PyTorch Lightning model to an ``onnx.ModelProto`` in memory, without permanently writing to disk.

    Parameters
    ----------
    lightning_model : DensityRatioLightning
        Trained model to convert. Must have parameters accessible via ``model.parameters()`` to determine the target device.

    input_dim : int
        Number of input features. Used to construct a random dummy input tensor for graph tracing.

    opset : int, optional
        ONNX opset version to target during export. Default ``17``.

    Returns
    -------
    onnx.ModelProto
        The exported ONNX model loaded into memory and ready to pass to :func:`predict_with_onnx`.

    Notes
    -----
    * A temporary ``.onnx`` file is written to the system's temp directory during export and deleted immediately after loading. The returned object is fully in-memory.
    * Dynamic batch axes are set for both input and output so the returned model accepts any batch size at inference.
    * This function differs from :func:`save_model` in that it does not persist the model to a user-specified path and does not handle scaler serialisation. Use :func:`save_model` when you need to save model artefacts for later reuse.
    """
    lightning_model.eval()

    dummy = torch.randn(1, input_dim, device=next(lightning_model.parameters()).device)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name

    torch.onnx.export(
        lightning_model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"}
        },
        opset_version=opset
    )

    return onnx.load(onnx_path)


def convert_logLR_to_score(logLR):
    """
    Convert a log-likelihood ratio to a probability score.

    Maps :math:`\\log(p_A / p_B)` to the relative probability
    :math:`s = p_A / (p_A + p_B)` via the sigmoid function:

    .. math::

        s = \\frac{1}{1 + e^{-\\log(p_A/p_B)}}

    Parameters
    ----------
    logLR : numpy.ndarray
        Array of log-likelihood ratio values, unbounded in range.

    Returns
    -------
    numpy.ndarray
        Probability scores in the range ``(0, 1)``.

    Notes
    -----
    * Use this function when the model was trained with ``use_log_loss=True``,
      which causes the network to regress :math:`\\log(p_A/p_B)` directly
      rather than a classification score. The output of this function is
      compatible with downstream methods that expect scores in ``(0, 1)``.
    * To recover the density ratio from the score, use
      :func:`convert_score_to_ratio`.
    """
    return 1.0/(1.0+np.exp(-logLR))

def convert_score_to_ratio(score):
    """
    Convert a probability score to a density ratio.

    Given a classifier score :math:`s = p_A / (p_A + p_B)`, returns the
    density ratio :math:`r = p_A / p_B` via:

    .. math::

        r = \\frac{s}{1 - s}

    Parameters
    ----------
    score : numpy.ndarray
        Probability scores in the range ``(0, 1)``. Values at exactly ``0``
        or ``1`` will produce ``0`` or ``inf`` respectively — clip inputs
        to a safe range such as ``[1e-9, 1 - 1e-9]`` if needed.

    Returns
    -------
    numpy.ndarray
        Density ratio values :math:`p_A / p_B`, unbounded above.
    """
    return score / (1.0 - score)

