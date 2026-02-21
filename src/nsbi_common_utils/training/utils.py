#import libraries
import os, importlib, sys, shutil
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 
import math
import pickle 

import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import torch
torch.set_float32_matmul_precision("high")
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from torch.utils.data import Subset

from nsbi_common_utils.lightning_tools import MultiClassLightning, DensityRatioLightning, PrintEpochMetrics, LossHistory, WeightedTensorDataset

from pathlib import Path
from typing import Union, Dict
from joblib import dump, load

import onnx
import onnxruntime as rt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.compose import ColumnTransformer

from nsbi_common_utils.calibration import HistogramCalibrator, IsotonicCalibrator
from nsbi_common_utils.plotting import plot_loss, plot_all_features, plot_all_features, plot_reweighted, plot_calibration_curve, plot_calibration_curve_ratio, plot_overfit_side_by_side
 
def save_model(lightning_model, 
               input_sample,
               path_to_save_model: Union[str, Path], 
               scaler_instance, 
               path_to_save_scaler: Union[str, Path],
               softmax_output: bool = False) -> None:
    
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

    # Load scaler
    scaler          = load(str(path_to_saved_scaler))

    # Load ONNX model
    model           = onnx.load(str(path_to_saved_model))

    return scaler, model


def predict_with_onnx(dataset, scaler, model, batch_size = 10_000):

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
    
    return preds

def convert_torch_to_onnx(lightning_model, input_dim, opset=17):

    lightning_model.eval()

    dummy = torch.randn(1, input_dim, device=next(lightning_model.parameters()).device)

    onnx_path = "__tmp_model.onnx"

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
    '''
    Convert regressed logLR into relative probabilities for compatibility with other methods
    '''
    return 1.0/(1.0+np.exp(-logLR))

