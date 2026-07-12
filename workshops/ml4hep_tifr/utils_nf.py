"""Normalizing-flow models and training helpers for the ML4HEP tutorial.

The workshop notebooks intentionally keep the model and training hyperparameters
visible.  This module contains the reusable implementation: preprocessing,
model construction, checkpoint management, training, density evaluation, and
sampling in the original feature coordinates.

Two discrete normalizing-flow architectures are supported:

* ``realnvp``: affine coupling layers implemented directly in PyTorch.
* ``quadratic_spline``: piecewise rational-quadratic coupling layers provided
  by the lightweight ``nflows`` package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


_FLOW_TYPE_ALIASES = {
    "realnvp": "realnvp",
    "affine": "realnvp",
    "quadratic_spline": "quadratic_spline",
    "rational_quadratic_spline": "quadratic_spline",
    "spline": "quadratic_spline",
}


def _canonical_flow_type(flow_type: str) -> str:
    try:
        return _FLOW_TYPE_ALIASES[str(flow_type).lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(set(_FLOW_TYPE_ALIASES.values())))
        raise ValueError(
            f"Unknown flow_type={flow_type!r}. Supported choices: {supported}."
        ) from exc


@dataclass
class Standardizer:
    """Per-feature affine standardization used before flow training."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        x = np.asarray(x, dtype=np.float32)
        mean = x.mean(axis=0).astype(np.float32)
        std = x.std(axis=0).astype(np.float32)
        std = np.where(std > 1.0e-6, std, 1.0).astype(np.float32)
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float32)
        return (z * self.std + self.mean).astype(np.float32)

    @property
    def log_det_x_to_z_standardization(self) -> float:
        # z = (x - mean) / std, so log |dz/dx| = -sum(log std).
        return float(-np.log(self.std).sum())


class _MLP(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden_features: int = 128,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = n_in
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(last, hidden_features), nn.ReLU()])
            last = hidden_features
        layers.append(nn.Linear(last, n_out))
        self.net = nn.Sequential(*layers)

        # Start each affine coupling layer close to the identity map.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _AffineCoupling(nn.Module):
    def __init__(
        self,
        n_features: int,
        mask: torch.Tensor,
        hidden_features: int = 128,
        hidden_layers: int = 2,
        scale_clip: float = 1.5,
    ) -> None:
        super().__init__()
        self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.float32))
        self.net = _MLP(
            n_features,
            2 * n_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
        )
        self.scale_clip = float(scale_clip)

    def _shift_and_log_scale(
        self, x_masked: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, log_scale = self.net(x_masked).chunk(2, dim=-1)
        inv_mask = 1.0 - self.mask
        log_scale = self.scale_clip * torch.tanh(log_scale) * inv_mask
        shift = shift * inv_mask
        return shift, log_scale

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map data space to base space for one affine coupling layer."""
        x_masked = x * self.mask
        shift, log_scale = self._shift_and_log_scale(x_masked)
        z = x_masked + (1.0 - self.mask) * (x - shift) * torch.exp(-log_scale)
        return z, -log_scale.sum(dim=-1)

    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map base space to data space for one affine coupling layer."""
        z_masked = z * self.mask
        shift, log_scale = self._shift_and_log_scale(z_masked)
        x = z_masked + (1.0 - self.mask) * (z * torch.exp(log_scale) + shift)
        return x, log_scale.sum(dim=-1)


class _RealNVP(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_coupling_layers: int = 8,
        hidden_features: int = 128,
        hidden_layers: int = 2,
        scale_clip: float = 1.5,
    ) -> None:
        super().__init__()
        base_mask = torch.tensor(
            [i % 2 for i in range(n_features)], dtype=torch.float32
        )
        masks = [
            base_mask if i % 2 == 0 else 1.0 - base_mask
            for i in range(n_coupling_layers)
        ]
        self.layers = nn.ModuleList(
            [
                _AffineCoupling(
                    n_features=n_features,
                    mask=mask,
                    hidden_features=hidden_features,
                    hidden_layers=hidden_layers,
                    scale_clip=scale_clip,
                )
                for mask in masks
            ]
        )
        self.n_features = int(n_features)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        total_log_det = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            z, log_det = layer(z)
            total_log_det = total_log_det + log_det
        base_log_prob = -0.5 * (
            z.pow(2) + math.log(2.0 * math.pi)
        ).sum(dim=-1)
        return base_log_prob + total_log_det

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        z = torch.randn(
            n, self.n_features, device=next(self.parameters()).device
        )
        x = z
        for layer in reversed(self.layers):
            x, _ = layer.inverse(x)
        return x


def _build_quadratic_spline(config: Mapping[str, Any]) -> nn.Module:
    """Build a rational-quadratic spline coupling flow using ``nflows``."""
    try:
        from nflows.distributions.normal import StandardNormal
        from nflows.flows.base import Flow
        from nflows.nn.nets import ResidualNet
        from nflows.transforms.base import CompositeTransform
        from nflows.transforms.coupling import (
            PiecewiseRationalQuadraticCouplingTransform,
        )
        from nflows.transforms.permutations import ReversePermutation
        from nflows.utils.torchutils import create_alternating_binary_mask
    except ImportError as exc:
        raise ImportError(
            "The quadratic-spline flow requires the 'nflows' package. "
            "Install it with `pip install nflows`."
        ) from exc

    n_features = int(config["n_features"])
    hidden_features = int(config["hidden_features"])
    hidden_layers = int(config["hidden_layers"])
    dropout_probability = float(config.get("dropout_probability", 0.0))

    def make_net(in_features: int, out_features: int) -> nn.Module:
        return ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            num_blocks=hidden_layers,
            activation=torch.nn.functional.relu,
            dropout_probability=dropout_probability,
            use_batch_norm=False,
        )

    transforms = []
    for layer_index in range(int(config["n_coupling_layers"])):
        mask = create_alternating_binary_mask(
            features=n_features, even=(layer_index % 2 == 0)
        )
        transforms.append(
            PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=make_net,
                num_bins=int(config.get("spline_num_bins", 8)),
                tails="linear",
                tail_bound=float(config.get("spline_tail_bound", 3.0)),
                apply_unconditional_transform=False,
            )
        )
        if layer_index + 1 < int(config["n_coupling_layers"]):
            transforms.append(ReversePermutation(features=n_features))

    return Flow(
        transform=CompositeTransform(transforms),
        distribution=StandardNormal(shape=[n_features]),
    )


def build_flow(model_config: Mapping[str, Any], device: torch.device) -> nn.Module:
    """Construct the requested flow architecture from a notebook-side config."""
    config = dict(model_config)
    flow_type = _canonical_flow_type(config.get("flow_type", "realnvp"))
    config["flow_type"] = flow_type

    if flow_type == "realnvp":
        flow = _RealNVP(
            n_features=int(config["n_features"]),
            n_coupling_layers=int(config["n_coupling_layers"]),
            hidden_features=int(config["hidden_features"]),
            hidden_layers=int(config["hidden_layers"]),
            scale_clip=float(config.get("scale_clip", 1.5)),
        )
    else:
        flow = _build_quadratic_spline(config)
    return flow.to(device)


def checkpoint_path(
    sample_name: str,
    model_dir: str | Path,
    flow_type: str,
) -> Path:
    """Return a model-specific checkpoint path without architecture collisions."""
    flow_type = _canonical_flow_type(flow_type)
    return Path(model_dir) / f"{flow_type}_{sample_name}.pt"


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    # PyTorch versions differ in whether torch.load exposes weights_only.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _save_flow(
    path: Path,
    flow: nn.Module,
    scaler: Standardizer,
    features: Sequence[str],
    model_config: Mapping[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": flow.state_dict(),
            "scaler_mean": scaler.mean,
            "scaler_std": scaler.std,
            "features": list(features),
            "model_config": dict(model_config),
        },
        path,
    )
    return path


def load_flow(
    sample_name: str,
    *,
    model_dir: str | Path,
    flow_type: str,
    device: torch.device,
    expected_features: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load a trained flow and its standardization from a checkpoint."""
    path = checkpoint_path(sample_name, model_dir, flow_type)
    checkpoint = _torch_load(path, device)
    saved_features = list(checkpoint["features"])
    if expected_features is not None and saved_features != list(expected_features):
        raise ValueError(
            f"Checkpoint features {saved_features} do not match "
            f"the requested features {list(expected_features)}."
        )

    # ``config`` was the key used by the original in-notebook RealNVP. Keep
    # those checkpoints loadable after moving the implementation here.
    saved_config = checkpoint.get("model_config", checkpoint.get("config"))
    if saved_config is None:
        raise ValueError(f"Checkpoint {path} does not contain a model config.")
    model_config = dict(saved_config)
    saved_flow_type = _canonical_flow_type(
        model_config.get("flow_type", "realnvp")
    )
    if saved_flow_type != _canonical_flow_type(flow_type):
        raise ValueError(
            f"Checkpoint contains {saved_flow_type!r}, not {flow_type!r}."
        )

    flow = build_flow(model_config, device)
    flow.load_state_dict(checkpoint["state_dict"])
    flow.eval()
    scaler = Standardizer(
        mean=np.asarray(checkpoint["scaler_mean"], dtype=np.float32),
        std=np.asarray(checkpoint["scaler_std"], dtype=np.float32),
    )
    return {
        "flow": flow,
        "scaler": scaler,
        "features": saved_features,
        "model_config": model_config,
        "path": path,
    }


def _choose_training_array(
    df: pd.DataFrame,
    features: Sequence[str],
    max_train_events: int | None,
    seed: int,
) -> np.ndarray:
    if max_train_events is not None and len(df) > max_train_events:
        df = df.sample(n=max_train_events, random_state=seed).reset_index(drop=True)
    return df[list(features)].to_numpy(dtype=np.float32)


def _make_loaders(
    x_scaled: np.ndarray,
    *,
    validation_fraction: float,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    if len(x_scaled) < 2:
        raise ValueError("At least two events are required to train a flow.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    n_total = len(x_tensor)
    n_val = min(n_total - 1, max(1, int(round(validation_fraction * n_total))))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_total, generator=generator)
    train_ds = TensorDataset(x_tensor[permutation[:n_train]])
    val_ds = TensorDataset(x_tensor[permutation[n_train:]])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader


def train_flow(
    sample_name: str,
    df_train: pd.DataFrame,
    *,
    features: Sequence[str],
    model_dir: str | Path,
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    device: torch.device,
    max_train_events: int | None = None,
    load_if_available: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Train (or load) one flow using notebook-selected hyperparameters."""
    model_config = dict(model_config)
    model_config["flow_type"] = _canonical_flow_type(
        model_config.get("flow_type", "realnvp")
    )
    if int(model_config["n_features"]) != len(features):
        raise ValueError("model_config['n_features'] must match len(features).")

    path = checkpoint_path(
        sample_name, model_dir, model_config["flow_type"]
    )
    if load_if_available and path.exists():
        print(f"Loading existing {sample_name} flow from {path}")
        return load_flow(
            sample_name,
            model_dir=model_dir,
            flow_type=model_config["flow_type"],
            device=device,
            expected_features=features,
        )

    training_config = dict(training_config)
    batch_size = int(training_config["batch_size"])
    n_epochs = int(training_config["n_epochs"])
    learning_rate = float(training_config["learning_rate"])
    weight_decay = float(training_config.get("weight_decay", 0.0))
    validation_fraction = float(training_config.get("validation_fraction", 0.2))
    patience = int(training_config.get("patience", n_epochs))
    gradient_clip = float(training_config.get("gradient_clip", 5.0))
    lr_scheduler_factor = float(
        training_config.get("lr_scheduler_factor", 0.2)
    )
    lr_scheduler_patience = int(
        training_config.get("lr_scheduler_patience", 2)
    )
    min_learning_rate = float(
        training_config.get("min_learning_rate", learning_rate * 1.0e-3)
    )

    if not 0.0 < lr_scheduler_factor < 1.0:
        raise ValueError("lr_scheduler_factor must be strictly between 0 and 1.")
    if lr_scheduler_patience < 0:
        raise ValueError("lr_scheduler_patience must be non-negative.")
    if not 0.0 <= min_learning_rate <= learning_rate:
        raise ValueError(
            "min_learning_rate must be non-negative and no larger than "
            "learning_rate."
        )

    x = _choose_training_array(
        df_train,
        features=features,
        max_train_events=max_train_events,
        seed=seed,
    )
    scaler = Standardizer.fit(x)
    x_scaled = scaler.transform(x)
    train_loader, val_loader = _make_loaders(
        x_scaled,
        validation_fraction=validation_fraction,
        batch_size=batch_size,
        seed=seed,
    )

    flow = build_flow(model_config, device)
    optimizer = torch.optim.AdamW(
        flow.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        threshold=1.0e-4,
        threshold_mode="abs",
        min_lr=min_learning_rate,
    )

    best_val = np.inf
    best_state = None
    stale_epochs = 0
    print(
        f"Training {sample_name} {model_config['flow_type']} flow "
        f"on {len(x_scaled):,} events"
    )
    for epoch in range(1, n_epochs + 1):
        flow.train()
        train_losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            loss = -flow.log_prob(batch).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                flow.parameters(), max_norm=gradient_clip
            )
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        flow.eval()
        val_losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                val_loss = -flow.log_prob(batch).mean()
                val_losses.append(float(val_loss.detach().cpu()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        learning_rate_before_step = float(optimizer.param_groups[0]["lr"])
        scheduler.step(val_loss)
        current_learning_rate = float(optimizer.param_groups[0]["lr"])
        print(
            f"  epoch {epoch:03d}: train NLL = {train_loss:.4f}, "
            f"val NLL = {val_loss:.4f}, lr = {current_learning_rate:.3e}"
        )

        if val_loss < best_val - 1.0e-4:
            best_val = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in flow.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if current_learning_rate < learning_rate_before_step:
            print(
                "    reducing learning rate: "
                f"{learning_rate_before_step:.3e} -> "
                f"{current_learning_rate:.3e}"
            )
            # Give the finer learning rate a fresh opportunity to improve the
            # validation loss before applying early stopping.
            stale_epochs = 0
        elif stale_epochs >= patience:
            print(f"  early stopping after {epoch} epochs")
            break

    if best_state is not None:
        flow.load_state_dict(best_state)
    flow.eval()
    saved = _save_flow(
        path,
        flow,
        scaler,
        features=features,
        model_config=model_config,
    )
    print(f"Saved {sample_name} flow to {saved}")
    return {
        "flow": flow,
        "scaler": scaler,
        "features": list(features),
        "model_config": model_config,
        "path": saved,
    }


@torch.no_grad()
def flow_log_prob_x(
    flow_pack: Mapping[str, Any],
    x: pd.DataFrame | np.ndarray,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Evaluate log densities in the original, unstandardized coordinates."""
    features = list(flow_pack["features"])
    if isinstance(x, pd.DataFrame):
        x_array = x[features].to_numpy(dtype=np.float32)
    else:
        x_array = np.asarray(x, dtype=np.float32)

    flow = flow_pack["flow"]
    scaler = flow_pack["scaler"]
    device = next(flow.parameters()).device
    flow.eval()

    chunks = []
    for start in range(0, len(x_array), batch_size):
        x_scaled = scaler.transform(x_array[start : start + batch_size])
        x_tensor = torch.tensor(
            x_scaled, dtype=torch.float32, device=device
        )
        log_p_scaled = flow.log_prob(x_tensor).detach().cpu().numpy()
        chunks.append(
            log_p_scaled + scaler.log_det_x_to_z_standardization
        )
    if not chunks:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(chunks)


@torch.no_grad()
def flow_sample_x(
    flow_pack: Mapping[str, Any],
    n: int,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Draw flow samples and return them in the original coordinates."""
    flow = flow_pack["flow"]
    scaler = flow_pack["scaler"]
    flow.eval()

    chunks = []
    remaining = int(n)
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        sample = flow.sample(current_batch).detach().cpu().numpy()
        chunks.append(scaler.inverse(sample))
        remaining -= current_batch
    if not chunks:
        return np.empty((0, len(flow_pack["features"])), dtype=np.float32)
    return np.concatenate(chunks, axis=0)
