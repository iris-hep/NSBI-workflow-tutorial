from __future__ import annotations

import argparse
import copy
import gc
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import multiprocessing as mp
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
CACHE_DIR = Path(__file__).with_name(".cache")
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
(CACHE_DIR / "matplotlib").mkdir(exist_ok=True)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import torch

# fix weight initialization seed
torch.manual_seed(42)

import uproot
from matplotlib import pyplot as plt

from nsbi_common_utils.lightning_tools import DensityRatioLightning
from nsbi_common_utils.training import (
    convert_score_to_ratio,
    density_ratio_trainer,
    predict_with_model,
)


# FEATURE_LIST = ["pt_j1", "delta_phi_jj", "met"]
# FEATURE_LIST = [
#     'pt_j1', 'eta_j1', 'phi_j1', 'e_j1', 'pt_j2', 'eta_j2', 'phi_j2',
#     'e_j2', 'm_jj', 'delta_eta_jj', 'delta_phi_jj', 'pt_a1', 'eta_a1',
#     'phi_a1', 'e_a1', 'pt_a2', 'eta_a2', 'phi_a2', 'e_a2', 'm_aa', 'pt_aa',
# ]

FEATURE_LIST = [
    "pt_j1",
    "eta_j1",
    "phi_j1",
    "pt_j2",
    "eta_j2",
    "phi_j2",
    "m_jj",
    "delta_eta_jj",
    "delta_phi_jj",
    "pt_a1",
    "eta_a1",
    "phi_a1",
    "pt_a2",
    "eta_a2",
    "phi_a2",
    "m_aa",
    "pt_aa",
]
DEFAULT_SYSTEMATIC_WEIGHT = "weight_scale_muf_nuisance_param_0_up"
MSE_DATASET_SIZE = 100_000
DEFAULT_DATASET_SIZES = [
    # 100,
    # 250,
    # 500,
    1_000,
    2_500,
    5_000,
    7_500,
    10_000,
    25_000,
    50_000,
    75_000,
    100_000,
    150_000,
    200_000,
    250_000,
    300_000,
    350_000,
    400_000,
    450_000,
    500_000,
    550_000,
    600_000,
    650_000,
    700_000,
    750_000,
    800_000,
    850_000,
    # 875_000,
]
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("artifacts") / "lora_exploration"

NOMINAL_TRAINING_KWARGS = {
    "hidden_layers": 4,
    "neurons": 256,
    "number_of_epochs": 20,
    "batch_size": 1024,
    "learning_rate": 1e-3,
    "scalerType": "StandardScaler",
    "ensemble_index": 0,
    "lr_scheduler": "step",
    "rnd_seed": 42,
    #
    "validation_split": 0.20,
    "holdout_split": 0.0,
}

SYSTEMATIC_TRAINING_KWARGS = {
    "hidden_layers": 4,
    "neurons": 256,
    "number_of_epochs": 20,
    "batch_size": 1024,
    "learning_rate": 1e-3,
    "scalerType": "StandardScaler",
    "ensemble_index": 0,
    "lr_scheduler": "step",
    "rnd_seed": 42,
    #
    "validation_split": 0.20,
    "holdout_split": 0.0,
}

LORA_KWARGS = {
    "lora_rank": 4,
    "lora_alpha": 8,
    "number_of_epochs": 20,
    "batch_size": 1024,
    "learning_rate": 1e-3,
    "scalerType": "StandardScaler",
    "ensemble_index": 0,
    "lr_scheduler": "step",
    "rnd_seed": 42,
    #
    "validation_split": 0.20,
    "holdout_split": 0.0,
}


class Benchmark(Enum):
    SM = 0
    W = 1
    NEG_W = 2
    WW = 3
    NEG_WW = 4


class EFTCoefficients(NamedTuple):
    cwl2: float
    cpwl2: float


benchmarks_to_eftcoeffs = {
    Benchmark.SM: EFTCoefficients(0.0, 0.0),
    Benchmark.W: EFTCoefficients(15.2, 0.1),
    Benchmark.NEG_W: EFTCoefficients(-15.2, 0.2),
    Benchmark.WW: EFTCoefficients(0.3, 15.1),
    Benchmark.NEG_WW: EFTCoefficients(0.4, -15.3),
}

benchmarks_to_name = {
    Benchmark.SM: "sm",
    Benchmark.W: "w",
    Benchmark.NEG_W: "neg_w",
    Benchmark.WW: "ww",
    Benchmark.NEG_WW: "neg_ww",
}

name_to_benchmark = {name: benchmark for benchmark, name in benchmarks_to_name.items()}


@dataclass(frozen=True)
class ArtifactPaths:
    base_dir: Path
    models_dir: Path
    plots_dir: Path


@dataclass(frozen=True)
class ComparisonTask:
    root_file: Path
    benchmark_name: str
    systematic_weight: str
    output_dir: Path
    target_dataset_size: int  # Size for target/benchmark dataset
    sm_dataset_size: int | None  # Size for SM dataset (None = use all available)
    mse_dataset_size: int
    seed: int
    nominal_bundle_path: Path
    systematic_training_kwargs: dict
    lora_kwargs: dict
    # Note: original_weight_sums is passed as a separate argument to run_comparison_task
    # to avoid pickle issues with multiprocessing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone version of the LoRA exploration notebook for SM-vs-benchmark training."
    )
    parser.add_argument(
        "root_file",
        type=Path,
        help="Path to the ROOT file containing the 'Events' tree.",
    )
    parser.add_argument(
        "--benchmark",
        "--target-benchmark",
        choices=sorted(name for name in name_to_benchmark if name != "sm"),
        default="ww",
        dest="benchmark",
        help=(
            "Target benchmark dataset to compare against the SM reference. "
            "This choice is used consistently for nominal training, "
            "systematic training from scratch, LoRA finetuning, and the "
            "dataset-size MSE sweep."
        ),
    )
    parser.add_argument(
        "--systematic-weight",
        "--systematic-variation",
        default=DEFAULT_SYSTEMATIC_WEIGHT,
        dest="systematic_weight",
        help=(
            "Systematic variation weight column to use throughout the workflow. "
            "This choice is used consistently for systematic training from "
            "scratch, LoRA finetuning, the dataset-size MSE sweep, and the "
            "final comparison plot."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saved models, plots, and comparison tables.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the one-time dataset shuffle used before prefix slicing. Ignored if --seeds is provided.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        metavar="SEED",
        help=(
            "List of random seeds to run the dataset-size sweep over. "
            "Each seed produces a separate CSV file. If provided, overrides --seed for the sweep."
        ),
    )
    parser.add_argument(
        "--compare-sizes",
        nargs="*",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Optional training-pool event counts per benchmark for the scratch-vs-LoRA "
            f"comparison. The fixed {MSE_DATASET_SIZE:,}-event MSE sample is reserved first."
        ),
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Disable diagnostic reweighted plots for the main training stages.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=12,
        help="Number of worker processes for the dataset-size comparison. Use 0 for auto.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of epochs used for nominal, scratch, and LoRA training.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override the learning rate used for nominal, scratch, and LoRA training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the batch size used for nominal, scratch, and LoRA training.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="Override the LoRA rank used for finetuning.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="Override the LoRA alpha used for finetuning. Defaults to 2 * lora-rank when omitted.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        default=None,
        help="Override the number of hidden layers used for nominal and scratch training.",
    )
    parser.add_argument(
        "--neurons",
        type=int,
        default=None,
        help="Override the number of neurons per hidden layer used for nominal and scratch training.",
    )
    parser.add_argument(
        "--callback-patience",
        type=int,
        default=None,
        help="Override the early-stopping patience in epochs used for nominal, scratch, and LoRA training.",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=("step", "plateau"),
        default=None,
        help="Override the learning-rate scheduler used for nominal, scratch, and LoRA training.",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=None,
        help=(
            "Override the scheduler patience in epochs. When omitted, the scheduler "
            "reuses --callback-patience or the training default."
        ),
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=None,
        help="Override the learning-rate reduction factor used by the scheduler.",
    )
    return parser.parse_args()


def phase_space_cuts(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["pt_j1"] < 500)
        & (df["pt_j2"] < 250)
        & (df["met"] < 100)
        & (df["e_a1"] < 2000)
        & (df["e_a2"] < 2000)
        & (df["e_j1"] < 4000)
        & (df["e_j1"] < 3500)
        & (np.abs(df["eta_a1"]) < 3.5)
        & (np.abs(df["eta_a2"]) < 3.5)
        & (df["m_jj"] < 4000)
        & (df["pt_a1"] < 400)
        & (df["pt_a2"] < 200)
        & (df["pt_aa"] < 500)
    ].copy()


def load_datasets(
    root_file: Path, shuffle_seed: int | None = None
) -> dict[Benchmark, pd.DataFrame]:
    """Load benchmark-partitioned datasets.

    When ``shuffle_seed`` is provided, each benchmark dataframe is shuffled
    exactly once here. All later dataset-size sweeps use deterministic prefix
    slices of these shuffled frames, so larger dataset sizes are guaranteed to
    contain the events from all smaller sizes.
    """
    resolved_root_file = root_file.expanduser().resolve()
    tree = uproot.open(resolved_root_file)["Events"]
    events = tree.arrays(library="pd")
    print(f"Loaded {len(events):,} total events from '{resolved_root_file}'.")
    print("Available columns:", sorted(events.columns.tolist()))
    datasets = {
        benchmark: events[events["sampling_benchmark_id"] == benchmark.value].copy()
        for benchmark in Benchmark
    }
    if shuffle_seed is not None:
        datasets = {
            benchmark: df.sample(frac=1.0, random_state=shuffle_seed).reset_index(
                drop=True
            )
            for benchmark, df in datasets.items()
        }
    return datasets


def split_training_and_mse_datasets(
    datasets: dict[Benchmark, pd.DataFrame],
    mse_dataset_size: int = MSE_DATASET_SIZE,
    return_weight_sums: bool = False,
) -> tuple[
    dict[Benchmark, pd.DataFrame],
    dict[Benchmark, pd.DataFrame],
    dict[Benchmark, float] | None,
]:
    """Reserve a fixed post-cut MSE sample and return the remaining training pool.

    The split is done after the one-time benchmark-level shuffle and after
    phase-space cuts, so the MSE sample contains exactly ``mse_dataset_size``
    usable events per benchmark. Dataset-size sweeps then slice only from the
    returned training pool, making each requested size the exact number of
    training-pool events per benchmark before the train/validation split.

    If return_weight_sums=True, also returns the original weight sums from the
    full post-cut dataset (including MSE holdout) for proper cross-section normalization.
    """
    training_datasets: dict[Benchmark, pd.DataFrame] = {}
    mse_datasets: dict[Benchmark, pd.DataFrame] = {}
    original_weight_sums: dict[Benchmark, float] = {}

    for benchmark, df in datasets.items():
        cut_df = phase_space_cuts(df).reset_index(drop=True)
        if len(cut_df) <= mse_dataset_size:
            raise ValueError(
                f"Need more than {mse_dataset_size} post-cut events for benchmark "
                f"'{benchmarks_to_name[benchmark]}' to reserve the fixed MSE sample, "
                f"but only {len(cut_df)} are available."
            )

        # Record original weight sum from full post-cut dataset (including MSE holdout)
        original_weight_sums[benchmark] = get_weight_sum_for_normalization(cut_df)

        mse_datasets[benchmark] = cut_df.iloc[:mse_dataset_size].copy()
        training_datasets[benchmark] = (
            cut_df.iloc[mse_dataset_size:].reset_index(drop=True).copy()
        )

    if return_weight_sums:
        return training_datasets, mse_datasets, original_weight_sums
    return training_datasets, mse_datasets, None


def stage_artifacts(
    output_dir: Path, benchmark: Benchmark, stage_name: str
) -> ArtifactPaths:
    base_dir = output_dir / benchmarks_to_name[benchmark] / stage_name
    models_dir = base_dir / "models"
    plots_dir = base_dir / "plots"
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return ArtifactPaths(base_dir=base_dir, models_dir=models_dir, plots_dir=plots_dir)


def trainer_output_paths(paths: ArtifactPaths) -> tuple[str, str]:
    return f"{paths.plots_dir.as_posix()}/", f"{paths.models_dir.as_posix()}/"


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def pretty_systematic_name(weight_column: str) -> str:
    match = re.match(r"^weight_scale_(.+)_param_0_(.+)$", weight_column)
    if match is None:
        return weight_column
    nuisance_name, direction = match.groups()
    return f"{nuisance_name} {direction}"


def prepare_training(
    datasets: dict[Benchmark, pd.DataFrame],
    benchmark: Benchmark,
    artifacts: ArtifactPaths,
    systematic_weight: str | None = None,
) -> tuple[density_ratio_trainer, pd.DataFrame]:
    reference = phase_space_cuts(datasets[Benchmark.SM])
    target = phase_space_cuts(datasets[benchmark])
    df = pd.concat([reference, target], ignore_index=True).copy()

    target_weight_column = f"weight_{benchmarks_to_name[benchmark]}"
    target_mask = df["sampling_benchmark_id"] == benchmark.value

    if systematic_weight is None:
        weights = np.where(target_mask, df[target_weight_column], df["weight_sm"])
        target_label = f"{benchmarks_to_name[benchmark]} (nominal)"
        output_name = f"ref_vs_{benchmarks_to_name[benchmark]}__nominal"
    else:
        if systematic_weight not in df.columns:
            raise KeyError(
                f"Weight column '{systematic_weight}' is not present in the dataset."
            )
        scale_factors = df[systematic_weight] / df["weight_sm"]
        weights = np.where(
            target_mask,
            df[target_weight_column] * scale_factors,
            df["weight_sm"],
        )
        target_label = f"{benchmarks_to_name[benchmark]} ({pretty_systematic_name(systematic_weight)})"
        output_name = f"ref_vs_{benchmarks_to_name[benchmark]}__{systematic_weight}"

    training_labels = np.where(target_mask, 1, 0).astype(np.int32)
    df["weights"] = np.asarray(weights, dtype=np.float64)
    df["train_labels"] = training_labels

    plots_dir, models_dir = trainer_output_paths(artifacts)
    trainer = density_ratio_trainer(
        dataset=df,
        weights=df["weights"].to_numpy(),
        training_labels=training_labels,
        features=FEATURE_LIST,
        features_scaling=FEATURE_LIST,
        sample_name=[target_label, "sm"],
        output_name=output_name,
        path_to_figures=plots_dir,
        path_to_models=models_dir,
    )
    trainer.skip_training_loss_plot = True
    trainer.use_torch_inference = True
    return trainer, df


def true_ratio(
    datasets: dict[Benchmark, pd.DataFrame],
    benchmark: Benchmark,
    systematic_weight: str | None = None,
) -> np.ndarray:
    sm_events = phase_space_cuts(datasets[Benchmark.SM])
    target_weight_column = f"weight_{benchmarks_to_name[benchmark]}"

    if systematic_weight is None:
        numerator = sm_events[target_weight_column]
    else:
        if systematic_weight not in sm_events.columns:
            raise KeyError(
                f"Weight column '{systematic_weight}' is not present in the dataset."
            )
        scale_factors = sm_events[systematic_weight] / sm_events["weight_sm"]
        numerator = sm_events[target_weight_column] * scale_factors

    denominator = sm_events["weight_sm"]
    return np.asarray(numerator / denominator, dtype=np.float64)


def ratio_mse(
    model,
    scaler,
    datasets: dict[Benchmark, pd.DataFrame],
    benchmark: Benchmark,
    systematic_weight: str | None = None,
) -> float:
    sm_events = phase_space_cuts(datasets[Benchmark.SM])
    score_pred = predict_with_model(
        sm_events[FEATURE_LIST].astype("float32"), scaler, model
    )
    ratio_pred = np.asarray(convert_score_to_ratio(score_pred), dtype=np.float64)
    truth = true_ratio(datasets, benchmark, systematic_weight=systematic_weight)

    valid = (
        np.isfinite(ratio_pred)
        & np.isfinite(truth)
        & (ratio_pred > 1e-9)
        & (truth > 1e-9)
    )
    if not np.any(valid):
        raise ValueError("No valid events were available for the MSE computation.")
    return float(np.mean((np.log(truth[valid]) - np.log(ratio_pred[valid])) ** 2))


def save_nominal_model_bundle(
    nominal_model, output_dir: Path, benchmark: Benchmark
) -> Path:
    bundle_path = (
        output_dir
        / benchmarks_to_name[benchmark]
        / "nominal"
        / "nominal_model_bundle.pt"
    )
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "hparams": dict(nominal_model.hparams),
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in nominal_model.state_dict().items()
        },
    }
    torch.save(bundle, bundle_path)
    return bundle_path


def load_nominal_model_bundle(bundle_path: Path) -> DensityRatioLightning:
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    model = DensityRatioLightning(**bundle["hparams"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model


def maybe_make_reweighted_plots(trainer: density_ratio_trainer) -> None:
    if len(getattr(trainer, "holdout_idx", ())) == 0:
        print("Skipping reweighted plots because holdout_split is 0.", flush=True)
        return
    trainer.make_reweighted_plots(FEATURE_LIST, "log", 50)


def train_nominal(
    datasets: dict[Benchmark, pd.DataFrame],
    benchmark: Benchmark,
    output_dir: Path,
    make_plots: bool,
):
    artifacts = stage_artifacts(output_dir, benchmark, "nominal")
    trainer, _ = prepare_training(
        datasets, benchmark, artifacts, systematic_weight=None
    )
    with working_directory(artifacts.base_dir):
        nominal_model = trainer.train(**NOMINAL_TRAINING_KWARGS)
        if make_plots:
            maybe_make_reweighted_plots(trainer)
    return nominal_model, trainer


def train_systematic_from_scratch(
    datasets: dict[Benchmark, pd.DataFrame],
    mse_datasets: dict[Benchmark, pd.DataFrame],
    benchmark: Benchmark,
    systematic_weight: str,
    output_dir: Path,
    make_plots: bool,
):
    artifacts = stage_artifacts(output_dir, benchmark, f"scratch_{systematic_weight}")
    trainer, _ = prepare_training(
        datasets,
        benchmark,
        artifacts,
        systematic_weight=systematic_weight,
    )
    with working_directory(artifacts.base_dir):
        trainer.train(**SYSTEMATIC_TRAINING_KWARGS)
        if make_plots:
            maybe_make_reweighted_plots(trainer)
    systematic_mse = ratio_mse(
        trainer.model_NN,
        trainer.scaler,
        mse_datasets,
        benchmark,
        systematic_weight=systematic_weight,
    )
    return trainer, systematic_mse


def lora_finetune_systematic(
    datasets: dict[Benchmark, pd.DataFrame],
    mse_datasets: dict[Benchmark, pd.DataFrame],
    benchmark: Benchmark,
    systematic_weight: str,
    nominal_model,
    output_dir: Path,
    make_plots: bool,
):
    artifacts = stage_artifacts(output_dir, benchmark, f"lora_{systematic_weight}")
    trainer, _ = prepare_training(
        datasets,
        benchmark,
        artifacts,
        systematic_weight=systematic_weight,
    )
    with working_directory(artifacts.base_dir):
        trainer.lora_finetune(
            model_to_finetune=copy.deepcopy(nominal_model),
            **LORA_KWARGS,
        )
        if make_plots:
            maybe_make_reweighted_plots(trainer)
    lora_mse = ratio_mse(
        trainer.model_NN,
        trainer.scaler,
        mse_datasets,
        benchmark,
        systematic_weight=systematic_weight,
    )
    return trainer, lora_mse


def subsample_datasets(
    datasets: dict[Benchmark, pd.DataFrame],
    dataset_size: int,
    seed: int,
) -> dict[Benchmark, pd.DataFrame]:
    """Build deterministic nested training samples for a given dataset size.

    The ``seed`` argument is intentionally unused. The one-time benchmark-level
    shuffle happens in :func:`load_datasets`; after that,
    :func:`split_training_and_mse_datasets` removes the fixed MSE sample and
    every sweep point is defined as ``training_df.iloc[:dataset_size]``. This
    guarantees that:

    1. the sample for size ``N2`` contains the events from any smaller size
       ``N1 < N2``;
    2. ``dataset_size`` is the exact number of training-pool events per
       benchmark before the train/validation split;
    3. the exact same sampled events are reused for from-scratch training and
       LoRA finetuning at that dataset size.
    """
    del seed
    sampled: dict[Benchmark, pd.DataFrame] = {}
    for benchmark, df in datasets.items():
        if len(df) < dataset_size:
            raise ValueError(
                f"Requested {dataset_size} events for benchmark '{benchmarks_to_name[benchmark]}', "
                f"but only {len(df)} are available in the training pool."
            )
        sampled[benchmark] = df.iloc[:dataset_size].copy()
    return sampled


def get_weight_sum_for_normalization(df: pd.DataFrame) -> float:
    """Get the base weight sum for cross-section normalization.

    Uses weight_sm if available (nominal SM weight), otherwise falls back to weights.
    """
    if "weight_sm" in df.columns:
        return df["weight_sm"].sum()
    elif "weights" in df.columns:
        return df["weights"].sum()
    else:
        # Fallback: sum of all weight columns
        weight_cols = [c for c in df.columns if c.startswith("weight_")]
        if weight_cols:
            return df[weight_cols[0]].sum()
        return float(len(df))  # Unweighted


def get_weight_column_sums(df: pd.DataFrame) -> dict[str, float]:
    """Get the sum of each weight column for normalization.

    Returns a dict mapping each weight column name to its sum.
    """
    result = {}
    weight_cols = [c for c in df.columns if c.startswith("weight_")]
    for col in weight_cols:
        result[col] = df[col].sum()
    return result


def subsample_datasets_imbalanced(
    datasets: dict[Benchmark, pd.DataFrame],
    target_benchmark: Benchmark,
    target_size: int,
    sm_size: int | None,
    original_weight_sums: dict[int, dict[str, float]] | None = None,
) -> dict[Benchmark, pd.DataFrame]:
    """Build training samples with different sizes for target and SM.

    SM uses full dataset (or specified sm_size), while target uses target_size.
    Each weight column (weight_sm, weight_systematic, etc.) is independently
    renormalized to preserve its original sum (cross section) for that benchmark.

    This ensures that after pd.concat, the combined dataset has:
    - SM weights summing to original SM total cross section
    - Target weights summing to original Target total cross section
    """
    sampled: dict[Benchmark, pd.DataFrame] = {}

    for benchmark, df in datasets.items():
        if benchmark == target_benchmark:
            # Target/benchmark: use specified (smaller) size
            size = target_size
            label = f"target_{benchmarks_to_name[benchmark]}"
        else:
            # SM: use full dataset or specified size
            size = sm_size if sm_size is not None else len(df)
            label = "sm"

        if len(df) < size:
            raise ValueError(
                f"Requested {size} events for {label} ('{benchmarks_to_name[benchmark]}'), "
                f"but only {len(df)} are available in the training pool."
            )

        sampled_df = df.iloc[:size].copy()

        # Get original weight sums for this benchmark
        if original_weight_sums is not None and benchmark.value in original_weight_sums:
            orig_sums = original_weight_sums[benchmark.value]
        else:
            orig_sums = get_weight_column_sums(df)

        # Renormalize each weight column independently to preserve its cross section
        sampled_sums = get_weight_column_sums(sampled_df)

        for weight_col, orig_sum in orig_sums.items():
            if (
                weight_col in sampled_sums
                and sampled_sums[weight_col] > 0
                and orig_sum > 0
            ):
                SF = orig_sum / sampled_sums[weight_col]
                sampled_df[weight_col] *= SF

        sampled[benchmark] = sampled_df

    return sampled


def valid_dataset_sizes(
    datasets: dict[Benchmark, pd.DataFrame],
    dataset_sizes: list[int],
    use_imbalanced: bool = False,
    target_benchmark: Benchmark | None = None,
) -> list[int]:
    """Filter dataset sizes to only those that fit in available data.

    For imbalanced mode, only check against target benchmark size.
    """
    if use_imbalanced and target_benchmark is not None:
        max_available = len(datasets[target_benchmark])
    else:
        max_available = min(len(df) for df in datasets.values())
    return [size for size in dataset_sizes if size <= max_available]


def resolve_compare_jobs(requested_jobs: int, task_count: int) -> int:
    if task_count <= 1:
        return 1
    if requested_jobs > 0:
        return min(requested_jobs, task_count)
    if torch.cuda.is_available():
        return 1
    return min(task_count, os.cpu_count() or 1)


def run_comparison_task(
    task: ComparisonTask,
    original_weight_sums: dict[int, dict[str, float]] | None = None,
) -> dict[str, float | int]:
    benchmark = name_to_benchmark[task.benchmark_name]
    datasets = load_datasets(task.root_file, shuffle_seed=task.seed)

    # Compute original weight column sums from full datasets (before any splitting)
    # This ensures each weight column preserves its cross section independently
    if original_weight_sums is None:
        original_weight_sums = {}
        for b, df in datasets.items():
            cut_df = phase_space_cuts(df)
            original_weight_sums[b.value] = get_weight_column_sums(cut_df)

    training_datasets, mse_datasets, _ = split_training_and_mse_datasets(
        datasets,
        mse_dataset_size=task.mse_dataset_size,
    )

    # Construct the per-size training sample using imbalanced sampling:
    # SM uses full dataset, target/benchmark uses specified size
    # Each weight column is renormalized independently to preserve its cross section
    sampled = subsample_datasets_imbalanced(
        training_datasets,
        target_benchmark=benchmark,
        target_size=task.target_dataset_size,
        sm_size=task.sm_dataset_size,
        original_weight_sums=original_weight_sums,
    )
    nominal_model = load_nominal_model_bundle(task.nominal_bundle_path)

    scratch_artifacts = stage_artifacts(
        task.output_dir,
        benchmark,
        f"compare/scratch_{task.systematic_weight}_target{task.target_dataset_size}",
    )
    scratch_trainer, _ = prepare_training(
        sampled,
        benchmark,
        scratch_artifacts,
        systematic_weight=task.systematic_weight,
    )
    with working_directory(scratch_artifacts.base_dir):
        scratch_trainer.train(**task.systematic_training_kwargs)
    scratch_mse = ratio_mse(
        scratch_trainer.model_NN,
        scratch_trainer.scaler,
        mse_datasets,
        benchmark,
        systematic_weight=task.systematic_weight,
    )

    lora_artifacts = stage_artifacts(
        task.output_dir,
        benchmark,
        f"compare/lora_{task.systematic_weight}_target{task.target_dataset_size}",
    )
    lora_trainer, _ = prepare_training(
        sampled,
        benchmark,
        lora_artifacts,
        systematic_weight=task.systematic_weight,
    )
    with working_directory(lora_artifacts.base_dir):
        lora_trainer.lora_finetune(
            model_to_finetune=nominal_model,
            **task.lora_kwargs,
        )
    lora_mse = ratio_mse(
        lora_trainer.model_NN,
        lora_trainer.scaler,
        mse_datasets,
        benchmark,
        systematic_weight=task.systematic_weight,
    )

    return {
        "target_dataset_size": task.target_dataset_size,
        "sm_dataset_size": task.sm_dataset_size,
        "dataset_size": task.target_dataset_size,  # For backward compatibility
        "mse_from_scratch": scratch_mse,
        "mse_lora": lora_mse,
        "mse_dataset_size": task.mse_dataset_size,
    }


def compare_dataset_sizes(
    root_file: Path,
    benchmark: Benchmark,
    systematic_weight: str,
    nominal_bundle_path: Path,
    output_dir: Path,
    dataset_sizes: list[int],
    seed: int,
    jobs: int,
    sm_dataset_size: int | None = None,
    mse_dataset_size: int = MSE_DATASET_SIZE,
) -> tuple[pd.DataFrame, Path]:
    """Run dataset-size sweep for a single seed and save to CSV.

    Uses full SM dataset (or sm_dataset_size if specified) while varying
    only the target/benchmark dataset size.
    """
    datasets = load_datasets(root_file, shuffle_seed=seed)

    # Compute original weight column sums from full post-cut datasets
    # This preserves cross section for EACH weight column independently
    original_weight_sums: dict[int, dict[str, float]] = {}
    for b, df in datasets.items():
        cut_df = phase_space_cuts(df)
        original_weight_sums[b.value] = get_weight_column_sums(cut_df)

    training_datasets, _, _ = split_training_and_mse_datasets(
        datasets,
        mse_dataset_size=mse_dataset_size,
    )

    # Validate dataset sizes against target/benchmark availability
    # (SM typically has more events, so we check against the limiting benchmark)
    sm_training = training_datasets[Benchmark.SM]
    target_training = training_datasets[benchmark]

    # Filter sizes that fit in target dataset
    dataset_sizes = [size for size in dataset_sizes if size <= len(target_training)]

    if not dataset_sizes:
        max_available = len(target_training)
        raise ValueError(
            "None of the requested dataset sizes fit in target/benchmark training pool. "
            f"Max available for target: {max_available}. "
            f"Consider reducing MSE_DATASET_SIZE or using smaller sizes."
        )

    # Check SM size constraint
    if sm_dataset_size is not None and sm_dataset_size > len(sm_training):
        raise ValueError(
            f"Requested SM dataset size {sm_dataset_size} exceeds available "
            f"{len(sm_training)} events."
        )

    tasks = [
        ComparisonTask(
            root_file=root_file,
            benchmark_name=benchmarks_to_name[benchmark],
            systematic_weight=systematic_weight,
            output_dir=output_dir,
            target_dataset_size=target_size,
            sm_dataset_size=sm_dataset_size,  # None = use all SM
            mse_dataset_size=mse_dataset_size,
            seed=seed,
            nominal_bundle_path=nominal_bundle_path,
            systematic_training_kwargs=dict(SYSTEMATIC_TRAINING_KWARGS),
            lora_kwargs=dict(LORA_KWARGS),
        )
        for target_size in dataset_sizes
    ]
    max_workers = resolve_compare_jobs(jobs, len(tasks))

    if max_workers == 1:
        # Single worker: pass original_weight_sums directly
        rows = [run_comparison_task(task, original_weight_sums) for task in tasks]
    else:
        # Multiprocessing: recompute weight sums in each worker
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=mp_context
        ) as executor:
            rows = list(executor.map(run_comparison_task, tasks))

    results = (
        pd.DataFrame(rows).sort_values("target_dataset_size").reset_index(drop=True)
    )
    # Include seed in output filename for unique CSV per seed
    results_path = (
        output_dir
        / benchmarks_to_name[benchmark]
        / f"comparison_{systematic_weight}_seed{seed}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    return results, results_path


def plot_dataset_size_comparison(
    results: pd.DataFrame,
    benchmark: Benchmark,
    systematic_weight: str,
    output_dir: Path,
) -> Path:
    figure_path = (
        output_dir
        / benchmarks_to_name[benchmark]
        / f"comparison_{systematic_weight}.png"
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    # Use target_dataset_size if available, otherwise fall back to dataset_size
    x_column = (
        "target_dataset_size"
        if "target_dataset_size" in results.columns
        else "dataset_size"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        results[x_column],
        results["mse_from_scratch"],
        marker="o",
        linewidth=2,
        label="From scratch",
    )
    ax.plot(
        results[x_column],
        results["mse_lora"],
        marker="s",
        linewidth=2,
        label="LoRA finetune",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Target/Benchmark Dataset size (SM uses full)")
    ax.set_ylabel("MSE")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


def print_dataset_summary(
    datasets: dict[Benchmark, pd.DataFrame], benchmark: Benchmark
) -> None:
    eft = benchmarks_to_eftcoeffs[benchmark]
    print(
        f"Benchmark: {benchmarks_to_name[benchmark]} (cwl2={eft.cwl2}, cpwl2={eft.cpwl2})"
    )
    for benchmark_name, df in datasets.items():
        print(
            f"Loaded {len(df):>9} events for {benchmarks_to_name[benchmark_name]:>6} "
            f"({len(phase_space_cuts(df)):>9} after cuts)"
        )


def print_split_summary(
    training_datasets: dict[Benchmark, pd.DataFrame],
    mse_datasets: dict[Benchmark, pd.DataFrame],
) -> None:
    print(f"Reserved {MSE_DATASET_SIZE:,} post-cut events per benchmark for MSE.")
    for benchmark, training_df in training_datasets.items():
        mse_df = mse_datasets[benchmark]
        print(
            f"Split {benchmarks_to_name[benchmark]:>6}: "
            f"{len(training_df):>9} training-pool events, {len(mse_df):>9} MSE events"
        )


def main() -> None:
    args = parse_args()
    benchmark = name_to_benchmark[args.benchmark]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.epochs is not None:
        NOMINAL_TRAINING_KWARGS["number_of_epochs"] = args.epochs
        SYSTEMATIC_TRAINING_KWARGS["number_of_epochs"] = args.epochs
        LORA_KWARGS["number_of_epochs"] = args.epochs

    if args.learning_rate is not None:
        NOMINAL_TRAINING_KWARGS["learning_rate"] = args.learning_rate
        SYSTEMATIC_TRAINING_KWARGS["learning_rate"] = args.learning_rate
        LORA_KWARGS["learning_rate"] = args.learning_rate

    if args.batch_size is not None:
        NOMINAL_TRAINING_KWARGS["batch_size"] = args.batch_size
        SYSTEMATIC_TRAINING_KWARGS["batch_size"] = args.batch_size
        LORA_KWARGS["batch_size"] = args.batch_size

    if args.lora_rank is not None:
        LORA_KWARGS["lora_rank"] = args.lora_rank
        if args.lora_alpha is None:
            LORA_KWARGS["lora_alpha"] = 2 * args.lora_rank

    if args.lora_alpha is not None:
        LORA_KWARGS["lora_alpha"] = args.lora_alpha

    if args.hidden_layers is not None:
        NOMINAL_TRAINING_KWARGS["hidden_layers"] = args.hidden_layers
        SYSTEMATIC_TRAINING_KWARGS["hidden_layers"] = args.hidden_layers

    if args.neurons is not None:
        NOMINAL_TRAINING_KWARGS["neurons"] = args.neurons
        SYSTEMATIC_TRAINING_KWARGS["neurons"] = args.neurons

    if args.callback_patience is not None:
        NOMINAL_TRAINING_KWARGS["callback_patience"] = args.callback_patience
        SYSTEMATIC_TRAINING_KWARGS["callback_patience"] = args.callback_patience
        LORA_KWARGS["callback_patience"] = args.callback_patience

    if args.lr_scheduler is not None:
        NOMINAL_TRAINING_KWARGS["lr_scheduler"] = args.lr_scheduler
        SYSTEMATIC_TRAINING_KWARGS["lr_scheduler"] = args.lr_scheduler
        LORA_KWARGS["lr_scheduler"] = args.lr_scheduler

    if args.scheduler_patience is not None:
        NOMINAL_TRAINING_KWARGS["scheduler_patience"] = args.scheduler_patience
        SYSTEMATIC_TRAINING_KWARGS["scheduler_patience"] = args.scheduler_patience
        LORA_KWARGS["scheduler_patience"] = args.scheduler_patience

    if args.scheduler_factor is not None:
        NOMINAL_TRAINING_KWARGS["callback_factor"] = args.scheduler_factor
        SYSTEMATIC_TRAINING_KWARGS["callback_factor"] = args.scheduler_factor
        LORA_KWARGS["callback_factor"] = args.scheduler_factor

    # Determine which seed(s) to use for the dataset-size sweep
    sweep_seeds = args.seeds if args.seeds is not None else [args.seed]

    datasets = load_datasets(args.root_file, shuffle_seed=args.seed)
    print_dataset_summary(datasets, benchmark)
    training_datasets, mse_datasets, _ = split_training_and_mse_datasets(datasets)
    print_split_summary(training_datasets, mse_datasets)
    print(f"Dataset-size sweep will use seeds: {sweep_seeds}")
    print(f"Sweep mode: Full SM + varying target/benchmark (imbalanced)")
    print(f"Artifacts will be written to: {output_dir}")
    print(f"Systematic weight: {args.systematic_weight}")

    make_plots = not args.skip_plots

    print("Starting stage 1: nominal SM-vs-benchmark training", flush=True)
    nominal_model, nominal_trainer = train_nominal(
        datasets=training_datasets,
        benchmark=benchmark,
        output_dir=output_dir,
        make_plots=make_plots,
    )
    nominal_bundle_path = save_nominal_model_bundle(
        nominal_model, output_dir, benchmark
    )
    del nominal_model
    gc.collect()
    nominal_model_for_eval = load_nominal_model_bundle(nominal_bundle_path)
    nominal_mse = ratio_mse(
        nominal_model_for_eval, nominal_trainer.scaler, mse_datasets, benchmark
    )
    del nominal_model_for_eval
    gc.collect()
    print(f"Nominal MSE: {nominal_mse:.6f}")
    print(f"Nominal model directory: {nominal_trainer.path_to_models}")

    # Compute the "reweighted" MSE: nominal model predicting systematic target
    # This shows how well the nominal model (trained on nominal) performs on systematic data
    nominal_model_for_reweight = load_nominal_model_bundle(nominal_bundle_path)
    reweighted_mse = ratio_mse(
        nominal_model_for_reweight,
        nominal_trainer.scaler,
        mse_datasets,
        benchmark,
        systematic_weight=args.systematic_weight,
    )
    del nominal_model_for_reweight
    gc.collect()
    print(f"Reweighted MSE (nominal model on systematic): {reweighted_mse:.6f}")

    # Save reweighted MSE to JSON for later plotting
    import json

    reweighted_mse_data = {
        "reweighted_mse": float(reweighted_mse),
        "nominal_mse": float(nominal_mse),
        "benchmark": benchmarks_to_name[benchmark],
        "systematic_weight": args.systematic_weight,
        "mse_dataset_size": MSE_DATASET_SIZE,
        "description": "MSE of nominal model predicting systematic-varied target (reweighted reference)",
    }
    reweighted_mse_path = (
        output_dir
        / benchmarks_to_name[benchmark]
        / f"reweighted_mse_{args.systematic_weight}.json"
    )
    reweighted_mse_path.parent.mkdir(parents=True, exist_ok=True)
    with open(reweighted_mse_path, "w") as f:
        json.dump(reweighted_mse_data, f, indent=2)
    print(f"Saved reweighted MSE to: {reweighted_mse_path}")

    print("Starting stage 2: systematic training from scratch", flush=True)

    scratch_trainer, scratch_mse = train_systematic_from_scratch(
        datasets=training_datasets,
        mse_datasets=mse_datasets,
        benchmark=benchmark,
        systematic_weight=args.systematic_weight,
        output_dir=output_dir,
        make_plots=make_plots,
    )
    print(f"Systematic-from-scratch MSE: {scratch_mse:.6f}")
    print(f"Scratch model directory: {scratch_trainer.path_to_models}")
    print("Starting stage 2: LoRA finetuning on the systematic variation", flush=True)

    nominal_model_for_lora = load_nominal_model_bundle(nominal_bundle_path)
    lora_trainer, lora_mse = lora_finetune_systematic(
        datasets=training_datasets,
        mse_datasets=mse_datasets,
        benchmark=benchmark,
        systematic_weight=args.systematic_weight,
        nominal_model=nominal_model_for_lora,
        output_dir=output_dir,
        make_plots=make_plots,
    )
    del nominal_model_for_lora
    gc.collect()
    print(f"LoRA-finetuned MSE: {lora_mse:.6f}")
    print(f"LoRA model directory: {lora_trainer.path_to_models}")

    if args.compare_sizes is not None:
        dataset_sizes = args.compare_sizes or DEFAULT_DATASET_SIZES
        compare_jobs = resolve_compare_jobs(
            args.jobs, len(dataset_sizes) * len(sweep_seeds)
        )
        print(f"Dataset-size comparison workers: {compare_jobs}")
        print("Starting stage 3: dataset-size MSE sweep (imbalanced mode)", flush=True)
        print(f"  Mode: Full SM dataset + varying target/benchmark sizes", flush=True)
        print(
            f"  Target sizes to test: {dataset_sizes[:5]}..."
            if len(dataset_sizes) > 5
            else f"  Target sizes: {dataset_sizes}",
            flush=True,
        )

        all_results = []
        for seed in sweep_seeds:
            print(f"\n  Running sweep for seed={seed}...", flush=True)
            results, results_path = compare_dataset_sizes(
                root_file=args.root_file.expanduser().resolve(),
                benchmark=benchmark,
                systematic_weight=args.systematic_weight,
                nominal_bundle_path=nominal_bundle_path,
                output_dir=output_dir,
                dataset_sizes=dataset_sizes,
                seed=seed,
                jobs=args.jobs,
                sm_dataset_size=None,  # Use full SM dataset
                mse_dataset_size=MSE_DATASET_SIZE,
            )
            results["seed"] = seed
            all_results.append(results)
            print(f"  Saved seed {seed} results to: {results_path}")
            # Print summary
            if "sm_dataset_size" in results.columns:
                for _, row in results.iterrows():
                    target_size = int(row["target_dataset_size"])
                    mse_s = row["mse_from_scratch"]
                    mse_l = row["mse_lora"]
                    print(
                        f"    Target N={target_size:,}: Scratch={mse_s:.4f}, LoRA={mse_l:.4f}"
                    )

        # Save combined results with all seeds
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_path = (
            output_dir
            / benchmarks_to_name[benchmark]
            / f"comparison_{args.systematic_weight}_all_seeds.csv"
        )
        combined_results.to_csv(combined_path, index=False)
        print(f"\nSaved combined results with all seeds to: {combined_path}")

        # Plot only uses the first seed results (legacy behavior preserved)
        figure_path = plot_dataset_size_comparison(
            results=all_results[0],
            benchmark=benchmark,
            systematic_weight=args.systematic_weight,
            output_dir=output_dir,
        )
        print(f"Saved comparison plot (first seed) to: {figure_path}")


if __name__ == "__main__":
    """
    Example usage:

    >>> python -u examples/madminer_dataset/lora_exploration.py ~/Downloads/combined_samples_full.root --target-benchmark ww --systematic-variation weight_scale_muf_nuisance_param_0_up --epochs 5 --batch-size 1024 --skip-plots --compare-sizes --jobs 8
    
    >>> python -u examples/madminer_dataset/lora_exploration.py ~/Downloads/combined_samples_full.root --target-benchmark ww --systematic-variation weight_scale_muf_nuisance_param_0_up --epochs 20 --batch-size 1024 --skip-plots --compare-sizes --jobs 12 --seed 1234
    
    Multi-seed example (runs sweep for each seed, producing separate CSVs):
    >>> python -u examples/madminer_dataset/lora_exploration.py ~/Downloads/combined_samples_full.root --target-benchmark ww --systematic-variation weight_scale_muf_nuisance_param_0_up --skip-plots --compare-sizes --jobs 12 --seed 1234 --lr-scheduler "plateau" --scheduler-patience 5 --scheduler-factor 0.5  --seeds 42 123 456 789 1000
    
    LATEST:

    >>> python -u examples/madminer_dataset/lora_exploration.py ~/Downloads/combined_samples_full.root --target-benchmark neg_ww --systematic-variation weight_scale_corr_nuisance_param_0_down --skip-plots  --jobs 12 --seed 1234 --lr-scheduler "plateau" --scheduler-patience 5 --scheduler-factor 0.5 --compare-sizes --seeds 42 123 456 789 1000 --lora-rank 4 --lora-alpha 4
    """
    main()
