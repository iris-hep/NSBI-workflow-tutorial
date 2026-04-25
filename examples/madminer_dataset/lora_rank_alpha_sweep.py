"""
LoRA Rank and Alpha Sweep Script

Runs dataset-size MSE sweeps for multiple LoRA rank and alpha combinations.
Each configuration is stored in a separate directory for easy comparison.

Example usage:
    python lora_rank_alpha_sweep.py ~/data.root --lora-ranks 2 4 8 --lora-alphas 4 8 16 --compare-sizes --epochs 20
"""

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
CACHE_DIR = Path(__file__).with_name(".cache_sweep")
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
    # 300_000,
    # 350_000,
    # 400_000,
    # 450_000,
    # 500_000,
    # 550_000,
    # 600_000,
    # 650_000,
    # 700_000,
    # 750_000,
    # 800_000,
    # 850_000,
]
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("artifacts") / "lora_sweep"

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
class SweepConfig:
    """Configuration for a single LoRA rank/alpha sweep run."""

    lora_rank: int
    lora_alpha: int
    output_dir: Path  # Base output directory (will create subdirs per config)


@dataclass(frozen=True)
class ComparisonTask:
    root_file: Path
    benchmark_name: str
    systematic_weight: str
    output_dir: Path
    dataset_size: int
    mse_dataset_size: int
    seed: int
    nominal_bundle_path: Path
    systematic_training_kwargs: dict
    lora_kwargs: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA Rank and Alpha Sweep: Run MSE vs dataset size for multiple LoRA configurations."
    )
    parser.add_argument(
        "root_file",
        type=Path,
        help="Path to the ROOT file containing the 'Events' tree.",
    )
    parser.add_argument(
        "--benchmark",
        default="ww",
        choices=sorted(name for name in name_to_benchmark if name != "sm"),
        help="Target benchmark dataset to compare against the SM reference.",
    )
    parser.add_argument(
        "--systematic-weight",
        default=DEFAULT_SYSTEMATIC_WEIGHT,
        help="Systematic variation weight column to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base directory for saved models and comparison tables.",
    )
    parser.add_argument(
        "--lora-ranks",
        nargs="+",
        type=int,
        default=None,
        metavar="RANK",
        help="List of LoRA ranks to sweep over (e.g., --lora-ranks 2 4 8 16).",
    )
    parser.add_argument(
        "--lora-alphas",
        nargs="+",
        type=int,
        default=None,
        metavar="ALPHA",
        help="List of LoRA alphas to sweep over (e.g., --lora-alphas 4 8 16 32). "
        "If not provided, defaults to 2*rank for each rank.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        metavar="SEED",
        help="List of random seeds to run. Default: 42.",
    )
    parser.add_argument(
        "--compare-sizes",
        nargs="*",
        type=int,
        default=None,
        metavar="N",
        help=f"Training-pool event counts to sweep. Default: {len(DEFAULT_DATASET_SIZES)} preset sizes.",
    )
    parser.add_argument(
        "--skip-nominal",
        action="store_true",
        help="Skip nominal training (use existing model if --reuse-nominal is set).",
    )
    parser.add_argument(
        "--reuse-nominal",
        type=Path,
        default=None,
        help="Path to existing nominal model bundle to reuse.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Disable diagnostic plots.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=12,
        help="Number of worker processes for parallel tasks.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override the learning rate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the batch size.",
    )
    parser.add_argument(
        "--create-summary-plot",
        action="store_true",
        help="Create a summary plot comparing all LoRA configurations.",
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
    resolved_root_file = root_file.expanduser().resolve()
    tree = uproot.open(resolved_root_file)["Events"]
    events = tree.arrays(library="pd")
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
) -> tuple[dict[Benchmark, pd.DataFrame], dict[Benchmark, pd.DataFrame]]:
    training_datasets: dict[Benchmark, pd.DataFrame] = {}
    mse_datasets: dict[Benchmark, pd.DataFrame] = {}

    for benchmark, df in datasets.items():
        cut_df = phase_space_cuts(df).reset_index(drop=True)
        if len(cut_df) <= mse_dataset_size:
            raise ValueError(
                f"Need more than {mse_dataset_size} post-cut events for benchmark "
                f"'{benchmarks_to_name[benchmark]}' but only {len(cut_df)} available."
            )
        mse_datasets[benchmark] = cut_df.iloc[:mse_dataset_size].copy()
        training_datasets[benchmark] = (
            cut_df.iloc[mse_dataset_size:].reset_index(drop=True).copy()
        )

    return training_datasets, mse_datasets


def stage_artifacts(
    output_dir: Path, benchmark: Benchmark, stage_name: str
) -> ArtifactPaths:
    base_dir = output_dir / benchmarks_to_name[benchmark] / stage_name
    models_dir = base_dir / "models"
    plots_dir = base_dir / "plots"
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return ArtifactPaths(base_dir=base_dir, models_dir=models_dir, plots_dir=plots_dir)


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


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
        scale_factors = df[systematic_weight] / df["weight_sm"]
        weights = np.where(
            target_mask,
            df[target_weight_column] * scale_factors,
            df["weight_sm"],
        )
        target_label = f"{benchmarks_to_name[benchmark]} (systematic)"
        output_name = f"ref_vs_{benchmarks_to_name[benchmark]}__{systematic_weight}"

    training_labels = np.where(target_mask, 1, 0).astype(np.int32)
    df["weights"] = np.asarray(weights, dtype=np.float64)
    df["train_labels"] = training_labels

    plots_dir, models_dir = (
        f"{artifacts.plots_dir.as_posix()}/",
        f"{artifacts.models_dir.as_posix()}/",
    )
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
        raise ValueError("No valid events for MSE computation.")
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


def train_nominal(
    datasets: dict[Benchmark, pd.DataFrame],
    benchmark: Benchmark,
    output_dir: Path,
    training_kwargs: dict,
):
    artifacts = stage_artifacts(output_dir, benchmark, "nominal")
    trainer, _ = prepare_training(
        datasets, benchmark, artifacts, systematic_weight=None
    )
    with working_directory(artifacts.base_dir):
        nominal_model = trainer.train(**training_kwargs)
    return nominal_model, trainer


def subsample_datasets(
    datasets: dict[Benchmark, pd.DataFrame],
    dataset_size: int,
) -> dict[Benchmark, pd.DataFrame]:
    sampled: dict[Benchmark, pd.DataFrame] = {}
    for benchmark, df in datasets.items():
        if len(df) < dataset_size:
            raise ValueError(
                f"Requested {dataset_size} events for benchmark '{benchmarks_to_name[benchmark]}', "
                f"but only {len(df)} are available."
            )
        sampled[benchmark] = df.iloc[:dataset_size].copy()
    return sampled


def valid_dataset_sizes(
    datasets: dict[Benchmark, pd.DataFrame],
    dataset_sizes: list[int],
) -> list[int]:
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


def run_sweep_for_config(
    rank: int,
    alpha: int,
    root_file: Path,
    benchmark: Benchmark,
    systematic_weight: str,
    base_output_dir: Path,
    nominal_bundle_path: Path,
    dataset_sizes: list[int],
    seeds: list[int],
    training_kwargs: dict,
) -> list[Path]:
    """Run dataset-size sweep for a single (rank, alpha) configuration."""

    # Create output directory for this specific configuration
    config_dir = base_output_dir / f"rank{rank}_alpha{alpha}"
    config_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Running sweep for LoRA rank={rank}, alpha={alpha}")
    print(f"Output directory: {config_dir}")
    print(f"{'=' * 60}")

    # Configure LoRA kwargs for this run
    lora_kwargs = {
        **LORA_KWARGS,
        "lora_rank": rank,
        "lora_alpha": alpha,
        **training_kwargs,
    }

    # Prepare systematic training kwargs
    systematic_kwargs = {**SYSTEMATIC_TRAINING_KWARGS, **training_kwargs}

    all_csv_paths = []

    for seed in seeds:
        print(f"\n  Running seed={seed}...", flush=True)

        # Load datasets with this seed
        datasets = load_datasets(root_file, shuffle_seed=seed)
        training_datasets_seed, mse_datasets_seed, _ = split_training_and_mse_datasets(
            datasets
        )

        # Filter to valid dataset sizes that fit in available training data
        valid_sizes = valid_dataset_sizes(training_datasets_seed, dataset_sizes)
        if not valid_sizes:
            max_available = min(len(df) for df in training_datasets_seed.values())
            print(f"\n    ERROR: No valid dataset sizes found.")
            print(f"    Available training events per benchmark: {max_available:,}")
            print(f"    Requested sizes: {dataset_sizes}")
            print(f"    Hint: Use --compare-sizes with values <= {max_available:,}")
            raise ValueError(
                f"No valid dataset sizes. Max available: {max_available}. "
                f"Consider reducing MSE_DATASET_SIZE or using smaller --compare-sizes values."
            )

        # Log if some sizes were filtered out
        excluded = [s for s in dataset_sizes if s not in valid_sizes]
        if excluded:
            print(
                f"\n    Note: Excluding {len(excluded)} dataset size(s) > available data: {excluded[:5]}{'...' if len(excluded) > 5 else ''}"
            )
            print(
                f"    Using {len(valid_sizes)} valid size(s): {valid_sizes[:5]}{'...' if len(valid_sizes) > 5 else ''}"
            )

        # Run comparison for each dataset size
        results_rows = []
        for dataset_size in valid_sizes:
            print(f"    Size={dataset_size:,}...", flush=True, end="")

            # Subsample training data
            sampled = subsample_datasets(training_datasets_seed, dataset_size)

            # Load nominal model fresh for each size (to avoid any state issues)
            nominal_model = load_nominal_model_bundle(nominal_bundle_path)

            # Train from scratch
            scratch_artifacts = stage_artifacts(
                config_dir,
                benchmark,
                f"scratch_{systematic_weight}_{dataset_size}_seed{seed}",
            )
            scratch_trainer, _ = prepare_training(
                sampled,
                benchmark,
                scratch_artifacts,
                systematic_weight=systematic_weight,
            )
            with working_directory(scratch_artifacts.base_dir):
                scratch_trainer.train(**systematic_kwargs)
            scratch_mse = ratio_mse(
                scratch_trainer.model_NN,
                scratch_trainer.scaler,
                mse_datasets_seed,
                benchmark,
                systematic_weight=systematic_weight,
            )

            # LoRA finetune
            lora_artifacts = stage_artifacts(
                config_dir,
                benchmark,
                f"lora_{systematic_weight}_{dataset_size}_seed{seed}",
            )
            lora_trainer, _ = prepare_training(
                sampled, benchmark, lora_artifacts, systematic_weight=systematic_weight
            )
            with working_directory(lora_artifacts.base_dir):
                lora_trainer.lora_finetune(
                    model_to_finetune=copy.deepcopy(nominal_model),
                    **lora_kwargs,
                )
            lora_mse = ratio_mse(
                lora_trainer.model_NN,
                lora_trainer.scaler,
                mse_datasets_seed,
                benchmark,
                systematic_weight=systematic_weight,
            )

            results_rows.append(
                {
                    "dataset_size": dataset_size,
                    "mse_from_scratch": scratch_mse,
                    "mse_lora": lora_mse,
                    "seed": seed,
                }
            )
            print(f" Scratch={scratch_mse:.4f}, LoRA={lora_mse:.4f}")

            # Cleanup
            del nominal_model, scratch_trainer, lora_trainer
            gc.collect()

        # Save results for this seed
        results = pd.DataFrame(results_rows)
        csv_path = (
            config_dir
            / benchmarks_to_name[benchmark]
            / f"comparison_{systematic_weight}_seed{seed}.csv"
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(csv_path, index=False)
        all_csv_paths.append(csv_path)
        print(f"  Saved: {csv_path}")

    # Combine all seeds into one file
    if len(seeds) > 1:
        combined = pd.concat([pd.read_csv(p) for p in all_csv_paths], ignore_index=True)
        combined_path = (
            config_dir
            / benchmarks_to_name[benchmark]
            / f"comparison_{systematic_weight}_all_seeds.csv"
        )
        combined.to_csv(combined_path, index=False)
        print(f"  Combined all seeds: {combined_path}")

    return all_csv_paths


def create_summary_plot(
    all_results: dict[tuple[int, int], pd.DataFrame],
    benchmark: Benchmark,
    systematic_weight: str,
    output_dir: Path,
) -> Path:
    """Create a summary plot comparing all LoRA configurations."""
    import mplhep as hep

    hep.style.use("CMS")

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for idx, ((rank, alpha), df) in enumerate(sorted(all_results.items())):
        # Aggregate across seeds if present
        if "seed" in df.columns:
            grouped = (
                df.groupby("dataset_size")
                .agg(
                    {
                        "mse_lora": ["mean", "std"],
                    }
                )
                .reset_index()
            )
            grouped.columns = ["dataset_size", "mse_lora_mean", "mse_lora_std"]
            x = grouped["dataset_size"]
            y = grouped["mse_lora_mean"]
            yerr = grouped["mse_lora_std"]
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color=colors[idx])
        else:
            x = df["dataset_size"]
            y = df["mse_lora"]

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label=f"rank={rank}, α={alpha}",
            color=colors[idx],
        )

    ax.set_xscale("log")
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("MSE (LoRA)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="LoRA Configuration", bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()
    plot_path = (
        output_dir
        / benchmarks_to_name[benchmark]
        / f"lora_sweep_summary_{systematic_weight}.png"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return plot_path


def main():
    args = parse_args()

    # Validate arguments
    if args.lora_ranks is None:
        args.lora_ranks = [4]  # Default single rank

    # If alphas not specified, default to 2*rank for each rank
    if args.lora_alphas is None:
        args.lora_alphas = [2 * r for r in args.lora_ranks]

    benchmark = name_to_benchmark[args.benchmark]
    base_output_dir = args.output_dir.expanduser().resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Build training kwargs overrides
    training_overrides = {}
    if args.epochs is not None:
        training_overrides["number_of_epochs"] = args.epochs
    if args.learning_rate is not None:
        training_overrides["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        training_overrides["batch_size"] = args.batch_size

    dataset_sizes = (
        args.compare_sizes if args.compare_sizes is not None else DEFAULT_DATASET_SIZES
    )

    print(f"LoRA Rank/Alpha Sweep")
    print(f"=====================")
    print(f"Benchmark: {args.benchmark}")
    print(f"Systematic: {args.systematic_weight}")
    print(f"Ranks: {args.lora_ranks}")
    print(f"Alphas: {args.lora_alphas}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {base_output_dir}")
    print()

    # Step 1: Train nominal model (once, shared across all configs)
    nominal_bundle_path = args.reuse_nominal

    if nominal_bundle_path is None and not args.skip_nominal:
        print("Training nominal model...", flush=True)
        datasets = load_datasets(args.root_file, shuffle_seed=args.seeds[0])
        training_datasets, mse_datasets, _ = split_training_and_mse_datasets(datasets)
        nominal_training_kwargs = {**NOMINAL_TRAINING_KWARGS, **training_overrides}
        nominal_model, _ = train_nominal(
            training_datasets, benchmark, base_output_dir, nominal_training_kwargs
        )
        nominal_bundle_path = save_nominal_model_bundle(
            nominal_model, base_output_dir, benchmark
        )
        del nominal_model
        gc.collect()
        print(f"Nominal model saved: {nominal_bundle_path}")
    elif args.skip_nominal and args.reuse_nominal is None:
        raise ValueError("Must provide --reuse-nominal if using --skip-nominal")
    else:
        print(f"Reusing nominal model: {nominal_bundle_path}")

    # Step 2: Run sweeps for each (rank, alpha) combination
    all_results = {}

    # Use zip to pair ranks with alphas, cycling if lengths differ
    from itertools import zip_longest

    # If single alpha specified for multiple ranks, use 2*rank
    if len(args.lora_alphas) == 1 and len(args.lora_ranks) > 1:
        rank_alpha_pairs = [(r, args.lora_alphas[0]) for r in args.lora_ranks]
    # If single rank specified for multiple alphas, use same rank
    elif len(args.lora_ranks) == 1 and len(args.lora_alphas) > 1:
        rank_alpha_pairs = [(args.lora_ranks[0], a) for a in args.lora_alphas]
    # Otherwise zip them, defaulting to 2*rank for missing alphas
    else:
        rank_alpha_pairs = []
        for r, a in zip_longest(args.lora_ranks, args.lora_alphas):
            if a is None:
                a = 2 * r
            rank_alpha_pairs.append((r, a))

    # Remove duplicates while preserving order
    seen = set()
    rank_alpha_pairs = [x for x in rank_alpha_pairs if not (x in seen or seen.add(x))]

    for rank, alpha in rank_alpha_pairs:
        csv_paths = run_sweep_for_config(
            rank=rank,
            alpha=alpha,
            root_file=args.root_file,
            benchmark=benchmark,
            systematic_weight=args.systematic_weight,
            base_output_dir=base_output_dir,
            nominal_bundle_path=nominal_bundle_path,
            dataset_sizes=dataset_sizes,
            seeds=args.seeds,
            training_kwargs=training_overrides,
        )

        # Load and aggregate results for this config
        if len(args.seeds) > 1:
            combined_path = (
                base_output_dir
                / f"rank{rank}_alpha{alpha}"
                / benchmarks_to_name[benchmark]
                / f"comparison_{args.systematic_weight}_all_seeds.csv"
            )
            all_results[(rank, alpha)] = pd.read_csv(combined_path)
        else:
            all_results[(rank, alpha)] = pd.read_csv(csv_paths[0])

    # Step 3: Create summary plot if requested
    if args.create_summary_plot:
        summary_path = create_summary_plot(
            all_results, benchmark, args.systematic_weight, base_output_dir
        )
        print(f"\nSummary plot saved: {summary_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("SWEEP COMPLETE")
    print(f"{'=' * 60}")
    print(f"Results saved in: {base_output_dir}")
    for (rank, alpha), df in sorted(all_results.items()):
        print(f"  rank={rank}, alpha={alpha}: {len(df)} rows")


if __name__ == "__main__":
    """
    Example usage:
    
    # Sweep over multiple ranks with default alpha=2*rank
    >>> python lora_rank_alpha_sweep.py ~/data.root --lora-ranks 2 4 8 16 --compare-sizes
    
    # Sweep over specific (rank, alpha) combinations
    >>> python lora_rank_alpha_sweep.py ~/data.root --lora-ranks 4 8 --lora-alphas 4 16 --compare-sizes
    
    # Multiple seeds for statistical uncertainty
    >>> python lora_rank_alpha_sweep.py ~/data.root --lora-ranks 4 8 --seeds 42 123 456 --compare-sizes
    
    # Create summary comparison plot
    >>> python lora_rank_alpha_sweep.py ~/data.root --lora-ranks 2 4 8 --compare-sizes --create-summary-plot
    """
    main()
