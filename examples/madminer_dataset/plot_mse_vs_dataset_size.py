from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(__file__).with_name(".cache")
CACHE_DIR.mkdir(exist_ok=True)

import os
import sys

os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
(CACHE_DIR / "matplotlib").mkdir(exist_ok=True)
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import mplhep as hep
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter


DEFAULT_ARTIFACTS_DIR = Path(__file__).with_name("artifacts")
DEFAULT_GLOB = "comparison*.csv"
SCRATCH_COLOR = "#D95F02"
LORA_COLOR = "#56B4E9"
GRID_COLOR = "#BFC7D5"
SCRATCH_FILL_COLOR = "#D95F0220"  # With alpha for fill
LORA_FILL_COLOR = "#56B4E920"  # With alpha for fill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone MSE-vs-dataset-size plotter for MadMiner comparison CSV files. "
            "Accepts multiple CSVs (e.g., from different seeds) and aggregates them "
            "by averaging MSE values per dataset size with shaded uncertainty bands."
        )
    )
    parser.add_argument(
        "csv_paths",
        nargs="*",
        type=Path,
        help=(
            "Optional comparison CSV paths. Multiple CSVs will be aggregated with mean "
            "and std dev shown as shaded bands. If omitted, the script scans "
            f"{DEFAULT_ARTIFACTS_DIR.relative_to(REPO_ROOT)} for '{DEFAULT_GLOB}'."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output image path. Defaults to '<csv stem>_pretty.png' for a single "
            "CSV, or 'comparison_mse_vs_dataset_size.png' next to the CSVs for multiple."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom figure title. Defaults to an inferred label from the CSV path.",
    )
    parser.add_argument(
        "--exclude-sizes",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help=(
            "One or more dataset sizes to exclude from the plot. "
            "Provide space-separated values, e.g., '--exclude-sizes 5000 10000 25000'."
        ),
    )
    parser.add_argument(
        "--reweighted-mse",
        type=Path,
        default=None,
        help=(
            "Path to reweighted MSE JSON file (e.g., reweighted_mse_*.json). "
            "If provided, draws a horizontal line showing the reweighted baseline MSE "
            "(nominal model predicting systematic target)."
        ),
    )
    return parser.parse_args()


def discover_csv_paths(paths: list[Path]) -> list[Path]:
    if paths:
        return [path.expanduser().resolve() for path in paths]
    return sorted(DEFAULT_ARTIFACTS_DIR.rglob(DEFAULT_GLOB))


def validate_frame(frame: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """Validate and return frame, detecting if it's raw or aggregated data."""
    # Check for raw data columns
    raw_columns = {"dataset_size", "mse_from_scratch", "mse_lora"}
    # Check for aggregated data columns
    agg_columns = {
        "dataset_size",
        "mse_from_scratch_mean",
        "mse_from_scratch_std",
        "mse_lora_mean",
        "mse_lora_std",
    }

    has_raw = raw_columns.issubset(frame.columns)
    has_agg = agg_columns.issubset(frame.columns)

    if not (has_raw or has_agg):
        missing_raw = raw_columns.difference(frame.columns)
        missing_agg = agg_columns.difference(frame.columns)
        raise ValueError(
            f"{csv_path} must have either raw columns {sorted(raw_columns)} "
            f"(missing: {sorted(missing_raw)}) or aggregated columns {sorted(agg_columns)} "
            f"(missing: {sorted(missing_agg)})"
        )

    # Tag frame with its type for later processing
    frame.attrs["is_aggregated"] = has_agg
    return frame.sort_values("dataset_size").reset_index(drop=True)


def aggregate_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate multiple result frames by dataset_size.

    Returns a DataFrame with mean and std for each metric per dataset_size.
    """
    # Combine all frames
    combined = pd.concat(frames, ignore_index=True)

    # Group by dataset_size and compute statistics
    grouped = (
        combined.groupby("dataset_size")
        .agg(
            {
                "mse_from_scratch": ["mean", "std", "count"],
                "mse_lora": ["mean", "std", "count"],
            }
        )
        .reset_index()
    )

    # Flatten column names
    grouped.columns = [
        "dataset_size",
        "mse_from_scratch_mean",
        "mse_from_scratch_std",
        "mse_from_scratch_count",
        "mse_lora_mean",
        "mse_lora_std",
        "mse_lora_count",
    ]

    return grouped.sort_values("dataset_size").reset_index(drop=True)


def format_dataset_size(value: float, _: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{int(value)}"


def filter_dataset_sizes(
    frame: pd.DataFrame, exclude_sizes: list[int] | None
) -> pd.DataFrame:
    """Filter out specified dataset sizes from the frame."""
    if exclude_sizes is None:
        return frame
    filtered = frame[~frame["dataset_size"].isin(exclude_sizes)].copy()
    if len(filtered) < len(frame):
        excluded = set(frame["dataset_size"]) & set(exclude_sizes)
        print(f"  Excluded dataset sizes: {sorted(excluded)}")
    return filtered.reset_index(drop=True)


def load_reweighted_mse(reweighted_mse_path: Path | None) -> float | None:
    """Load reweighted MSE from JSON file.

    The reweighted MSE is the MSE of the nominal model predicting the systematic target,
    which serves as a baseline for comparison.
    """
    if reweighted_mse_path is None:
        return None
    try:
        with open(reweighted_mse_path.expanduser().resolve()) as f:
            data = json.load(f)
        # Try "reweighted_mse" first, fallback to "nominal_mse" for backwards compatibility
        mse = data.get("reweighted_mse") or data.get("nominal_mse")
        if mse is not None:
            print(f"Loaded reweighted MSE: {mse:.6f} from {reweighted_mse_path}")
            return float(mse)
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load reweighted MSE from {reweighted_mse_path}: {e}")
    return None


def default_output_path(csv_paths: list[Path]) -> Path:
    if len(csv_paths) == 1:
        return csv_paths[0].with_name(f"{csv_paths[0].stem}_pretty.png")
    shared_parent = csv_paths[0].parent
    return shared_parent / "comparison_mse_vs_dataset_size.png"


def apply_style() -> None:
    hep.style.use("CMS")
    plt.rcParams.update(
        {
            "axes.edgecolor": "#5F5A4F",
            "axes.labelcolor": "#2D2A26",
            "xtick.color": "#2D2A26",
            "ytick.color": "#2D2A26",
            "text.color": "#2D2A26",
            "axes.labelsize": 22,
            "legend.frameon": False,
            "legend.fontsize": 16,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "font.size": 14,
            "lines.solid_capstyle": "round",
        }
    )


REWEIGHTED_LINE_COLOR = "#7570B3"  # Purple for reweighted reference line


def compute_ylim(
    frame: pd.DataFrame,
    has_uncertainty: bool,
    reweighted_mse: float | None = None,
    padding: float = 0.05,
) -> tuple[float, float]:
    """Compute appropriate y-axis limits based on data range.

    Args:
        frame: DataFrame with MSE data
        has_uncertainty: Whether frame contains uncertainty bands
        reweighted_mse: Optional reweighted MSE value to include in range
        padding: Fraction of range to add as padding (default 10%)

    Returns:
        (ymin, ymax) tuple
    """
    if has_uncertainty:
        # Include uncertainty bands in range calculation
        all_values = np.concatenate(
            [
                frame["mse_from_scratch_mean"].values,
                frame["mse_lora_mean"].values,
                (frame["mse_from_scratch_mean"] - frame["mse_from_scratch_std"]).values,
                (frame["mse_from_scratch_mean"] + frame["mse_from_scratch_std"]).values,
                (frame["mse_lora_mean"] - frame["mse_lora_std"]).values,
                (frame["mse_lora_mean"] + frame["mse_lora_std"]).values,
            ]
        )
    else:
        all_values = np.concatenate(
            [
                frame["mse_from_scratch"].values,
                frame["mse_lora"].values,
            ]
        )

    # Include reweighted MSE in range calculation if provided
    if reweighted_mse is not None and np.isfinite(reweighted_mse):
        all_values = np.append(all_values, reweighted_mse)

    # Filter out NaN and Inf values
    valid_values = all_values[np.isfinite(all_values)]

    if len(valid_values) == 0:
        return 0.0, 1.0  # Fallback

    ymin = valid_values.min()
    ymax = valid_values.max()

    # Add padding
    yrange = ymax - ymin
    if yrange < 0.01:  # Handle near-constant data
        yrange = 0.1

    ymin_padded = max(0, ymin - padding * yrange)  # Don't go below 0
    ymax_padded = ymax + padding * yrange

    # Round to nice numbers
    # For log scale like behavior on small ranges
    if ymax_padded <= 0.5:
        ymax_padded = np.ceil(ymax_padded * 20) / 20  # Round to 0.05
    elif ymax_padded <= 2:
        ymax_padded = np.ceil(ymax_padded * 10) / 10  # Round to 0.1
    else:
        ymax_padded = np.ceil(ymax_padded * 2) / 2  # Round to 0.5

    # Ensure reasonable minimum
    ymin_padded = np.floor(ymin_padded * 10) / 10

    return ymin_padded , ymax_padded 


def plot_single_axis(
    ax: plt.Axes,
    frame: pd.DataFrame,
    has_uncertainty: bool = False,
    reweighted_mse: float | None = None,
) -> None:
    x_values = frame["dataset_size"]

    if has_uncertainty:
        # Plot with uncertainty bands (aggregated data)
        scratch_mean = frame["mse_from_scratch_mean"]
        scratch_std = frame["mse_from_scratch_std"]
        lora_mean = frame["mse_lora_mean"]
        lora_std = frame["mse_lora_std"]

        # Plot CARL (from scratch) with uncertainty band
        ax.fill_between(
            x_values,
            scratch_mean - scratch_std,
            scratch_mean + scratch_std,
            color=SCRATCH_FILL_COLOR,
            zorder=2,
        )
        ax.plot(
            x_values,
            scratch_mean,
            color=SCRATCH_COLOR,
            linewidth=2.2,
            marker="o",
            markersize=8,
            label=r"CARL $p_\mathrm{c} (x|\nu^\pm) / p_\mathrm{SM} (x)$",
            zorder=3,
        )

        # Plot LoRA with uncertainty band
        ax.fill_between(
            x_values,
            lora_mean - lora_std,
            lora_mean + lora_std,
            color=LORA_FILL_COLOR,
            zorder=2,
        )
        ax.plot(
            x_values,
            lora_mean,
            color=LORA_COLOR,
            linewidth=2.2,
            marker="s",
            markersize=7.5,
            label=r"CARL $p_\mathrm{c} (x|\nu^\pm) / p_\mathrm{SM} (x)$ + LoRA $p_\mathrm{c} (x|\nu^\pm) / p_\mathrm{SM} (x)$",
            zorder=4,
        )
    else:
        # Simple single-run plot
        scratch = frame["mse_from_scratch"]
        lora = frame["mse_lora"]

        ax.plot(
            x_values,
            scratch,
            color=SCRATCH_COLOR,
            linewidth=2.2,
            marker="o",
            markersize=8,
            label=r"CARL $p_\mathrm{c} (x|\nu^\pm) / p_\mathrm{SM} (x)$",
            zorder=3,
        )
        ax.plot(
            x_values,
            lora,
            color=LORA_COLOR,
            linewidth=2.2,
            marker="s",
            markersize=7.5,
            label=r"CARL $p_\mathrm{c} (x|\nu^\pm) / p_\mathrm{SM} (x)$ + LoRA $p_\mathrm{c} (x|\nu^\pm) / p_\mathrm{SM} (x)$",
            zorder=4,
        )

    # Plot reweighted MSE as horizontal dashed line if provided
    if reweighted_mse is not None:
        ax.axhline(
            reweighted_mse,
            color=REWEIGHTED_LINE_COLOR,
            linestyle="--",
            linewidth=2,
            label="Nominal Network",
            zorder=5,
        )

    # ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(format_dataset_size))
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=1)
    ax.grid(True, which="minor", color=GRID_COLOR, linewidth=1)
    ax.set_ylabel(r"Expected MSE on log $r$")

    # Auto-compute ylim based on data (including reweighted MSE)
    ymin, ymax = compute_ylim(frame, has_uncertainty, reweighted_mse)
    ax.set_ylim(ymin, ymax)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)


def plot_csvs(
    csv_paths: list[Path],
    output_path: Path,
    title: str | None,
    exclude_sizes: list[int] | None = None,
    reweighted_mse: float | None = None,
) -> Path:
    apply_style()
    frames = [
        filter_dataset_sizes(validate_frame(pd.read_csv(path), path), exclude_sizes)
        for path in csv_paths
    ]
    # Check that at least one frame has data remaining
    if all(len(f) == 0 for f in frames):
        raise ValueError(
            f"All dataset sizes were excluded. Available sizes: "
            f"{sorted(pd.read_csv(csv_paths[0])['dataset_size'].unique())}"
        )

    # Check if frames contain seed column (multiple runs of same config)
    has_seed_column = "seed" in frames[0].columns
    # Check if frames are already aggregated (have mean/std columns)
    all_aggregated = all(frame.attrs.get("is_aggregated", False) for frame in frames)

    if has_seed_column or (len(frames) > 1 and not all_aggregated):
        # Multiple raw seed runs - aggregate and plot with uncertainty
        if has_seed_column:
            print(f"Aggregating {len(frames)} seed runs...")
        else:
            print(f"Aggregating {len(frames)} CSV files...")
        aggregated = aggregate_frames(frames)
        has_uncertainty = True
    elif all_aggregated:
        # Already aggregated data - use directly
        if len(frames) == 1:
            print("Using pre-aggregated data with uncertainty bands...")
            aggregated = frames[0]
            has_uncertainty = True
        else:
            # Multiple aggregated frames - would need merging logic
            print("Multiple aggregated CSVs detected. Combining them...")
            aggregated = pd.concat(frames, ignore_index=True)
            # Group by dataset_size and average the means, pool the stds
            aggregated = aggregate_frames([aggregated])  # Re-aggregate to be safe
            has_uncertainty = True
    else:
        # Single raw CSV - legacy behavior without uncertainty
        aggregated = None
        has_uncertainty = False

    if has_uncertainty:
        # Create single plot with uncertainty bands
        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        plot_single_axis(
            ax, aggregated, has_uncertainty=True, reweighted_mse=reweighted_mse
        )
        ax.legend(loc="upper right")
        ax.set_xlabel("Dataset size")
        ax.xaxis.set_label_coords(1.0, -0.11)
        ax.xaxis.label.set_horizontalalignment("right")

        if title:
            fig.suptitle(title, fontsize=17, fontweight="semibold", y=0.995)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
        else:
            fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        # Save aggregated data to CSV (skip if input was already a single aggregated file)
        if not (len(frames) == 1 and frames[0].attrs.get("is_aggregated", False)):
            csv_output_path = output_path.with_suffix(".csv")
            aggregated.to_csv(csv_output_path, index=False)
            print(f"Saved aggregated data to: {csv_output_path}")
    else:
        # Single raw CSV - legacy behavior
        figure_height = 5.8
        fig, axes = plt.subplots(1, 1, figsize=(10.5, figure_height))
        axes = [axes]

        for ax, frame in zip(axes, frames):
            plot_single_axis(
                ax, frame, has_uncertainty=False, reweighted_mse=reweighted_mse
            )
            ax.legend(loc="upper right")
            ax.set_xlabel("Dataset size")
            ax.xaxis.set_label_coords(1.0, -0.11)
            ax.xaxis.label.set_horizontalalignment("right")

        if title:
            fig.suptitle(title, fontsize=17, fontweight="semibold", y=0.995)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
        else:
            fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    return output_path


def main() -> None:
    args = parse_args()
    csv_paths = discover_csv_paths(args.csv_paths)
    if not csv_paths:
        search_root = DEFAULT_ARTIFACTS_DIR.relative_to(REPO_ROOT)
        raise SystemExit(f"No comparison CSVs found under {search_root}.")

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else default_output_path(csv_paths)
    )
    # Load reweighted MSE if provided
    reweighted_mse = load_reweighted_mse(args.reweighted_mse)

    saved_path = plot_csvs(
        csv_paths,
        output_path,
        title=args.title,
        exclude_sizes=args.exclude_sizes,
        reweighted_mse=reweighted_mse,
    )

    print(f"Saved plot to {saved_path}")
    for csv_path in csv_paths:
        print(f"  - used {csv_path}")


if __name__ == "__main__":
    """
    Example:

    >>> python examples/madminer_dataset/plot_mse_vs_dataset_size.py examples/madminer_dataset/artifacts/lora_exploration/neg_ww/comparison_weight_scale_corr_nuisance_param_0_down_seed*.csv --output NegWW_corr_nuisance_param_0_down.png --reweighted-mse examples/madminer_dataset/artifacts/lora_exploration/neg_ww/reweighted_mse_weight_scale_corr_nuisance_param_0_down.json --exclude-sizes 1000 2500 5000 7500 10000
    """
    
    main()
