"""Plotting helpers for the ML4HEP TIFR normalizing-flow tutorial.

The notebook remains responsible for sampling events and evaluating learned or
analytic probabilities.  Functions here only turn those prepared arrays into
figures, keeping the pedagogical data flow visible in the exercise.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import chi2, norm


__all__ = [
    "plot_flow_pair_closure",
    "plot_log_density_truth_binned",
    "plot_log_density_truth_scatter",
    "plot_log_prob_cdf_closure",
    "plot_log_prob_closure",
    "plot_mu_hat_toys",
    "plot_profile_scan",
    "plot_profile_scan_comparison",
    "plot_t_mu_toys",
]


def _highest_density_contour_levels(histogram, probabilities=(0.95, 0.68)):
    """Return density thresholds enclosing the requested probability masses."""
    values = np.asarray(histogram, dtype=float).ravel()
    values = values[np.isfinite(values) & (values > 0.0)]
    if len(values) == 0:
        return np.array([])

    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    cumulative /= cumulative[-1]
    levels = []
    for probability in probabilities:
        index = min(np.searchsorted(cumulative, probability), len(ordered) - 1)
        levels.append(ordered[index])

    levels = np.unique(np.sort(levels))
    return levels[(levels > 0.0) & (levels < ordered[0])]


def plot_flow_pair_closure(
    sample_name,
    mc,
    generated,
    feature_names,
    n_bins_1d=60,
    n_bins_2d=35,
    contour_smoothing=1.0,
    quantile_range=(0.005, 0.995),
):
    """Pair plot of prepared held-out and flow-generated event arrays."""
    mc = np.asarray(mc)
    generated = np.asarray(generated)
    feature_names = list(feature_names)
    if mc.ndim != 2 or generated.ndim != 2 or mc.shape[1] != generated.shape[1]:
        raise ValueError("mc and generated must be 2D arrays with matching columns.")
    if mc.shape[1] != len(feature_names):
        raise ValueError("feature_names must match the number of array columns.")

    q_low, q_high = quantile_range
    if not 0.0 <= q_low < q_high <= 1.0:
        raise ValueError("quantile_range must satisfy 0 <= low < high <= 1.")

    n_features = len(feature_names)
    limits = []
    for feature_index in range(n_features):
        low = min(
            np.quantile(mc[:, feature_index], q_low),
            np.quantile(generated[:, feature_index], q_low),
        )
        high = max(
            np.quantile(mc[:, feature_index], q_high),
            np.quantile(generated[:, feature_index], q_high),
        )
        limits.append((low, high))

    corr_mc = np.corrcoef(mc, rowvar=False)
    corr_generated = np.corrcoef(generated, rowvar=False)
    fig, axes = plt.subplots(
        n_features,
        n_features,
        figsize=(2.55 * n_features, 2.55 * n_features),
        squeeze=False,
    )

    for row in range(n_features):
        for column in range(n_features):
            ax = axes[row, column]
            x_low, x_high = limits[column]

            if row == column:
                bins = np.linspace(x_low, x_high, n_bins_1d + 1)
                ax.hist(
                    mc[:, column],
                    bins=bins,
                    density=True,
                    histtype="step",
                    color="black",
                    lw=1.8,
                    label="held-out MC",
                )
                ax.hist(
                    generated[:, column],
                    bins=bins,
                    density=True,
                    histtype="step",
                    color="C1",
                    lw=1.8,
                    ls="--",
                    label="flow sample",
                )
                ax.set_xlim(x_low, x_high)
                if row == 0:
                    ax.set_ylabel("density")

            elif row > column:
                y_low, y_high = limits[row]
                x_edges = np.linspace(x_low, x_high, n_bins_2d + 1)
                y_edges = np.linspace(y_low, y_high, n_bins_2d + 1)
                x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
                y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

                hist_mc, _, _ = np.histogram2d(
                    mc[:, column], mc[:, row], bins=(x_edges, y_edges)
                )
                hist_generated, _, _ = np.histogram2d(
                    generated[:, column], generated[:, row], bins=(x_edges, y_edges)
                )
                if contour_smoothing > 0.0:
                    hist_mc = gaussian_filter(hist_mc, sigma=contour_smoothing)
                    hist_generated = gaussian_filter(
                        hist_generated, sigma=contour_smoothing
                    )

                levels_mc = _highest_density_contour_levels(hist_mc)
                levels_generated = _highest_density_contour_levels(hist_generated)
                if len(levels_mc):
                    ax.contour(
                        x_centers,
                        y_centers,
                        hist_mc.T,
                        levels=levels_mc,
                        colors="black",
                        linewidths=1.5,
                    )
                if len(levels_generated):
                    ax.contour(
                        x_centers,
                        y_centers,
                        hist_generated.T,
                        levels=levels_generated,
                        colors="C1",
                        linestyles="--",
                        linewidths=1.5,
                    )
                ax.set_xlim(x_low, x_high)
                ax.set_ylim(y_low, y_high)

            else:
                rho_mc = corr_mc[row, column]
                rho_generated = corr_generated[row, column]
                delta_rho = rho_generated - rho_mc
                ax.set_axis_off()
                ax.text(
                    0.5,
                    0.62,
                    rf"$\rho_{{\rm MC}}={rho_mc:+.3f}$",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )
                ax.text(
                    0.5,
                    0.44,
                    rf"$\rho_{{\rm flow}}={rho_generated:+.3f}$",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="C1",
                )
                ax.text(
                    0.5,
                    0.25,
                    rf"$\Delta\rho={delta_rho:+.3f}$",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight="bold",
                )

            if row >= column:
                ax.tick_params(
                    axis="x",
                    labelbottom=(row == n_features - 1),
                    labelsize=8,
                )
                ax.tick_params(
                    axis="y",
                    labelleft=(column == 0),
                    labelsize=8,
                )
                if row == n_features - 1:
                    ax.set_xlabel(feature_names[column])
                if column == 0 and row > 0:
                    ax.set_ylabel(feature_names[row])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(f"{sample_name}: feature and correlation closure", y=0.999)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    return fig


def plot_log_prob_closure(
    sample_name,
    log_p_mc,
    log_p_generated,
    n_bins=60,
    quantile_range=(0.001, 0.999),
    flow_color="C1",
):
    """Histogram and ratio for two prepared learned-log-density arrays."""
    log_p_mc = np.asarray(log_p_mc)
    log_p_generated = np.asarray(log_p_generated)
    if len(log_p_mc) == 0 or len(log_p_generated) == 0:
        raise ValueError("The log-probability arrays must be non-empty.")

    q_low, q_high = quantile_range
    if not 0.0 <= q_low < q_high <= 1.0:
        raise ValueError("quantile_range must satisfy 0 <= low < high <= 1.")
    x_min = min(
        np.quantile(log_p_mc, q_low),
        np.quantile(log_p_generated, q_low),
    ) - 5.0
    x_max = max(
        np.quantile(log_p_mc, q_high),
        np.quantile(log_p_generated, q_high),
    ) + 5.0

    edges = np.linspace(x_min, x_max, n_bins + 1)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mc_counts, _ = np.histogram(log_p_mc, bins=edges)
    generated_counts, _ = np.histogram(log_p_generated, bins=edges)

    mc_norm = mc_counts.sum()
    generated_norm = generated_counts.sum()
    if mc_norm == 0 or generated_norm == 0:
        raise ValueError("No log-probability entries fall inside the plotting range.")
    mc_density = mc_counts / (mc_norm * widths)
    mc_error = np.sqrt(mc_counts) / (mc_norm * widths)
    generated_density = generated_counts / (generated_norm * widths)

    fig, (ax, ratio_ax) = plt.subplots(
        2,
        1,
        figsize=(6.4, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )
    ax.stairs(
        generated_density,
        edges,
        color=flow_color,
        lw=2,
        fill=True,
        alpha=0.22,
        label=f"flow sample ({len(log_p_generated):,} events)",
    )
    ax.stairs(generated_density, edges, color=flow_color, lw=2)
    ax.errorbar(
        centers,
        mc_density,
        yerr=mc_error,
        fmt="o",
        ms=3.5,
        color="black",
        capsize=1.5,
        label=f"held-out MC ({len(log_p_mc):,} events)",
    )
    ax.set_ylabel("normalized events\n/ unit log density")
    ax.set_title(f"{sample_name}: log-density closure")
    ax.legend(fontsize=9)

    valid = (mc_counts > 0) & (generated_counts > 0)
    ratio = mc_density[valid] / generated_density[valid]
    ratio_error = ratio * np.sqrt(
        1.0 / mc_counts[valid] + 1.0 / generated_counts[valid]
    )
    ratio_ax.errorbar(
        centers[valid],
        ratio,
        yerr=ratio_error,
        fmt="o",
        ms=3.5,
        color="black",
        capsize=1.5,
    )
    ratio_ax.axhline(1.0, color=flow_color, lw=1.5)
    ratio_ax.set_xlabel(r"$\log \hat p_{\rm flow}(x)$")
    ratio_ax.set_ylabel("MC / flow")
    ratio_ax.set_xlim(edges[0], edges[-1])
    ratio_ax.grid(axis="y", alpha=0.25)
    fig.align_ylabels((ax, ratio_ax))
    fig.subplots_adjust(hspace=0.05)
    return fig


def plot_log_prob_cdf_closure(
    sample_name,
    log_p_mc,
    log_p_generated,
    n_cdf_points=200,
    cdf_range=(0.001, 0.999),
    color="C0",
):
    """P-P plot and CDF ratio for prepared learned-log-density arrays."""
    log_p_mc = np.asarray(log_p_mc)
    log_p_generated = np.asarray(log_p_generated)
    flow_cdf = np.linspace(cdf_range[0], cdf_range[1], n_cdf_points)
    if not 0.0 < flow_cdf[0] < flow_cdf[-1] < 1.0:
        raise ValueError("cdf_range must lie strictly inside (0, 1).")

    thresholds = np.quantile(log_p_generated, flow_cdf)
    sorted_mc = np.sort(log_p_mc)
    mc_cdf = np.searchsorted(sorted_mc, thresholds, side="right") / len(sorted_mc)
    cdf_sigma = np.sqrt(
        flow_cdf
        * (1.0 - flow_cdf)
        * (1.0 / len(log_p_mc) + 1.0 / len(log_p_generated))
    )
    lower = np.clip(flow_cdf - cdf_sigma, 0.0, 1.0)
    upper = np.clip(flow_cdf + cdf_sigma, 0.0, 1.0)

    fig, (ax, ratio_ax) = plt.subplots(
        2,
        1,
        figsize=(6.2, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )
    ax.fill_between(
        flow_cdf,
        lower,
        upper,
        color=color,
        alpha=0.18,
        label=r"pointwise sampling expectation ($\pm1\sigma$)",
    )
    ax.plot(flow_cdf, mc_cdf, color=color, lw=2, label="held-out MC versus flow")
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color="black",
        ls="--",
        lw=1.5,
        label="perfect closure",
    )
    ax.set_ylabel(r"MC CDF $F_{\rm MC}(\log \hat p)$")
    ax.set_title(f"{sample_name}: log-density CDF closure")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")

    cdf_ratio = mc_cdf / flow_cdf
    ratio_sigma = cdf_sigma / flow_cdf
    ratio_ax.fill_between(
        flow_cdf,
        np.clip(1.0 - ratio_sigma, 0.0, None),
        1.0 + ratio_sigma,
        color=color,
        alpha=0.18,
    )
    ratio_ax.plot(flow_cdf, cdf_ratio, color=color, lw=2)
    ratio_ax.axhline(1.0, color="black", ls="--", lw=1.5)
    ratio_ax.set_xlabel(r"flow CDF $F_{\rm flow}(\log \hat p)$")
    ratio_ax.set_ylabel(r"$F_{\rm MC}/F_{\rm flow}$")
    ratio_ax.set_xlim(0.0, 1.0)
    ratio_ax.grid(axis="y", alpha=0.25)

    ratio_span = max(0.05, 1.1 * np.max(np.abs(cdf_ratio - 1.0) + ratio_sigma))
    ratio_ax.set_ylim(max(0.0, 1.0 - ratio_span), 1.0 + ratio_span)
    fig.align_ylabels((ax, ratio_ax))
    fig.subplots_adjust(hspace=0.05)
    return fig


def plot_log_density_truth_scatter(sample_name, log_p_truth, log_p_flow):
    """Scatter plot of prepared analytic and learned log-density arrays."""
    log_p_truth = np.asarray(log_p_truth)
    log_p_flow = np.asarray(log_p_flow)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(log_p_truth, log_p_flow, s=4, alpha=0.25)
    lo = min(log_p_truth.min(), log_p_flow.min())
    hi = max(log_p_truth.max(), log_p_flow.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("truth log density")
    ax.set_ylabel("flow log density")
    ax.set_title(sample_name)
    fig.tight_layout()
    return fig


def plot_log_density_truth_binned(
    sample_name,
    calibration,
    edges,
    log_p_truth=None,
    log_p_flow=None,
    show_scatter=False,
):
    """Plot a prepared binned log-density calibration table."""
    edges = np.asarray(edges)
    valid = calibration["count"] > 0
    fig, (ax, residual_ax) = plt.subplots(
        2,
        1,
        figsize=(6.4, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    if show_scatter:
        if log_p_truth is None or log_p_flow is None:
            raise ValueError("Raw log-density arrays are required for show_scatter=True.")
        ax.scatter(log_p_truth, log_p_flow, s=3, alpha=0.10)

    flow_step = calibration["flow_mean"].to_numpy()
    delta_step = calibration["delta_mean"].to_numpy()
    ax.step(edges, np.r_[flow_step, flow_step[-1]], where="post", lw=2, label="bin mean")
    ax.errorbar(
        calibration.loc[valid, "truth_mean"],
        calibration.loc[valid, "flow_mean"],
        yerr=calibration.loc[valid, "flow_sem"],
        fmt="o",
        ms=3,
        capsize=2,
    )
    lo = min(edges[0], np.nanmin(flow_step))
    hi = max(edges[-1], np.nanmax(flow_step))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect calibration")
    ax.set_ylabel(r"$\langle \log \hat p_{\rm flow} \rangle$")
    ax.set_title(f"{sample_name}: binned log-density calibration")
    ax.legend(fontsize=8)

    residual_ax.step(edges, np.r_[delta_step, delta_step[-1]], where="post", lw=2)
    residual_ax.errorbar(
        calibration.loc[valid, "truth_mean"],
        calibration.loc[valid, "delta_mean"],
        yerr=calibration.loc[valid, "delta_sem"],
        fmt="o",
        ms=3,
        capsize=2,
    )
    residual_ax.axhline(0.0, color="k", lw=1)
    residual_ax.set_xlabel(r"truth $\log p(x)$ bin")
    residual_ax.set_ylabel(r"$\langle \Delta \log p \rangle$")
    residual_ax.set_xlim(edges[0], edges[-1])
    return fig


def plot_profile_scan(scan, t_mu, label="Flow densities", reference_mu=1.0):
    """Plot one prepared profile-likelihood scan."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(scan, t_mu, lw=2, label=label)
    ax.axvline(reference_mu, color="black", ls=":", lw=1, alpha=0.7)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$\mu_{\rm signal}$")
    ax.set_ylabel(r"$t_\mu$")
    ax.legend()
    return fig


def plot_profile_scan_comparison(
    scan_flow,
    t_mu_flow,
    mu_min_flow,
    scan_truth,
    t_mu_truth,
    mu_min_truth,
    reference_mu=1.0,
):
    """Compare prepared learned- and analytic-density likelihood scans."""
    mu_min_flow = float(np.asarray(mu_min_flow).reshape(-1)[0])
    mu_min_truth = float(np.asarray(mu_min_truth).reshape(-1)[0])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        np.asarray(scan_flow) - mu_min_flow + reference_mu,
        t_mu_flow,
        lw=2,
        label="Flow densities",
    )
    ax.plot(
        np.asarray(scan_truth) - mu_min_truth + reference_mu,
        t_mu_truth,
        lw=2,
        ls="--",
        label="Analytic reco densities",
    )
    ax.axvline(reference_mu, color="black", ls=":", lw=1, alpha=0.7)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$\mu_{\rm signal}$")
    ax.set_ylabel(r"$t_\mu$")
    ax.legend()
    return fig


def plot_t_mu_toys(toy_results, n_bins=35):
    """Compare prepared toy test statistics with Wilks/Cowan expectations."""
    hypotheses = sorted(toy_results["mu_true"].unique())
    fig, axes = plt.subplots(1, len(hypotheses), figsize=(6.2 * len(hypotheses), 4.6))
    axes = np.atleast_1d(axes)

    for ax, mu_true in zip(axes, hypotheses):
        values = toy_results.loc[toy_results["mu_true"] == mu_true, "t_mu"].to_numpy()
        upper = max(float(chi2.ppf(0.999, df=1)), 1.001 * float(values.max()))
        edges = np.linspace(0.0, upper, n_bins + 1)
        toy_probability = np.histogram(values, bins=edges)[0] / len(values)
        chi2_probability = np.diff(chi2.cdf(edges, df=1))

        if np.isclose(mu_true, 0.0):
            asymptotic_probability = 0.5 * chi2_probability
            asymptotic_probability[0] += 0.5
            theory_label = r"Cowan: $\frac{1}{2}\delta(0)+\frac{1}{2}\chi^2_1$"
        else:
            asymptotic_probability = chi2_probability
            theory_label = r"Wilks: $\chi^2_1$"

        ax.stairs(
            toy_probability,
            edges,
            color="C0",
            lw=2,
            fill=True,
            alpha=0.22,
            label="flow toys",
        )
        ax.stairs(
            asymptotic_probability,
            edges,
            color="C3",
            lw=2,
            ls="--",
            label=theory_label,
        )
        ax.set_xlabel(r"$t_\mu$")
        ax.set_ylabel("probability / bin")
        ax.set_title(rf"toys generated at $\mu={mu_true:g}$")
        ax.legend(fontsize=9)
        if np.isclose(mu_true, 0.0):
            zero_fraction = np.mean(values < 1.0e-10)
            ax.text(
                0.97,
                0.72,
                rf"toy $P(t_0=0)={zero_fraction:.3f}$",
                ha="right",
                transform=ax.transAxes,
            )

    fig.tight_layout()
    return fig


def plot_mu_hat_toys(toy_results, sigma_by_mu, n_bins=35):
    """Compare prepared bounded estimators with the Wald/Cowan prediction."""
    hypotheses = sorted(toy_results["mu_true"].unique())
    fig, axes = plt.subplots(1, len(hypotheses), figsize=(6.2 * len(hypotheses), 4.6))
    axes = np.atleast_1d(axes)

    for ax, mu_true in zip(axes, hypotheses):
        values = toy_results.loc[toy_results["mu_true"] == mu_true, "mu_hat"].to_numpy()
        sigma = float(sigma_by_mu[float(mu_true)])
        upper = max(
            float(mu_true) + 5.0 * sigma,
            5.0 * sigma,
            1.001 * float(values.max()),
        )
        edges = np.linspace(0.0, upper, n_bins + 1)
        toy_probability = np.histogram(values, bins=edges)[0] / len(values)
        gaussian_cdf = norm.cdf((edges - float(mu_true)) / sigma)
        asymptotic_probability = np.diff(gaussian_cdf)
        boundary_mass = float(norm.cdf(-float(mu_true) / sigma))
        asymptotic_probability[0] += boundary_mass

        ax.stairs(
            toy_probability,
            edges,
            color="C2",
            lw=2,
            fill=True,
            alpha=0.22,
            label="flow toys",
        )
        ax.stairs(
            asymptotic_probability,
            edges,
            color="C3",
            lw=2,
            ls="--",
            label=rf"bounded Wald ($\sigma_\mu={sigma:.3g}$)",
        )
        ax.axvline(float(mu_true), color="black", ls=":", lw=1.5)
        ax.set_xlabel(r"$\hat\mu$")
        ax.set_ylabel("probability / bin")
        ax.set_title(rf"toys generated at $\mu={mu_true:g}$")
        ax.legend(fontsize=9)

    fig.tight_layout()
    return fig
