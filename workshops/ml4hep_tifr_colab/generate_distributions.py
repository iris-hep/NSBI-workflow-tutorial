"""Gaussian-mixture event generation for the ML4HEP-TIF NSBI tutorial.

Unlike the ``nsbi_atlas_workshop`` example -- where every feature is a single
unimodal Gaussian -- here each sample is a *mixture* of several correlated,
differently-oriented Gaussian components.  This gives

  * multi-modal marginals (bumps that a single Gaussian cannot capture), and
  * curved iso-density contours from mixing rotated components,

which make the joint density genuinely non-trivial to model -- a good stress
test for a Normalizing Flow density estimator (planned follow-up), while still
being cheap to sample and to reason about analytically.

We only generate the two samples the workflow actually uses -- ``background``
and ``signal`` -- plus an independent ``data`` draw for visualisation.

Numerical-stability design
---------------------------
Density-ratio estimation is only well behaved when the reference (denominator)
has support everywhere the numerator does.  To guarantee bounded ratios when
we train one sample against another, **every** sample here contains a common,
broad ``BASE`` component (see ``utils.BASE_MEAN`` / ``utils.BASE_SIGMA``)
carrying a non-negligible mixing fraction.  Because that component alone already
covers the whole region of interest with p(x) > 0, no phase-space pocket exists
where one density vanishes while another does not -- so p_a(x)/p_b(x) never
blows up.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import background_components, signal_components, smearing_parameters

parser = argparse.ArgumentParser()
parser.add_argument("--n_bkg", type=int, default=1_000_000)
parser.add_argument("--n_sig", type=int, default=100_000)
args = parser.parse_args()

n_bkg = args.n_bkg
n_sig = args.n_sig

features = ["z1", "z2", "z3", "z4", "z5"]
reco = ["x1", "x2", "x3", "x4", "x5"]

# Total expected yields (define the signal strength of the measurement).
LAM_BKG = 1_000_000.0
LAM_SIG = 1_100.0

SIGNAL_COLOR = "xkcd:hot pink"


# The mixture definitions (build_cov, BASE_*, background_components,
# signal_components) live in utils.py so the parameter-fitting notebook can
# import the exact same distributions to compute the "truth" density ratios.
def sample_mixture(components, n, rng):
    """Sample ``n`` events from a Gaussian mixture.

    ``components`` is a list of ``(fraction, mean, cov)``.  Fractions need not
    sum to 1 exactly; they are renormalised.  Component assignment is
    multinomial so the empirical fractions match the requested ones.
    """
    fracs = np.array([c[0] for c in components], dtype=float)
    fracs = fracs / fracs.sum()
    counts = rng.multinomial(n, fracs)
    chunks = []
    for (_, mean, cov), c in zip(components, counts):
        if c > 0:
            chunks.append(rng.multivariate_normal(mean, cov, size=c))
    x = np.concatenate(chunks, axis=0)
    rng.shuffle(x)  # avoid block-ordering by component
    return x

def add_reco_smearing(df, rng):
    """Add x1,...,x5 as independently smeared versions of y1,...,y5."""
    scale, resolution = smearing_parameters()

    y = df[features].to_numpy(dtype=float)

    x = rng.normal(
        loc=y * scale[None, :],
        scale=resolution[None, :],
        size=y.shape,
    )

    df[reco] = x
    return df

rng = np.random.default_rng(42)

# --- Background ---
background = pd.DataFrame(
    sample_mixture(background_components(), n_bkg, rng), columns=features
)
background["fold"] = rng.integers(0, 2, size=n_bkg)
background["label"] = 0
background["weight"] = LAM_BKG / n_bkg  # total yield = LAM_BKG

# --- Signal ---
signal = pd.DataFrame(
    sample_mixture(signal_components(), n_sig, rng), columns=features
)
signal["fold"] = rng.integers(0, 2, size=n_sig)
signal["label"] = 1
signal["weight"] = LAM_SIG / n_sig  # total yield = LAM_SIG

# --- Pseudo-data: an independent draw from the background mixture ---
data = pd.DataFrame(
    sample_mixture(background_components(), n_bkg, rng), columns=features
)
data["fold"] = rng.integers(0, 2, size=n_bkg)
data["label"] = 0

background = add_reco_smearing(background, rng)
signal = add_reco_smearing(signal, rng)
data = add_reco_smearing(data, rng)

os.makedirs("dataframes", exist_ok=True)
background.to_parquet("dataframes/background.parquet", index=False)
signal.to_parquet("dataframes/signal.parquet", index=False)
data.to_parquet("dataframes/data.parquet", index=False)

os.makedirs("plots", exist_ok=True)


def feature_bins(feat):
    lo = min(background[feat].quantile(0.005), signal[feat].quantile(0.005))
    hi = max(background[feat].quantile(0.995), signal[feat].quantile(0.995))
    return np.linspace(lo, hi, 61)


# --- Plot 1: per-feature signal vs background (with pseudo-data points) ---
for feat in features:
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = feature_bins(feat)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    ax.errorbar(
        bin_centers,
        np.histogram(data[feat], bins=bins, density=True)[0],
        yerr=0,
        fmt="k.",
        capsize=3,
        ms=4,
        lw=1.2,
        label="Data",
        zorder=5,
    )
    ax.hist(
        background[feat],
        bins=bins,
        density=True,
        histtype="stepfilled",
        lw=2,
        color="black",
        alpha=0.15,
        label="Background",
    )
    ax.hist(
        signal[feat],
        bins=bins,
        density=True,
        histtype="step",
        lw=2,
        color=SIGNAL_COLOR,
        label="Signal",
    )
    ax.set_xlabel(feat, loc="right")
    ax.set_ylabel("Density", loc="top")
    ax.legend(fontsize=8)
    fig.savefig(f"plots/{feat}.pdf", bbox_inches="tight")
    plt.close(fig)

# --- Plot 2: 2D views exposing the multi-modal, correlated structure ---
# A LOG colour scale is essential here: on a linear scale the sharp background
# peak saturates the colour map and hides the broad, low-but-nonzero density
# floor (the shared BASE component). That floor is precisely what makes the
# support overlap total and the density ratios bounded, so we must show it.
from matplotlib.colors import LogNorm

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (fi, fj) in zip(axes, [(0, 1), (2, 3)]):
    ax.hist2d(
        background[features[fi]],
        background[features[fj]],
        bins=80,
        cmap="Greys",
        norm=LogNorm(),
    )
    ax.scatter(
        signal[features[fi]][:2000],
        signal[features[fj]][:2000],
        s=3,
        color=SIGNAL_COLOR,
        alpha=0.4,
        label="signal",
    )
    ax.set_xlabel(features[fi])
    ax.set_ylabel(features[fj])
    ax.legend(fontsize=8)
fig.suptitle("Background density on a LOG scale (grey) with signal overlaid")
fig.savefig("plots/2d_structure.pdf", bbox_inches="tight")
plt.close(fig)

print(
    f"Done. Generated background ({n_bkg:,}, yield {LAM_BKG:g}) and "
    f"signal ({n_sig:,}, yield {LAM_SIG:g}). Saved to dataframes/, plots to plots/"
)
