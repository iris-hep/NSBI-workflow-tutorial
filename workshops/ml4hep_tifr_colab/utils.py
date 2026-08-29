import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURES = ["x1", "x2", "x3", "x4", "x5"]
_BASIS_COLORS = ["xkcd:lilac", "xkcd:hot pink", "#4ad9d9"]
_COLOR_1SIGMA = "xkcd:light teal"
_COLOR_2SIGMA = "xkcd:light yellow"

# ---------------------------------------------------------------------------
# Lagrange interpolation
# ---------------------------------------------------------------------------


def lagrange_weights(v, nodes):
    weights = []
    for i, ni in enumerate(nodes):
        w = 1.0
        for j, nj in enumerate(nodes):
            if i != j:
                w *= (v - nj) / (ni - nj)
        weights.append(w)
    return weights


# ---------------------------------------------------------------------------
# Train / inference splitting
# ---------------------------------------------------------------------------


def split_train_inference(df, train_fraction=0.5, n_train=None, seed=0):
    """Deterministically partition ``df`` into disjoint (train, inference) sets.

    Density ratios must be *trained* and the likelihood *evaluated* on
    different events -- otherwise the fit inherits the classifier's overfitting.
    Call this with **identical** ``train_fraction``/``n_train`` and ``seed`` in
    the training notebooks (2a, 2b) and the fitting notebook (3): the shared
    seed fixes the permutation, so the "train" half in one notebook and the
    "inference" half in the other are guaranteed non-overlapping.

    Parameters
    ----------
    df : pandas.DataFrame
        Sample to split (e.g. background or signal).
    train_fraction : float
        Fraction of ``df`` assigned to the training subset. Ignored if
        ``n_train`` is given.
    n_train : int or None
        Absolute number of events for the training subset. Takes priority over
        ``train_fraction`` when not None.
    seed : int
        Seed for the permutation. Must match across notebooks.

    Returns
    -------
    (train, inference) : tuple[pandas.DataFrame, pandas.DataFrame]
        Disjoint subsets. If a ``weight`` column is present, each subset's
        weights are rescaled so the subset still sums to the full expected
        yield (the per-event weights represent an expected yield, which the
        physical measurement expects regardless of how many MC events we use).
    """
    n = len(df)
    k = int(n_train) if n_train is not None else int(round(train_fraction * n))
    k = max(0, min(n, k))
    perm = np.random.default_rng(seed).permutation(n)
    train = df.iloc[np.sort(perm[:k])].reset_index(drop=True)
    inference = df.iloc[np.sort(perm[k:])].reset_index(drop=True)
    if "weight" in df.columns:
        total = df["weight"].sum()
        for sub in (train, inference):
            s = sub["weight"].sum()
            if s > 0:
                sub["weight"] = sub["weight"] * (total / s)
    return train, inference


# ---------------------------------------------------------------------------
# Gaussian-mixture model definition
# ---------------------------------------------------------------------------
# This is the SINGLE SOURCE OF TRUTH for the toy distributions.  Both
# ``generate_distributions.py`` (which samples from these mixtures) and the
# parameter-fitting notebook (which evaluates the *analytic* density ratios to
# separate training quality from inference-statistics effects) import these
# definitions, so the "truth" ratios are guaranteed to match the data.


def build_cov(sigmas, correlations=None):
    """Full covariance from per-feature sigmas and a sparse dict of pairwise
    correlations ``{(i, j): rho}``.  Off-diagonal correlations "rotate" the
    components; mixing rotated components gives the curved density contours."""
    sigmas = np.asarray(sigmas, dtype=float)
    corr = np.eye(len(sigmas))
    if correlations:
        for (i, j), rho in correlations.items():
            corr[i, j] = corr[j, i] = rho
    return np.outer(sigmas, sigmas) * corr


# Shared broad component -- present in EVERY sample.  Guarantees overlapping
# support so trained (and truth) density ratios stay bounded everywhere.
BASE_MEAN = np.array([2.5, 2.0, 3.0, 1.5, 2.5])
BASE_SIGMA = np.array([2.5, 2.4, 2.5, 2.4, 2.5])
BASE_COV = build_cov(BASE_SIGMA)
BASE_FRAC = 0.20


def background_components():
    """Bimodal, correlated background mixture: two modes plus the base."""
    bg1 = (
        0.45,
        np.array([2.0, 1.0, 2.5, 0.8, 2.0]),
        build_cov([1.0, 0.9, 1.2, 0.9, 1.0], {(0, 1): 0.6, (3, 4): 0.5}),
    )
    bg2 = (
        0.35,
        np.array([4.2, 3.1, 4.6, 2.6, 3.6]),
        build_cov([0.9, 1.0, 1.0, 0.9, 1.1], {(0, 1): -0.5, (2, 3): 0.5}),
    )
    base = (BASE_FRAC, BASE_MEAN, BASE_COV)
    return [bg1, bg2, base]


def signal_components(v=10):
    """Signal mixture at parameter value ``v`` (default ``v=10``, the signal
    used in the measurement).  Two modes whose means slide with ``t = v / 10``
    (so v=0 sits close to the background and v=10 is the most displaced), plus
    the shared base component."""
    t = v / 10.0
    sig_a = (
        0.50,
        np.array([2.2, 1.3, 2.8, 1.0, 2.2]) + t * np.array([2.3, 1.9, 1.5, 1.3, 1.8]),
        build_cov([0.9, 0.8, 1.0, 0.8, 0.9], {(0, 2): 0.5, (1, 4): 0.4}),
    )
    sig_b = (
        0.30,
        np.array([3.5, 2.5, 3.8, 2.0, 3.0]) + t * np.array([0.8, 0.7, 1.2, 0.6, 1.0]),
        build_cov([0.8, 0.9, 0.9, 0.8, 1.0], {(0, 1): 0.5, (2, 4): -0.4}),
    )
    base = (BASE_FRAC, BASE_MEAN, BASE_COV)
    return [sig_a, sig_b, base]


def mixture_density(X, components):
    """Evaluate the mixture pdf at points ``X`` (shape ``(n, n_dim)``).

    ``components`` is a list of ``(fraction, mean, cov)``; fractions are
    renormalised to sum to 1.  Returns an array of shape ``(n,)``.
    """
    from scipy.stats import multivariate_normal

    X = np.asarray(X, dtype=float)
    fracs = np.array([c[0] for c in components], dtype=float)
    fracs = fracs / fracs.sum()
    p = np.zeros(len(X))
    for (_, mean, cov), f in zip(components, fracs):
        p += f * multivariate_normal(mean, cov).pdf(X)
    return p

def smearing_parameters():
    scale = np.array([1.2, 1.1, 0.99, 0.96, 1.01])
    resolution = np.array([1.0, 0.1, 0.9, 1.3, 0.2])
    return [scale, resolution]

def reference_density(X, lam_bkg, lam_sig0):
    """Analytic density of the *training reference* (numerator of nothing, the
    common denominator).  The reference used in notebooks 2a/2b is
    ``concat(background, signal_0)`` weighted by physics weights, so as a
    normalised density it is the yield-weighted mixture of the two.
    """
    p_bkg = mixture_density(X, background_components())
    p_sig0 = mixture_density(X, signal_components(0))
    return (lam_bkg * p_bkg + lam_sig0 * p_sig0) / (lam_bkg + lam_sig0)


# ---------------------------------------------------------------------------
# MLP Classifier
# ---------------------------------------------------------------------------


class Classifier(nn.Module):
    def __init__(self, n_features, hidden_size=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def classifier_setup(nodes):
    """Return (classifier_names, classifier_colors) from node list."""
    names = [f"signal_{n:g}" for n in nodes]
    colors = {name: c for name, c in zip(names, _BASIS_COLORS)}
    return names, colors


# ---------------------------------------------------------------------------
# Model loading and scoring
# ---------------------------------------------------------------------------


def load_models(clf_name, n_folds, device=None):
    if device is None:
        device = torch.device("cpu")
    models = []
    for fold_idx in range(n_folds):
        m = Classifier(n_features=len(FEATURES))
        m.load_state_dict(
            torch.load(
                f"models/classifier_{clf_name}_fold{fold_idx}.pt",
                map_location=device,
            )
        )
        m.eval()
        models.append(m)
    return models


def score_with_models(X, models):
    """Average sigmoid scores across fold models."""
    X_t = torch.tensor(X, dtype=torch.float32)
    scores = []
    for model in models:
        with torch.no_grad():
            s = torch.sigmoid(model(X_t)).squeeze().numpy()
            scores.append(np.atleast_1d(s))
    return np.mean(scores, axis=0)


def get_outoffold_scores(clf_name, df, n_folds, device=None):
    """
    Out-of-fold scores using the pre-assigned ``fold`` column.
    Each event is scored only by the fold model that did NOT see it
    during training.
    """
    if device is None:
        device = torch.device("cpu")
    X = df[FEATURES].values
    folds = df["fold"].values
    all_scores = np.empty(len(X))
    for fold_idx in range(n_folds):
        val_mask = folds == fold_idx
        m = Classifier(n_features=len(FEATURES))
        m.load_state_dict(
            torch.load(
                f"models/classifier_{clf_name}_fold{fold_idx}.pt",
                map_location=device,
            )
        )
        m.eval()
        X_t = torch.tensor(X[val_mask], dtype=torch.float32).to(device)
        with torch.no_grad():
            s = torch.sigmoid(m(X_t)).squeeze().cpu().numpy()
        all_scores[val_mask] = np.atleast_1d(s)
    return all_scores


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_eval_dataframes(classifier_names):
    bkg_eval = pd.read_parquet("eval_dataframes/background_eval.parquet")
    data_eval = pd.read_parquet("eval_dataframes/data_eval.parquet")
    sig_evals = {
        name: pd.read_parquet(f"eval_dataframes/{name}_eval.parquet")
        for name in classifier_names
    }
    return bkg_eval, sig_evals, data_eval


def load_dataframes(classifier_names):
    bkg_eval = pd.read_parquet("dataframes/background.parquet")
    data_eval = pd.read_parquet("dataframes/data.parquet")
    sig_evals = {
        name: pd.read_parquet(f"dataframes/{name}.parquet") for name in classifier_names
    }
    return bkg_eval, sig_evals, data_eval


# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------


def clip_and_renorm(h):
    """
    Clip a signal histogram to non-negative values, then rescale to preserve
    the total expected yield (= the sum before clipping).

    Negative bins arise from Lagrange interpolation with negative weights and
    are unphysical.  Clipping alone biases the normalization downward, making
    limits artificially weaker.  Renormalizing after clipping keeps the total
    signal yield correct while zeroing out unphysical bins.

    If the pre-clip sum is non-positive (degenerate morphing point), the
    histogram is left all-zero so the caller can detect h.sum() < 1e-6.
    """
    expected = float(h.sum())
    h = np.clip(h, 0, None)
    if expected > 0 and h.sum() > 0:
        h *= expected / h.sum()
    return h


def weighted_quantile_edges(values, weights, n_bins):
    """
    Return n_bins+1 edges so each bin carries approximately equal total
    background weight.  The first edge is pinned to 0.
    """
    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]
    cum_w = np.cumsum(weights[sorted_idx])
    cum_w /= cum_w[-1]  # normalise to [0, 1]
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.interp(quantiles, cum_w, sorted_vals)
    edges[0] = 0.0  # first edge at 0
    return edges


def compute_ratio_bin_edges(r_bkg, bkg_weights, n_bins_nd, min_bkg):
    """
    Compute per-dimension bin edges with guaranteed minimum background per cell.

    Starts from equal-weight quantile edges, then iteratively merges adjacent
    bins in the dimension that eliminates the most under-threshold cells.
    Dimensions may end up with different numbers of bins.
    """
    n_dim = r_bkg.shape[1]
    edges = [
        weighted_quantile_edges(r_bkg[:, d], bkg_weights, n_bins_nd)
        for d in range(n_dim)
    ]
    h, _ = np.histogramdd(r_bkg, bins=edges, weights=bkg_weights)
    n_under = int((h < min_bkg).sum())
    while n_under > 0:
        best_dim, best_idx, best_remaining = None, None, n_under
        for d in range(n_dim):
            if len(edges[d]) <= 2:
                continue
            for i in range(1, len(edges[d]) - 1):
                trial = [e for e in edges]
                trial[d] = np.delete(edges[d], i)
                h_trial, _ = np.histogramdd(r_bkg, bins=trial, weights=bkg_weights)
                n_bad = int((h_trial < min_bkg).sum())
                if n_bad < best_remaining or (
                    n_bad == best_remaining
                    and best_dim is not None
                    and len(trial[d]) > len(edges[best_dim]) - 1
                ):
                    best_dim, best_idx, best_remaining = d, i, n_bad
        if best_dim is None:
            break
        edges[best_dim] = np.delete(edges[best_dim], best_idx)
        h, _ = np.histogramdd(r_bkg, bins=edges, weights=bkg_weights)
        n_under = int((h < min_bkg).sum())
    bin_counts = [len(e) - 1 for e in edges]
    print(
        f"  Edge merging: {n_bins_nd}^{n_dim} -> {bin_counts}  "
        f"({int(np.prod(bin_counts))} cells, all >= {min_bkg} bkg)"
    )
    return edges


def make_ratio_histogram(r_values, weights, bins):
    """Histogram an (N, n_dim) ratio array into the shared nD bins."""
    h, _ = np.histogramdd(r_values, bins=bins, weights=weights)
    return h
