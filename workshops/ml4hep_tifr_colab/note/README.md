# Hybrid NSBI note

This directory contains the APS Physical Review D–style draft

> A hybrid neural simulation-based inference method for LHC analyses

by Rafael Coelho Lopes de Sá and Jay Sandesara.

The note is intentionally a complete conceptual and methodological draft with explicit result placeholders. Do not replace `pending` values or placeholder panels until the corresponding completed Exercise 5 run has been inspected.

## Build

The source uses REVTeX 4.2 and BibTeX:

```bash
make
```

or directly:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

`make clean` removes intermediate TeX files while retaining `main.pdf`.

## Filling the result figures

Create publication-quality composite PDFs with these exact paths:

- `figures/reference_flow_closure.pdf`
- `figures/hybrid_density_closure.pdf`
- `figures/inference_asimov_toys.pdf`

If a file is absent, the manuscript compiles with a labeled placeholder panel. The scripts exported by Exercise 5 to `exercise5_figures_scripts/` are standalone and contain the numerical artist data, so they can be adjusted without rerunning the trained models. The final composites should be assembled from those outputs.

The minimum result set is:

1. reference-flow marginal and joint/correlation closure;
2. signal/reference and background/reference ratio calibration and reweighting closure;
3. reconstructed hybrid densities compared with analytic truth;
4. hybrid and analytic profile-likelihood scans;
5. the weighted Asimov fit at `mu_A = 1`, including `q_0,A` and `sigma_A`;
6. toy distributions of `mu_hat` and `q_0` compared with the Asimov/Wald predictions.

After reviewing the full notebook output, replace the `pending` entries in Table I and edit any claims in the abstract, demonstration, and conclusion to match the observed closure.
