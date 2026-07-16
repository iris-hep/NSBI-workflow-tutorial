# Hybrid NSBI note

This directory contains the APS Physical Review D–style draft

> A hybrid neural simulation-based inference method for LHC analyses

by Rafael Coelho Lopes de Sá and Jay Sandesara.

The note contains the conceptual development, finite-quadrature derivations, and numerical results from the completed Exercise 5 and Exercise 6 runs. It includes the hybrid-density, weighted-Asimov, pseudo-experiment, and neural-importance-sampling demonstrations.

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

## Result figures

The current draft includes the following composites assembled from the executed notebook outputs:

- `figures/hybrid_reweighting_closure.png`
- `figures/hybrid_profile.png`
- `figures/asimov_toys.png`
- `figures/nis_proposal.png`
- `figures/nis_reweighting.png`
- `figures/nis_convergence.png`

The scripts exported by Exercise 5 to `exercise5_figures_scripts/` are standalone and contain their numerical plotting data, so the figures can be refined without rerunning the trained models. The plots produced by Exercise 6 are saved alongside its NIS outputs.

The present result set shows:

1. signal/reference and background/reference reweighting closure;
2. hybrid and analytic profile-likelihood scans;
3. exact finite-quadrature Asimov closure at `mu_A = 1`;
4. toy distributions of `mu_hat` and `q_0` compared with the Asimov/Wald predictions;
5. NIS proposal and importance-reweighting validation; and
6. direct-reference versus NIS convergence for `q_0,A` and the complete expected scan.
