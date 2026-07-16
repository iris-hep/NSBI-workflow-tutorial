# Hybrid NSBI note

This directory contains the APS Physical Review D–style draft

> A hybrid neural simulation-based inference method for LHC analyses

by Rafael Coelho Lopes de Sá and Jay Sandesara.

The manuscript includes the reviewed outputs of the successfully executed Exercise 5 and Exercise 6 notebooks: hybrid-density inference, weighted unbinned Asimov closure, pseudo-experiment validation, and the neural importance-sampling efficiency study.

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

The figures currently used by `main.tex` are stored in `figures/` and documented in `figures/README.md`. They are direct notebook outputs and are intended as scientifically accurate draft figures. They can be restyled for publication using the data-complete standalone scripts exported by Exercise 5 and the saved Exercise 6 plotting outputs.

The selected result set covers:

1. signal/reference and background/reference reweighting closure;
2. hybrid and analytic profile-likelihood scans;
3. the weighted Asimov prediction compared with `mu_hat` and `q_0` toys;
4. neural proposal and importance-reweighting validation;
5. equal-size convergence of direct and neural-importance Asimov quadratures.
