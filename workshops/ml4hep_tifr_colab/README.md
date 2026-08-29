# ML4HEP-TIFR NSBI tutorial — Google Colab edition

These are **Colab-ready notebooks** for the ML4HEP-TIFR NSBI tutorial. Each
notebook has two cells at the top:

1. an **"Open in Colab"** badge, and
2. a **setup cell** that installs the dependencies, pulls the
   `nsbi_common_utils` package plus the tutorial helpers (`utils.py`,
   `generate_distributions.py`), generates the dataset, and `cd`s into the
   tutorial directory so every relative path in the notebook body resolves
   exactly like a local run.

Use these when the local `pixi` environment isn't available — no install, just a
browser.

## Open in Colab

| Notebook | |
|---|---|
| Exercise 1 — Summary statistics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iris-hep/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_1_summary_statistics.ipynb) |
| Exercise 2.1 — Visualise the data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iris-hep/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_2_1_visualize_data.ipynb) |
| Exercise 2.2a — SigvsRef training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iris-hep/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_2_2a_SigvsRef_training.ipynb) |
| Exercise 2.2b — BkgvsRef training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iris-hep/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_2_2b_BkgvsRef_training.ipynb) |
| Exercise 2.3 — Parameter fitting | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iris-hep/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_2_3_parameter_fitting.ipynb) |
| Exercise 3 — Parameterised CARL | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iris-hep/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_3_parameterized_carl.ipynb) |
| Exercise 4 — Normalizing flows | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iris-hep/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_4_normalizing_flows_density_estimation_direct_likelihood.ipynb) |

# Notes for running on Colab

- **Use a GPU runtime** for the training notebooks (*Runtime → Change runtime
  type → GPU*). The setup cell installs `pytorch-lightning onnx onnxruntime
  onnxscript iminuit mplhep`; everything else (torch, jax, scikit-learn, …)
  ships with Colab. (`onnxscript` is needed by recent `torch.onnx.export`.)
- The setup cell generates a **Colab-sized dataset** (`N_BKG = N_SIG =
  300_000`). Raise those for less Monte-Carlo noise in the fit; lower
  `number_of_epochs` / `N_TRAIN` in the training cells for a quicker pass.
- **Every notebook is self-contained except `Exercise_2_3`**, which *loads* the
  density-ratio models trained by `Exercise_2_2a` and `Exercise_2_2b`. Because
  each Colab notebook is a fresh runtime, either run 2.2a and 2.2b in the same
  runtime first, or set `USE_DRIVE = True` in the setup cell of all three so the
  trained `models_*/` folders persist to your Google Drive.
