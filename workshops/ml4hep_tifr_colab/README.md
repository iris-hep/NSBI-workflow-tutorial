# ML4HEP-TIFR NSBI tutorial — Google Colab edition

These are the **Colab-ready tutorial notebooks**. Exercises 1--4 mirror the
original ML4HEP-TIFR sequence, while Exercises 5--7 extend the tutorial with
hybrid neural ratio estimation, efficient Asimov sampling, and a study of
model misspecification. Each notebook includes:

1. exactly one **"Open in Colab"** badge, and
2. a **setup cell** near the beginning that installs the dependencies, pulls the
   `nsbi_common_utils` package plus the tutorial helpers (`utils.py`,
   `generate_distributions.py`), creates or reuses the legacy working directory,
   and generates the dataset there when it is missing.

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
| Exercise 5 — Hybrid flow and density ratios | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iris-hep/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_5_Hybrid_NormalizingFlow_DensityRatio.ipynb) |
| Exercise 6 — Neural importance-sampled Asimov data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iris-hep/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_6_NeuralImportanceSampling_Asimov.ipynb) |
| Exercise 7 — Asimov closure and misspecification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iris-hep/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_7_Asimov_Misspecification_Coverage.ipynb) |

## Notes for running on Colab

- The helper code is loaded from `workshops/ml4hep_tifr_colab`, but generated
  samples, trained models, saved densities, and plots remain in the legacy
  `workshops/ml4hep_tifr` working directory. The setup cell creates that
  directory when needed and reuses it when it already exists, including in
  Google Drive, so earlier training outputs remain available.
- **Use a GPU runtime** for the training notebooks (*Runtime → Change runtime
  type → GPU*). The setup cell installs `pytorch-lightning onnx onnxruntime
  onnxscript iminuit mplhep`; everything else (torch, jax, scikit-learn, …)
  ships with Colab. (`onnxscript` is needed by recent `torch.onnx.export`.)
- Dataset sizes are configured near the top of each setup cell. Lower them, or
  lower `number_of_epochs` / `N_TRAIN` in the training cells, for a quicker
  pass; raise them for less Monte-Carlo noise in the fit.
- **Every notebook is self-contained except `Exercise_2_3`**, which *loads* the
  density-ratio models trained by `Exercise_2_2a` and `Exercise_2_2b`. Because
  each Colab notebook is a fresh runtime, either run 2.2a and 2.2b in the same
  runtime first, or set `USE_DRIVE = True` in the setup cell of all three so the
  trained `models_*/` folders persist to your Google Drive.
- Exercises 6 and 7 load the expensive PRESEL, reference-flow, and ratio
  checkpoints produced by Exercise 5. Run Exercise 5 first and keep
  `USE_DRIVE = True` in all three notebooks so those checkpoints persist.
