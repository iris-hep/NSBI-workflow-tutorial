FAIR Universe Dataset
--

The tabular dataset used in this demonstration is hosted on Zenodo (https://zenodo.org/records/15131565), and is created using the particle physics simulation tools Pythia 8.2 and Delphes 3.5.0. The dataset provides events for the $H\to \tau\tau$ analysis, where the signal process is sub-dominant compared to the very large $Z\to \tau\tau$ and other backgrounds - good challenge to test the sensitivty of NSBI techniques.

## Download saved models and processed data

If you want to skip the data loading and preprocessing stages — and optionally use pre-trained ensemble networks — download the pre-processed bundle from [https://cernbox.cern.ch/s/mltUDvzKdisEEpJ](https://cernbox.cern.ch/s/mltUDvzKdisEEpJ) and extract it:

```bash
tar -xvf saved_datasets.tar.gz
```

This creates a `saved_datasets/` directory with the processed `.root` files (and optionally the trained models). **If you use this download, do not re-run notebooks 1 (data loader) and 2 (data preprocessing).** They regenerate exactly the artifacts you just downloaded and would overwrite them. Start from notebook 3 (or notebook 4 if you also pulled pre-trained models).

> **Path note:** the shipped `config.pipeline.yaml` uses absolute paths under `/projects/Physics_Cranmer/saved_datasets/` (and the fit configs `config_fit_nsbi.yml` / `config_fit_histogram.yml` likewise). If you extract `saved_datasets.tar.gz` somewhere else (e.g. next to this README), repoint the `saved_data_path` / `output.dir` keys in `config.pipeline.yaml` and the `SamplePath` / `Models` / `Ratios` / `AsimovWeights` entries in the fit configs to your local extraction directory before running the notebooks or scripts.

## Three ways to run this example

The pipeline runs through any of three equivalent entry points, all driven by this directory's `config.pipeline.yaml`:

1. **Notebooks (interactive)** — step through stages 1-7:

   ```bash
   cd examples/FAIR_universe_Higgs_tautau
   jupyter lab     # or: jupyter notebook
   ```

   Open `1_data_loader.ipynb` through `7_parameter_fitting_with_systematics.ipynb` in order. Best for learning and inspecting intermediate outputs.

2. **Scripts (headless CLI)** — run each stage as a script. From this directory:

   ```bash
   python scripts/data_loader.py --config config.pipeline.yaml
   python scripts/data_preprocessing.py --config config.pipeline.yaml
   python scripts/preselection_network.py --config config.pipeline.yaml
   python scripts/neural_likelihood_ratio_estimation.py \
       --config config.pipeline.yaml --process htautau --ensemble_index 0
   python scripts/systematic_uncertainty_training.py \
       --config config.pipeline.yaml --process htautau --systematic JES --direction Up
   python scripts/data_nn_eval.py --config config.pipeline.yaml
   python scripts/parameter_fitting.py --config config.pipeline.yaml
   ```

   Stages 4 and 5 are per-process / per-systematic — repeat them across the values you want trained, or use Snakemake for automatic fan-out.

3. **Snakemake (parallel on a cluster)** — orchestrates the scripts as a parallel DAG (see next section).

If you downloaded the pre-processed bundle above, **skip stages 1 and 2** in whichever path you choose — they regenerate exactly that data.

See [`docs/basics/workflow.rst`](../../docs/basics/workflow.rst) for the full workflow reference: notebook order, CLI submission, monitoring, partial reruns, and adapting to other clusters.

## Running the pipeline with Snakemake on HTCondor

The whole pipeline is one [Snakemake](https://snakemake.readthedocs.io/) workflow. From the repository root:

```bash
snakemake --snakefile examples/FAIR_universe_Higgs_tautau/Snakefile \
          --profile  examples/FAIR_universe_Higgs_tautau/profiles/chtc
```

That submits every required job to HTCondor (via the `snakemake-executor-plugin-htcondor`), waits for completion, and produces the final `parameter_fitting` outputs. Re-running the same command after a failure resumes from where it stopped — sentinels under `/projects/.../sentinels_FAIR_higgs/` track which (process, fold, ensemble) jobs are done.

To target a different cluster, copy `profiles/chtc/` to `profiles/<your-cluster>/`, change `executor:` to the appropriate snakemake plugin (`slurm`, `cluster-generic`, etc.) in `config.yaml`, and adjust `default-resources`.

See [`docs/basics/workflow.rst`](../../docs/basics/workflow.rst) for the full reference (rule structure, partial reruns, troubleshooting).

### Workflow chart

![NSBI workflow](../../docs/_images/toolkit_workflow_AGCstyle.png)
