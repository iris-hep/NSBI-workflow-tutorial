Running the Workflow
====================

There are three equivalent ways to run the NSBI pipeline, all driven by the same configuration file and the same underlying ``nsbi_common_utils`` library:

1. **Notebooks** — step through the analysis interactively, one stage per notebook. Best for learning the method, inspecting intermediate outputs, and developing.
2. **Scripts** — run each stage as a command-line script. Same logic as the notebooks, scriptable and headless. Best for reproducible single-machine runs.
3. **Snakemake** — orchestrate the scripts as a parallel DAG on a cluster. Best for production: it fans out the embarrassingly-parallel training jobs (per process, ensemble member, k-fold split, systematic) and submits them to your batch system. Infrastructure-agnostic — the same ``Snakefile`` runs on HTC (HTCondor), HPC (SLURM), Kubernetes, or a laptop, by swapping the executor profile.

Below is an example workflow using the FAIR Universe :math:`H\to \tau\tau` dataset.

All three paths are driven by a single configuration file, ``config.pipeline.yaml``, located at the root of each example directory (e.g. ``examples/FAIR_universe_Higgs_tautau/config.pipeline.yaml``). This file defines dataset paths, training hyperparameters, ensemble sizes, systematic variations, and fit settings. Inspect the example config to understand the available options.

Pipeline overview
-----------------

.. image:: /_static/DAG_overview.svg
   :alt: NSBI Workflow Overview
   :align: center
   :width: 100%

Interactive execution (notebooks)
---------------------------------

Each pipeline stage has a numbered notebook in the example directory. Run them in order — each reads ``config.pipeline.yaml`` and the outputs of the previous stage:

.. code-block:: text

   1_data_loader.ipynb                       # download + balance + write per-process ROOT files
   2_data_preprocessing.ipynb                # feature engineering, write processed ntuples
   3_preselection_network.ipynb             # train the signal/control-region classifier
   4_neural_likelihood_ratio_estimation.ipynb  # train nominal density-ratio ensembles
   5_systematic_uncertainty_training.ipynb   # train systematic-variation networks
   6_evaluate_networks.ipynb                 # evaluate all models on the Asimov dataset
   7_parameter_fitting_with_systematics.ipynb  # build the workspace and run the fit

Launch Jupyter from the example directory so the relative ``config.pipeline.yaml`` path resolves:

.. code-block:: bash

   cd examples/FAIR_universe_Higgs_tautau
   jupyter lab            # or: jupyter notebook

The notebooks read the same ``config.pipeline.yaml`` as the scripts and Snakemake, so the ensemble size and other hyperparameters are governed there. The notebooks operate on a single fold (k-fold loops are only wired into the scripts and Snakemake); set ``num_folds: 1`` in the config when running the notebook path. For large ensembles or a full k-fold run, prefer the scripts or Snakemake — the per-job parallelism is what makes those modes scale.

Local (sequential) execution
-----------------------------

For a headless, scriptable run of the same stages, use the command-line scripts. From the example directory
(``examples/FAIR_universe_Higgs_tautau/``):

.. code-block:: bash

   # 1. Load and preprocess data
   python scripts/data_loader.py --config config.pipeline.yaml
   python scripts/data_preprocessing.py --config config.pipeline.yaml

   # 2. Train preselection network (region classifier)
   python scripts/preselection_network.py --config config.pipeline.yaml

   # 3. Train nominal density-ratio ensembles (per process)
   python scripts/neural_likelihood_ratio_estimation.py \
       --config config.pipeline.yaml --process htautau --ensemble_index 0

   # 4. Train systematic variation networks
   python scripts/systematic_uncertainty_training.py \
       --config config.pipeline.yaml --process htautau --systematic JES --direction Up

   # 5. Evaluate all trained models on the Asimov dataset
   python scripts/data_nn_eval.py --config config.pipeline.yaml

   # 6. Build workspace and fit
   python scripts/parameter_fitting.py --config config.pipeline.yaml

Steps 3 and 4 are embarrassingly parallel across processes, ensemble members, k-fold splits, and systematic variations. Snakemake fans them out automatically when running on a cluster — see below.

Cluster execution (Snakemake on HTCondor)
------------------------------------------

A single ``Snakefile`` at the root of each example directory defines all pipeline rules; a profile under ``profiles/<cluster>/`` configures the executor and per-rule resource defaults. From the repository root:

.. code-block:: bash

   snakemake --snakefile examples/FAIR_universe_Higgs_tautau/Snakefile \
             --profile  examples/FAIR_universe_Higgs_tautau/profiles/chtc

That single command builds the full DAG (parameter_fitting at the leaf, all training and preprocessing as dependencies), submits each rule's jobs to HTCondor via the `snakemake-executor-plugin-htcondor <https://github.com/snakemake/snakemake-executor-plugin-htcondor>`_, and waits for completion.

.. warning::

   The provided ``profiles/chtc/`` profile, the ``Snakefile`` resource blocks, and the paths in ``config.pipeline.yaml`` are written for the `CHTC <https://chtc.cs.wisc.edu/>`_ pool at UW-Madison and a specific user account. **They will not run as-is on another system.** Before launching, you must edit them for your site — see *Adapting to your cluster* below for the exact files and fields to change.

Submitting and monitoring
^^^^^^^^^^^^^^^^^^^^^^^^^

A practical CLI workflow for launching and watching a run:

.. code-block:: bash

   # 1. Preview what would run, without submitting anything.
   #    --list-target-rules / -n give a dry-run summary of the DAG.
   snakemake --snakefile examples/FAIR_universe_Higgs_tautau/Snakefile \
             --profile  examples/FAIR_universe_Higgs_tautau/profiles/chtc \
             -n

   # 2. Launch. Run inside tmux/screen (or nohup) so it survives disconnects;
   #    a long campaign can run for hours.
   tmux new -s nsbi
   snakemake --snakefile examples/FAIR_universe_Higgs_tautau/Snakefile \
             --profile  examples/FAIR_universe_Higgs_tautau/profiles/chtc \
             > run.log 2>&1

The number of jobs submitted concurrently is set by ``jobs:`` in the profile (``profiles/chtc/config.yaml``); raise it to fill more cluster slots, lower it to be gentle on the scheduler.

Monitor progress from another shell:

.. code-block:: bash

   tail -f run.log                              # snakemake driver output
   condor_q                                     # jobs in the batch queue
   ls <sentinel_dir>/.done_* | wc -l            # completed stages (sentinel count)

If the driver process is killed (disconnect, node policy, time limit), the submitted jobs keep running on the cluster and write their sentinels independently. Just re-run the same ``snakemake`` command — it reads the sentinels and resubmits only what's missing (see *Resuming and partial reruns* below). For unattended long runs, wrap the command in a restart loop so it relaunches automatically after a driver kill.

File layout
^^^^^^^^^^^

.. code-block:: text

   examples/FAIR_universe_Higgs_tautau/
     Snakefile                            # all pipeline rules + wildcard fan-out
     config.pipeline.yaml                 # single source of truth for paths and hyperparameters
     profiles/
       chtc/
         config.yaml                      # snakemake profile: executor=htcondor, default resources
         job_wrapper.sh                   # per-job entrypoint inside the container
     scripts/                             # the same python scripts used by the local-execution path

Rule structure replaces DAGMan PRE / SUBDAG
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each pipeline step is one Snakemake rule. Per-job fan-out (``process × fold × ensemble_index`` for nominal training; ``process × systematic × direction × fold × ensemble_index`` for systematics) is expressed as **wildcard expansion** at Snakefile-parse time, computed from the config:

.. code-block:: python

   BASIS_PROCESSES  = config["neural_likelihood_ratio_estimation"]["basis_processes_to_train"]
   N_ENSEMBLE       = config["neural_likelihood_ratio_estimation"]["num_ensemble_members_training"]
   NUM_FOLDS        = config.get("data_preprocessing", {}).get("num_folds", 1)

This replaces the legacy DAGMan pattern of (a) generating DAG files dynamically via ``SCRIPT PRE`` hooks, (b) submitting them as ``SUBDAG EXTERNAL`` nested workflows. K-fold cross-validation is just bumping ``num_folds`` in the config — no DAG regeneration step needed.

Sentinel-driven completion tracking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The DAG dependency edges are tracked via per-rule sentinel files on shared storage (e.g. ``/projects/.../sentinels/.done_train_<process>_fold<F>_<E>``), not via HTCondor's ``transfer_output_files``. This decouples DAG state from HTCondor's job lifecycle and survives driver restarts cleanly. Sentinel paths and the shared-FS prefix are declared in the profile:

.. code-block:: yaml

   # profiles/chtc/config.yaml
   shared-fs-usage: none
   htcondor-shared-fs-prefixes: "/staging,/projects"

HTCondor resource specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Per-rule resource requests, ClassAds, and transfer specs are declared in the rule's ``resources:`` block. The htcondor plugin emits them verbatim into the submit description:

.. code-block:: python

   resources:
       request_memory          = "16GB",
       request_disk            = "32GB",
       request_gpus            = 1,
       gpus_minimum_capability = _r(7.0),
       classad_WantGPULab      = _r(True),
       classad_GPUJobLength    = "medium",
       requirements            = gpu_requirements(driver_version="12.4"),
       allowed_job_duration    = 9000,    # replaces periodic_hold + periodic_release
       max_retries             = 3,       # replaces DAGMan RETRY
       htcondor_transfer_input_files = COMMON_TRANSFER,

``allowed_job_duration`` + ``max_retries`` together replace the legacy ``periodic_hold`` / ``periodic_release`` retry idiom that the plugin doesn't expose.

Resuming and partial reruns
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sentinels make resumption trivial — Snakemake skips any rule whose output sentinel already exists. After a failed or killed driver, re-running the same command picks up where it left off and only submits the missing jobs. A few useful flags:

- ``--forcerun <rule>`` — re-execute a specific rule even if its sentinel exists. Does **not** re-run upstream rules unless their inputs are themselves missing.
- ``--rerun-triggers mtime`` — only re-run on file-modification-time changes, not on Snakefile/code edits. Pass this when you've edited the Snakefile for an unrelated reason and don't want every rule to re-trigger.
- ``--unlock`` — clear a stale ``LockException`` left by a previously killed driver.
- ``--cleanup-metadata <output paths>`` — mark previously-incomplete outputs as complete in the Snakemake metadata store. Needed if a previous driver died after the EP-side job touched its sentinel but before Snakemake recorded completion.
- ``--touch`` — bring outputs up-to-date in Snakemake's metadata without actually running anything. Useful for adopting an existing tree of artifacts.

Adapting to your cluster
^^^^^^^^^^^^^^^^^^^^^^^^

The rule *logic* is portable, but the site-specific values are not. Before running anywhere other than CHTC, edit the following. **None of these will work unchanged on your system.**

**1. The profile —** ``profiles/chtc/config.yaml`` (or copy it to ``profiles/<your-cluster>/``):

- ``executor:`` — set to the plugin for your batch system (``htcondor``, ``slurm``, ``cluster-generic``, ``kubernetes``); remove it for local execution. See the `plugin catalog <https://snakemake.github.io/snakemake-plugin-catalog/>`_.
- ``container_image:`` in ``default-resources`` — points at a specific Apptainer ``.sif`` under one user's ``/staging``. Replace with your own image, or remove if you don't run in a container.
- ``htcondor-shared-fs-prefixes:`` and ``shared-fs-usage:`` — set to the shared-filesystem paths visible on your execute nodes.
- ``job_wrapper:`` — path to the per-job entrypoint; adjust if you relocate the profile.
- ``classad_*`` flags (``WantFlocking``, ``want_campus_pools``, ``want_ospool``, ``WantGlidein``) — these are CHTC-pool-specific and meaningless elsewhere; remove or replace with your pool's ClassAds.
- ``jobs:`` — concurrency cap; tune to your fair-share / scheduler limits.

**2. The Snakefile —** ``examples/FAIR_universe_Higgs_tautau/Snakefile``:

- ``SENTINEL_DIR`` — currently ``/projects/Physics_Cranmer/sentinels_FAIR_higgs``. Point it at a directory on a filesystem shared between submit and execute nodes for your site.
- ``gpu_requirements(...)`` / machine-exclude lists — contain CHTC hostnames (``gpulab2001.chtc.wisc.edu``, etc.). Remove or replace with your site's ``requirements`` expression.
- ``classad_GPUJobLength``, ``allowed_job_duration``, ``request_*`` — review the resource asks against your queue's limits.

**3. The pipeline config —** ``config.pipeline.yaml`` and the fit configs (``config_fit_nsbi.yml``, ``config_fit_histogram.yml``):

- All dataset and output paths use absolute ``/projects/Physics_Cranmer/...`` locations. Repoint every ``saved_data_path``, ``output.dir``, ``SamplePath``, ``Models``/``Ratios``, and ``AsimovWeights`` entry to your storage layout.

For sites without a batch system at all, the local sequential commands (the scripts, or the notebooks) still work and can be wrapped in your site's job-submission idiom directly.
