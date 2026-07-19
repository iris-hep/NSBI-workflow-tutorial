# Toolkit for Simulation-Based Inference at the LHC

The full documtation can be found here: [https://toolkit-for-simulation-based-inference.readthedocs.io](https://toolkit-for-simulation-based-inference.readthedocs.io/en/latest/index.html#)

-----

## Table of Contents

- [Setup](#setup)
- [Introduction](#introduction)
- [Library](#library)
- [License](#license)

## Setup

We will use `pixi` to setup the environment for the workflow. The specifications are defined in the `pixi.toml` file. If `pixi` is not installed on your machine follow the instructions in [pixi setup guide](https://pixi.sh/latest/installation/).

On Linux machines with CUDA-capable GPUs, install the GPU environment with:
```
pixi install -e nsbi-env-gpu
```
The `nsbi-env-gpu` environment is restricted to `linux-64` and will fail on macOS.

On macOS, install the CPU environment instead:
```
pixi install -e nsbi-env
```

A jupyter kernel can then be created by running the command for the environment you installed:
```
pixi run -e nsbi-env python -m ipykernel install --user --name nsbi-env --display-name "Python (pixi: nsbi-env)"
```

For the Linux GPU environment:
```
pixi run -e nsbi-env-gpu python -m ipykernel install --user --name nsbi-env-gpu --display-name "Python (pixi: nsbi-env-gpu)"
```

For cloning this repo you may want to disable auto-replacing the LFS pointer files with the actual files as they're pretty large, use the following git clone command to do so:

```shell
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:iris-hep/NSBI-workflow-tutorial.git --depth=1
```

## Introduction

Simulation-Based Inference (SBI) or Neural Simulation-Based Inference (NSBI) refers to set of statistical techniques that allow statistical inference directly using high-dimensional data. This circumvents the need to build low-dimensional summaries as is traditionally done and which can lose sensitive information. 

This toolkit helps facilitate the application of a type of SBI that is scalable for LHC-style analysis with high-dimensional parameter spaces, where the systematic uncertainty modeling is done via certain domain-specific assumptions. This is done via easy-to-use APIs for the various stages in the analysis as well as providing an end-to-end workflow orchestratation pipeline steered via human-readable configuration files. 

``nsbi-common-utils`` provides building blocks for SBI analysis tailored to the statistical models typical at the ATLAS and CMS experiments. It implements semi-parametric approach to SBI where the statistical models are built using a combination of non-parametric and parametric methods targeting different parts. The toolkit has a modular structure, and offers APIs for dataset preparation, density-ratio estimation, model building and profiled-likelihood ratio fitting.

The semi-parametric model and workflow is related to the SBI analysis recently published by ATLAS:

<br/>

- An implementation of neural simulation-based inference for parameter estimation in ATLAS (https://arxiv.org/pdf/2412.01600)

<br/>

- Measurement of off-shell Higgs boson production in the $H\to ZZ \to 4\ell$ decay channel using a neural simulation-based inference technique in 13 TeV p-p collisions with the ATLAS detector (https://arxiv.org/pdf/2412.01548)

<br/>
<br/>

![NLL_PE_ObsExp_StatSyst_Comp](https://github.com/user-attachments/assets/3c518b0b-90cb-4bcc-8830-a2783777010f)

<br/>
<br/>

We demonstrate the usage of `nsbi-common-utils` applied to a full-scale LHC-style analysis in the `examples/`. The workflow currently uses the Higgs to tau tau dataset from FAIR universe challenge. More open datasets will be added in the future. 

## Library

To use the library `nsbi_common_utils` developed here in general cases outside of this tutorial, do:

```console
python -m pip install --upgrade 'nsbi-common-utils @ git+https://github.com/iris-hep/NSBI-workflow-tutorial.git'
```

Workflow bluprint (**tentative**):

![NSBI_flowchart](https://github.com/user-attachments/assets/f9bd85be-10d8-487b-a7ed-1bdd3028fb4e)

## Acknowledgements

[![NSF-1836650](https://img.shields.io/badge/NSF-1836650-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1836650)
[![PHY-2323298](https://img.shields.io/badge/PHY-2323298-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=2323298)


This work is being supported by the U.S. National Science Foundation (NSF) cooperative agreements [OAC-1836650](https://nsf.gov/awardsearch/showAward?AWD_ID=1836650) and [PHY-2323298](https://nsf.gov/awardsearch/showAward?AWD_ID=2323298) (IRIS-HEP).



## License

`nsbi-common-utils` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
