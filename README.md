# Neural Simulation-Based Inference Workflow demonstration

-----

## Table of Contents

- [Setup](#setup)
- [Introduction](#introduction)
- [Library](#library)
- [License](#license)

## Setup

We will use `pixi` to setup the environment for the workflow. The specifications are defined in the `pixi.toml` file. If `pixi` is not installed on your machine follow the instructions in [pixi seutp guide](https://pixi.sh/latest/installation/). Then proceed to install the environment with:
```
pixi install -e nsbi-env-gpu
```
Currently the environment can only be built on machines with GPU. 

A jupyter kernel can then be created by running:
```
pixi run -e nsbi-env-gpu python -m ipykernel install --user --name nsbi-env-gpu --display-name "Python (pixi: nsbi-env-gpu)"
```

## Introduction

The notebooks presented here aim to serve as a guide to use the `nsbi-common-utils` library being developed (currently not deployed to PyPi). The core model and workflow is related to the NSBI analysis recently published by ATLAS:

<br/>

- An implementation of neural simulation-based inference for parameter estimation in ATLAS (https://arxiv.org/pdf/2412.01600)

<br/>

- Measurement of off-shell Higgs boson production in the $H\to ZZ \to 4\ell$ decay channel using a neural simulation-based inference technique in 13 TeV p-p collisions with the ATLAS detector (https://arxiv.org/pdf/2412.01548)

<br/>
<br/>

![NLL_PE_ExpOnly_PEHist_Comp](https://github.com/user-attachments/assets/5aba909b-efc5-4a66-b171-9aa2c8c4d6f4) ![NLL_PE_ObsExp_StatSyst_Comp](https://github.com/user-attachments/assets/3c518b0b-90cb-4bcc-8830-a2783777010f)

<br/>
<br/>

We demonstrate the usage of `nsbi-common-utils` applied to a full-scale LHC analysis. The physics results presented in the notebooks only serve as examples of the workflow. The code in this tutorial is partially derived from the original ATLAS analysis code written by Jay Sandesara [[git](https://github.com/JaySandesara)], R.D. Schaffer [[git](https://gitlab.cern.ch/schaffer)] and Arnaud Maury [[git](https://github.com/Maury98)].

The workflow currently uses the Higgs to tau tau dataset from FAIR universe challenge. More open datasets will be added in the future. 

## Library

To use the library `nsbi_common_utils` developed here in general cases outside of this tutorial, do:

```console
python -m pip install --upgrade 'nsbi-common-utils @ git+https://github.com/iris-hep/NSBI-workflow-tutorial.git'
```

The library tools are steered by a configuration file - an example can be found in the FAIR Universe Challenge workflow [config.yml](https://github.com/iris-hep/NSBI-workflow-tutorial/blob/main/FAIR_universe_Higgs_tautau/config.yml).

Workflow bluprint (**tentative**):

![NSBI_flowchart](https://github.com/user-attachments/assets/f9bd85be-10d8-487b-a7ed-1bdd3028fb4e)

## Acknowledgements

[![NSF-1836650](https://img.shields.io/badge/NSF-1836650-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1836650)
[![PHY-2323298](https://img.shields.io/badge/PHY-2323298-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=2323298)


This work is being supported by the U.S. National Science Foundation (NSF) cooperative agreements [OAC-1836650](https://nsf.gov/awardsearch/showAward?AWD_ID=1836650) and [PHY-2323298](https://nsf.gov/awardsearch/showAward?AWD_ID=2323298) (IRIS-HEP).



## License

`nsbi-common-utils` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
