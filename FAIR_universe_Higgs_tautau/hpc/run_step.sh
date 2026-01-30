#!/usr/bin/env bash
set -e

STEP=$1

export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
python -m pip install --user --no-deps -e .

cd FAIR_universe_Higgs_tautau

python ${STEP}.py
