#!/usr/bin/env bash
set -e

STEP=$1

python -m pip install --no-deps -e .

cd FAIR_universe_Higgs_tautau

python ${STEP}.py
