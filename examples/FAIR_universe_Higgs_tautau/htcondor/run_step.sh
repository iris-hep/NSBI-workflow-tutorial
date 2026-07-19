#!/usr/bin/env bash
set -e

STEP=$1
CONFIG=$2
shift 2  # remaining args are step-specific (e.g. --process, --ensemble_index, etc.)

WORK_DIR=FAIR_universe_Higgs_tautau

export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
python -m pip install --no-deps --user -e .

echo "PATH=$PATH"
which python
python -c "import sys; print(sys.executable)"

cd $WORK_DIR

python -u scripts/${STEP}.py --config ${CONFIG} "$@"
