#!/usr/bin/env bash
set -e

STEP=$1
CONFIG=$2
SKIP=${3:-0}

if [ "$SKIP" = "1" ]; then
    echo "Skipping step: ${STEP}"
    exit 0
fi

export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
python -m pip install --no-deps -e .

echo "PATH=$PATH"
which python
python -c "import sys; print(sys.executable)"

cd FAIR_universe_Higgs_tautau

python scripts/${STEP}.py --config ${CONFIG}
