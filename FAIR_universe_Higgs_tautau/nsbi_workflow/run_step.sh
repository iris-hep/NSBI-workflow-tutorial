#!/usr/bin/env bash
set -e

STEP=$1
CONFIG=$2

python -m nsbi_workflow.${STEP} --config ${CONFIG}

