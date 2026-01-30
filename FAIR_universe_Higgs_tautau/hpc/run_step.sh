#!/usr/bin/env bash
set -e

STEP=$1

cd "$(dirname "$0")/.."

python ${STEP}.py
