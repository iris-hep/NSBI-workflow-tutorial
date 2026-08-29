#!/bin/bash

# Fail early if there's an issue
set -e

# When .cache files are created, they need to know where HOME is to write there.
# In this case, that should be the HTCondor scratch dir the job is executing in.
export HOME=$(pwd)

# EP-side bootstrap (was repeated in every rule's shell block in the Snakefile).
# The container ships snakemake + deps via pixi; here we (re)install the package
# from the transferred `src/` so the EP runs the current source. hatch-vcs can't
# read a git tag on the EP (no .git transferred), so pin a placeholder version.
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
python -m pip install --no-deps --user -e .

# Pass any arguments to Snakemake
exec snakemake "$@"
