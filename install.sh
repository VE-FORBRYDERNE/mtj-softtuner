#!/bin/bash
cd "${0%/*}"
git submodule sync
git submodule update --remote --init --recursive
${1:-python3} -m pip uninstall -y mtj-softtuner
${1:-python3} -m pip install .
