#!/bin/bash
cd "${0%/*}"
git submodule sync
git submodule update --init --recursive
python3 -m pip uninstall -y mtj-softtuner
python3 -m pip install .
