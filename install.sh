#!/bin/bash
cd "${0%/*}"
Format=" "
hash="$Format:%H$"
if [[ $hash != "" && $hash != *" "* && ! -d ".git" ]]; then
    git init
    git config commit.gpgsign false
    git remote add origin https://github.com/ve-forbryderne/mtj-softtuner
    git fetch -qa
    git checkout -qb _alpha
    git add .
    git commit -qm "alpha"
    git checkout -q $hash
    git merge --allow-unrelated-histories --squash -X theirs _alpha -m "omega" &> /dev/null
    git branch -qD _alpha
    git reset -q HEAD
    git checkout -q -- mtj_softtuner/_version.py
    sed -i '4s/.*/hash="\$Format:%H\$"/' install.sh
fi
git submodule sync
git submodule update --remote --init --recursive
${1:-python3} -m pip uninstall -y mtj-softtuner
${1:-python3} -m pip install .
