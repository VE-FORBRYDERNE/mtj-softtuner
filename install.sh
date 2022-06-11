#!/bin/bash
set -e
cd "$(dirname ${BASH_SOURCE[0]})"
Format=" "
hash="$Format:%H$"
if [[ $hash != "" && $hash != *" "* && ! -d ".git" ]]; then
    # This executes only if you downloaded this repository using GitHub's API
    # (e.g. by using the "Download ZIP" link on this repository's GitHub page
    # or the source code downloads on this repository's GitHub releases page)
    # rather than by cloning this repository using `git clone`.
    # In that case, the downloaded source code won't be a git repository, so
    # this will initialize a git repository based on the one on GitHub,
    # checkout to the correct commit and then replay all your changes, if any,
    # on top.
    # We need to do this so that Versioneer can display the correct version
    # string.
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
    git config --unset commit.gpgsign
    p='s/^hash[[:blank:]]*=[[:blank:]]*".*$/hash="\$For'
    p+='mat:%H\$"/'
    sed -i $p install.sh
fi
git submodule sync
git submodule update --remote --init --recursive
${1:-python3} -m pip uninstall -y mtj-softtuner
${1:-python3} -m pip install .
apt install aria2 2> /dev/null  # For faster model downloads in Colab
