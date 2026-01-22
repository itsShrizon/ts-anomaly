#!/usr/bin/env bash
# pull SKAB and NASA C-MAPSS locally. both are free.
set -e
mkdir -p data/raw/skab data/raw/turbofan

if [ ! -d data/raw/skab/.git ]; then
  git clone --depth 1 https://github.com/waico/SKAB.git data/raw/skab
fi

if [ ! -f data/raw/turbofan/train_FD001.txt ]; then
  curl -L -o /tmp/cmapss.zip \
    https://ti.arc.nasa.gov/m/project/prognostic-repository/CMAPSSData.zip
  unzip -o /tmp/cmapss.zip -d data/raw/turbofan
fi
