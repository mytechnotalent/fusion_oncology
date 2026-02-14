#!/bin/bash

rm -rf venv
python3 -m venv venv
source ./venv/bin/activate
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -e .
