#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python data/download_dataset.py
python data/prepare.py
python training/train.py --epochs 8 --batch-size 32
python training/evaluate.py
uvicorn app.main:app --reload
