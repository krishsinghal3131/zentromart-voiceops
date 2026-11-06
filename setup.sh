#!/usr/bin/env bash
cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
python3 run.py doctor
