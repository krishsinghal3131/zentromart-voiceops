#!/usr/bin/env bash
# usage: ./run.sh path/to/audio_folder [output_folder]
cd "$(dirname "$0")"
python3 run.py batch --audio_dir "$1" --out "${2:-sample_outputs}"
