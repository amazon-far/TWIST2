#!/bin/bash
SCRIPT_DIR=$(dirname $(realpath $0))
cd "$SCRIPT_DIR"
exec python3 gui_train.py
