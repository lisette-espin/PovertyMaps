#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../libs"

# 1. Load parameters (arguments)
while getopts ":r:y:j:" opt; do
  case $opt in
    r) root="$OPTARG"
    ;;
    y) years="$OPTARG"
    ;;
    j) njobs="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

echo "python batch_movement.py -r $root -y $years -j $njobs"

# 2. Get Facebook Movement features
python batch_movement.py -r "$root" -y "$years" -j "$njobs"