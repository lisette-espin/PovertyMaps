#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../libs"

option='none'

# 1. Load parameters (arguments)
while getopts ":r:y:o:" opt; do
  case $opt in
    r) root="$OPTARG"
    ;;
    y) years="$OPTARG"
    ;;
    o) option="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# 2. Change survey (DHS) cluster locations
echo "python batch_locations.py -r $root -y $years -o $option"
python batch_locations.py -r "$root" -y "$years" -o "$option"