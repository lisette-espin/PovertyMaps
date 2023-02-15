#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../libs"

# 1. Load parameters (arguments)
while getopts ":r:l:" opt; do
  case $opt in
    r) root="$OPTARG"
    ;;
    l) load="-load"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

echo "python batch_pplaces.py -r $root $load"

# 2. Get populated places
python batch_pplaces.py -r "$root" $load