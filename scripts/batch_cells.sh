#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../libs"

meters=0
distance=-1 #meters

# 1. Load parameters (arguments)
while getopts ":r:y:d:m:" opt; do
  case $opt in
    r) root="$OPTARG"
    ;;
    y) years="$OPTARG"
    ;;
    d) distance="$OPTARG"
    ;;
    m) meters="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

echo "python batch_cells.py -r $root -y $years -d $distance -m $meters"

# 2. Get Cellular features
python batch_cells.py -r "$root" -y "$years" -d $distance -m "$meters"