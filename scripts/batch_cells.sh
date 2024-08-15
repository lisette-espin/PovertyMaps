#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../libs"

meters=0
distance=-1 #meters
njobs=1 #num of job in parallel

# 1. Load parameters (arguments)
while getopts ":r:y:d:m:n:" opt; do
  case $opt in
    r) root="$OPTARG"
    ;;
    y) years="$OPTARG"
    ;;
    d) distance="$OPTARG"
    ;;
    m) meters="$OPTARG"
    ;;
    n) njobs="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

echo "python batch_cells.py -r $root -y $years -d $distance -m $meters -n $njobs"

# 2. Get Cellular features
python batch_cells.py -r "$root" -y "$years" -d $distance -m "$meters" -n $njobs