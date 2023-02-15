#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../libs"

meters=0

# 1. Load parameters (arguments)
while getopts ":r:y:m:" opt; do
  case $opt in
    r) root="$OPTARG"
    ;;
    y) years="$OPTARG"
    ;;
    m) meters="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# 2. Get population features
echo "python batch_population.py -r $root -y $years -m $meters"
python batch_population.py -r "$root" -y "$years" -m "$meters"