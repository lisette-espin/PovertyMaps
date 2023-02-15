#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../libs"

# 1. Load parameters (arguments)
while getopts ":r:y:s:k:" opt; do
  case $opt in
    r) root="$OPTARG"
    ;;
    y) years="$OPTARG"
    ;;
    s) secretfn="$OPTARG"
    ;;
    k) keyfn="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

echo "python batch_staticmaps.py -r $root -y $years -s $secretfn -k $keyfn"

# 2. Get OpenStreetMap features
python batch_staticmaps.py -r "$root" -y "$years" -s "$secretfn" -k "$keyfn"