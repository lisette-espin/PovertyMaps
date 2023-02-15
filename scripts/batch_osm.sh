#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../libs"

width=0

# 1. Load parameters (arguments)
while getopts ":r:y:w:" opt; do
  case $opt in
    r) root="$OPTARG"
    ;;
    y) years="$OPTARG"
    ;;
    w) width="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

echo "python batch_osm.py -r $root -y $years -w $width"

# 2. Get OpenStreetMap features
python batch_osm.py -r "$root" -y "$years" -w $width