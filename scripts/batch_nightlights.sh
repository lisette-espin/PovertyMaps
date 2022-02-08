#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../libs"

# 1. Load parameters (arguments)
while getopts ":r:y:a:p:" opt; do
  case $opt in
    r) root="$OPTARG"
    ;;
    y) years="$OPTARG"
    ;;
    a) apikey="$OPTARG"
    ;;
    p) projectid="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# 2. Get Cellular features
python batch_nightlights.py -r "$root" -y "$years" -a "$apikey" -p "$projectid"