#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../libs"

radius=0 # kilometer 
unit="kilometer" # kilometer or mile

# 1. Load parameters (arguments)
while getopts ":r:y:t:i:u:n:" opt; do
  case $opt in
    r) root="$OPTARG"
    ;;
    y) years="$OPTARG"
    ;;
    t) tokensdir="$OPTARG"
    ;;
    i) radius="$OPTARG"
    ;;
    u) unit="$OPTARG"
    ;;
    n) njobs="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

echo "python batch_marketing.py -r $root -y $years -t $tokensdir -i $radius -u $unit -n $njobs"

# 2. Get Facebook Marketing API features
python batch_marketing.py -r "$root" -y "$years" -t "$tokensdir" -i $radius -u "$unit" -n $njobs