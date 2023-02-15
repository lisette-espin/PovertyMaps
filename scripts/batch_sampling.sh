#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../libs"

seeds=''

# 1. Load parameters (arguments)
while getopts ":r:y:o:t:k:e:s:" opt; do
  case $opt in
    r) root="$OPTARG"
    ;;
    y) years="$OPTARG"
    ;;
    o) dhsloc="$OPTARG"
    ;;
    t) trainyear="$OPTARG"
    ;;
    k) kfolds="$OPTARG"
    ;;
    e) repeat="$OPTARG"
    ;;
    s) seeds="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# 2. Split 
echo "python batch_sampling.py -r $root -y $years -o $dhsloc -t $trainyear -k $kfolds -e $repeat -s $seeds"
python batch_sampling.py -r "$root" -y "$years" -o "$dhsloc" -t "$trainyear" -k "$kfolds" -e "$repeat" -s "$seeds"