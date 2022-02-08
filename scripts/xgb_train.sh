#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../libs"

# 1. Load parameters (arguments)
while getopts ":r:y:d:t:r:v:" opt; do
  case $opt in
    r) root="$OPTARG"
    ;;
    y) years="$OPTARG"
    ;;
    o) dhsloc="$OPTARG"
    ;;
    t) traintype="$OPTARG"
    ;;
    k) isregression="$OPTARG"
    ;;
    e) viirnorm="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# 2. Split 
python xgb_train.py -r "$root" -years "$years" -dhsloc "$dhsloc" -traintype "$traintype" -isregression "$isregression" -viirsnorm "$viirsnorm" 