#!/bin/bash
echo $PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:../libs"

# 1. Load parameters (arguments)

gsize=0
shift=''
while getopts ":r:c:y:n:s:" opt; do
  case $opt in
    r) root="$OPTARG"
    ;;
    c) code="$OPTARG"
    ;;
    y) years="$OPTARG"
    ;;
    n) njobs=$OPTARG
    ;;
    s) gsize=$OPTARG
    ;;
    f) shift="-f"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

echo "python batch_gt.py -r $root -c $code -y $years -n  $njobs -s $gsize"

# 2. Load ground-truth data and compute wealth index (mean and std)
python batch_gt.py -r "$root" -c "$code" -y "$years" -n  $njobs -s $gsize $shift