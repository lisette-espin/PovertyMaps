#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../libs"

# 1. Load parameters (arguments)
size='' # string WxH. W: width, H: heigh, both in pixels (x must be included)
while getopts ":r:y:z:s:k:" opt; do
  case $opt in
    r) root="$OPTARG"
    ;;
    y) years="$OPTARG"
    ;;
    z) size="$OPTARG"
    ;;
    s) secretfn="$OPTARG"
    ;;
    k) keyfn="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

echo "python batch_staticmaps.py -r $root -y $years -z $size -s $secretfn -k $keyfn"

# 2. Get OpenStreetMap features
python batch_staticmaps.py -r "$root" -y "$years" -z "$size" -s "$secretfn" -k "$keyfn"