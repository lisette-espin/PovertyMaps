#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../libs"

meters=0

# 1. Load parameters (arguments)
while getopts ":r:y:m:a:p:s:c:" opt; do
  case $opt in
    r) root="$OPTARG"
    ;;
    y) years="$OPTARG"
    ;;
    m) meters="$OPTARG"
    ;;
    a) apikey="$OPTARG"
    ;;
    p) projectid="$OPTARG"
    ;;
    s) servicename="$OPTARG"
    ;;
    c) clientsecret="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

echo "python batch_nightlights.py -r $root -y $years -m $meters -a $apikey -p $projectid -s $servicename -c $clientsecret"

# 2. Get nightlight intensity features
python batch_nightlights.py -r "$root" -y "$years" -m "$meters" -a "$apikey" -p "$projectid" -s "$servicename" -c "$clientsecret"