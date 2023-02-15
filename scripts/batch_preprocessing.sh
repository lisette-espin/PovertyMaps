#!/bin/bash
echo $PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:../libs"

# Load parameters (arguments)
 
years='' # years of gt data (if empty, then pplaces)
dhsloc='none' # none or rc
trainyear='all'
kfolds=5
repeat=5
seeds='' #comma-separated seeds

while getopts ":r:c:y:o:t:k:e:s:" opt; do
  case $opt in
    r) root="$OPTARG"
    ;;
    c) code="$OPTARG"
    ;;
    y) years="$OPTARG"
    ;;
    o) dhsloc="$OPTARG"
    ;;
    t) trainyear="$OPTARG" # all, newest, oldest (2016)
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

###########################################################################################################

# Validation country
if [[ -z "$root" ]] || [[ -z "$code" ]]
then
  echo "-r root and -c code arguments missing."
  exit
fi

# Log file
fnlog="../logs/$code-preprocess-$(date +"%FT%H%M").log"
echo "params: -r $root -c $code -y $years -o $dhsloc -t $trainyear -k $kfolds -e $repeat -s $seeds" 2>&1 | tee -a "$fnlog"
echo "LOG FILE: $fnlog"

###########################################################################################################

ccodes=("SL" "UG" "ZW")
for i in "${ccodes[@]}"
do
    if [ "$i" == "$code" ] ; then
        
        ## 1. re-arregement cluster location
        echo "====================================================================" 2>&1 | tee -a "$fnlog"
        echo "python batch_locations.py -r $root -y $years -o $dhsloc"  2>&1 | tee -a "$fnlog"
        python batch_locations.py -r "$root" -y "$years" -o "$dhsloc"  2>&1 | tee -a "$fnlog"
        
        # 2. spliting data: train-val (cross-val) and test 
        echo "====================================================================" 2>&1 | tee -a "$fnlog"
        echo "python batch_sampling.py -r $root -y $years -o $dhsloc -t $trainyear -k $kfolds -e $repeat -s $seeds"  2>&1 | tee -a "$fnlog"
        python batch_sampling.py -r "$root" -y "$years" -o "$dhsloc" -t "$trainyear" -k $kfolds -e $repeat -s "$seeds"

    fi
done

# 3. end
echo "====================================================================" 2>&1 | tee -a "$fnlog"
echo "LOG FILE: $fnlog"