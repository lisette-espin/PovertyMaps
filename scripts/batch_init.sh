#!/bin/bash
echo $PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:../libs"

# Load parameters (arguments)
 
years='' # years of gt data (if empty, then pplaces)
njobs=1 # parallel (gt & movement)
load='' # 1 load populated places, none download from scratch
gsize=0 # grid size gt data in meters
shift='' # shift or not grid
dhsloc='none' # none or rc
trainyear='all'
kfolds=5
repeat=5
seeds='' #comma-separated seeds

while getopts ":r:c:y:n:l:g:f:" opt; do
  case $opt in
    r) root="$OPTARG"
    ;;
    c) code="$OPTARG"
    ;;
    y) years="$OPTARG"
    ;;
    n) njobs=$OPTARG
    ;;
    l) load="-load"
    ;;
    g) gsize=$OPTARG
    ;;
    f) shift="-f"
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
echo "params: -r $root -c $code -y $years -n $njobs -l $load -g $gsize -f $shift" 2>&1 | tee -a "$fnlog"
echo "LOG FILE: $fnlog"

###########################################################################################################

ccodes=("SL" "UG" "ZW")
for i in "${ccodes[@]}"
do
    if [ "$i" == "$code" ] ; then
        # 1. prepare
        echo "====================================================================" 2>&1 | tee -a "$fnlog"
        echo "./batch_prepare.sh $root"  2>&1 | tee -a "$fnlog"
        ./batch_prepare.sh "$root" 2>&1 | tee -a "$fnlog"
        
        # 2. clusters
        echo "====================================================================" 2>&1 | tee -a "$fnlog"
        echo "python batch_gt.py -r $root -c $code -y $years -n  $njobs -s $gsize $shift" 2>&1 | tee -a "$fnlog"
        python batch_gt.py -r "$root" -c "$code" -y "$years" -n  $njobs -s $gsize $shift 2>&1 | tee -a "$fnlog"

        # 3. populated places
        echo "====================================================================" 2>&1 | tee -a "$fnlog"
        echo "python batch_pplaces.py -r $root $load" 2>&1 | tee -a "$fnlog"
        python batch_pplaces.py -r "$root" $load 2>&1 | tee -a "$fnlog"
      
    fi
done

# 3. end
echo "====================================================================" 2>&1 | tee -a "$fnlog"
echo "LOG FILE: $fnlog"