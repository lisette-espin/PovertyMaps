#!/bin/bash
echo $PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:../libs"

# Load parameters (arguments)

timevar='none'
viirs=''
cnn_name='none'
layer_id=0
weighted=''

while getopts ":r:c:y:l:t:a:f:i:k:v:n:e:w" opt; do
  case $opt in
    r) root="$OPTARG"
    ;;
    c) code="$OPTARG"
    ;;
    y) years="$OPTARG"
    ;;
    l) dhsloc=$OPTARG    # none, rc, ruc
    ;;
    t) ttype="$OPTARG"   # newest, oldest, all
    ;;
    a) yatt="$OPTARG"    # comma-separated attributes: mean_wi, std_wi
    ;;
    f) fsource="$OPTARG" # all, OCI, FBM, FBP, FBMV, NTLL, OSM
    ;;
    i) timevar="$OPTARG" # none, deltatime, gdp, gdpp, gdppg, gdppgp, gdpg, gdpgp, gni, gnip
    ;;
    k) cv="$OPTARG"      # cross-validation: 4
    ;;
    v) viirs="-viirs"    # norm viir or not
    ;;
    n) cnn_name="$OPTARG"     # cnn model name: offaug_cnn_mp_dp_relu_sigmoid_adam_mean_std_regression, noaug_cnn_mp_dp_relu_sigmoid_adam_mean_std_regression
    ;;
    e) layer_id="$OPTARG"   # 19
    ;;
    w) weighted="-weighted" # sample weights 1
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
fnlog="../logs/$code-train-$(date +"%FT%H%M")-catboost.log"
echo "params: -r $root -c $code -y $years -l $dhsloc -t $ttype -a $yatt -f $fsource -i $timevar -k $cv -v $viirs -n $cnn_name -e $layer_id -w $weighted" 2>&1 | tee -a "$fnlog"
echo "LOG FILE: $fnlog"


###########################################################################################################

ccodes=("SL" "UG" "ZW")
for i in "${ccodes[@]}"
do
    if [ "$i" == "$code" ] ; then
    
      # 1. train xgb
      echo "====================================================================" 2>&1 | tee -a "$fnlog"
      echo "python xgb_train.py -root $root -years $years -dhsloc $dhsloc -ttype $ttype -yatt $yatt -fsource $fsource -timevar $timevar -cv $cv $viirs -cnn_name $cnn_name -layer_id $layer_id $weighted"  2>&1 | tee -a "$fnlog" 
      python xgb_train.py -root "$root" -years "$years" -dhsloc "$dhsloc" -ttype "$ttype" -yatt "$yatt" -fsource "$fsource" -timevar "$timevar" -cv $cv $viirs -cnn_name "$cnn_name" -layer_id $layer_id $weighted 2>&1 | tee -a "$fnlog"
    
    fi
done

# 2. end
echo "====================================================================" 2>&1 | tee -a "$fnlog"
echo "LOG FILE: $fnlog"