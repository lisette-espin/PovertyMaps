#!/bin/bash
echo $PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:../libs"

# Load parameters (arguments)
 
years='' # years of gt data (if empty, then pplaces)
njobs=1 # parallel (gt & movement)
width=0 # width bbox openstreetmap data in meters
meters=0 # comma separated widths (bbox) to query
distance=-1 # max distance between cells (same antenna)
radius=0 # radius for FBM
unit='kilometer' # kilometer or mile
secretfn="../resources/maps/staticmaps/secret"
keyfn="../resources/maps/staticmaps/api_key"
tokensdir='../resources/Facebook/MarketingAPI/tokens'
apikey="" # GEE VIIRS
projectid="" # GEE VIIRS
servicename="" # GEE VIIRS

while getopts ":r:c:y:n:f:w:d:m:i:u:t:" opt; do
  case $opt in
    r) root="$OPTARG"
    ;;
    c) code="$OPTARG"
    ;;
    y) years="$OPTARG"
    ;;
    n) njobs=$OPTARG ##
    ;;
    w) width="$OPTARG"
    ;;
    d) distance="$OPTARG"
    ;;
    m) meters="$OPTARG"
    ;;
    i) radius="$OPTARG"
    ;;
    u) unit="$OPTARG"
    ;;
    t) tokensdir="$OPTARG"
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

# Kind of locations
if [ -z "$years" ]
then
  kind='pplaces'
else
  kind='clusters'
fi

# Log file
fnlog="../logs/$code-$kind-$(date +"%FT%H%M")-features.log"
echo "params: -r $root -c $code -y $years -n $njobs -w $width -d $distance -m $meters -i $radius -u $unit -t $tokensdir" 2>&1 | tee -a "$fnlog"
echo "LOG FILE: $fnlog"

###########################################################################################################

# 2. Load population density 
echo "====================================================================" 2>&1 | tee -a "$fnlog"
echo "python batch_population.py -r $root -y $years -m $meters" 2>&1 | tee -a "$fnlog" 
python batch_population.py -r "$root" -y "$years" -m "$meters" 2>&1 | tee -a "$fnlog"

# 3. Get nightlight intensity features
echo "====================================================================" 2>&1 | tee -a "$fnlog"
echo "python batch_nightlights.py -r $root -y $years -m $meters -a $apikey -p $projectid -s $servicename" 2>&1 | tee -a "$fnlog"
python batch_nightlights.py -r "$root" -y "$years" -m "$meters" -a "$apikey" -p "$projectid" -s "$servicename" 2>&1 | tee -a "$fnlog"

# 4. Load cell-antennas 
echo "====================================================================" 2>&1 | tee -a "$fnlog"
echo "python batch_cells.py -r $root -y $years -d $distance -m $meters" 2>&1 | tee -a "$fnlog"
python batch_cells.py -r "$root" -y "$years" -d $distance -m $meters 2>&1 | tee -a "$fnlog"

# 5. Get Facebook Movement features
echo "====================================================================" 2>&1 | tee -a "$fnlog"
echo "python batch_movement.py -r $root -y $years -j $njobs" 2>&1 | tee -a "$fnlog"
python batch_movement.py -r "$root" -y "$years" -j $njobs 2>&1 | tee -a "$fnlog"

# 6. Get satellite images features
echo "====================================================================" 2>&1 | tee -a "$fnlog"
echo "python batch_staticmaps.py -r $root -y $years -s $secretfn -k $keyfn" 2>&1 | tee -a "$fnlog"
python batch_staticmaps.py -r "$root" -y "$years" -s "$secretfn" -k "$keyfn" 2>&1 | tee -a "$fnlog"

# 7. Get OpenStreetMap features
echo "====================================================================" 2>&1 | tee -a "$fnlog"
echo "python batch_osm.py -r $root -y $years -w $width" 2>&1 | tee -a "$fnlog"
python batch_osm.py -r "$root" -y "$years" -w $width 2>&1 | tee -a "$fnlog"

# 8. Get Facebook Marketing API features
echo "====================================================================" 2>&1 | tee -a "$fnlog"
echo "python batch_marketing.py -r $root -y $years -t $tokensdir -i $radius -u $unit" 2>&1 | tee -a "$fnlog"
python batch_marketing.py -r "$root" -y "$years" -t "$tokensdir" -i $radius -u "$unit" 2>&1 | tee -a "$fnlog"

# 9. end
echo "====================================================================" 2>&1 | tee -a "$fnlog"
echo "LOG FILE: $fnlog"