###############################################################################
# Constants
###############################################################################
import re
import os
import sys
import numpy as np
from traitlets.traitlets import TraitType
from datetime import timedelta
from datetime import datetime
from pytz import timezone
import time

###############################################################################
# Constants
###############################################################################

from utils.constants import *

###############################################################################
# Functions
###############################################################################
   
def validate_grid_size(**kwargs):
  key = 'grid_size'
  
  if key not in kwargs:
    print(key)
    raise Exception("grid_sizr parameter missing")
  
  try:
    key = float(kwargs[key])
    if key <= 0:
      raise Exception("grid_size must be positive and non-zero")
  except Exception as ex:
    print(f"[ERROR] validations.py | validate_grid_size | {ex}")
    sys.exit(0)

def validate_not_empty(var, name):
  if var in NONE:
    raise Exception(f"- {name} is missing.")
  
def get_country_code(root):
  return root.split('/')[-2 if root.endswith('/') else -1]

def validate_years(years):
  if type(years)==int:
    years = str(years)
  if type(years)==str:
    years = [y.strip('').replace(' ','') for y in years.split(',')] if type(years)==str else years
  if type(years)==list:
    years = [int(y) for y in years]
    return years
  raise Exception("wrong type for years")

def validate_traintype(traintype):
  if traintype not in TRAIN_YEAR_TYPES:
    raise Exception("train type does not exist.")

def validate_years_traintype(years, traintype):
  if traintype not in [TTYPE_ALL,TTYPE_DOWNSAMPLE] and len(years)==1:
    print(f"ERROR: traintype:{traintype} | years:{years} | You need at least 2 years.")
    raise Exception("train type and years do not match.")

def valid_file_year(fn, years):
  all_years_in_fn = re.findall(r'\d+', os.path.basename(fn))
  if len(all_years_in_fn) != len(years):
    return False
  iyears = set([int(y) for y in years])
  fyears = set([int(y) for y in all_years_in_fn])
  return len(iyears.intersection(fyears))==len(years)
  
def validate_timevar(timevar):
  if timevar in NONE or timevar in NO:
    return None
  if timevar in TIMEVAR_OPTIONS:
    return timevar
  print(f"[ERROR] validations.py | validate_timevar | timevar:{timevar}")
  raise Exception("timevar does not exist.")
      
def validate_none(var, extended=True):
  if extended and var in NONE or var in NO:
      return None
  elif not extended and var in NONE:
      return None
  return var

def is_regression(y_attributes):
  r = [int(y in REGRESSION_VARS) for y in y_attributes]
  
  if sum(r) == len(r):
    return True

  if sum(r) > 0:
    return False

  raise Exception("Combined prediction (regression and classification) not supported.")
  
def get_valid_output_names(y_attributes):
  if type(y_attributes)==str:
    y_attributes = [y.strip('').replace(' ','') for y in y_attributes.strip('').replace(' ','').split(',')]
  if type(y_attributes)==list:
    if len(y_attributes) not in [1,2]:
      raise Exception("Only 1 or 2 outputs are supported at the moment")
    return y_attributes
  raise Exception("wrong type for y_attributes")

def get_column_id(df):
  for c in ['gtID','OSMID']:
    if c in df.columns:
      return c
  raise Exception("column id gtID or OSMID does not exist.")
  
def get_places_kind(df):
  if 'gtID' in df.columns:
    return GT_PLACE
  if 'OSMID' in df.columns:
    return P_PLACE
  raise Exception('Kind of place is neither grount-truth cluster (no gtID columns) not a pplace (no OSMID column)')
  
def delete_nonprojected_variables(df, fn, del_geometry=False):
  # This ie being used in: libs.maps.opencellid and libs.facebook.population, libs.facebook.movement, nightlights
  import gc
  # removing these columns
  data_col_ids = ['gtID','OSMID','geometry']
  data_cols = [c for c in df.columns if c not in data_col_ids and not c.startswith("FBMV_") and not c.startswith("FBM_") and not c.startswith("NTLL_")]
  fb_mv = ['original_index']
  try:
    cols_to_remove = data_cols + fb_mv
    if del_geometry:
      cols_to_remove += ['geometry']
    df = df.drop(columns=cols_to_remove, errors='ignore')
    gc.collect()
  except Exception as ex:
    print(f"{fn} | delete_nonprojected_variables | ",ex)
  
  cols = df.columns.values.tolist()
  first_col = [c for c in cols if c in data_col_ids]
  cols = [c for c in cols if c not in first_col]
  cols = first_col +  cols
  return df.loc[:,cols]

def validate_source(source, model=None):
  if source not in ['all',None] and source not in FEATURE_SOURCES:
    raise Exception ("Not a valid source")

  if model is not None:
    if source in FEATURE_SOURCES and model not in ['CB', 'xgb', 'xgb-{source}', 'catboost']:
      raise Exception ("Not a valid model")
    
##############################################################################################################
# OpenCellId
##############################################################################################################

def validate_max_distance_antenna(distance):
  # in meters
  if distance in NONE:
    distance = MAX_DISTANCE_ANTENNA_METERS
  else:
    if type(distance)==str:
      distance = float(distance.strip('').replace(' ',''))
    if distance < 0:
      distance = MAX_DISTANCE_ANTENNA_METERS
      
  print("- Max distance in meters (antennas same tower): ", distance)
  return distance
  
def validate_meters(meters):
  if meters in NONE or meters in NO:
    numbers = METERS
  else:
    try:
      numbers = [float(m.strip('').replace(' ','')) for m in meters.strip().split(',')]
      if len(numbers)==0:
        numbers = METERS
    except Exception as ex:
      print(f"[ERROR] validations.py | validate_meters | {ex}")
    
  print("- BBox width's in meters: ", numbers)
  return numbers

##############################################################################################################
# FBM
##############################################################################################################


def validate_radius(radius, unit):
  from facebook.marketing import FacebookMarketing
  
  if radius in NONE or radius in NO:
    radius = MILE_TO_M / KM_TO_M # 1 mile in Km (see line below)
    unit = FacebookMarketing.UNIT_KM
    
  if unit in NONE:
    raise Exception("The unit -u must be specified: kilometer of mile")
    
  print(f"- Radius set to: {radius} {unit}")
  return radius, unit


def validate_time_intervals(sintervals):
  intervals = None
  terminate = False
  try:
    intervals = [(int(pair.split('-')[0]),int(pair.split('-')[1])) for pair in sintervals.split(',')]
    for i in np.arange(len(intervals)):
      for j in np.arange(0,i):
        if (intervals[i][0]>=intervals[j][0] and intervals[i][0]<intervals[j][1]) or (intervals[i][1]>intervals[j][0] and intervals[i][1]<intervals[j][1]) or (intervals[j][0]>=intervals[i][0] and intervals[j][0]<intervals[i][1]) or (intervals[j][1]>intervals[i][0] and intervals[j][1]<intervals[i][1]):
          print(f"[ERROR] validations.py | FBM_validate_intervals | overlap: {intervals[i]} and {intervals[j]}")
          terminate = True
  except Exception as ex:
    print(f"[ERROR] validations.py | FBM_validate_intervals | {ex}")
  if terminate:
    raise Exception("Time intervals overlap.")
  return intervals

def get_current_time_in_country(ccode):
    
  if ccode not in COUNTRIES:
    try:
      ccode = [c for c,obj in COUNTRIES.items() if obj['code']==ccode][0]
    except:
      raise Exception("ccode does not exist. Please add it to constants.py") 

  # Current time in UTC
  now_utc = datetime.now(timezone('UTC'))

  # Convert to country time zone
  now_in_country = now_utc.astimezone(timezone(COUNTRIES[ccode]['tz']))
  return now_in_country

def get_current_interval(ccode, intervals):
  dnow = get_current_time_in_country(ccode)
  print(f'- Time now in {ccode}: {dnow}')
  current_interval = [i for i in intervals if (i[0]<i[1] and dnow.hour>=i[0] and dnow.hour<i[1]) or (i[0]>i[1] and dnow.hour>=i[0] and dnow.hour<i[1]+24)]
  current_interval = current_interval[0] if len(current_interval)>0 else None
  print(f"- Current interval: {current_interval} (all: {intervals})")
  return current_interval, dnow

def get_seconds_to_wait_to_target(ccode, intervals, verbose=True):
  current_interval, t1 = get_current_interval(ccode, intervals)
  next_workday = False

  if  t1.isoweekday() not in WORKDAYS:
    # next working day
    next_workday = True
    next_interval = sorted({i[0]:i for i in intervals}.items())
    next_interval = next_interval[0][1]

  elif current_interval is not None and t1.hour == current_interval[0]:
    # Today, now
    if verbose:
        print(f"In {ccode} is now {t1}.")
        print('Ready to go!')
    return timedelta(0)

  else:
    next_interval = sorted({i[0]-t1.hour:i for i in intervals if t1.hour<i[0]}.items())
    if len(next_interval) == 0:
      # next working day
      next_workday = True
      next_interval = sorted({i[0]:i for i in intervals}.items())
    next_interval = next_interval[0][1]


  # next_interval
  days_until_next_working_day = 7-t1.isoweekday()+1 if next_workday else 0
  t2 = datetime(t1.year,t1.month,t1.day+days_until_next_working_day,next_interval[0],0,0) 
  waiting_time = t2.replace(tzinfo=None)-t1.replace(tzinfo=None)

  t0 = datetime.now()
  t3 = t0 + waiting_time

  if verbose:
    print(f"In {ccode} is now ({'workday' if t1.day in WORKDAYS else 'weekend'}) {t1}.")
    print(f"- Waiting {waiting_time} until it is {t2} in {ccode}.")
    print(f"Your current time is {t0}.")
    print(f"- Waiting {waiting_time} until it is {t3} in your current time.")
  return waiting_time
