###############################################################################
# Dependencies
###############################################################################
import os
import time
import glob
import argparse
import itertools
import numpy as np
from tqdm import tqdm
from datetime import datetime

from maps import geo
from utils import ios
from google.viirs import VIIRS
from utils import validations

###############################################################################
# Constants
###############################################################################

from utils.constants import GTID
from utils.constants import LON
from utils.constants import LAT

###############################################################################
# Functions
###############################################################################

def run(root, years, meters, api_key, project_id, service_account):
  # validation
  validations.validate_not_empty(root,'root')
  meters = validations.validate_meters(meters)
  
  # data
  fn_places = ios.get_places_file(root, years)
  df_places = ios.load_csv(fn_places, index_col=0)

  if GTID in df_places.columns:
    years = df_places.year.unique()
    today = None
  else:
    today = datetime.today()
    years = np.array([today.year-1])

  total = len(years)*len(meters)
  rad_gte_thres = 10.0
  newyear = -1
  cachedir = os.path.join(root,'cache','VIIRS{}'.format("" if today is None else 'PP'))
  print()
  
  for year, m in tqdm(itertools.product(years,meters), total=total):
    # constructor per year
    if newyear != year:
      nl = VIIRS(source="viirs", year=year, api_key=api_key, project_id=project_id, service_account=service_account)
      nl.auth()
      if today is None:
        # year of ground-turth data
        nl.filter(select='mean', date_start='{}-01-01'.format(year), date_end='{}-01-01'.format(year+1))
      else:
        # latest data (populated places)
        nl.filter(select='mean', date_start='{}-01-01'.format(year), date_end='{}-01-01'.format(year+2))
      newyear = year
      
    # points
    if GTID in df_places.columns:
      index = df_places.query("year==@year").index.values
      allpoints = df_places.loc[index,[LON,LAT]].values
    else:
      index = df_places.index.values
      allpoints = df_places.loc[index,[LON,LAT]].values 

    tmp = nl.fast_lights_at_points(allpoints=allpoints, buffer_meters=m, rad_gte_thres=rad_gte_thres, cache_dir=cachedir)
    tmp.set_index(index, inplace=True)
    df_places.loc[index,tmp.columns.values] = tmp
  
  df_places = validations.delete_nonprojected_variables(df_places, os.path.basename(__file__), del_geometry=True)
  print(df_places.head())
  print(df_places.shape)

  # save
  fn_places_new = fn_places.replace(".csv","_NTLL.csv")
  ios.save_csv(df_places, fn_places_new)

###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", help="Country's main folder.", type=str, required=True)
    parser.add_argument("-y", help="Year or years separated by comma (E.g. 2016,2019).", type=str, required=False, default=None)
    parser.add_argument("-m", help="Comma separated bbox width (eg. 1000,2000,3000).", type=str, default=None, required=False)
    
    parser.add_argument("-a", help="Path to API_KEY.", type=str, required=False)
    parser.add_argument("-p", help="Path to PROJECT_ID.", type=str, required=False)
    parser.add_argument("-s", help="Path to SERVICE_ACCOUNT.", type=str, required=False)
    
    args = parser.parse_args()
    for arg in vars(args):
      print("{}: {}".format(arg, getattr(args, arg)))
      
    start_time = time.time()
    run(args.r, args.y, args.m, args.a, args.p, args.s)
    print("--- %s seconds ---" % (time.time() - start_time))