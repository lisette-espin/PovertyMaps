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
from ses.data import delete_nonprojected_variables

###############################################################################
# Functions
###############################################################################

def run(root, years, api_key, project_id):
  fn_places = ios.get_places_file(root, years)
  df_places = ios.load_csv(fn_places, index_col=0)

  if 'DHSYEAR' in df_places.columns:
    years = df_places.DHSYEAR.unique()
    today = None
  else:
    today = datetime.today()
    years = np.array([today.year-1])

  meters = geo.METERS
  total = len(years)*len(meters)
  rad_gte_thres = 10.0
  newyear = -1
  CACHEDIR = os.path.join(root,'cache','VIIRS{}'.format("" if today is None else 'PP'))
  print()
  
  for year, meters in tqdm(itertools.product(years,meters), total=total):
    # constructor per year
    if newyear != year:
      nl = VIIRS(source="viirs", api_key=api_key, project_id=project_id)
      nl.auth()
      if today is None:
        # year of survey data
        nl.filter(select='mean', date_start='{}-01-01'.format(year), date_end='{}-01-01'.format(year+1))
      else:
        # latest data (populated places)
        nl.filter(select='mean', date_start='{}-01-01'.format(year), date_end='{}-01-01'.format(year+2))
      newyear = year
      
    # points
    if 'DHSYEAR' in df_places.columns:
      index = df_places.query("DHSYEAR==@year").index.values
      allpoints = df_places.loc[index,['LONGNUM','LATNUM']].values
    else:
      index = df_places.index.values
      allpoints = df_places.loc[index,['lon','lat']].values 

    tmp = nl.fast_lights_at_points(allpoints=allpoints, buffer_meters=meters, rad_gte_thres=rad_gte_thres, cache_dir=CACHEDIR)
    tmp.set_index(index, inplace=True)
    df_places.loc[index,tmp.columns.values] = tmp
  
  df_places = delete_nonprojected_variables(df_places, os.path.basename(__file__), del_geometry=True)
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
    parser.add_argument("-y", help="Year or years separated by comma (E.g. 2016,2019).", type=str, required=True)
    parser.add_argument("-a", help="Path to API_KEY.", type=str, required=True)
    parser.add_argument("-p", help="Path to PROJECT_ID.", type=str, required=True)
    args = parser.parse_args()
    for arg in vars(args):
      print("{}: {}".format(arg, getattr(args, arg)))
      
    start_time = time.time()
    run(args.r, args.y, args.a, args.p)
    print("--- %s seconds ---" % (time.time() - start_time))