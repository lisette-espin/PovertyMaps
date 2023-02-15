###############################################################################
# Dependencies
###############################################################################
import os
import time
import glob
import argparse
import pandas as pd
from tqdm import tqdm
from collections import Counter

from utils import ios
from utils import validations
from google.staticmaps import StaticMaps
from utils import validations

###############################################################################
# Constants
###############################################################################

from utils.constants import *

###############################################################################
# Functions
###############################################################################

def run(root, years, secretfn, keyfn):
  # validation
  validations.validate_not_empty(root,'root')
  
  # survey data
  fn_places = ios.get_places_file(root, years)
  df_places = ios.load_csv(fn_places, index_col=0)
  
  # Google Maps Statis API setup
  SECRET = ios.read_txt(secretfn)
  KEY = ios.read_txt(keyfn)
  folder = validations.get_places_kind(df_places)
  CACHEDIR = os.path.join(root,'results','staticmaps',folder)

  ### query API
  query_api(df_places, SECRET, KEY, CACHEDIR)
  print(f"- Images stored in: {CACHEDIR}")

def query_api(df_places, SECRET, KEY, CACHEDIR):
  results = []
  counter = 0
  for id,row in tqdm(df_places.iterrows(), total=df_places.shape[0]):
    sm = StaticMaps(key=KEY, secret=SECRET, lat=row.lat, lon=row.lon, size=SIZE, zoom=ZOOM, scale=SCALE, maptype=MAPTYPE)
    prefix = StaticMaps.get_prefix(row) #'Y{}-C{}-U{}'.format(int(row.year), int(row.cluster), int(row.rural)) if GTID in df_places else 'OSMID{}'.format(row.OSMID)
    results.append((id, sm.retrieve_and_save(CACHEDIR, prefix=prefix, verbose=False, load=True)))
    
    counter += 1
    if counter % 500 == 0:
      time.sleep(1)

  status_counts = Counter([code for fn,code in results])
  print('')
  print("1:ok | 0:already exist | -1:error")
  print(status_counts) # 1:ok, 0:already exists, -1:error

###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", help="Country's main folder.", type=str, required=True)
    parser.add_argument("-y", help="Year or years separated by comma (E.g. 2016,2019).", type=str, default=None, required=False)
    parser.add_argument("-s", help="Path to secret Google Maps Static API.", type=str, required=True)
    parser.add_argument("-k", help="Path to key Google Maps Static API", type=str, required=True)
    
    args = parser.parse_args()
    for arg in vars(args):
      print("{}: {}".format(arg, getattr(args, arg)))

    start_time = time.time()
    run(args.r, args.y, args.s, args.k)
    print("--- %s seconds ---" % (time.time() - start_time))