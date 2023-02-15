###############################################################################
# Dependencies
###############################################################################
import os
import time
import glob
import argparse
import numpy as np

from maps.geoviz import GeoViz
from maps.osm import OSM
from utils import ios
from utils import validations
from utils.constants import LON
from utils.constants import LAT
from utils.constants import GTID
from utils.constants import OSMID

###############################################################################
# Functions
###############################################################################

def run(root, years, width):
  # validation
  validations.validate_not_empty(root,'root')
  
  # data
  fn_places = ios.get_places_file(root, years)
  df_places = ios.load_csv(fn_places, index_col=0)

  id = GTID if GTID in df_places.columns else OSMID if OSMID in df_places.columns else None
  subfolder = f"OSM{'PP' if id=='OSMID' else ''}"
  if id is None:
    raise Exception("gtID or OSMID columns not in dataset.")
  places = df_places[[id,LAT,LON]].values #id,lat,lon
  cachedir = os.path.join(root,'cache',subfolder)
  print(f"cache_dir: {cachedir}")
  
  osm = OSM()
  osm.get_features(places, width, fn=None, overwrite=True, cache_dir=cachedir)
  df_places_new = osm.features
  df_places_new.rename(columns={'id':id}, inplace=True)
  print(df_places_new.head())
  print(df_places_new.shape)

  fn_places_new = fn_places.replace(".csv","_OSM.csv")
  ios.save_csv(df_places_new, fn_places_new)

###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", help="Country's main folder.", type=str, required=True)
    parser.add_argument("-y", help="Year or years separated by comma (E.g. 2016,2019).", type=str, required=False)
    parser.add_argument("-w", help="Width of the bounding box in meters (E.g. 1600).", type=float, default=0, required=False)
    
    args = parser.parse_args()
    for arg in vars(args):
      print("{}: {}".format(arg, getattr(args, arg)))

    start_time = time.time()
    run(args.r, args.y, args.w)
    print("--- %s seconds ---" % (time.time() - start_time))