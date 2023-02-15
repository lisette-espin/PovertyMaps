###############################################################################
# Dependencies
###############################################################################
import os
import time
import glob
import argparse

import numpy as np
import geopandas as gpd
from shapely.ops import nearest_points
from shapely.strtree import STRtree
from pyproj import Transformer

from utils import ios
from maps import opencellid
from maps import geo
from utils import validations

###############################################################################
# Functions
###############################################################################

def run(root, years, distance, meters, njobs=1):
  validations.validate_not_empty(root,'root')
  fn_cells, fn_places = ios.get_data_and_places_file(root, years, 'connectivity')

  distance = validations.validate_max_distance_antenna(distance)
  meters = validations.validate_meters(meters)
  df_places_new = opencellid.update_opencellid_features(fn_cells, fn_places, distance, meters, njobs)
  print(df_places_new.head())
  print(df_places_new.shape)

  fn_places_new = fn_places.replace(".csv","_cells.csv")
  ios.save_csv(df_places_new, fn_places_new)


###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", help="Country's main folder.", type=str, required=True)
    parser.add_argument("-y", help="Year or years separated by comma (E.g. 2016,2019).", type=str, required=False)
    parser.add_argument("-d", help="Maximum distance in meters between antennas to determine same tower. (eg. 10).", type=float, default=None, required=False)
    parser.add_argument("-m", help="Comma separated bbox width (eg. 1000,2000,3000).", type=str, default=None, required=False)
    parser.add_argument("-n", help="Njobs to run in parallel.", type=int, default=1, required=False)
    
    args = parser.parse_args()
    for arg in vars(args):
      print("{}: {}".format(arg, getattr(args, arg)))
      
    start_time = time.time()
    run(args.r, args.y, args.d, args.m, args.n)
    print("--- %s seconds ---" % (time.time() - start_time))