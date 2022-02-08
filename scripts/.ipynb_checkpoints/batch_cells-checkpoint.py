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

###############################################################################
# Functions
###############################################################################

def run(root, years, distance):
  fn_cells = os.path.join(root,"connectivity","cell_towers_{}.csv".format(root.split("/")[-1]))
  fn_cells, fn_places = ios.get_data_and_places_file(root, years, 'connectivity')

  df_places_new = opencellid.update_opencellid_features(fn_cells, fn_places, distance)
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
    parser.add_argument("-d", help="Maximum distance in meters between antennas to determine same tower. (E.g. 5).", type=float, default=5.0)
    
    args = parser.parse_args()
    for arg in vars(args):
      print("{}: {}".format(arg, getattr(args, arg)))
      
    start_time = time.time()
    run(args.r, args.y, args.d)
    print("--- %s seconds ---" % (time.time() - start_time))