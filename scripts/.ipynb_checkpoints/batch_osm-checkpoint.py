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

###############################################################################
# Functions
###############################################################################

def run(root, years):
  fn_places = ios.get_places_file(root, years)
  df_places = ios.load_csv(fn_places, index_col=0)

  if 'DHSID' in df_places.columns:
    # survey
    id = 'DHSID'
    places = df_places[[id,'LATNUM','LONGNUM']].values #id,lat,lon
    cachedir = os.path.join(root,'cache','OSM')
  else:
    # populated places
    id = 'OSMID'
    places = df_places[[id,'lat','lon']].values #id,lat,lon
    cachedir = os.path.join(root,'cache','OSMPP')
  
  osm = OSM()
  osm.get_features(places, fn=None, overwrite=True, cache_dir=cachedir)
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
    
    args = parser.parse_args()
    for arg in vars(args):
      print("{}: {}".format(arg, getattr(args, arg)))

    start_time = time.time()
    run(args.r, args.y)
    print("--- %s seconds ---" % (time.time() - start_time))