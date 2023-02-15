###############################################################################
# Dependencies
###############################################################################
import os
import time
import glob
import argparse
import pandas as pd

from maps.geoviz import GeoViz
from maps.osm import OSM
from utils import viz
from utils import ios
from utils import validations

###############################################################################
# Constants
###############################################################################

from utils.constants import * 

###############################################################################
# Functions
###############################################################################

def run(root, load=False):
  # validation
  validations.validate_not_empty(root,'root')
  
  # data
  fn_places = ios.get_places_file(root)

  country = root.split("/")[-2 if root.endswith('/') else -1]
  print("country:",country)
  code = COUNTRIES[country]['code']
  print("code:",code)
  cachedir = os.path.join(root,'cache','OSMPP')

  osmppl = OSM()
  osmppl.get_populated_places_by_country(code, fn=fn_places, load=load, cache_dir=cachedir)
  print(osmppl.ppl.shape)
  print(fn_places)

###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", help="Country's main folder.", type=str, required=True)
    parser.add_argument("-load", help="Load if exists otherwise keep querying", action='store_true')
    
    args = parser.parse_args()
    for arg in vars(args):
      print("{}: {}".format(arg, getattr(args, arg)))
      
    start_time = time.time()
    run(args.r, args.load)
    print("--- %s seconds ---" % (time.time() - start_time))