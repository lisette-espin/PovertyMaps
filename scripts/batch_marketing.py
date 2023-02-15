###############################################################################
# Dependencies
###############################################################################
import os
import time
import glob
import argparse
import numpy as np
from maps import geo
from maps.geoviz import GeoViz
from maps.osm import OSM
from utils import ios
from facebook.marketing import FacebookMarketing
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

def run(root, years, tokens_dir, radius=None, unit=None):
  # validation
  validations.validate_not_empty(root,'root')
  validations.validate_not_empty(tokens_dir,'tokens_dir')
  ccode = validations.get_country_code(root)
  radius, unit = validations.validate_radius(radius, unit)
  
  # survey data
  fn_places = ios.get_places_file(root, years)
  df_places = ios.load_csv(fn_places, index_col=0)

  # FBM setup
  profiles = FacebookMarketing.get_all_profiles()
  tokens = FacebookMarketing.load_tokens(tokens_dir)
  
  # query API
  fn_places_new = fn_places.replace(".csv","_FBM.csv")
  id,lat,lon = validations.get_column_id(df_places), LAT, LON
  cachedir = os.path.join(root,'cache','FBM{}'.format('' if GTID in df_places else 'PP'))
  df_places = FacebookMarketing.query(df_places, profiles, radius, unit, id, lat, lon, ccode, tokens, cachedir, fn_places_new)
  
  # final save
  df_places = validations.delete_nonprojected_variables(df_places, os.path.basename(__file__), True) 
  ios.save_csv(df_places, fn_places_new)


###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", help="Country's main folder.", type=str, required=True)
    parser.add_argument("-y", help="Year or years separated by comma (eg. 2016,2019).", type=str, default=None, required=False)
    parser.add_argument("-t", help="Directory where all token files are in the form filenameID.json", type=str, required=True)
    parser.add_argument("-i", help="Radius for search", type=float, default=None, required=False)
    parser.add_argument("-u", help="Unit to account for radius: kilometer or mile", type=str, default=None, required=False)
    
    # parser.add_argument("-i", help="Time intervals (eg., 0-8 or 0,8,16) ", type=str, default=None, required=False)
    # parser.add_argument("-o", help="Hour to collect data in country (HH:01,02,..,12,13,..22,23)", type=int, default=15, required=False)
    # parser.add_argument("-a", help="Delta time relaxation: collect from h-a to h+a (eg. 3)", type=int, default=3, required=False)
    # parser.add_argument("-x", help="How many times to collect the reach estimates (e.g., 3)", type=int, default=3, required=False)
    
    args = parser.parse_args()
    for arg in vars(args):
      print("{}: {}".format(arg, getattr(args, arg)))

    start_time = time.time()
    run(args.r, args.y, args.t, args.i, args.u)
    print("--- %s seconds ---" % (time.time() - start_time))