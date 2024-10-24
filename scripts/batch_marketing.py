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
import pandas as pd
from joblib import Parallel
from joblib import delayed
    
###############################################################################
# Constants
###############################################################################
 
from utils.constants import GTID
from utils.constants import LON
from utils.constants import LAT

###############################################################################
# Functions
###############################################################################

def run(root, years, tokens_dir, radius=None, unit=None, n_jobs=1):
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
  # tokens = FacebookMarketing.load_tokens(tokens_dir)
  
  # query API
  fn_places_new = fn_places.replace(".csv","_FBM.csv")
  id,lat,lon = validations.get_column_id(df_places), LAT, LON
  cachedir = os.path.join(root,'cache','FBM{}'.format('' if GTID in df_places else 'PP'))
  print(f'- cache_dir: {cachedir}')
  
  if n_jobs <= 1:
    print("=== Sequential querying (it might take a while) ===")
    df_places = FacebookMarketing.query(df_places, profiles, radius, unit, id, lat, lon, ccode, 
                                        tokens_dir, cachedir, fn_places_new)
  else:
    ############ @TODO: NOT TESTED YET
    print("[ERROR] Not implemented")
#     print(f"=== Parallel querying ({n_jobs} jobs in parallel, ~15sec per place,  ) ===")
#     df_places_chunks = np.array_split(df_places, n_jobs)
#     tokens_chunks = np.array_split(tokens, n_jobs)
#     print(f"- {df_places_chunks.shape[0]} pplaces chunks")
#     print(f"- {tokens_chunks.shape[0]} tokens chunks")
    
#     results = Parallel(n_jobs=n_jobs)(delayed(FacebookMarketing.query)(df_places_chunks[i], 
#                                                                       profiles, radius, unit, id, lat, lon, 
#                                                                       ccode, tokens_chunks[i], cachedir, fn_places_new) for 
#                                      i in np.arrange(n_jobs))
#     df_places = pd.concat(results)
    
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
    parser.add_argument("-n", help="Njobs (parallelize)", type=int, default=1, required=False)
    
    # parser.add_argument("-i", help="Time intervals (eg., 0-8 or 0,8,16) ", type=str, default=None, required=False)
    # parser.add_argument("-o", help="Hour to collect data in country (HH:01,02,..,12,13,..22,23)", type=int, default=15, required=False)
    # parser.add_argument("-a", help="Delta time relaxation: collect from h-a to h+a (eg. 3)", type=int, default=3, required=False)
    # parser.add_argument("-x", help="How many times to collect the reach estimates (e.g., 3)", type=int, default=3, required=False)
    
    args = parser.parse_args()
    for arg in vars(args):
      print("{}: {}".format(arg, getattr(args, arg)))

    start_time = time.time()
    run(args.r, args.y, args.t, args.i, args.u, args.n)
    print("--- %s seconds ---" % (time.time() - start_time))