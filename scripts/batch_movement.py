import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

###############################################################################
# Dependencies
###############################################################################
import os
import gc
import sys
import glob
import time
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from collections import Counter
from joblib import Parallel, delayed

from utils import ios
from utils import validations
from facebook.movement import FacebookMovement
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # hide warning messages from TF

###############################################################################
# Constants
###############################################################################

from utils.constants import *

###############################################################################
# Functions
###############################################################################

def run(root, years, njobs=1):
  # validation
  validations.validate_not_empty(root,'root')
  
  # survey data
  fn_places = ios.get_places_file(root, years)
  df_places = ios.load_csv(fn_places, index_col=0)

  # params setup
  country = root.split("/")[-2 if root.endswith('/') else -1]
  code = COUNTRIES[country]['code']
  path = os.path.join(root, "movement/Facebook")

  # generate movement graph
  mvnet = FacebookMovement(country_name=country, country_code=code, path=path)
  mvnet.load_movements(njobs=njobs)
  mvnet.generate_network(includenan=True)
  print(mvnet.M.shape, mvnet.A.shape, mvnet.D.shape)
  print(mvnet.M.count_nonzero(), mvnet.A.sum())
  print(mvnet.M.sum())
  print(mvnet.D.min(), mvnet.D.max(), mvnet.D.mean())

  # generate features for each cluster
  fn_places_new = fn_places.replace(".csv","_FBMV.csv")
  df_places_new = mvnet.get_features(df_places)

  # final save
  ios.save_csv(df_places_new, fn_places_new)


###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", help="Country's main folder.", type=str, required=True)
    parser.add_argument("-y", help="Year or years separated by comma (E.g. 2016,2019).", type=str, required=False, default=None)
    parser.add_argument("-j", help="n_jobs for parallel computing (E.g. 10)", type=int, default=1)
    
    args = parser.parse_args()
    for arg in vars(args):
      print("{}: {}".format(arg, getattr(args, arg)))
    
    start_time = time.time()
    run(args.r, args.y, args.j)
    print("--- %s seconds ---" % (time.time() - start_time))