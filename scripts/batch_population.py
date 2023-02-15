###############################################################################
# Dependencies
###############################################################################
import os
import time
import glob
import argparse
import numpy as np
from tqdm import tqdm

from maps import geo
from utils import ios
from facebook.population import FacebookPopulation
from utils import validations

###############################################################################
# Functions
###############################################################################

def run(root, years, meters):
  # validation
  validations.validate_not_empty(root,'root')
  meters = validations.validate_meters(meters)
  
  # data
  fn_pop, fn_places = ios.get_data_and_places_file(root, years, 'population')

  # load data
  fb = FacebookPopulation(fn_pop=fn_pop, fn_places=fn_places)
  
  print('Loading data...')
  with tqdm(total=2) as pbar:
    fb.load_data()
    pbar.update(1)
    fb.project_data()
    pbar.update(1)

  # update features
  df_places_new = fb.update_population_features(meters)
  print(df_places_new.head())
  print(df_places_new.shape)

  # save
  fn_places_new = fn_places.replace(".csv","_population.csv")
  ios.save_csv(df_places_new, fn_places_new)


###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", help="Country's main folder.", type=str, required=True)
    parser.add_argument("-y", help="Year or years separated by comma (E.g. 2016,2019).", type=str, required=False, default=None)
    parser.add_argument("-m", help="Comma separated bbox width (eg. 1000,2000,3000).", type=str, default=None, required=False)
    
    args = parser.parse_args()
    for arg in vars(args):
      print("{}: {}".format(arg, getattr(args, arg)))
    start_time = time.time()
    
    run(args.r, args.y, args.m)
    print("--- %s seconds ---" % (time.time() - start_time))