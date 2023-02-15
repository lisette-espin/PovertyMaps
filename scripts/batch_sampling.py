###############################################################################
# Dependencies
###############################################################################
import os
import time
import glob
import argparse

import numpy as np
import pandas as pd

from ses import locations as loc
from utils import ios
from utils import validations

###############################################################################
# Constants
###############################################################################

from utils.constants import *

###############################################################################
# Functions
###############################################################################

def run(root, years, dhsloc, traintype, kfolds, epochs=None, random_states=None):
  # validation
  validations.validate_not_empty(root,'root')
  
  # 1. identify years/survey
  prefix = ios.get_prefix_surveys(root=root, years=years)

  # 2. identifiying the dataset to train on
  validations.validate_traintype(traintype)
  validations.validate_years_traintype(years, traintype)
  
  # 3. pre-process
  null = NONE + NO
  epochs = None if epochs in null else epochs
  random_states = None if random_states in null else random_states
  
  if epochs is None and random_states is None:
    raise Exception("Either epochs or random_states must be given.")

  if random_states is not None:
    if type(random_states) == str:
      random_states = [int(rs) for rs in random_states.replace(" ","").strip(" ").split(",")]
    epochs = len(random_states)

  # 4. load main data
  fn_mapping = glob.glob(os.path.join(root,'results','features',f'{prefix}_{dhsloc}_cluster_pplace_ids.csv'))[0]
  df_mapping = ios.load_csv(fn_mapping)
  print(f"{df_mapping.shape[0]} records loaded.")

  # in case years is a subset
  years = validations.validate_years(years)
  df_mapping = df_mapping.query("cluster_year in @years").copy()
  print(f"{df_mapping.shape[0]} records in {years}.")
  prefix = "_".join([pre for pre in prefix.split('_') for y in years if str(y) in pre])
  
  # internal stratified sampling
  bins = 10
  df_mapping.loc[:,'ses'] = pd.cut(df_mapping.loc[:,f"mean_{WEALTH}"], bins=bins, labels=np.arange(bins), precision=0, retbins=False, ordered=True)
  
  # 5. shuffle, split
  for epoch in np.arange(1,epochs+1,1):
    random_state = np.random.randint(0,2**32 - 1,1)[0] if random_states is None else random_states[epoch-1]
    df, rs = loc.split_kfold_stratified(df_mapping, 'ses', dhsloc, traintype, kfolds, random_state)

    print("random_state:",rs)
    print(f"records run #{epoch}",df.shape[0])
    print(df['test'].value_counts())
    for k in np.arange(kfolds-1):
      fold = f'fold{k+1}'
      print(df[fold].value_counts())

    #. 6 save
    path = os.path.dirname(fn_mapping)
    path = path.replace("/features",f"/samples/{prefix}_{traintype}_{dhsloc}/epoch{epoch}-rs{rs}")
    ios.validate_path(path)
    fn = os.path.join(path, f'data.csv')
    ios.save_csv(df, fn)

###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", help="Country's main folder.", type=str, required=True)
    parser.add_argument("-y", help="Years (comma-separated): 2016,2019", type=str, required=True)
    parser.add_argument("-o", help="Type of DHS location: none, cc, ccur, gc, gcur, rc", type=str, default='none')
    parser.add_argument("-t", help="Years to include in training: all, newest, oldest", type=str, default='none')
    parser.add_argument("-k", help="K-folds", type=int, required=True)
    parser.add_argument("-e", help="# Epochs (repeats / runids)", type=int, default=None, required=False)
    parser.add_argument("-s", help="Comma separated random seeds", default=None, required=False)

    args = parser.parse_args()
    for arg in vars(args):
      print("{}: {}".format(arg, getattr(args, arg)))
      
    start_time = time.time()
    run(args.r, args.y, args.o, args.t, args.k, args.e, args.s)
    print("--- %s seconds ---" % (time.time() - start_time))