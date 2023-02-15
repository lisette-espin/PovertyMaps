################################################################
# Dependencies: System
################################################################
import os
import re
import time
import glob
import json
import h5py
import ntpath
import pickle
import joblib
import numpy as np
import pandas as pd
from shutil import copyfile
from numpy import savez_compressed

################################################################
# Constants
################################################################

from utils.constants import NONE

################################################################
# Functions: Places (survey or pplaces) files
################################################################

def get_data_and_places_file(root, years, source):

  # population or cell data
  if source not in ['connectivity','population']:
    raise Exception("Source does not exist.")

  if source == 'connectivity':
    fn_data = glob.glob(os.path.join(root,source,"cell_towers_*.csv"))
  elif source == 'population':
    fn_data = glob.glob(os.path.join(root,source,"Facebook","*_general_2020_csv.zip"))
    # fn_data = glob.glob(os.path.join(root,source,"Facebook","population_*.csv.zip")) # old: 2019
  
  if len(fn_data) > 0:
    fn_data = fn_data[0]
  else:
    raise Exception("Data not found.")

  # places data (dhs or populated places)
  fn_places = get_places_file(root, years)

  return fn_data, fn_places

def get_places_file(root, years=None, households=False, verbose=True):
  fn_places = None
  
  if root in NONE:
    raise Exception("Survey data not specified.")
    
  if years in NONE:
    fn_places = os.path.join(root,'results','features','pplaces','PPLACES.csv')
  else:
    if type(years)==str:
      years = years.strip(" ").replace(" ","").split(",")
    
    if households:
      files = glob.glob(os.path.join(root,'results','features','households',"*_household.csv"))
    else:
      files = glob.glob(os.path.join(root,'results','features','clusters',"*_cluster.csv"))

    if len(files) is None:
       raise Exception("Survey data not found.")

    fn_places = None
    
    if verbose:
      print(files)
    
    # exact number of years
    for fn in files:
      allyears = re.findall(r'\d+', os.path.basename(fn))
      nyears = len(allyears) # how many 4-digit numbers are in fn
      years_in_fn = sum([1 for year in years if str(year) in fn]) # from 'years' how many are in fn
        
      if nyears == len(years) and years_in_fn == len(years):
        fn_places = fn 
        if verbose:
          print(f'[INFO] you will load {years}, fn_places: {fn_places}')
        break
    
    # if none, then extract from combination if exists
    if fn_places is None:
      for fn in files:
        allyears = re.findall(r'\d+', os.path.basename(fn))
        nyears = len(allyears) # how many 4-digit numbers are in fn
        years_in_fn = sum([1 for year in years if str(year) in fn]) # from 'years' how many are in fn
        if years_in_fn == len(years) and nyears >= len(years):
          fn_places = fn
          if nyears > len(years):
            print(f"WARNING: You will load {allyears} data for {years}")
          break

    if fn_places is None:
       raise Exception("Survey file for year not found.")

  return fn_places

################################################################
# Functions: Files and paths 
################################################################

def get_prefix_surveys(df=None, root=None, years=None):
  
  if df is None and root is None and years is None:
    raise Exception("You must pass either df or root and years")

  if df is None and root is not None and years is not None:
    fn = get_places_file(root, years)
    df = load_csv(fn)

  if df is None:
    raise Exception("Something went wrong df is None and it should not be.")
  
  # @ TODO: only use new (dsource,year)
  if 'dsource' in df.columns:
    columns = ['dsource','year'] # new
  elif 'survey' in df.columns:
      columns = ['survey','year']
      print(f'[WARNING] This should not happen (root:{root}, years:{years}, survey)')
  else:
      columns = ['SURVEY','DHSYEAR'] 
      print(f'[WARNING] This should not happen (root:{root}, years:{years}, SURVEY)')

  prefix = "_".join(["{}{}".format(group[0],group[1]) for group, r in df.groupby(columns)])
  return prefix


def getfn(path, ext=True):
    fn = ntpath.basename(path)
    return fn if ext else os.path.splitext(fn)[0]

def get_random_fn(path, ext='.png', n=1):
    fns = np.random.choice([fn for fn in os.listdir(path) if fn.endswith(ext)],n)
    fns = [os.path.join(path,fn) for fn in fns]
    return fns if n>1 else fns[0]

def get_files(path, endswith=None, abspath=True):
    files = [os.path.join(path,fn) if not abspath else os.path.abspath(os.path.join(path,fn)) for fn in os.listdir(path) if endswith is None or fn.endswith(endswith)]
    return files

def validate_path(path):
  try:
    if not os.path.exists(path):
      os.makedirs(path)
  except FileExistsError:
    pass
  except Exception as ex:
    print(f"[ERROR] validate_path | ios.py | {ex} | {path}")
    
def copy(sourcefile, targetpath):
    try:
        targetfile = os.path.join(targetpath, os.path.basename(sourcefile))
        copyfile(sourcefile, targetfile)
        return targetfile
    except Exception as ex:
        print(ex)
    return None

def exists(fn):
    return os.path.exists(fn)

def create_path(path):
    os.makedirs(path, exist_ok=True)

################################################################
# Functions: Content
################################################################

def showtxt(fn):
    if os.path.exists(fn):
        with open(fn, 'r') as f:
            print("".join(f.readlines()))
    else:
        print("{} does not exist.".format(fn))

def read_txt_to_list(fn, verbose=True):
    content = []
    try:
        with open(fn,'r') as f:
            content = [line.strip() for line in f.readlines()]
    except Exception as ex:
        if verbose:
            print(ex)
        content = None

    return content

def write_list_to_txt(content, fn, verbose=True):
    try:
        with open(fn,'w') as f:
          f.write("\n".join([str(i) for i in content]))
        if verbose:
          print("{} saved!".format(fn))
    except Exception as ex:
        if verbose:
            print(ex)
        
def read_list_of_json(fn, verbose):
  content = []
  try:
      with open(fn,'r') as f:
          content = [json.loads(line.strip()) for line in f.readlines()]
  except Exception as ex:
      if verbose:
          print(ex)
      content = None

  return content
  
################################################################
# Functions: Read & Write
################################################################

def load_pyobj(fn, verbose=True):
  obj = None
  try:
    obj = joblib.load(open(fn, "rb"))
    if verbose:
      print("{} loaded!".format(fn))
  except Exception as ex:
    if verbose:
      print(ex)
  return obj
  
  return 

def save_pyobj(obj, fn, verbose=True):
  try:
    joblib.dump(obj, open(fn, "wb"))
    if verbose:
      print("{} saved!".format(fn))
  except Exception as ex:
    print(ex) 

def save_csv(df, fn, index=True, verbose=True):
    try:
        df.to_csv(fn, index=index)
        if verbose:
          print("{} saved!".format(fn))
    except Exception as ex:
        print(ex)

def load_csv(fn, index_col=0, verbose=False):
    df = None
    try:
        df = pd.read_csv(fn, index_col=index_col, low_memory=False)
        if verbose:
            print("{} loaded!".format(fn))
    except Exception as ex:
        if verbose:
            print(ex)
    return df

def save_json(obj, fn, mode='w', verbose=True):
  try:
    with open(fn,mode) as f:
      json.dump(obj, f)
      if mode=='a':
        f.write('\n')
    if verbose:
      print("{} saved!".format(fn))
  except Exception as ex:
      print(ex)

def load_json(fn, verbose=False):
    try:
        with open(fn, 'r') as f:
            return json.load(f)
    except Exception as ex:
        if verbose:
            print(ex)
        pass
    return None

def write_txt(content, fn):
    try:
        with open(fn, 'w') as f:
            f.write(content)
    except Exception as ex:
        print(ex)

def read_txt(fn):
  try:
    with open(fn,'r') as f:
      content = f.read()
  except Exception as ex:
    print(ex)
    content = None
    
  return content

def save_compressed_array(arr, fn):
  try:
    savez_compressed(fn, arr)
  except Exception as ex:
    print(ex)
  return

def load_array(fn):
  try:
    arr = np.load(fn)
  except Exception as ex:
    print(ex)
    arr = None
  return arr

def write_pickle(data, fn):
    try:
        pickle.dump(data, open(fn, 'wb'))
        print("{} saved!".format(fn))
    except Exception as ex:
        print(ex)

def read_pickle(fn):
    try:
        with open(fn,'rb') as f:
            data = pickle.load(f)
        print("{} loaded!".format(fn))
        return data
    except Exception as ex:
        print(ex)
        return None

def save_h5(data, fn):
  try:
    hf = h5py.File(fn, 'w')
    hf.create_dataset('data', data=data, compression="gzip", compression_opts=9)
    hf.close()
  except Exception as ex:
    print(ex)

def read_h5(fn):
  try:
    hf = h5py.File(fn, 'r')
    for key in hf.keys():
      print(key) #Names of the groups in HDF5 file.
    data = hf.get('data')
    print('data:',data)
    return data
  except Exception as ex:
    print(ex)
