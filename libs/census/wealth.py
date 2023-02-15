
##############################################################################
# DEPENDENCIES
##############################################################################

import gc
import os
import glob
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm 
from pqdm.threads import pqdm 

from utils import validations

##############################################################################
# CONSTANTS
##############################################################################

from utils.constants import SES_LABELS
from utils.constants import GTID
from utils.constants import LON
from utils.constants import LAT
from utils.constants import RURAL
from utils.constants import YEAR
from utils.constants import CLUSTER
from utils.constants import WEALTH

##############################################################################
# CLASS
##############################################################################

class WI(object):
  
  def __init__(self, root, country_code, years, indicator=None, **kwargs):
    self.indicator = indicator # real indicator name from source
    self.df_survey = None
    self.df_cluster = None
    self.root = root
    self.country_code = country_code
    self.years = validations.validate_years(years)
    self.kwargs = kwargs
    # unified column names
    self.col_ccode = 'ccode' # country code
    self.col_source = 'dsource' # data source
    self.col_year = YEAR
    self.col_id = GTID
    self.col_lon = LON
    self.col_lat = LAT
    self.col_cluster = CLUSTER # id per year
    self.col_rural = RURAL
    self.col_indicator = WEALTH
    self.col_counts = 'counts' # number of households or data points involved in the aggregation

  ###############################################
  # Load data
  ###############################################

  def load_data(self):
    pass

  def commpute_indicators(self, njobs=1):
    pass 
  
  def set_categories(self, bins):
    ### Here add 4-bin categories
    self.df_cluster = WI.add_ses_categories(self.df_cluster, f'mean_{self.col_indicator}', bins)

  def rename_columns(self):
    pass

  def clean_columns(self):
    subset = [self.col_ccode, self.col_source, self.col_year, self.col_cluster, self.col_lon, self.col_lat, 
              self.col_rural, self.col_counts, f"mean_{self.col_indicator}", f"std_{self.col_indicator}", 
              f"mean_{self.col_indicator}_bin", f"mean_{self.col_indicator}_cat", f"mean_{self.col_indicator}_cat_id"]
    self.df_cluster.loc[:,self.col_id] = self.df_cluster.apply(lambda row: f"{row[self.col_ccode]}{row[self.col_year]:4d}{row[self.col_cluster]:010}", axis=1)
    self.df_cluster = self.df_cluster[[self.col_id]+subset]
    
    subset = [self.col_ccode, self.col_source, self.col_year, self.col_indicator]
    subset.extend([c for c in self.df_survey.columns if c not in subset])
    self.df_survey = self.df_survey[subset].copy()
    
  ###############################################
  # Static methods
  ###############################################

  @staticmethod
  def add_ses_categories(df, col_wi, bins=None, minv=None, maxv=None):
    labels = SES_LABELS
    n = len(labels)
    
    if type(bins)==int:
      if minv is None or maxv is None:
        raise Exception("[ERROR] wealth.py | add_ses_categories | minv and maxv must be passed when bins is int.")
      steps = int(np.ceil(maxv / n))
      bins = pd.IntervalIndex.from_tuples([(b,b+steps+(1 if b==maxv-steps else 0)) for b in np.arange(minv,maxv,steps)], closed='left')
    
    if len(labels) != len(bins) or len(labels) != bins.shape[0]:
      raise Error("[ERROR] wealth.py | add_ses_categories | bins are not 4")
    
    print(f"[INFO] wealth.py | add_ses_categories | bins:{bins}, labels:{labels}")
    map = {b:l for b,l in zip(*[[v.left for v in bins],labels])}
    df.loc[:,f'{col_wi}_bin'] = pd.cut(df[col_wi], bins=bins, precision=0, retbins=False)
    df.loc[:,f'{col_wi}_cat'] = df.loc[:,f'{col_wi}_bin'].apply(lambda v: map[v.left])
    df.loc[:,f'{col_wi}_cat_id'] = df.loc[:,f'{col_wi}_cat'].apply(lambda v: labels.index(v))
    return df

  @staticmethod
  def discretize_in_n_bins(arr, n=10, minv=0, maxv=100):
    labels = np.arange(n)
    steps = int(np.ceil(maxv / n))
    bins = pd.IntervalIndex.from_tuples([(b,b+steps+(1 if b==maxv-steps else 0)) for b in np.arange(minv,maxv,steps)], closed='left')
    #map = {b:l for b,l in zip(*[[v.left for v in bins],labels])}
    #codes = [map[i.left] for i in tmp]
    tmp = pd.cut(arr, bins=bins, precision=0, retbins=False, ordered=True, labels=labels) #.codes
    return tmp
    