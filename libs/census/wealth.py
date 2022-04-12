
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
    subset = [self.col_ccode, self.col_source, self.col_year, self.col_cluster, self.col_lon, self.col_lat, self.col_rural, self.col_counts, 
              f"mean_{self.col_indicator}", f"std_{self.col_indicator}", f"mean_{self.col_indicator}_bin", f"mean_{self.col_indicator}_cat", f"mean_{self.col_indicator}_cat_id"]
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
    
########################### OLD CODE ########################

### @deprecated
  # def assign_quantile_based_SES_to_clusters(self, labels):
  #   column = 'mean_iwi'
  #   # SES based on quantiles (% of observations in bins)
  #   self.df_cluster.loc[:,'SESq'] = pd.qcut(self.df_cluster[column], q=len(labels), labels=labels, precision=0)
  #   # SES based on iwi ranges
  #   #bins = pd.IntervalIndex.from_tuples([(0, 25), (25, 50), (50, 75), (75,101)], closed='left')
  #   #self.df_cluster.loc[:,'SES-bin'] = pd.cut(self.df_cluster[column], bins, include_lowest=True, precision=0, right=False)
  #   #self.df_cluster.loc[:,'SES'] = self.df_cluster.loc[:,'SES-bin'].apply(lambda ses: labels[ses.left//25])
  #   self.df_cluster.loc[:,'SES'] = pd.cut(self.df_cluster[column], bins=len(labels), labels=labels, include_lowest=True, precision=0, right=False)
  #   bins = pd.cut(self.df_cluster[column], bins=len(labels), include_lowest=True, precision=0, right=False)
  #   bins = bins.values.categories
  #   self.bins = {'left':'closed', 'right':'open', 'bins':{i:[b.left, b.right] for i,b in enumerate(bins)}}



  # def load_surveys(self, fn_data, validate=True):
  #   self.load_surveys_from_stata(fn_data, validate)
  #   self.load_surveys_from_geo(fn_data, validate)
    
  # def load_surveys_from_stata(self, fn_data, validate=True):

  #   ### 1. reading stata file
  #   self.df_survey = pd.DataFrame(columns=COLS_SURVEY, index=[])
  #   for year,obj in tqdm(fn_data.items(), total=len(fn_data.keys())):
  #     tmp = pd.read_stata(obj['survey'], convert_categoricals=False)
  #     tmp.loc[:,'year'] = year
  #     self.df_survey = self.df_survey.append(tmp.loc[:,COLS_SURVEY], ignore_index=True)

  #   ### 2. remove refugee data
  #   if validate:
  #     self.df_survey = self.df_survey.query("hv025 in [1,2]").copy() # 3 is for refugee

  #   ### 3. cast types
  #   if validate:
  #     bool_cols = ['hv208','hv209','hv221','hv243a','hv212','hv210','hv206','hv243e','hv211','hv243d','hv243a']
  #     self.df_survey.loc[:,bool_cols] = self.df_survey.loc[:,bool_cols].astype(np.bool)
      
  #     int_cols = ['hv001','hv002','hv024','hv025','hv201','hv213','hv205','hv216']
  #     self.df_survey.loc[:,int_cols] = self.df_survey.loc[:,int_cols].astype(np.int16)
      
  #     float_cols = ['hv270','hv271','hv005']
  #     self.df_survey.loc[:,float_cols] = self.df_survey.loc[:,float_cols].astype(np.float)


  # def load_surveys_from_geo(self, fn_data, validate=True):
  #   df = pd.DataFrame(columns=COLS_CLUSTER, index=[])

  #   for year,obj in tqdm(fn_data.items(), total=len(fn_data.keys())):
      
  #     geo = GeoViz(obj['shp'], obj['dbf'])
  #     for r in geo.sf.records():
  #       r = r.as_dict()
  #       tmp = pd.DataFrame(r,index=[0])
  #       df = df.append(tmp.loc[:,COLS_CLUSTER], ignore_index=True)

  #   if validate:
  #     self.df_cluster = df.copy() # query("DHSCC == @CC").copy() # check whether this is necesary
  #     self.df_cluster.loc[:,'DHSYEAR'] = self.df_cluster.DHSYEAR.astype(np.int16)
  #     self.df_cluster.loc[:,'DHSCLUST'] = self.df_cluster.DHSCLUST.astype(np.int16)
  #     self.df_cluster.loc[:,'URBAN_RURA'] = self.df_cluster.URBAN_RURA.apply(lambda c: 1 if c=='U' else 2 if c=='R' else 0).astype(np.int16)
  #     self.df_cluster = self.df_cluster.merge(self.df_survey[['year','hv001']].drop_duplicates(), left_on=['DHSYEAR','DHSCLUST'], right_on=['year','hv001'])
  #     self.df_cluster.drop(columns=['year','hv001'], inplace=True)
  #     ### Remove invalid clusters (lon and lat = 0)
  #     self.df_cluster = self.df_cluster.query("LATNUM!=0 and LONGNUM!=0").copy()
  #     DHSCLUSTS = self.df_cluster.DHSCLUST.values
  #     self.df_survey = self.df_survey.query("hv001 in @DHSCLUSTS")



  # def compute_IWI(self, update_cluster=False):
  #   ### The International Wealth Index (IWI)
  #   ### Jeroen Smits and Roel Steendijk

  #   self.df_survey.loc[:,'iwi'] = 0

  #   # water
  #   w1 = [32, 42, 43, 51, 81, 96]
  #   w2 = [31, 61, 62, 63, 14, 21, 41, 71,92]
  #   w3 = [71, 72, 11, 12, 13, 91]

  #   # toilet
  #   t1 = [14,15,23,42,43,31,96] 
  #   t2 = [21,22,41] 
  #   t3 = [11,12,13]

  #   # floor
  #   f1 = [11,12,96] 
  #   f2 = [34,32,21,22] 
  #   f3 = [31,33,35,36,37]

  #   for id,row in tqdm(self.df_survey.iterrows(), total=self.df_survey.shape[0]):
  #     iwi = 0

  #     ### normal variables
  #     for k,b in BETAS.items():
  #       x = row.loc[k]

  #       if k == 'hv201':   # water
  #         x = 1 if x in w1 else 2 if x in w2 else 3 if x in w3 else 0
  #         b = b[x] if x>0 else 0
  #         x = 1
  #       elif k == 'hv205': # toilet
  #         x = 1 if x in t1 else 2 if x in t2 else 3 if x in t3 else 0
  #         b = b[x] if x>0 else 0
  #         x = 1
  #       elif k == 'hv213': # floor
  #         x = 1 if x in f1 else 2 if x in f2 else 3 if x in f3 else 0
  #         b = b[x] if x>0 else 0
  #         x = 1
  #       elif k == 'hv216': # sleeping rooms
  #         x = 1 if x in [0,1] else 2 if x==2 else 3 if x>=3 else 0
  #         b = b[x] if x>0 else 0
  #         x = 1
  #       elif x in [9,99]:
  #         x = 0
  #       else:
  #         x = int(x)

  #       iwi += (b * x)
      
  #     ### special variables (expensive utensils)
  #     has_expensive = False
  #     for k in ['hv212', 'hv243e', 'hv211', 'hv243d', 'hv243a']:
  #       x = row.loc[k]==1
  #       has_expensive = has_expensive or x
  #     iwi += BETA_EXPENSIVE_UTENSILS * int(has_expensive)
      
  #     ### special variables (cheap utensils)
  #     has_cheap = has_expensive
  #     for k in ['hv208', 'hv209', 'hv221', 'hv210', 'hv212']:
  #       x = row.loc[k]==1
  #       has_cheap = has_cheap or x
  #     has_cheap = has_cheap or row.loc['hv205'] in t3 # toilet
  #     has_cheap = has_cheap or row.loc['hv213'] in f3 # floor
  #     iwi += (BETA_CHEAP_UTENSILS * int(has_cheap))

  #     self.df_survey.loc[id,'iwi'] = iwi

  #   # WORLDWIDE
  #   self.df_survey.loc[:,'iwi'] += CONSTANT
  #   self.df_survey.loc[:,'iwi'] = self.df_survey.loc[:,'iwi'].round(2)

  #   # PER COUNTRY
  #   #self.df_survey.loc[:,'iwi'] += abs(self.df_survey.iwi.min())
  #   #self.df_survey.loc[:,'iwi'] /= self.df_survey.iwi.max()
  #   #self.df_survey.loc[:,'iwi'] *= 100
  #   #self.df_survey.loc[:,'iwi'] = self.df_survey.loc[:,'iwi'].round(1)

  #   if update_cluster:
  #     self.df_cluster.drop(columns=['mean_iwi'], inplace=True, errors='ignore')
  #     tmp = self.df_survey.groupby(['year','hv001']).iwi.mean().reset_index()
  #     self.df_cluster = self.df_cluster.merge(tmp, left_on=['DHSYEAR','DHSCLUST'], right_on=['year','hv001']).drop(columns=['year','hv001'])
  #     self.df_cluster.rename(columns={'iwi':'mean_iwi'}, inplace=True)

