### How to read a questionary: 
### https://dhsprogram.com/data/Guide-to-DHS-Statistics/Organization_of_DHS_Data.htm#Recode_Files
### https://dhsprogram.com/data/Guide-to-DHS-Statistics/index.htm#t=Household_Characteristics.htm
### https://dhsprogram.com/data/Guide-to-DHS-Statistics/index.htm#t=Household_Possessions.htm
### https://dhsprogram.com/data/Guide-to-DHS-Statistics/index.htm#t=Analyzing_DHS_Data.htm%23Householdsbc-2&rhtocid=_4_4_1
### https://www.dhsprogram.com/data/Using-DataSets-for-Analysis.cfm#CP_JUMP_14042
### https://www.dhsprogram.com/pubs/pdf/DHSG4/Recode7_DHS_10Sep2018_DHSG4.pdf
### https://dhsprogram.com/pubs/pdf/FR333/FR333.pdf (2016)
### https://dhsprogram.com/pubs/pdf/MIS34/MIS34.pdf (2018)
### https://dhsprogram.com/data/Guide-to-DHS-Statistics/index.htm#t=Wealth_Quintiles.htm
### IWI: https://link.springer.com/article/10.1007/s11205-014-0683-x#Tab1

### hhid: household id
### hv000: country code (remove numbers)
### hv001: cluster number (matches DHSCLUST in dbf file)
### hv002: household number (in cluster)
### hv024: region of residence
### hv025: urban (1) rural (2) refugee (3)

### hv208: has TV
### hv209: has fridge
### hv221: has telephone (land-line)
### hv212: has a car/truck
### hv210: has a bike
### hv206: has electricity

### hv201: source of drinking water (1,2,3: l,m,h)
### hv213: floor material
### hv205: type of toilet facility

### Water supply (hv201)
### Toilet facility (hv205)
### Floor quality (hv213)
### hv216: number of sleeping rooms (1,2,3: 0-1, 2, 3>)

### cheap utensils: hv208, hv209, hv221, hv210, hv212
### expensive utensils: hv212, hv243e, hv211, hv243d, (hv243a mobile phone ?)

### hv270: Wealth index combined (1 poorest, 2, 3, 4, 5 richest)
### hv271: Wealth index factor score combined (Gini coefficient of wealth index: hv271 – see Calculation)
### hv005: Household sample weight

### COMPUTE WGT = V005/1000000.
### WEIGHT by WGT

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
from fiona.io import ZipMemoryFile

from census.wealth import WI

##############################################################################
# CONSTANTS
##############################################################################

from utils.constants import WATER
from utils.constants import TOILET
from utils.constants import FLOOR
from utils.constants import BETAS
from utils.constants import BETA_CHEAP_UTENSILS
from utils.constants import BETA_EXPENSIVE_UTENSILS
from utils.constants import CONSTANT

from utils.constants import COLS_SURVEY
from utils.constants import COLS_CLUSTER

##############################################################################
# CLASS
##############################################################################

class DHSMIS(WI):
  
  def __init__(self, root, country_code, years, indicator='iwi', **kwargs):
    super().__init__(root, country_code, years, indicator, **kwargs)

  ###############################################
  # Main inheret methods
  ###############################################

  def load_data(self):
    super().load_data()
    self._load_surveys()

  def commpute_indicators(self, njobs=1):
    self._compute_IWI(njobs)

  def rename_columns(self):
    self.df_cluster.rename(columns={'DHSYEAR':self.col_year, 
                                    'DHSCLUST':self.col_cluster, 
                                    'URBAN_RURA':self.col_rural, 
                                    'LATNUM':self.col_lat, 
                                    'LONGNUM':self.col_lon, 
                                    'SURVEY':self.col_source,
                                    'mean_iwi':f"mean_{self.col_indicator}",
                                    'std_iwi':f"std_{self.col_indicator}",
                                    'DHSCC':self.col_ccode,
                                    'counts':self.col_counts}, inplace=True)

    self.df_survey.rename(columns={'iwi':self.col_indicator,
                                   'year':self.col_year, 
                                   'survey':self.col_source,
                                   'country':self.col_ccode}, inplace=True)
  
  def set_categories(self):
    bins = pd.IntervalIndex.from_tuples([(0, 25), (25, 50), (50, 75), (75,101)], closed='left')
    super().set_categories(bins)
    
  ###############################################
  # other methods
  ###############################################

  def _load_surveys(self):
    self._load_households()
    self._load_clusters()
    
  def _load_households(self):

    ### 1. reading stata file
    df_survey = None
    for year in tqdm(self.years):
      for fn in glob.glob(os.path.join(self.root,f"survey/*/{year}/{self.country_code}HR*")):
          
        if 'LSMS' in fn:
          continue

        survey_name = fn.split("/survey/")[-1].split('/')[0]
        
        if fn.endswith(".zip"):
          zf = zipfile.ZipFile(fn)
          fn = [f for f in zf.namelist() if f.endswith('.DTA')][0]
          fn = zf.open(fn) # not a filename
        else:
          fn = glob.glob(os.path.join(self.root,f"survey/*/{year}/{self.country_code}HR*/{self.country_code}HR*.DTA"))[0]

        tmp = pd.read_stata(fn, convert_categoricals=False, convert_dates=False, order_categoricals=False)
        tmp.loc[:,'year'] = year
        tmp.loc[:,'survey'] = survey_name
        tmp.loc[:,'country'] = self.country_code
        
        # some
        for c in COLS_SURVEY:
          if c not in tmp.columns:
            tmp.loc[:,c] = 0
        tmp = tmp.loc[:,COLS_SURVEY]

        df_survey = tmp.copy() if df_survey is None else df_survey.append(tmp, ignore_index=True)

    ### 2. drop nan
    self.df_survey = df_survey.dropna(subset=COLS_SURVEY)
    del(df_survey)
    gc.collect()

    ### 3. remove refugee data
    self.df_survey = self.df_survey.query("hv025 in [1,2]").copy() # 3 is for refugee

    ### 4. cast types
    bool_cols = ['hv208','hv209','hv221','hv243a','hv212','hv210','hv206','hv243e','hv211','hv243d']
    self.df_survey.loc[:,bool_cols] = self.df_survey.loc[:,bool_cols].astype(np.bool)
    
    int_cols = ['year','hv001','hv002','hv024','hv025','hv201','hv213','hv205','hv216']
    self.df_survey.loc[:,int_cols] = self.df_survey.loc[:,int_cols].astype(np.int16)
    
    float_cols = ['hv270','hv271','hv005']
    self.df_survey.loc[:,float_cols] = self.df_survey.loc[:,float_cols].astype(np.float)
    

  def _load_clusters(self):
    print("loading cluster")
    
    ### 1. reading shape files
    self.df_cluster = gpd.GeoDataFrame()
    for year in tqdm(self.years):
      for fn in glob.glob(os.path.join(self.root,f"survey/*/{year}/{self.country_code}GE*")):
        if fn.endswith(".zip"):
          pass
        else:
          fn = glob.glob(os.path.join(self.root,f"survey/*/{year}/{self.country_code}GE*/{self.country_code}GE*.shp"))[0]
        
        tmp = gpd.read_file(fn)
        tmp.loc[:,'SURVEY'] = fn.split("/survey/")[-1].split('/')[0]
        self.df_cluster = self.df_cluster.append(tmp, ignore_index=True)
    
    ### 2. validate
    self.df_cluster.loc[:,'DHSYEAR'] = self.df_cluster.DHSYEAR.astype(np.int16)
    self.df_cluster.loc[:,'DHSCLUST'] = self.df_cluster.DHSCLUST.astype(np.int16)
    self.df_cluster.loc[:,'URBAN_RURA'] = self.df_cluster.URBAN_RURA.apply(lambda c: 0 if c=='U' else 1 if c=='R' else -1).astype(np.int16) 
    # -1 never happens (3 is refugee but is filter out in survey)
    self.df_cluster = self.df_cluster.merge(self.df_survey[['year','survey','hv001']].drop_duplicates(), 
                                            left_on=['DHSYEAR','SURVEY','DHSCLUST'], right_on=['year','survey','hv001'])
    self.df_cluster.drop(columns=['year','survey','hv001'], inplace=True)
    

    ### 3 Remove invalid clusters (lon and lat = 0)
    self.df_cluster = self.df_cluster.query("LATNUM!=0 and LONGNUM!=0").copy()
    DHSCLUSTS = self.df_cluster.DHSCLUST.values
    self.df_survey = self.df_survey.query("hv001 in @DHSCLUSTS")
    self.county_code = self.df_cluster.DHSCC.unique()[0]
    
    

  def _iwi_household(self, obj):
    id,row = obj
    
    # water (hv201)
    w1 = WATER[self.country_code][row.year]['low'] #low   1
    w2 = WATER[self.country_code][row.year]['mid'] #mid   2
    w3 = WATER[self.country_code][row.year]['high'] #high 3

    # toilet (hv205)
    t1 = TOILET[self.country_code][row.year]['low'] #low
    t2 = TOILET[self.country_code][row.year]['mid'] #mid
    t3 = TOILET[self.country_code][row.year]['high'] #high

    # floor (hv213)
    f1 = FLOOR[self.country_code][row.year]['low'] #low
    f2 = FLOOR[self.country_code][row.year]['mid'] #mid
    f3 = FLOOR[self.country_code][row.year]['high'] #high
    
    iwi = 0

    ### normal variables
    for k,beta in BETAS.items():
        x = row.loc[k]
        original = x

        if x in [96,99]: # 9, 99?
            x = 1
            b = 0
            
        elif k == 'hv201':   # water
            x = 1 if x in w1 else 2 if x in w2 else 3 if x in w3 else 0
            if x==0:
                print(id, k, row.year, original, 'zero')
            b = beta[x] if x>0 else 0
            x = 1
            
        elif k == 'hv205': # toilet
            x = 1 if x in t1 else 2 if x in t2 else 3 if x in t3 else 0
            if x==0:
                print(id, k, row.year, original, 'zero')
            b = beta[x] if x>0 else 0
            x = 1
            
        elif k == 'hv213': # floor
            x = 1 if x in f1 else 2 if x in f2 else 3 if x in f3 else 0
            if x==0:
                print(id, k, row.year, original, 'zero')
            b = beta[x] if x>0 else 0
            x = 1
            
        elif k == 'hv216': # sleeping rooms
            x = 1 if x in [0,1] else 2 if x==2 else 3 if x>=3 else 0
            if x==0:
                print(id, k, row.year, original, 'zero')
            b = beta[x] if x>0 else 0
            x = 1
            
        else:
            x = int(x)
            b = beta
            
        if x==9:
            print(id, k, row.year, original, 'nine')

        iwi += (b * x)
    
    ### special variables (expensive utensils)
    has_expensive = False
    for k in ['hv212', 'hv243e', 'hv211', 'hv243d', 'hv243a']:
        x = row.loc[k]==1
        has_expensive = has_expensive or x
    iwi += BETA_EXPENSIVE_UTENSILS * int(has_expensive)
    
        
    ### special variables (cheap utensils)
    has_cheap = has_expensive
    for k in ['hv208', 'hv209', 'hv221', 'hv210', 'hv212']:
        x = row.loc[k]==1
        has_cheap = has_cheap or x
    has_cheap = has_cheap or row.loc['hv205'] in t3 # toilet
    has_cheap = has_cheap or row.loc['hv213'] in f3 # floor
    iwi += (BETA_CHEAP_UTENSILS * int(has_cheap))
    
    return iwi + CONSTANT

  def _compute_IWI(self, njobs=-1):
    ### The International Wealth Index (IWI)
    ### Jeroen Smits and Roel Steendijk
    results = pqdm(self.df_survey.iterrows(), self._iwi_household, n_jobs=njobs)
    self.df_survey.loc[:,'iwi'] = results
    self.df_survey.iwi = self.df_survey.iwi.astype(np.float32) #.round(2) 

    self.df_cluster.drop(columns=['mean_iwi','std_iwi'], inplace=True, errors='ignore')
    tmp = self.df_survey.groupby(['year','survey','hv001']).iwi.agg(['mean','std','size']).reset_index()
    self.df_cluster = self.df_cluster.merge(tmp, left_on=['DHSYEAR','SURVEY','DHSCLUST'], right_on=['year','survey','hv001']).drop(columns=['year','survey','hv001'])
    self.df_cluster.rename(columns={'mean':'mean_iwi', 'std':'std_iwi', 'size':'counts'}, inplace=True)
