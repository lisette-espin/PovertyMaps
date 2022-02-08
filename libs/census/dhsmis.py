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

##############################################################################
# CONSTANTS
# Betas: https://link.springer.com/article/10.1007/s11205-014-0683-x/tables/1
# https://globaldatalab.org/iwi/using/
##############################################################################

### Water supply (hv201)
### - high quality is bottled water or water piped into dwelling or premises;
### - middle quality is public tap, protected well, tanker truck, etc.;
### - low quality is unprotected well, spring, surface water, etc.

WATER = {'UG':{2006:{'high':[11,12,71],'mid':[13,31,33,34,41,61],'low':[20,21,22,23,30,32,35,36,40,42,43,44,45,46,51,62,91]},
               2009:{'high':[10,11,12,71],'mid':[13,31,33,34,41,61],'low':[20,21,22,23,30,32,35,40,42,43,44,45,46,51,62]},
               2011:{'high':[10,11,12,71],'mid':[13,31,33,34,41,61,71,72],'low':[20,21,22,23,30,32,35,36,40,42,43,44,45,46,51,62]},
               2014:{'high':[11,12,91],'mid':[13,31,41,61,63,71],'low':[21,22,32,42,43,44,51,62,81]},
               2016:{'high':[11,12,13,91],'mid':[14,31,41,61,63,72,92],'low':[21,32,42,43,51,71,81]},
               2018:{'high':[11,12,13,91],'mid':[14,31,41,61,63,72,92],'low':[21,32,42,43,51,71,81]}},
         'SL':{2008:{'high':[10,11,12,71], 'mid':[13,31,41,61], 'low':[20,21,30,32,40,42,43,51,62]},
               2013:{'high':[10,11,12,91,71], 'mid':[13,31,41,61,92], 'low':[20,21,30,32,40,42,43,51,62]},
               2016:{'high':[10,11,12,13,71], 'mid':[14,31,41,61,72], 'low':[20,21,30,32,40,42,43,51,62]},
               2019:{'high':[10,11,12,13,71], 'mid':[14,31,41,61,72], 'low':[20,21,30,32,40,42,43,51,62,81]}}}

### Toilet facility (hv205)
### - high quality is any kind of private flush toilet; 
### - middle quality is public toilet, improved pit latrine, etc.;
### - low quality is traditional pit latrine, hanging toilet, or no toilet facility.

TOILET = {'UG':{2006:{'high':[10,11],'mid':[21,23],'low':[20,22,24,25,30,31,41,42,43]},
               2009:{'high':[10,11],'mid':[21,23],'low':[20,22,24,25,30,31,41,42,43]},
               2011:{'high':[1,10,11],'mid':[2,4,21,23],'low':[3,5,6,7,8,9,20,22,24,25,30,31,41,43,44]},
               2014:{'high':[11,12,13,14,15],'mid':[21,22],'low':[23,24,25,31,41,42,43,51,61]},
               2016:{'high':[11,12,13,14,15],'mid':[21,22],'low':[23,31,41,42,43,51,61]},
               2018:{'high':[11,12,13,14,15],'mid':[21,22],'low':[23,31,41,42,43,51,61]}},
         'SL':{2008:{'high':[10,11,12,13,14,15], 'mid':[21,22], 'low':[20,23,30,31,41,42,43,71]},
               2013:{'high':[10,11,12,13,14,15], 'mid':[21,22], 'low':[20,23,30,31,41,42,43]},
               2016:{'high':[10,11,12,13,14,15], 'mid':[21,22], 'low':[20,23,30,31,41,42,43]},
               2019:{'high':[10,11,12,13,14,15], 'mid':[21,22], 'low':[20,23,30,31,41,42,43]}}}

### Floor quality (hv213)
### - high quality is finished floor with parquet, carpet, tiles, ceramic etc.; 
### - middle quality is cement, concrete, raw wood, etc. 
### - low quality is none, earth, dung etc., 
FLOOR = {'UG':{2006:{'high':[30,31,33],'mid':[34,35,36],'low':[10,11,12,20,]},
               2009:{'high':[30,31,33],'mid':[34,35,36],'low':[10,11,12,20]},
               2011:{'high':[30,31,33],'mid':[34,35,36],'low':[10,11,12,20]},
               2014:{'high':[30,31,32],'mid':[21,22,33,34,35],'low':[10,11,12,20]},
               2016:{'high':[30,31,33,35],'mid':[21,22,32,34,36,37],'low':[10,11,12,20,]},
               2018:{'high':[30,31,33,35],'mid':[21,22,32,34,36,37],'low':[10,11,12,20]}},
         'SL':{2008:{'high':[30,31,32,33,35], 'mid':[13,21,22,34], 'low':[10,11,12,20]},                          
               2013:{'high':[30,31,32,33,35], 'mid':[21,22,34], 'low':[10,11,12,20]},
               2016:{'high':[31,32,33,35], 'mid':[21,22,34], 'low':[11,12]},
               2019:{'high':[30,31,32,33,35], 'mid':[21,22,34], 'low':[10,11,12,20]}}}

COLS_SURVEY = ['year','survey','hhid','hv001','hv002','hv024','hv025','hv270','hv271','hv005','hv243e','hv211','hv243d','hv243a']
BETAS = {'hv208':8.612657,  # tv
         'hv209':8.429076,  # fridge
         'hv221':7.127699,  # telephone
         'hv212':4.651382,  # car
         'hv210':1.846860,  # bike
         'hv206':8.056664,  # electricity 
         'hv201':{1:-6.306477,2:-2.302023,3:7.952443},  # water
         'hv213':{1:-7.558471,2:1.227531, 3:6.107428},  # floor 
         'hv205':{1:-7.439841,2:-1.090393,3:8.140637},  # toilet
         'hv216':{1:-3.699681,2:0.384050, 3:3.445009}}  # sleeping rooms
BETA_CHEAP_UTENSILS = 4.118394     # cheap utensils
BETA_EXPENSIVE_UTENSILS = 6.507283 # expensive utensils 

CONSTANT = 25.004470
COLS_SURVEY.extend(BETAS.keys())
COLS_CLUSTER = ['DHSCC','DHSYEAR','DHSCLUST','URBAN_RURA','LATNUM','LONGNUM','SOURCE','ALT_GPS','ALT_DEM','DATUM']

##############################################################################
# CLASS
##############################################################################

class DHSMIS(object):
  
  def __init__(self, root, code):
    self.df_survey = None
    self.df_cluster = None
    self.bins = None
    self.root = root
    self.country_code = code

  ###############################################
  # Load data from STATA
  ###############################################

  def load_surveys(self, years):
    self.load_households(years)
    self.load_clusters(years)

  def load_households(self, years):
    years = years.split(',') if type(years)==str else years
    years = [int(y) for y in years]

    ### 1. reading stata file
    df_survey = None
    for year in tqdm(years):
      for fn in glob.glob(os.path.join(self.root,f"survey/*/{year}/{self.country_code}HR*")):
          
        if 'LSMS' in fn:
          continue

        if fn.endswith(".zip"):
          zf = zipfile.ZipFile(fn)
          fn = [f for f in zf.namelist() if f.endswith('.DTA')][0]
          fn = zf.open(fn) # not a filename
        else:
          fn = glob.glob(os.path.join(self.root,f"survey/*/{year}/{self.country_code}HR*/{self.country_code}HR*.DTA"))[0]

        tmp = pd.read_stata(fn, convert_categoricals=False, convert_dates=False, order_categoricals=False)
        tmp.loc[:,'year'] = year
        tmp.loc[:,'survey'] = fn.split("/survey/")[-1].split('/')[0]
      
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
    bool_cols = ['hv208','hv209','hv221','hv243a','hv212','hv210','hv206','hv243e','hv211','hv243d','hv243a']
    self.df_survey.loc[:,bool_cols] = self.df_survey.loc[:,bool_cols].astype(np.bool)
    
    int_cols = ['year','hv001','hv002','hv024','hv025','hv201','hv213','hv205','hv216']
    self.df_survey.loc[:,int_cols] = self.df_survey.loc[:,int_cols].astype(np.int16)
    
    float_cols = ['hv270','hv271','hv005']
    self.df_survey.loc[:,float_cols] = self.df_survey.loc[:,float_cols].astype(np.float)
    

  def load_clusters(self, years):
    years = years.split(',') if type(years)==str else years
    years = [int(y) for y in years]

    ### 1. reading shape files
    self.df_cluster = gpd.GeoDataFrame()
    for year in tqdm(years):
      for fn in glob.glob(os.path.join(self.root,f"survey/*/{year}/{self.country_code}GE*")):
        if fn.endswith(".zip"):
          fname = os.path.basename(fn).replace('.zip','')
          fn = f"zip:///{fn}!{fname}.shp"
        else:
          fn = glob.glob(os.path.join(self.root,f"survey/*/{year}/{self.country_code}GE*/{self.country_code}GE*.shp"))[0]
        tmp = gpd.read_file(fn)
        tmp.loc[:,'SURVEY'] = fn.split("/survey/")[-1].split('/')[0]
        self.df_cluster = self.df_cluster.append(tmp, ignore_index=True)
    
    ### 2. validate
    self.df_cluster.loc[:,'DHSYEAR'] = self.df_cluster.DHSYEAR.astype(np.int16)
    self.df_cluster.loc[:,'DHSCLUST'] = self.df_cluster.DHSCLUST.astype(np.int16)
    self.df_cluster.loc[:,'URBAN_RURA'] = self.df_cluster.URBAN_RURA.apply(lambda c: 1 if c=='U' else 2 if c=='R' else 0).astype(np.int16)
    self.df_cluster = self.df_cluster.merge(self.df_survey[['year','survey','hv001']].drop_duplicates(), left_on=['DHSYEAR','SURVEY','DHSCLUST'], right_on=['year','survey','hv001'])
    self.df_cluster.drop(columns=['year','survey','hv001'], inplace=True)
    
    ### 3 Remove invalid clusters (lon and lat = 0)
    self.df_cluster = self.df_cluster.query("LATNUM!=0 and LONGNUM!=0").copy()
    DHSCLUSTS = self.df_cluster.DHSCLUST.values
    self.df_survey = self.df_survey.query("hv001 in @DHSCLUSTS")
    self.county_code = self.df_cluster.DHSCC.unique()[0]

  def _iwi_household(self, obj):
    id,row = obj
    
    # water (hv201)
    w1 = WATER[self.country_code][row.year]['low'] #low
    w2 = WATER[self.country_code][row.year]['mid'] #mid
    w3 = WATER[self.country_code][row.year]['high'] #high

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
    
    return iwi

  def compute_IWI(self, njobs=-1):
    ### The International Wealth Index (IWI)
    ### Jeroen Smits and Roel Steendijk
    results = pqdm(self.df_survey.iterrows(), self._iwi_household, n_jobs=njobs)
    self.df_survey.loc[:,'iwi'] = results
    self.df_survey.loc[:,'iwi'] += CONSTANT
    self.df_survey.iwi = self.df_survey.iwi.astype(np.float32) #.round(2) 
        
    self.df_cluster.drop(columns=['mean_iwi','std_iwi'], inplace=True, errors='ignore')
    tmp = self.df_survey.groupby(['year','survey','hv001']).iwi.agg(['mean','std']).reset_index()
    self.df_cluster = self.df_cluster.merge(tmp, left_on=['DHSYEAR','SURVEY','DHSCLUST'], right_on=['year','survey','hv001']).drop(columns=['year','survey','hv001'])
    self.df_cluster.rename(columns={'mean':'mean_iwi', 'std':'std_iwi'}, inplace=True)

  def set_categories(self):
    # labels = ['poor','lower_middle','upper_middle','rich']
    # bins = pd.IntervalIndex.from_tuples([(0, 25), (25, 50), (50, 75), (75,101)], closed='left')
    # map = {b:l for b,l in zip(*[[v.left for v in bins],labels])}
    # self.df_cluster.loc[:,'iwi_bin'] = pd.cut(self.df_cluster.mean_iwi, bins=bins, precision=0, retbins=False)
    # self.df_cluster.loc[:,'iwi_cat'] = self.df_cluster.loc[:,'iwi_bin'].apply(lambda v: map[v.left])
    # self.df_cluster.loc[:,'iwi_cat_id'] = self.df_cluster.loc[:,'iwi_cat'].apply(lambda v: labels.index(v))
    self.df_cluster = DHSMIS.add_ses_categories(self.df_cluster, 'mean_iwi')

  @staticmethod
  def add_ses_categories(df, col_iwi):
    from utils.constants import SES_LABELS
    labels = SES_LABELS
    bins = pd.IntervalIndex.from_tuples([(0, 25), (25, 50), (50, 75), (75,101)], closed='left')
    map = {b:l for b,l in zip(*[[v.left for v in bins],labels])}
    df.loc[:,'iwi_bin'] = pd.cut(df[col_iwi], bins=bins, precision=0, retbins=False)
    df.loc[:,'iwi_cat'] = df.loc[:,'iwi_bin'].apply(lambda v: map[v.left])
    df.loc[:,'iwi_cat_id'] = df.loc[:,'iwi_cat'].apply(lambda v: labels.index(v))
    return df

  @staticmethod
  def discretize_in_n_bins(arr, n=10, minv=0, maxv=100):
    labels = np.arange(n)
    steps = int(np.ceil(maxv / n))
    bins = pd.IntervalIndex.from_tuples([(b,b+steps+(1 if b==maxv-steps else 0)) for b in np.arange(minv,maxv,steps)], closed='left')
    map = {b:l for b,l in zip(*[[v.left for v in bins],labels])}
    return pd.cut(arr, bins=bins, precision=0, retbins=False, ordered=True).codes

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

