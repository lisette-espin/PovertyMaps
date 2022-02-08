import pandas as pd
import numpy as np
from tqdm import tqdm

USDOLAR_TO_SHILLING = 0.000387 # Google Finance 2012/10 https://www.theglobaleconomy.com/Uganda/Dollar_exchange_rate/
#USDOLAR_TO_SHILLING = 0.00027 # 2021-03-21 Google:Data provided by Morningstar for Currency and Coinbase for Cryptocurrency

### How to read a questionary: 
### UNPS_2010-11_Basic-Info-2014.pdf
### https://microdata.worldbank.org/index.php/catalog/2166/data-dictionary/F81?file_name=GSEC14
### https://microdata.worldbank.org/index.php/catalog/2166/download/31660 

### HHID: household id
### comm: cluster number
### sregion: region of residence
### urban: urban or rural

### h14q2(Television) - h14q3(Yes)
### h14q2(Household Appliances e.g. Kettle, Flat iron, etc.) - h14q3(Yes) / has fridge 
### h14q2(Mobile phone) - h14q3(Yes): Mobile phone
### h14q2(Motor vehicle) - h14q3(Yes)
### h14q2(Bicycle) - h14q3(Yes)
### h10q1: has electricity 

### h9q7: source of drinking water (1,2,3: l,m,h)
### h9q6: floor material
### h9q22: type of toilet facility

### h9q3: number of -sleeping- rooms (1,2,3: 0-1, 2, 3>)

### UNPS 2011-12 Consumption Aggregate:
### welfare: cpexp30 adjusted by equiv - Consumption Aggregate
### cpexp30: Monthly HH Expenditures in 05/06 Prices, Spatially/Temporally Adjusted in 11/12
### equiv: Household Adult Equivalence Scale
### poor: =1 if welfare<spline

### Water supply 
### - high quality is bottled water or water piped into dwelling or premises; (Private connection to pipeline (Tap))
### - middle quality is public tap, protected well, tanker truck, etc.; (Public taps, Protected well/spring, Vendor/Tanker truck, Bore-hole, Public taps)
### - low quality is unprotected well, spring, surface water, etc. (Unprotected well/spring,  "River, stream, lake, pond", Gravity flow scheme, Rain water, Other)
### Toilet facility 
### - high quality is any kind of private flush toilet; (Flush toilet private)
### - middle quality is public toilet, improved pit latrine, etc.; (Flush toilet shared, VIP latrine shared, VIP latrine private)
### - low quality is traditional pit latrine, hanging toilet, or no toilet facility. (Covered pit latrine private, Covered pit latrine shared, Uncovered pit latrine, Bush, Other)
### Floor quality 
### - high quality is finished floor with parquet, carpet, tiles, ceramic etc.; (Mosaic or tiles)
### - middle quality is cement, concrete, raw wood, etc. (Cement, Bricks, Stone, Wood)
### - low quality is none, earth, dung etc., (Earth, Earth and cow dung, Other)


### The International Wealth Index (IWI) by Jeroen Smits and Roel Steendijk
BETAS = {'television':8.612657,
         'refrigerator':8.429076,
         'phone':7.127699,
         'car':4.651382,
         'bike':1.846860,
         'electricity':8.056664,
         'water':{1:-6.306477,2:-2.302023,3:7.952443},
         'floor':{1:-7.558471,2:1.227531,3:6.107428},
         'toilet':{1:-7.439841,2:-1.090393,3:8.140637},
         'rooms':{1:-3.699681,2:0.384050,3:3.445009}}

class LSMS(object):
  # ID: HHID (household-id)
  
  def __init__(self):
    self.df_geovars = None
    self.df_gsec1 = None
    self.df_gsec8 = None
    self.df_gsec9a = None
    self.df_gsec10a = None
    self.df_gsec14 = None
    self.df_aggcons = None
    self.df_survey = None
    self.df_merged = None

  ###############################################
  # Load data from STATA
  ###############################################

  def load_geovars_from_stata(self, fn):
    self.df_geovars = pd.read_stata(fn)
    self.validate_lat_lon()

  #def load_consagg_from_stata(self, fn):
  #  self.df_consagg = pd.read_stata(fn)

  def load_gsec1_from_stata(self, fn):
    ### comm: cluster ID (from Cwest)
    self.df_gsec1 = pd.read_stata(fn)

  def load_gsec8_from_stata(self, fn):
    ### labour force:
    self.df_gsec8 = pd.read_stata(fn)

  def load_gsec9a_from_stata(self, fn):
    ### household conditions and water and sanitation
    self.df_gsec9a = pd.read_stata(fn)

  def load_gsec10a_from_stata(self, fn):
    ### household energy
    self.df_gsec10a = pd.read_stata(fn)

  def load_gsec14_from_stata(self, fn):
    ### household asset type
    self.df_gsec14 = pd.read_stata(fn)

  def load_aggcons_from_stata(self, fn):
    self.df_aggcons = pd.read_stata(fn)

  ###############################################
  # Methods
  ###############################################

  def validate_lat_lon(self, drop=True):
    self.df_geovars.loc[:,'valid'] = self.df_geovars.nogps.apply(lambda c:int(c!=1))
    if drop:
      self.df_geovars.drop(self.df_geovars[self.df_geovars.valid == 0].index, inplace=True)
      
  def get_valid_lat_lon(self):
    return self.df_geovars.query("valid==1")[['lat_mod','lon_mod']].values

  def main_join(self, df):
    return pd.merge(self.df_geovars[['HHID','lat_mod','lon_mod','valid']], df, on='HHID') 

  def merge_all_necessary_data(self):
    self.df_survey_iwi = pd.merge(self.df_geovars[['HHID','lat_mod','lon_mod','valid']], self.df_gsec1[['HHID','urban','sregion','comm']], on='HHID') 
    self.df_survey_iwi.loc[:,'urban'] = self.df_survey_iwi.urban.apply(lambda c: int(c=='Urban'))

    ### Section 14: Household assets
    tmp = pd.pivot(self.df_gsec14[['HHID','h14q2','h14q3']],index='HHID',columns='h14q2',values='h14q3')
    
    drop_cols_sec14 = ['House','Other buildings','Land','Furniture/furnishigs','Radio/Cassette','Generators','Solar panel/electric inverters','Motor cycle','Boat','Other Transport equipment','Jewelry and Watches','Computer','Internet Access','Other electronic equipment','Other household assets e.g. lawn mowers, etc.','Other 1 (specify)','Other 2 (specify)']
    tmp.drop(columns=drop_cols_sec14, inplace=True)
    self.df_survey_iwi = pd.merge(self.df_survey_iwi, tmp, on='HHID') 

    ### Section 9a: Quality of water, floor, toilet, # of rooms
    tmp = self.df_gsec9a[['HHID','h9q6','h9q7','h9q22','h9q3']].copy()

    wh = ['Private connection to pipeline (Tap)']
    wm = ['Public taps', 'Protected well/spring', 'Vendor/Tanker truck', 'Bore-hole', 'Public taps']
    wl = ['Unprotected well/spring',  'River, stream, lake, pond', 'Gravity flow scheme', 'Rain water', 'Other (specify)']
    tmp.loc[:,'h9q7'] = tmp.h9q7.apply(lambda c: 1 if c in wl else 2 if c in wm else 3 if c in wh else 0)

    fh = ['Mosaic or tiles']
    fm = ['Cement', 'Bricks', 'Stone', 'Wood']
    fl = ['Earth', 'Earth and cow dung', 'Other (specify)']
    tmp.loc[:,'h9q6'] = tmp.h9q6.apply(lambda c: 1 if c in fl else 2 if c in fm else 3 if c in fh else 0)

    th = ['Flush toilet private']
    tm = ['Flush toilet shared', 'VIP latrine shared', 'VIP latrine private']
    tl = ['Covered pit latrine private', 'Covered pit latrine shared', 'Uncovered pit latrine', 'Bush', 'Other (specify)']
    tmp.loc[:,'h9q22'] = tmp.h9q22.apply(lambda c: 1 if c in tl else 2 if c in tm else 3 if c in th else 0)

    tmp.loc[:,'h9q3'] = tmp.h9q3.apply(lambda c: 1 if c <= 1 else 2 if c == 2 else 3)

    self.df_survey_iwi = pd.merge(self.df_survey_iwi, tmp, on='HHID') 

    ### Section 10a: Electricity
    tmp = self.df_gsec10a[['HHID','h10q1']].copy()
    self.df_survey_iwi = pd.merge(self.df_survey_iwi, tmp, on='HHID') 

    ### Consumption Aggregate: 
    self.df_survey_iwi = pd.merge(self.df_survey_iwi, self.df_aggcons[['HHID','welfare','cpexp30','equiv','poor']], on='HHID') 

    ### rename columns:
    self.df_survey_iwi.rename(columns={'Household appliances':'refrigerator',
                                      'Television':'television',
                                      'Bicycle':'bike',
                                      'Motor vehicle':'car',
                                      'Mobile phone':'phone',
                                      'h9q6':'floor',
                                      'h9q7':'water',
                                      'h9q22':'toilet',
                                      'h9q3':'rooms',
                                      'h10q1':'electricity',
                                      }, inplace=True)

    ### coding
    self.df_survey_iwi.replace('Yes',1,inplace=True)
    self.df_survey_iwi.replace('No',0,inplace=True)
    self.df_survey_iwi.loc[:,'iwi'] = 0

  def compute_IWI(self):
    ### The International Wealth Index (IWI)
    ### Jeroen Smits and Roel Steendijk

    self.merge_all_necessary_data()

    #columns = ['hhid','cluster','region','urban','tv','fridge','phone','car','bike','electricity','water','floor','toilet','rooms','wealth','iwi']
    #self.df_survey = pd.DataFrame(columns=columns)

    for id,row in self.df_survey_iwi.iterrows():
      yr = 0
      for k,b in BETAS.items():
        x = row.loc[k]

        if np.isnan(x):
          continue

        if k in ['water','floor','toilet','rooms']:
          b = b[int(x)]

        yr += b * x
      self.df_survey_iwi.loc[id:,'iwi'] = yr

    self.df_survey_iwi.loc[:,'iwi'] += abs(self.df_survey_iwi.iwi.min())
    self.df_survey_iwi.loc[:,'iwi'] /= self.df_survey_iwi.iwi.max()
    self.df_survey_iwi.loc[:,'iwi'] *= 100

  ###############################################
  # Cross-data
  ###############################################

  def validate_clusters(self):
    from shapely.geometry import MultiPoint
    
    errors = []
    df_clusters = pd.DataFrame(columns=['lat','lon','comm','urban']) 

    tmp = self.main_join(self.df_gsec1)
    for cluster,df in tmp.groupby('comm'):
      points = MultiPoint(df[['lon_mod','lat_mod']].values)
      try:
        urban = 1 if df.urban.mode()[0]=='Urban' else 0
        df_clusters = df_clusters.append(pd.DataFrame({'lat':points.centroid.y,
                                                      'lon':points.centroid.x,
                                                      'comm':cluster,
                                                      'urban':urban
                                                      }, index=[0]), ignore_index=1)
      except Exception as ex:
        errors.append((cluster,ex))
    return df_clusters, errors

  def get_clusters(self, hue=None):
    
    cols_1 = ['lat_mod','lon_mod']
    cols_2 = ['HHID','comm']


    if hue is not None and hue in self.df_gsec1.columns:
      cols_2.append(hue)
      cols_1.append(hue)
    else:
      cols_1.append('comm')
    
    tmp = self.main_join(self.df_gsec1[cols_2])

    if hue == 'urban':
      tmp.loc[:,'urban'] = tmp.urban.apply(lambda x: int(x.lower()=='urban')).astype(np.float)

    tmp = tmp.groupby('comm').mean().reset_index()[cols_1]
    tmp.loc[:,'comm'] = np.arange(1,tmp.shape[0]+1)

    if hue is not None:
      tmp.drop(columns=['comm'], inplace=True)
      if hue == 'urban':
        tmp.loc[:,'urban'] = tmp.urban.apply(lambda x: 'Urban' if x==1 else 'Rural')
    return tmp.values
    
  def get_first_job_weekly_income(self, convert_to_dollar=False, cut=None, qcut=None):
    columns = ['HHID','h8q31a','h8q31b','h8q31c','result_code']
    tmp = self.df_gsec8[columns].query("result_code=='Completed'")
    tmp.h8q31a = tmp.h8q31a.fillna(0)
    tmp.h8q31b = tmp.h8q31b.fillna(0)

    def get_weekly_value(row):
      summ = row.h8q31a + row.h8q31b
      if row.h8q31c == 'Hour':
        summ = summ * (1 / (7*24))
      elif row.h8q31c == 'Day':
        summ = summ * (1 / (7))
      elif row.h8q31c == 'Month':
        summ = summ * (30/7)
      elif row.h8q31c == 'Week':
        summ = summ
      else:
        summ = -1
      return summ

    tmp.loc[:,'h8q31'] = tmp.apply(lambda row: get_weekly_value(row)  , axis=1)
    tmp = tmp.query("h8q31 > 0")
    tmp = tmp.groupby("HHID")[['h8q31']].sum().reset_index()
    tmp = self.main_join(tmp)

    if convert_to_dollar:
      tmp.loc[:,'h8q31'] = tmp.h8q31 * USDOLAR_TO_SHILLING

    if cut is not None:
      # to especify bin edges.
      # i.e., bins are of constant size (ranges)
      tmp['h8q31'] = pd.cut(tmp['h8q31'], bins=cut, right=False, precision=0)
    elif qcut is not None:
      # to especify bin edges.
      # i.e., bins are of constant size (ranges)
      tmp['h8q31'] = pd.qcut(tmp['h8q31'], q=qcut, precision=0)

    return tmp[['lat_mod','lon_mod','h8q31']].values

  def get_cross_welfare(self, convert_to_dollar=False, qcut=None, cut=None):
    tmp = self.main_join(self.df_aggcons[['HHID','welfare']])
    
    if convert_to_dollar:
      tmp.loc[:,'welfare'] = tmp.welfare * USDOLAR_TO_SHILLING
    tmp.sort_values("welfare", ascending=True, inplace=True)

    if qcut is not None:
      # distribution of the data in bins is equal 
      # i.e., all bins will have the same number of obs.
      tmp['welfare'] = pd.qcut(tmp['welfare'], q=qcut, precision=0)
      
    if cut is not None:
      # to especify bin edges.
      # i.e., bins are of constant size (ranges)
      tmp['welfare'] = pd.cut(tmp['welfare'], bins=cut, right=False, precision=0)
      
    return tmp[['lat_mod','lon_mod','welfare']].values

  def get_cross_poor(self):
    tmp = self.main_join(self.df_aggcons[['HHID','poor']])
    return tmp[['lat_mod','lon_mod','poor']].values

  def get_cross_welfare_and_poor(self, convert_to_dollar=None, log=False):
    tmp = self.main_join(self.df_aggcons[['HHID','welfare','poor']])

    if convert_to_dollar is not None:
      tmp.loc[:,'welfare'] = tmp.welfare * USDOLAR_TO_SHILLING
    
    if log:
      tmp.loc[:,'welfare'] = np.log(tmp.welfare)

    return tmp[['lat_mod','lon_mod','poor','welfare']].values


