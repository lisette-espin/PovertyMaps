###
# https://code.earthengine.google.com/7ce7678f72b8d90ef305504609c2f813
# https://code.earthengine.google.com/3dc6ac4e72eb0456745c85698e91cc7e
##

##############################################################################
# DEPENDENCIES
##############################################################################
import ee
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyproj import Geod
from shapely.geometry import Point
from shapely.geometry import LineString
from utils import ios
from utils.constants import NONE

##############################################################################
# CONSTANTS
##############################################################################
COLLECTIONS = {'ols':{'name':'NOAA/DMSP-OLS/NIGHTTIME_LIGHTS',
                      'key':'stable_lights',
                      'available_from':'1992-01-01',
                      'available_to':'2014-01-01',
                      'resolution':'30 arc seconds',
                      'min':0,
                      'max':63},
               'viirs':{'name':'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG',
                        'key':'avg_rad',
                        'available_from':'2014-01-01',
                        'available_to':'2021-03-01', #2021-07-09
                        'resolution':'15 arc seconds',
                        'min':-1.5,
                        'max':193565}}

SELECT_OPTIONS = ['first','mean','median','max']

SCALE = 30  # meters
MAXELEMENTS = 5000
SLEEP = 60 * 1 # 1 minute

##############################################################################
# MAIN CLASS
##############################################################################
class VIIRS(object):

  def __init__(self, source, year, api_key, project_id, service_account):
    '''
    Constructor
    '''
    if source not in COLLECTIONS:
      raise Exception("source "+ source +" is not supported.")

    self.year = year
    self.collectionstr = COLLECTIONS[source]['name']
    self.key = COLLECTIONS[source]['key']
    self.available_from = COLLECTIONS[source]['available_from']
    self.available_to = COLLECTIONS[source]['available_to']
    self.collection = None
    self.lights = None
    self.points = None
    self.intensity = None
    self.API_KEY = None if api_key in NONE else ios.read_txt(api_key)
    self.PROJECT_ID = None if project_id in NONE else ios.read_txt(project_id)
    self.SERVICE_ACCOUNT = None if project_id in NONE else ios.read_txt(service_account)


  def auth(self):
    try:
      ee.Initialize()
    except:
      if self.API_KEY is None:
        ee.Authenticate()
        ee.Initialize()
      else:
        #credentials = ee.ServiceAccountCredentials(service_account, '.private-key.json')
        #ee.Initialize(credentials)
        ee.Initialize(project=self.PROJECT_ID, cloud_api_key=self.API_KEY)
    
  ###
  # Filters layers by date and aggregates by 'select' from the collection.
  ###
  def filter(self, date_start=None, date_end=None, select='first'):
    '''
    Filters data collection by date, and aggregates 
    the filtered layers by 'select'.
    The resulting aggregated data is stored in self.lights
    '''
    if select not in SELECT_OPTIONS:
      raise Exception("select option not implemented.")

    date_start = date_start if date_start else self.available_from
    date_end = date_end if date_end else self.available_to

    self.collection = ee.ImageCollection(self.collectionstr)
    self.collection = self.collection.filterDate(date_start, date_end)
    
    if select == 'first':
      self.image = self.collection.first()
    elif select == 'mean':
      self.image = self.collection.mean()
    elif select == 'median':
      self.image = self.collection.median()
    elif select == 'max':
      self.image = self.collection.max()

    self.image = self.image.addBands(ee.Image.pixelArea())

  ###
  # Aggregates luminosity within rectangular areas around 'points'
  ###
  def fast_lights_at_points(self, allpoints, buffer_meters=1.0, rad_gte_thres=1.0, scale=SCALE, cache_dir=None):

    maxe = MAXELEMENTS if buffer_meters<10000 else 1000
    df = pd.DataFrame()
    nbins = int(np.ceil(allpoints.shape[0] / maxe))
    for binid,points in enumerate(np.array_split(allpoints, nbins)):
      
      tmp = load_cache_results(cache_dir, self.year, allpoints.shape[0], buffer_meters, rad_gte_thres, scale, binid, nbins)
      if tmp is not None:
        df = df.append(tmp)
      else:
        done = False
        nibins = 1
        while not done:
          
          for nibinid, chunk in enumerate(np.array_split(points,nibins)):

            tmp = load_cache_results(cache_dir, self.year, allpoints.shape[0], buffer_meters, rad_gte_thres, 
                                     scale, binid, nbins, nibinid, nibins)
            if tmp is not None:
              df = df.append(tmp)
            else:
              try:
                # 1. areas around points
                rectangles = [ee.Geometry.Point([lon, lat]).buffer(buffer_meters).bounds() for lon,lat in chunk]
                
                # 2. stats (reducers)
                tc = ee.Reducer.count()
                ts = ee.Reducer.sum()
                tmi = ee.Reducer.min()
                tma = ee.Reducer.max()
                tme = ee.Reducer.mean()
                tmd = ee.Reducer.median()
                l3 = ee.Reducer.intervalMean(0,33)
                u3 = ee.Reducer.intervalMean(66,100)
                
                # 3. totals
                total_cols = ['avg_rad_min','avg_rad_max','avg_rad_mean','avg_rad_median','avg_rad_l3_mean',
                              'avg_rad_u3_mean','area_sum','avg_rad_count','avg_rad_sum']
                total_reducers = tc.combine(reducer2=ts, sharedInputs=True).combine(reducer2=tmi, sharedInputs=True).combine(reducer2=tma, sharedInputs=True).combine(reducer2=tme, sharedInputs=True).combine(reducer2=tmd, sharedInputs=True).combine(reducer2=l3, sharedInputs=True, outputPrefix="l3_").combine(reducer2=u3, sharedInputs=True, outputPrefix='u3_')
                totals = self.image.reduceRegions(scale=scale,
                                                  collection=ee.FeatureCollection(rectangles),
                                                  reducer=total_reducers).getInfo()
                
                # 4. constrained areas (rad >= rad_gte_trhes)
                masked = self.image.updateMask(self.image.select("avg_rad").gte(rad_gte_thres))
                masked_cols = ['area_sum','avg_rad_count','avg_rad_sum']
                masked_reducers = tc.combine(reducer2=ts, sharedInputs=True)
                masks = masked.reduceRegions(scale=scale,
                                            collection=ee.FeatureCollection(rectangles),
                                            reducer=masked_reducers).getInfo()
                
                df_totals = pd.DataFrame([v['properties'] for v in totals['features']])[total_cols]
                df_masks = pd.DataFrame([v['properties'] for v in masks['features']])[masked_cols]
                df_masks.rename(columns={'area_sum':'cons_area_sum', 'avg_rad_count':'cons_avg_rad_count', 
                                         'avg_rad_sum':'cons_avg_rad_sum'}, inplace=True)
                tmp = pd.concat([df_totals, df_masks], axis=1)
                cache_results(tmp, cache_dir, self.year, allpoints.shape[0], buffer_meters, rad_gte_thres, scale, binid, nbins, nibinid, nibins)
                df = df.append(tmp)
                
                del(totals)
                del(masks)
                del(masked)
                del(df_totals)
                del(df_masks)
                time.sleep(SLEEP)
                done = True
              except Exception as ex:
                print(ex)
                print('current chunk size: {}'.format(len(chunk)))
                done = False
                nibins += 1
                print("repartition -> nibins: {}".format(nibins))
                break

    df.loc[:,'frac_area'] = df.apply(lambda row: row.cons_area_sum/row.area_sum, axis=1)
    df.loc[:,'frac_pixels'] = df.apply(lambda row: row.cons_avg_rad_count/row.avg_rad_count, axis=1)
    df.loc[:,'frac_sum_rad'] = df.apply(lambda row: row.cons_avg_rad_sum/row.avg_rad_sum, axis=1)
    
    # 5. keeping necesary columns
    km = round(buffer_meters/1000.,2)
    cols = {}
    for c in df.columns:
      if c in ['area_sum','avg_rad_count','avg_rad_sum'] or c.startswith("cons_"):
        continue
      cols[c] = c.replace("avg_rad","NTLL_{}km".format(km)).replace("frac","NTLL_{}km_rad_gte_{}".format(km,rad_gte_thres))
    df.rename(columns=cols, inplace=True)
    
    return df[[c for c in df.columns if c.startswith("NTLL")]]

def get_cache_fn(path, year, data_size, buffer_meters, rad_gte_thres, scale, binid, nbins, nibinid=0, nibins=1):
  fn = "year{}_size{}_bmeter{}_radthres{}_scale{}_b{}_{}{}.csv".format(year, data_size, buffer_meters, rad_gte_thres, scale, binid+1, nbins, '' if nibinid==0 and nibins==1 else '_nib{}_{}'.format(nibinid+1,nibins))
  return os.path.join(path,fn)

def load_cache_results(path, year, data_size, buffer_meters, rad_gte_thres, scale, binid, nbins, nibinid=0, nibins=1):
  fn = get_cache_fn(path, year, data_size, buffer_meters, rad_gte_thres, scale, binid, nbins, nibinid, nibins)
  if ios.exists(fn):
    return ios.load_csv(fn, verbose=False)
  return None

def cache_results(df, path, year, data_size, buffer_meters, rad_gte_thres, scale, binid, nbins, nibinid=0, nibins=1):
  fn = get_cache_fn(path, year, data_size, buffer_meters, rad_gte_thres, scale, binid, nbins, nibinid, nibins)
  ios.save_csv(df, fn, verbose=False)
