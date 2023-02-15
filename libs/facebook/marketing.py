### https://developers.facebook.com/docs/marketing-api/audiences/reference/basic-targeting
### https://developers.facebook.com/docs/marketing-api/audiences/reference/advanced-targeting
### https://developers.facebook.com/docs/marketing-api/audiences/reference/targeting-search
### https://developers.facebook.com/docs/marketing-api/reference/reach-estimate/
### ### create app: https://developers.facebook.com/apps/?show_reminder=true

################################################################################
# Dependencies
################################################################################

import os
import time
import datetime
import pandas as pd
from tqdm import tqdm
from itertools import cycle
from collections import defaultdict
from facebook_business.adobjects.targetingsearch import TargetingSearch
from facebook_business.adobjects.adaccount import AdAccount
from facebook_business.adobjects.adrule import AdRule
from facebook_business.api import FacebookAdsApi
from facebook_business import session
from facebook_business.exceptions import FacebookRequestError

from utils import ios
from utils import validations

################################################################################
# Constants
################################################################################

from utils.constants import *

################################################################################
# Class
################################################################################

SLEEP = 60

class FacebookMarketing(object):
  ### BEHAVIORS
  BEHAVIOR_FREQUENT_TRAVELER = {'behaviors':[{'id':6002714895372, 'name':'Frequent Travelers'}]}
  BEHAVIOR_SMALL_BUSINESS_OWNER = {'behaviors':[{'id':6002714898572, 'name':'Small business owners'}]}
  BEHAVIOR_COMMUTER = {'behaviors':[{'id':6013516370183, 'name':'Commuters'}]}
  BEHAVIOR_LIVES_ABROAD = {'behaviors':[{'id':6015559470583, 'name':'Lives abroad'}]}
  BEHAVIOR_FREQUENT_INTERNATIONAL_TRAVELER = {'behaviors':[{'id':6022788483583, 'name':'Frequent international travelers'}]}
  BEHAVIOR_NETWORK_2G = {'behaviors':[{'id':6017253486583, 'name':'Facebook access (network type): 2G'}]}
  BEHAVIOR_NETWORK_3G = {'behaviors':[{'id':6017253511583, 'name':'Facebook access (network type): 3G'}]}
  BEHAVIOR_NETWORK_4G = {'behaviors':[{'id':6017253531383, 'name':'Facebook access (network type): 4G'}]}
  BEHAVIOR_FEATURE_PHONE = {'behaviors':[{'id':6004383149972, 'name':'Facebook access (mobile): feature phones'}]}
  BEHAVIOR_OLD_DEVICE_OS = {'behaviors':[{'id':6004854404172, 'name':'Facebook access: older devices and OS'}]}
  BEHAVIOR_MOBILE_ACCESS = {'behaviors':[{'id':6004382299972, 'name':'Facebook access (mobile): all mobile devices'}]}
  BEHAVIOR_BROWSER_ACCESS = {'behaviors':[{'id':6015547847583, 'name':'Facebook access (browser): Firefox'},
                                          {'id':6015547900583, 'name':'Facebook access (browser): Chrome'},
                                          {'id':6015593608983, 'name':'Facebook access (browser): Safari'},
                                          {'id':6015593652183, 'name':'Facebook access (browser): Opera'},
                                          {'id':6015593776783, 'name':'Facebook access (browser): Internet Explorer'}]}
  BEHAVIOR_SMARTPHONE_TABLET = {'behaviors':[{'id':6004383049972, 'name':'smartphones and tablets'}]}
  BEHAVIOR_WIFI = {'behaviors':[{'id':6015235495383, 'name':'Facebook access (network type): WiFi'}]}
  BEHAVIOR_TECHNOLOGY_EARLY_ADOPTERS = {'behaviors':[{'id':6003808923172, 'name':'Technology early adopters'}]}
  BEHAVIOR_RETURNED_TRAVEL_1WEEK = {'behaviors':[{'id':6008261969983, 'name':'Returned from travels 1 week ago'}]}
  ### LIFE-EVENTS
  LIFEEVENT_AWAY_FROM_HOME = {'life_events':[{'id':6003053860372, 'name':'Away from hometown'},{'id':6003053857372, 'name':'Away from family'}]}
  LIFEEVENT_ENGAGED = {'life_events':[{'id':6002714398772, 'name':'Newly-engaged (6 months)'},
                                      {'id':6003050210972, 'name':'Newly engaged (1 year)'},
                                      {'id':6012631862383,'name':'Newly engaged (3 months)'}]}
  ### INDUSTRY
  INDUSTRY_GOV = {'industries':[{'id':6019621029983, 'name':'Government Employees (Global)'}]}
  INDUSTRY_BUSINESS = {'industries':[{'id':6009003307783, 'name':'Business and Finance'}]}
  INDUSTRY_LEGAL = {'industries':[{'id':6008888972183, 'name':'Legal Services'}]}
  INDUSTRY_IT = {'industries':[{'id':6008888961983, 'name':'IT and Technical Services'}]}
  ### USER-DEVICE
  DEVICE_MOTOROLA = {'user_device':["motorola"]}
  DEVICE_AMAZON = {'user_device':["amazon"]}
  DEVICE_NOKIA = {'user_device':["nokia"]}
  DEVICE_MICROSOFT = {'user_device':["microsoft"]}
  ### USER-OS
  OS_IOS = {'user_os':['iOS']}
  OS_ANDROID = {'user_os':['Android']}
  OS_WINDOWS = {'user_os':['Windows','Windows Phone']}
  OS_WINDOWS_DESKTOP = {'user_os':['Windows']}
  OS_WINDOWS_PHONE = {'user_os':['Windows Phone']}
  ### EDUCATION
  EDU_HIGHSCHOOL = {'education_statuses':[1,4,13,4]}
  EDU_BACHELOR = {'education_statuses':[2,5,6]}
  EDU_MASTER = {'education_statuses':[3,7,8,9]}
  EDU_PROFESSIONAL = {'education_statuses':[10]}
  EDU_PHD = {'education_statuses':[11]}
  ### INSTERESTS
  INTEREST_CASINO = {'interests':[{'id':6003248338072, 'name':'Casino games'},{'id':6003012317397, 'name':'Gambling'}]}
  ### DISTACE_UNITS
  UNIT_KM = 'kilometer'
  UNIT_ML = 'mile'
  
  def __init__(self, app_id, app_secret, access_token, ad_account_id, timestamp):
    self.app_id = app_id
    self.app_secret = app_secret
    self.access_token = access_token
    self.ad_account_id = ad_account_id
    self.fb_session = None
    self.appsecret_proof = None
    self.api = None
    self.timestamp = timestamp

  def init(self, api_version=FBM_API_VERSION):
    self.fb_session = session.FacebookSession(app_id=self.app_id, app_secret=self.app_secret, access_token=self.access_token)
    self.appsecret_proof = self.fb_session._gen_appsecret_proof()  
    FacebookAdsApi.init(access_token=self.access_token, api_version=api_version)

  def extract_reach_estimate(self, lon, lat, ccode, radius=1.0, unit=UNIT_KM, specific_params=None, cache_dir='/mydrive/cache'):
    '''
    Potential reach (the number of monthly active people) on Facebook that match 
    the audience you defined through your audience targeting selections.
    '''
    code = None 
    
    results = load_cache_reach_estimates(lon,lat,radius,unit,specific_params,cache_dir)
    
    status = needs_to_be_queried(results)
    if status == FBM_LOADED_DONE:
      #print(f"DONE: {len(results)}")
      return get_value(results), FBM_LOADED_DONE
    elif status == FBM_NEEDS_QUERY:
      pass
      #print(f'NEEDS TO BE QUERY: {len(results)}')
    else:
      print(f"status: {status}")
      raise Exception("[ERROR] marketing.py | extract_reach_estimate | something went wrong (status).")
      
    fields = []
    params = {'targeting_spec': {'geo_locations':{'custom_locations':[{'latitude': lat, 'longitude': lon, 
                                                                       'radius': radius, 'distance_unit': unit}], 
                                                  "location_types":["home"]},
                                 'age_min':13,'age_max':65},
              'appsecret_proof': self.appsecret_proof,
              'optimization_goal':'REACH'} # 'optimization_goal':'REACH' # only for get_delivery_estimate 
    
    if specific_params is not None:
      params['targeting_spec'].update(specific_params)
    
    try:
      # get_reach_estimate --> MAU monthly active users
      # get_delivery_estimate --> MAU & DAU monthly and daily active users
      result = AdAccount(self.ad_account_id).get_delivery_estimate(fields=fields,params=params)
      code = FBM_QUERIED
    except FacebookRequestError as fex:
      #code:100  subcode:1487851 (wrong location)
      #code:80004 subcode:2446079 (quota)
      code = FBM_SKIP_LOC if fex.api_error_code() == FBM_ERR_CODE_WRONG_LOC and fex.api_error_subcode() == FBM_ERR_SUBCODE_WRONG_LOC else FBM_QUOTA
      print('[get_reach_estimate]',fex.api_error_code(), fex.api_error_subcode(), fex.get_message())
      print('')
      result = None

    if result is not None:
      if len(result) > 1:
        print(id)
        raise Exception("This should not happen, or?")
    
      result = result[0]
      result['timestamp'] = self.timestamp
      
      #print("NEW RESPONSE", "code:", code)
      
      cache_reach_estimate(result,lon,lat,radius,unit,specific_params,cache_dir)

    return get_active_users(result), code
    
  def get_all_targeting_categories(self, verbose=False):
    '''
    Target based on a user's actions or past purchase behavior. 
    Retrieve all possible behavior targeting options with type=adTargetingCategory&class=behaviors.
    https://developers.facebook.com/docs/marketing-api/audiences/reference/targeting-search#behaviors
    '''
    all_categories = TargetingSearch.search(params={'type': TargetingSearch.TargetingSearchTypes.targeting_category,
                                                    'appsecret_proof': self.appsecret_proof})
    columns = ['id','name','description','ttype','platform','audience_size']
    df_targeting_categories = pd.DataFrame(columns=columns)
    for r in all_categories:
      id,name,description,audience_size,ttype,platform = "","","","",0,""
      try:
        id,name,description,audience_size,ttype = r['id'], r['name'],r['description'],r['audience_size'],r['type']
      except:
        try:
          id,name,audience_size,ttype = r['id'], r['name'],r['audience_size'],r['type']
        except:
          try:
            name,audience_size,ttype = r['name'],r['audience_size'],r['type']
          except:
            try:
              name,platform,ttype = r['name'],r['platform'],r['type']
            except:
              if verbose:
                print(r)
      df_targeting_categories = df_targeting_categories.append({'id':id,'name':name,'description':description,'ttype':ttype,'platform':platform,'audience_size':audience_size}, ignore_index=True)
      if verbose:
          print("id:{:15s}\t name:{:70s}\t description:{:150s}\t audience_size:{:10d}\t platform:{:15s}\t type:{}".format(r['id'], r['name'],r['description'],r['audience_size'],r['type']))
    if verbose:
      print(len(all_categories), df_targeting_categories.shape)
    return df_targeting_categories
  
  def get_radius_suggestion(self, lon, lat, unit):
    '''
    To target around a specific location, get a suggested radius reach enough people with suggested_radius:
    https://developers.facebook.com/docs/marketing-api/audiences/reference/targeting-search#radius
    '''
    results = TargetingSearch.search(params={
                'latitude':lat,'longitude':lon,'distance_unit':unit,
                'type': TargetingSearch.TargetingSearchTypes.radius_suggestion,
                'appsecret_proof': self.appsecret_proof,  
            })
    return results
  
  ##################################################################
  # Static methods
  ##################################################################
  
  @staticmethod
  def get_all_profiles():
    return {'FBM_frequent_traveler':FacebookMarketing.BEHAVIOR_FREQUENT_TRAVELER,
            'FBM_small_business_owner':FacebookMarketing.BEHAVIOR_SMALL_BUSINESS_OWNER,
            'FBM_commuter':FacebookMarketing.BEHAVIOR_COMMUTER,
            'FBM_lives_abroad':FacebookMarketing.BEHAVIOR_LIVES_ABROAD,
            'FBM_frequent_int_traveler':FacebookMarketing.BEHAVIOR_FREQUENT_INTERNATIONAL_TRAVELER,
            'FBM_network_2G':FacebookMarketing.BEHAVIOR_NETWORK_2G,
            'FBM_network_3G':FacebookMarketing.BEHAVIOR_NETWORK_3G,
            'FBM_network_4G':FacebookMarketing.BEHAVIOR_NETWORK_4G,
            'FBM_feature_phone':FacebookMarketing.BEHAVIOR_FEATURE_PHONE,
            'FBM_old_device_os':FacebookMarketing.BEHAVIOR_OLD_DEVICE_OS,
            'FBM_mobile_access':FacebookMarketing.BEHAVIOR_MOBILE_ACCESS,
            'FBM_browser_access':FacebookMarketing.BEHAVIOR_BROWSER_ACCESS,
            'FBM_smartphone_tablet':FacebookMarketing.BEHAVIOR_SMARTPHONE_TABLET,
            'FBM_wifi':FacebookMarketing.BEHAVIOR_WIFI,
            'FBM_tech_early_adopter':FacebookMarketing.BEHAVIOR_TECHNOLOGY_EARLY_ADOPTERS,
            'FBM_returned_travel_1week':FacebookMarketing.BEHAVIOR_RETURNED_TRAVEL_1WEEK,
            'FBM_away_from_home':FacebookMarketing.LIFEEVENT_AWAY_FROM_HOME,
            'FBM_engaged':FacebookMarketing.LIFEEVENT_ENGAGED,
            'FBM_ind_gov':FacebookMarketing.INDUSTRY_GOV,
            'FBM_ind_business':FacebookMarketing.INDUSTRY_BUSINESS,
            'FBM_ind_legal':FacebookMarketing.INDUSTRY_LEGAL,
            'FBM_ind_it':FacebookMarketing.INDUSTRY_IT,
            'FBM_device_motorola':FacebookMarketing.DEVICE_MOTOROLA,
            'FBM_device_amazon':FacebookMarketing.DEVICE_AMAZON,
            'FBM_device_nokia':FacebookMarketing.DEVICE_NOKIA,
            'FBM_device_microsoft':FacebookMarketing.DEVICE_MICROSOFT,
            'FBM_os_ios':FacebookMarketing.OS_IOS,
            'FBM_os_android':FacebookMarketing.OS_ANDROID,
            'FBM_os_windows':FacebookMarketing.OS_WINDOWS,
            'FBM_os_windows_desktop':FacebookMarketing.OS_WINDOWS_DESKTOP,
            'FBM_os_windows_phone':FacebookMarketing.OS_WINDOWS_PHONE,
            'FBM_highschool':FacebookMarketing.EDU_HIGHSCHOOL,
            'FBM_bachelor':FacebookMarketing.EDU_BACHELOR,
            'FBM_master':FacebookMarketing.EDU_MASTER,
            'FBM_professional':FacebookMarketing.EDU_PROFESSIONAL,
            'FBM_phd':FacebookMarketing.EDU_PHD,
            'FBM_casino':FacebookMarketing.INTEREST_CASINO}

  @staticmethod
  def load_tokens(tokens_dir):
    files = [fn for fn in os.listdir(tokens_dir) if fn.endswith(".json") or os.path.isfile(os.path.join(tokens_dir,fn))]
    tokens = {}
    for id, fn in enumerate(files):
      fn = os.path.join(tokens_dir,fn)
      try:
        obj = ios.load_json(fn)
        if obj is not None:
          tokens[id+1] = obj
      except Exception as ex:
        print(ex)
        pass
    return tokens

  @staticmethod
  def query(df_places, profiles, radius, unit, id, lat, lon, ccode, tokens, cache_dir, fn=None):
  
    tstart, tmax = reset_start_max_quota_wait() # keeping track when the query started (after all tokens have been used, it waits or continues based on the 2hour constraint for each token)
    df_places = df_places.loc[:,[id, lon,lat]]

    maxtokens = len(tokens)
    tokenids = cycle(np.arange(1,maxtokens+1,1))
    status = defaultdict(int)
    
    tokenid = next(tokenids)
    ts = validations.get_current_time_in_country(ccode)
    fbm = fbconnect(tokens, tokenid, ts)
    counter = 0
    for id, row in tqdm(df_places.iterrows(), total=df_places.shape[0]):
      counter += 1
      #print("========== {} of {} ==========".format(counter, df_places.shape[0]))

      ### QUERY
      for k,v in profiles.items():
        
        code = FBM_QUOTA
        while code == FBM_QUOTA:
          value, code = fbm.extract_reach_estimate(lon=row[lon], lat=row[lat], ccode=ccode, 
                                                   radius=radius, unit=unit, specific_params=v, cache_dir=cache_dir) 

          status[tokenid] = code == FBM_QUOTA

          if code in [FBM_SKIP_LOC, FBM_QUOTA, FBM_NOT_YET]:
            print('k:', k, 'code:', code, 'lon:',row[lon], 'lat:',row[lat]) # bad: 1 quota, 2 bad location, 3: queried but not xtimes
            df_places.loc[id,k] = np.nan
          elif code in [FBM_LOADED_DONE, FBM_QUERIED]:
            df_places.loc[id,k] = value # good: -1 loaded, 0 queried

          if code == FBM_QUOTA:
            # to try again due to quota
            print("k:{}, token:{}, rowid:{}, result:fail ({})".format(k, tokenid, id, code))

            if sum(list(status.values())) == maxtokens:
              # if all tokens have reached quota,then wait
              status = {k:0 for k,v in status.items()} # reset status
              current_time = datetime.datetime.now().strftime("%H:%M:%S")
              delta = get_seconds_to_wait(tmax)

              if delta.days < 0:
                # if negative, then the FB_MAX_HOURS have passed already since the first token was queried
                continue

              print("[APIs' quota reached] tstart:{} - tmax:{} | now:{} -> sleeping {} ({} seconds)...".format(tstart, 
                                                                                                               tmax, current_time, 
                                                                                                               delta, delta.seconds))

              ### sleep 
              time.sleep(delta.seconds)
              tstart, tmax = reset_start_max_quota_wait()

            # use next token
            tokenid = next(tokenids)
            ts = validations.get_current_time_in_country(ccode)
            fbm = fbconnect(tokens, tokenid, ts)

          elif code == FBM_SKIP_LOC:
            # skip all queries for wrong locations
            df_places.loc[id,k] = FBM_DEFAULT_WRONG_LOCATION
            break
            
        if code == FBM_SKIP_LOC:
          break

    return df_places


################################################################################
# Functions
################################################################################

def get_active_users(result):
  if result is None:
    return None
  
  if 'users' in result:
    return result['users']
  if 'users_upper_bound' in result:
    return (result['users_upper_bound']+result['users_lower_bound'])/2
  if 'estimate_mau_lower_bound' in result:
    return (result['estimate_mau_lower_bound']+result['estimate_mau_upper_bound'])/2
  
  print(result)
  raise Exception("[ERROR] marketing.py | get_active_users | result not found.")
  
def get_value(results):
  
  if results is None or len(results)==0:
    return None
  
  try:
    obj = [get_active_users(obj) for obj in results]
    return np.mean(obj)
  except Exception as ex:
    print(f"[ERROR] marketing.py | get_value | This should not happen: {ex}")
  
def needs_to_be_queried(results):
  
  try:
    if get_value(results) is None:
      return FBM_NEEDS_QUERY
    return FBM_LOADED_DONE
  except Exception as ex:
    print(f"[ERROR] marketing.py | needs_to_be_queried | {ex}")
    
def reset_start_max_quota_wait():
  tstart = datetime.datetime.now()
  tmax = tstart + datetime.timedelta(hours=FBM_HQUOTA, minutes=5)
  return tstart, tmax

def get_seconds_to_wait(tmax):
  ## due to API constraints
  tnow = datetime.datetime.now()
  delta = tmax-tnow
  return delta
    
def fbconnect(tokens, tokenid, ts):
  ### TOKENS
  token = tokens[tokenid]
  access_token = token['access_token']
  app_secret = token['app_secret']
  ad_account_id = token['ad_account_id']
  app_id = token['app_id']
  
  ### CONNECT
  fbm = FacebookMarketing(app_id=app_id,
                          app_secret=app_secret,
                          access_token=access_token,
                          ad_account_id=ad_account_id,
                          timestamp=f"{ts}")
  fbm.init()
  return fbm



def get_cache_fn(lon,lat,radius,unit,specific_params,cache_dir):
  ''' 
  ''' 
  fn = "FBM_LO{}-LA{}-R{}-U{}-<PARAMS>".format(lon,lat,radius,unit)

  if 'education_statuses' in specific_params:
    tmp = 'EDU{}'.format("_".join([str(s) for s in specific_params['education_statuses']]))
  elif 'user_os' in specific_params:
    tmp = 'OS{}'.format("_".join(["".join([s[:3] for s in o.lower().split(" ")]) for o in specific_params['user_os']]))
  elif 'user_device' in specific_params:
    tmp = 'DEV{}'.format("_".join(["".join([s[:3] for s in o.lower().split(" ")]) for o in specific_params['user_device']]))
  elif 'industries' in specific_params:
    tmp = "IND{}".format("_".join([str(v['id']) for v in specific_params['industries']]))
  elif 'life_events' in specific_params:
    tmp = "LIF{}".format("_".join([str(v['id']) for v in specific_params['life_events']]))
  elif 'behaviors' in specific_params:
    tmp = "BEH{}".format("_".join([str(v['id']) for v in specific_params['behaviors']]))
  elif 'interests' in specific_params:
    tmp = "INT{}".format("_".join([str(v['id']) for v in specific_params['interests']]))

  fn = fn.replace("<PARAMS>",tmp)
  fn = os.path.join(cache_dir, fn)
  return fn

def load_cache_reach_estimates(lon,lat,radius,unit,specific_params,cache_dir):
  '''
  '''
  fn = get_cache_fn(lon,lat,radius,unit,specific_params,cache_dir)
  try:
    if ios.exists(fn):
      return ios.read_list_of_json(fn, verbose=False)
    return []
  except Exception as ex:
    print(ex)
    return None

def cache_reach_estimate(result,lon,lat,radius,unit,specific_params,cache_dir):
  '''
  '''
  fn = get_cache_fn(lon,lat,radius,unit,specific_params,cache_dir)
  ios.save_json(dict(result), fn, mode='a', verbose=False)
  
  
# type:behaviors
# id:6002714895372  	 name:Frequent Travelers
# id:6002714898572  	 name:Small business owners
# id:6013516370183  	 name:Commuters 
# id:6015559470583  	 name:Lives abroad
# id:6022788483583  	 name:Frequent international travelers
# id:6017253486583  	 name:Facebook access (network type): 2G
# id:6017253511583  	 name:Facebook access (network type): 3G
# id:6017253531383  	 name:Facebook access (network type): 4G
# id:6004383149972  	 name:Facebook access (mobile): feature phones
# id:6004382299972  	 name:Facebook access (mobile): all mobile devices
# id:6004383049972  	 name:Facebook access (mobile): smartphones and tablets
# id:6004854404172  	 name:Facebook access: older devices and OS
# id:6015235495383  	 name:Facebook access (network type): WiFi 
# id:6003808923172  	 name:Technology early adopters
# id:6015547847583  	 name:Facebook access (browser): Firefox
# id:6015547900583  	 name:Facebook access (browser): Chrome
# id:6015593608983  	 name:Facebook access (browser): Safari
# id:6015593652183  	 name:Facebook access (browser): Opera
# id:6015593776783  	 name:Facebook access (browser): Internet Explorer
# id:6008261969983     name:Returned from travels 1 week ago

# type:life_events
# id:6003053860372  	 name:Away from hometown
# id:6003053857372  	 name:Away from family
# id:6002714398772     name:Newly-engaged (6 months)
# id:6003050210972	   name:Newly engaged (1 year)
# id:6012631862383    name:Newly engaged (3 months)

# type:industries
# id:6019621029983  	 name:Government Employees (Global)
# id:6009003307783  	 name:Business and Finance 
# id:6008888961983  	 name:IT and Technical Services
# id:6008888972183  	 name:Legal Services

# type:user_device
# name:iPad, ipad, iphone, iPhone, ipod, iPod
# name:android, Android_Smartphone, Android_Tablet
# name:motorola
# name:amazon
# name:nokia
# name:microsoft

# type:user_os
# name:user_os                                           	 platform:Android        	 type:user_os
# name:user_os                                           	 platform:Windows        	 type:user_os
# name:user_os                                           	 platform:Windows Phone  	 type:user_os
# name:user_os                                           	 platform:iOS            	 type:user_os

# type:education_statuses
# 1: HIGH_SCHOOL
# 2: UNDERGRAD
# 3: ALUM
# 4: HIGH_SCHOOL_GRAD
# 5: SOME_COLLEGE
# 6: ASSOCIATE_DEGREE
# 7: IN_GRAD_SCHOOL
# 8: SOME_GRAD_SCHOOL
# 9: MASTER_DEGREE
# 10: PROFESSIONAL_DEGREE
# 11: DOCTORATE_DEGREE
# 12: UNSPECIFIED
# 13: SOME_HIGH_SCHOO


# type:interests
# id:6003248338072      name:Casino games
# id:6003012317397      name:Gambling




