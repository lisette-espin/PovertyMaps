### https://developers.facebook.com/docs/marketing-api/audiences/reference/basic-targeting
### https://developers.facebook.com/docs/marketing-api/audiences/reference/advanced-targeting
### https://developers.facebook.com/docs/marketing-api/audiences/reference/targeting-search
### https://developers.facebook.com/docs/marketing-api/reference/reach-estimate/
### create app: https://developers.facebook.com/apps/?show_reminder=true

################################################################################
# Dependencies
################################################################################

import os
import time
import pandas as pd
from facebook_business.adobjects.targetingsearch import TargetingSearch
from facebook_business.adobjects.adaccount import AdAccount
from facebook_business.adobjects.adrule import AdRule
from facebook_business.api import FacebookAdsApi
from facebook_business import session
from facebook_business.exceptions import FacebookRequestError

from utils import ios

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
  LIFEEVENT_ENGAGED = {'life_events':[{'id':6002714398772, 'name':'Newly-engaged (6 months)'},{'id':6003050210972, 'name':'Newly engaged (1 year)'},{'id':6012631862383,'name':'Newly engaged (3 months)'}]}
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
  


  def __init__(self, app_id, app_secret, access_token, ad_account_id):
    self.app_id = app_id
    self.app_secret = app_secret
    self.access_token = access_token
    self.ad_account_id = ad_account_id
    self.fb_session = None
    self.appsecret_proof = None

  def init(self):
    self.fb_session = session.FacebookSession(app_id=self.app_id, app_secret=self.app_secret, access_token=self.access_token)
    self.appsecret_proof = self.fb_session._gen_appsecret_proof()
    FacebookAdsApi.init(access_token=self.access_token)

  def get_reach_estimate(self, lon, lat, radius=1.0, unit=UNIT_KM, specific_params=None, cache_dir='/mydrive/cache'):
    '''
    Potential reach (the number of monthly active people) on Facebook that match 
    the audience you defined through your audience targeting selections.
    '''
    code = -1 #-1 loaded | 0: queried | 1:tryagain (quota) | 2:skip location 
    results = None

    result = load_cache_reach_estimate(lon,lat,radius,unit,specific_params,cache_dir)
    if result is not None:
      return result, code

    fields = []
    params = {'targeting_spec': {'geo_locations':{'custom_locations':[{'latitude': lat, 'longitude': lon, 'radius': radius, 'distance_unit': unit}], 
                                                  "location_types":["home"]},
                                 'age_min':13,'age_max':65},
              'appsecret_proof': self.appsecret_proof}
    if specific_params is not None:
      params['targeting_spec'].update(specific_params)
    
    try:
      results = AdAccount(self.ad_account_id).get_reach_estimate(
                          fields=fields,
                          params=params)
      code = 0
    except FacebookRequestError as fex:
      #code:100  subcode:1487851 (wrong location)
      #code:80004 subcode:2446079 (quota)
      code = 2 if fex.api_error_code() == 100 and fex.api_error_subcode() == 1487851 else 1
      print(fex.api_error_code(), fex.api_error_subcode(), fex.get_message())
      
    if results is not None:
      if len(results) > 1:
        print(id)
        raise Exception("This should not happen, or?")
    
      result = results[0]
      cache_reach_estimate(result,lon,lat,radius,unit,specific_params,cache_dir)

    return result, code
    
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


################################################################################
# Functions
################################################################################

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

def load_cache_reach_estimate(lon,lat,radius,unit,specific_params,cache_dir):
  '''
  '''
  fn = get_cache_fn(lon,lat,radius,unit,specific_params,cache_dir)
  try:
    return ios.load_json(fn, verbose=False)
  except:
    return None

def cache_reach_estimate(result,lon,lat,radius,unit,specific_params,cache_dir):
  '''
  '''
  fn = get_cache_fn(lon,lat,radius,unit,specific_params,cache_dir)
  ios.save_json(dict(result), fn, verbose=False)
  
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




