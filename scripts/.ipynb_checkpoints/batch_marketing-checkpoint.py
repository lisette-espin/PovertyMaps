###############################################################################
# Dependencies
###############################################################################
import os
import time
import glob
import argparse
import numpy as np
from itertools import cycle
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

from maps import geo
from maps.geoviz import GeoViz
from maps.osm import OSM
from utils import ios
from facebook.marketing import FacebookMarketing
from ses.data import delete_nonprojected_variables

###############################################################################
# Constants
###############################################################################

SLEEP = 60 * 20 # 45 minutes  

###############################################################################
# Functions
###############################################################################

def run(root, years, tokens_dir):
  # survey data
  fn_places = ios.get_places_file(root, years)
  df_places = ios.load_csv(fn_places, index_col=0)

  # FBM setup
  profiles = get_all_profiles()
  tokens = load_tokens(tokens_dir)
  radius = geo.MILE_TO_M / geo.KM_TO_M
  unit = FacebookMarketing.UNIT_KM
  
  # query API
  is_pplaces = 'LATNUM' not in df_places.columns
  fn_places_new = fn_places.replace(".csv","_FBM.csv")
  id,lat,lon = ('OSMID','lat','lon') if is_pplaces else ('DHSID','LATNUM',"LONGNUM")
  cachedir = os.path.join(root,'cache','FBM{}'.format('PP' if is_pplaces else ''))
  df_places_new = query(df_places, profiles, radius, unit, id, lat, lon, tokens, cachedir, fn_places_new)

  # final save
  # df_places_new.drop(columns=[lat,lon], inplace=True)
  df_places_new = delete_nonprojected_variables(df_places_new, os.path.basename(__file__), True) 
  ios.save_csv(df_places_new, fn_places_new)

def query(df_places, profiles, radius, unit, id, lat, lon, tokens, cache_dir, fn=None):
  df_places = df_places.loc[:,[id, lon,lat]]

  maxtokens = len(tokens)
  tokenids = cycle(np.arange(1,maxtokens+1,1))
  status = defaultdict(int)

  tokenid = next(tokenids)
  fbm = fbconnect(tokens, tokenid)
  counter = 0
  for id, row in tqdm(df_places.iterrows(), total=df_places.shape[0]):
    counter += 1
    #print("========== {} of {} ==========".format(counter, df_places.shape[0]))

    ### QUERY
    for k,v in profiles.items():
      code = None
      while code not in [-1,0,2]:
        result, code = fbm.get_reach_estimate(lon=row[lon], lat=row[lat], radius=radius, unit=unit, specific_params=v, cache_dir=cache_dir) 
        value = None if result is None else result['users']
        df_places.loc[id,k] = value
        status[tokenid] = code == 1

        if code not in [-1,0]:
          print('code:', code)
        elif code == 0:
          print("downloaded!")
        
        if status[tokenid]:
          print("token:{}, rowid:{}, result:fail".format(tokenid, id))
          
          if sum(list(status.values())) == maxtokens:
            status = {k:0 for k,v in status.items()}
            current_time = datetime.now().strftime("%H:%M:%S")
            print('{}: sleeping {}min...'.format(current_time, SLEEP/60))
            
            ### save intermediate results (update)
            if fn is not None:
              ios.save_csv(df_places.drop(columns=[lon,lat], inplace=False), fn , index=True)
            ### sleep 
            time.sleep(SLEEP)

          tokenid = next(tokenids)
          fbm = fbconnect(tokens, tokenid)

      if code==2:
        # skip wrong location
        break

  return df_places

# def get_places_file(root, years):
#   fn_places = None
#   if years in ['',None,np.nan]:
#     fn_places = os.path.join(root,'results','features','pplaces','PPLACES.csv')
#   else:
#     for year in years.strip(" ").replace(" ","").split(","):
#       tmp = glob.glob(os.path.join(root,'results','features','clusters',"*_iwi_cluster.csv"))
#       if len(tmp) > 0:
#         fn_places = tmp[0]
#         if year not in fn_places:
#           raise Exception("Survey file for year not found.")
#       else:
#         raise Exception("No data not found.")
#   return fn_places

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

def fbconnect(tokens, tokenid):
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
                          ad_account_id=ad_account_id)
  fbm.init()
  return fbm

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

###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", help="Country's main folder.", type=str, required=True)
    parser.add_argument("-y", help="Year or years separated by comma (E.g. 2016,2019).", type=str, required=True)
    parser.add_argument("-t", help="Directory where all token files are in the form filenameID.json", type=str, required=True)
    
    args = parser.parse_args()
    for arg in vars(args):
      print("{}: {}".format(arg, getattr(args, arg)))

    start_time = time.time()
    run(args.r, args.y, args.t)
    print("--- %s seconds ---" % (time.time() - start_time))