###############################################################################
# Dependencies
###############################################################################
import os
import time
import glob
import argparse

import pandas as pd
import numpy as np
import geopandas as gpd

from utils import ios
from maps import geo
from ses import locations as loc
from utils import validations
from utils.constants import RURAL
from utils.constants import OSMID

###############################################################################
# Functions
###############################################################################

def run(root, years, option):
  # validations
  validations.validate_not_empty(root,'root')
  option = loc.validate(option)

  # 1-2-3-4. load clusters and pplaces
  fn_cluster = ios.get_places_file(root, years)
  fn_pplaces = ios.get_places_file(root)
  gdf_cluster_m, gdf_pplaces_m = loc.get_data(fn_cluster, fn_pplaces)
  print('=====================')
  print("clusters", gdf_cluster_m.shape)
  print("pplaces", gdf_pplaces_m.shape)
  
  # 5. identify closest pplace within urban/rural
  df_results, dhs_notchanged = loc.distances(gdf_cluster_m, gdf_pplaces_m, option)
  print('=====================')
  print('df_results', df_results.shape)
  print('dhs_notchanged', len(dhs_notchanged) if dhs_notchanged else None)

  ## 6. keep those within displacement (or keep cluster)
  df_valid = loc.restrict_displacement(df_results)
  
  # 7. group by pplace
  gdf_cluster_m_new = loc.groupby(gdf_cluster_m, gdf_pplaces_m, df_valid, option)

  # summary
  print('=====================')
  print('cluster:',gdf_cluster_m.shape)
  print('pplaces:',gdf_pplaces_m.shape)
  print('results (match):',df_results.shape)
  print('valid:',df_valid.shape)
  print('ignored:', gdf_cluster_m.shape[0]-df_valid.shape[0], (gdf_cluster_m.shape[0]-df_valid.shape[0])*100/gdf_cluster_m.shape[0])
  print(gdf_cluster_m[~gdf_cluster_m.index.isin(df_valid.cluster_id.values)].groupby(RURAL).size())
  print('final:',gdf_cluster_m_new.shape, gdf_cluster_m_new.cluster_id.nunique(), gdf_cluster_m_new[OSMID].nunique(), gdf_cluster_m_new.shape[0])
  print(gdf_cluster_m_new.groupby('cluster_rural').size())
  
  # final number of clusters and remanining pplaces
  cluster_OSMIDs = set(gdf_cluster_m_new[OSMID].values)
  pplace_OSMIDs = set(gdf_pplaces_m[OSMID].values) - cluster_OSMIDs
  print('=====================')
  print('clusters changed: ', len(cluster_OSMIDs), 'remained pplaces: ', len(pplace_OSMIDs))
  print('empty: ', cluster_OSMIDs.intersection(pplace_OSMIDs))

  # 8. save
  path = fn_cluster.split("/clusters/")[0]
  prefix = ios.get_prefix_surveys(df=gdf_cluster_m)
  fn_cluster_new = os.path.join(path, '{}_{}_cluster_pplace_ids.csv'.format(prefix,option))
  ios.save_csv(gdf_cluster_m_new, fn_cluster_new)

  if dhs_notchanged is not None and len(dhs_notchanged) > 0:
    fn_cluster_notchanged = os.path.join(path, '{}_{}_cluster_ids_notchanged.txt'.format(prefix,option))
    ios.write_list_to_txt(dhs_notchanged, fn_cluster_notchanged)


###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", help="Country's main folder.", type=str, required=True)
    parser.add_argument("-y", help="Year or years separated by comma (E.g. 2016,2019).", type=str, required=True)
    parser.add_argument("-o", help="Option: cc (change to closest), ccur (change to closest urban/rural), gc (group closest), gcur (group closest urban/rural).", type=str, required=True)
    
    args = parser.parse_args()
    for arg in vars(args):
      print("{}: {}".format(arg, getattr(args, arg)))
      
    start_time = time.time()
    run(args.r, args.y, args.o)
    print("--- %s seconds ---" % (time.time() - start_time))