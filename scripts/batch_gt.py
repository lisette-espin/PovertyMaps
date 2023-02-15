###############################################################################
# Dependencies
###############################################################################
import os
import glob
import time
import argparse

from census.dhsmis import DHSMIS
from census.enemdu import ENEMDU
from census.ingatlan import INGATLAN

from utils import viz
from utils import ios
from utils import validations

###############################################################################
# Functions
###############################################################################

def get_instance(root, code, years, **kwargs):
  if code in ['SL','UG','ZW']:
    return DHSMIS(root,code, years, **kwargs)
  if code in ['EC']:
    return ENEMDU(root,code, years, **kwargs)
  if code in ['HU']:
    validations.validate_grid_size(**kwargs)
    return INGATLAN(root,code, years, **kwargs)
  raise Exception("code does not exist.")

def run(root, code, years, njobs=1, **kwargs):
  validations.validate_not_empty(root,'root')
  country = get_instance(root, code, years, **kwargs)
  
  # @TODO: load clusters if file exists
  
  print("\n1. Loading data...")
  country.load_data()

  print("\n2. Computing wealth per household and cluster...")
  country.commpute_indicators(njobs)
  country.rename_columns()

  print("----- survey/points -----")
  print(country.df_survey.head())
  print("----- clusters/agg -----")
  print(country.df_cluster.head())
  
  if country.df_survey.shape[0] > 0:
    print("- shape survey: {}".format(country.df_survey.shape))
    print("- years survey: {}".format(country.df_survey.groupby('year').size()))
  print("- shape cluster: {}".format(country.df_cluster.shape))
  print("- years cluster: {}".format(country.df_cluster.groupby('year').size()))

  print("\n3. Assigning SES scores")
  country.set_categories()
  print(country.df_cluster.head())
  
  print("\n4. Saving...")
  country.clean_columns()
  save_results(root, country)

def save_results(root, country):
  prefix = ios.get_prefix_surveys(country.df_survey)
  print(prefix)

  # household wealth
  if country.df_survey.shape[0] > 0:
    fn = os.path.join(root,"results","features","households","{}_{}_household.csv".format(prefix,country.indicator))
    ios.save_csv(country.df_survey, fn)

  # cluster wealth
  fn = os.path.join(root,"results","features","clusters","{}_{}_cluster.csv".format(prefix,country.indicator))
  ios.save_csv(country.df_cluster, fn)
    
  # plots
  labels = ['poor','lower_middle','upper_middle','rich']

  # @TODO: add in filename prefix (to distinguish plots per year)
  
  if country.df_survey.shape[0] > 0:
    fn = os.path.join(root,"results","plots",f"{prefix}_{country.col_indicator}_distribution_per_household.pdf")
    viz.plot_distribution(country.df_survey, country.col_indicator, quantiles=False, nbins=10, ylog=False, fn=fn, show=False)

    fn = os.path.join(root,"results","plots",f"{prefix}_{country.col_indicator}_distribution_per_household_quantiles.pdf")
    viz.plot_distribution(country.df_survey, country.col_indicator, quantiles=True, nbins=len(labels), labels=labels, ylog=False, fn=fn, show=False)

  fn = os.path.join(root,"results","plots",f"{prefix}_{country.col_indicator}_distribution_per_cluster.pdf")
  viz.plot_distribution(country.df_cluster, f'mean_{country.col_indicator}', quantiles=False, nbins=10, ylog=False, fn=fn, show=False)

  fn = os.path.join(root,"results","plots",f"{prefix}_{country.col_indicator}_distribution_per_cluster_quantiles.pdf")
  viz.plot_distribution(country.df_cluster, f'mean_{country.col_indicator}', quantiles=True, nbins=len(labels), labels=labels, ylog=False, fn=fn, show=False)

###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", help="Country's main folder.", type=str, required=True)
    parser.add_argument("-c", help="Country code name: UG, SL, EC, HU.", type=str, required=True)
    parser.add_argument("-y", help="Year or years separated by comma (E.g. 2016,2019).", type=str, required=True)
    parser.add_argument("-n", help="N parallel jobs.", type=int, default=1, required=False)
    parser.add_argument("-s", help="Grid size in meters (e.g., for Hungary)", type=float, default=None, required=False)
    parser.add_argument("-f", help="Whether or not to shift the cells", action='store_true')
    
    args = parser.parse_args()
    for arg in vars(args):
      print("{}: {}".format(arg, getattr(args, arg)))

    start_time = time.time()
    run(args.r, args.c, args.y, args.n, grid_size=args.s, shift=args.f)
    print("--- %s seconds ---" % (time.time() - start_time))


#dict_fn_data = get_survey_files(root, years)
# def get_survey_files(root, years):
#   dict_fn_data = {}
#   years = years.strip(" ").replace(" ","").split(",")
#   for year in years:
#     dta = glob.glob(os.path.join(root.replace("\\", ""), "survey/*/{}/*DT/*.DTA".format(year)))
#     shp = glob.glob(os.path.join(root.replace("\\", ""), "survey/*/{}/*FL/*.shp".format(year)))
#     dbf = glob.glob(os.path.join(root.replace("\\", ""), "survey/*/{}/*FL/*.dbf".format(year)))
#     if len(dta)>0 and len(shp)>0 and len(dbf)>0:
#       dict_fn_data[int(year)] = {"survey":dta[0], "shp":shp[0], "dbf":dbf[0]}
#     else:
#       raise Exception("There are no files.")
#   return dict_fn_data
