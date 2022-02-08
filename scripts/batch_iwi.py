###############################################################################
# Dependencies
###############################################################################
import os
import glob
import time
import argparse

from census.dhsmis import DHSMIS
from utils import viz
from utils import ios

###############################################################################
# Functions
###############################################################################

def run(root, code, years, njobs=1):
  print("\n1. Loading data...")
  country = DHSMIS(root, code)
  country.load_surveys(years)
  print(country.df_survey.head())
  print("- shape survey: {}".format(country.df_survey.shape))
  print("- shape cluster: {}".format(country.df_cluster.shape))
  print("- years survey: {}".format(country.df_survey.groupby('year').size()))
  print("- years cluster: {}".format(country.df_cluster.groupby('DHSYEAR').size()))
  
  print("\n2. Computing IWI per household and cluster...")
  country.compute_IWI(njobs)
  print(country.df_survey.head())
  print(country.df_cluster.head())

  print("\n3. Assigning SES scores")
  country.set_categories()
  print(country.df_cluster.head())
  
  print("\n4. Saving...")
  save_results(root, country)

def save_results(root, country):
  # prefix = "_".join(["".join(obj['survey'].split("/")[-4:-2]) for obj in dict_fn_data.values()])
  prefix = ios.get_prefix_surveys(country.df_survey)
  print(prefix)

  # household iwi
  fn = os.path.join(root,"results","features","households","{}_iwi_household.csv".format(prefix))
  ios.save_csv(country.df_survey, fn)

  # cluster iwi
  fn = os.path.join(root,"results","features","clusters","{}_iwi_cluster.csv".format(prefix))
  ios.save_csv(country.df_cluster, fn)
    
  # plots
  labels = ['poor','lower_middle','upper_middle','rich']

  fn = os.path.join(root,"results","plots","iwi_distribution_per_household.pdf")
  viz.plot_distribution(country.df_survey, 'iwi', quantiles=False, nbins=10, ylog=False, fn=fn, show=False)

  fn = os.path.join(root,"results","plots","iwi_distribution_per_household_quantiles.pdf")
  viz.plot_distribution(country.df_survey, 'iwi', quantiles=True, nbins=len(labels), labels=labels, ylog=False, fn=fn, show=False)

  fn = os.path.join(root,"results","plots","iwi_distribution_per_cluster.pdf")
  viz.plot_distribution(country.df_cluster, 'mean_iwi', quantiles=False, nbins=10, ylog=False, fn=fn, show=False)

  fn = os.path.join(root,"results","plots","iwi_distribution_per_cluster_quantiles.pdf")
  viz.plot_distribution(country.df_cluster, 'mean_iwi', quantiles=True, nbins=len(labels), labels=labels, ylog=False, fn=fn, show=False)

###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", help="Country's main folder.", type=str, required=True)
    parser.add_argument("-c", help="Country code name: UG, SL.", type=str, required=True)
    parser.add_argument("-y", help="Year or years separated by comma (E.g. 2016,2019).", type=str, required=True)
    parser.add_argument("-n", help="N parallel jobs.", type=int, default=1, required=False)
    
    args = parser.parse_args()
    for arg in vars(args):
      print("{}: {}".format(arg, getattr(args, arg)))

    start_time = time.time()
    run(args.r, args.c, args.y)
    print("--- %s seconds ---" % (time.time() - start_time))
