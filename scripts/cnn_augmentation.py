#export PYTHONPATH=/env/python:/home/leespinn/code/SES-Inference/libs/

###############################################################################
# Dependencies
###############################################################################
import os
import time
import argparse

from ses.images import Augmentation
from utils import system
from utils import validations

###############################################################################
# Functions
###############################################################################

def run(root, years, dhsloc, njobs=1, probaug=None):
  # validation
  validations.validate_not_empty(root,'root')
  
  # data
  aug = Augmentation(root, years, dhsloc, probaug)
  aug.load_data()
  aug.generate(njobs)
      

###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-r", help="Country's main folder.", type=str, required=True)
  parser.add_argument("-years", help="Years (comma-separated): 2016,2019", type=str, required=True)
  parser.add_argument("-dhsloc", help="DHS cluster option (None, cc, ccur, gc, gcur, rc).", type=str, default=None, required=False)
  parser.add_argument("-probaug", help="Probability of augmentation (0, .., 1] ", type=float, required=False, default=None)
  parser.add_argument("-njobs", help="Number of parallel processes.", type=int, required=False, default=1)
  parser.add_argument("-shutdown", help="Python script that shutsdown the server after training.", type=str, default=None, required=False)
    
  args = parser.parse_args()
  for arg in vars(args):
    print("{}: {}".format(arg, getattr(args, arg)))

  start_time = time.time()
  try:
    run(args.r, args.years, args.dhsloc, args.njobs, args.probaug)
  except Exception as ex:
    print(ex)
  print("--- %s seconds ---" % (time.time() - start_time))

  if args.shutdown:
    system.google_cloud_shutdown(args.shutdown)

