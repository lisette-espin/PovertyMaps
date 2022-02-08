### https://github.com/ameasure/IPUMS-helper

from xml.dom import minidom
import gzip
import pandas as pd
from joblib import Parallel
from joblib import delayed

def get_key(fn):
  key = ''
  with open(fn,'r') as f:
    key = f.readlines()[0].strip('').replace('\n','')
  return key

def _convert_to_df(row):
  return pd.DataFrame(row, index=[1])

def convert_to_df(rows, njobs=1):
  results = Parallel(n_jobs=njobs)(delayed(_convert_to_df)(row) for row in rows)
  return pd.concat(results, ignore_index=True)

def row_generator(datapath, ddipath):
    ''' Maps each line of the data file to the variables and values
        it represents '''
    
    # get mapping
    pmap = pos_map(ddipath)
    if datapath.endswith(".gz"):
      f = gzip.open(datapath, 'r')
    else:
      f = open(datapath, 'r')

    for line in f:
        # apply mapping
        row = {}
        for var in pmap.keys():
            start = pmap[var]['spos']
            end = pmap[var]['epos']
            dec = pmap[var]['dec']
            if dec:
                mid = end - dec
                row[var] = line[start:mid].decode("utf-8") +"."+ line[mid:end].decode("utf-8")
            else:
                row[var] = line[start : end].decode("utf-8")
        # yield mapping
        yield row


def pos_map(ddipath):
    ''' Returns a dictionary mapping the variable names to their positions
        and decimal places in the data file '''
    m = minidom.parse(ddipath)    
    vmap = {}
    varNodes = m.getElementsByTagName('var')
    for varNode in varNodes:
        locNode = varNode.getElementsByTagName('location')[0]
        name = varNode.attributes.getNamedItem('ID').value
        vmap[name] = {
            'spos' : int(locNode.attributes.getNamedItem('StartPos').value) - 1,
            'epos' : int(locNode.attributes.getNamedItem('EndPos').value),
            'dec' : int(varNode.attributes.getNamedItem('dcml').value)
            }
    return vmap
