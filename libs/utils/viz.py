################################################################
# Dependencies: System
################################################################
import collections
import os
import string
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pandas.plotting import scatter_matrix
from sklearn.metrics import confusion_matrix

################################################################
# Dependencies: Local
################################################################

from data_utilities import wv_util as wv
from data_utilities import aug_util as au
from utils.constants import *

################################################################
# Functions: Images
################################################################

def imgshow_big(path):
  import cv2
  import matplotlib.pyplot as plt
  #%matplotlib inline

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()

def imgshow(fn):
  image = mpimg.imread(fn) 
  plt.gca().clear()
  plt.imshow(image)
  plt.axis("off")
  plt.show()
  plt.close()

def imglblshow(fn_img, bboxes, classes, allclasses, showlabels):
  imgarr = wv.get_image(fn_img)
  
  ax = plt.subplot(1,1,1)
  plt.axis('off')
  #xmin,ymin,xmax,ymax
  labelled = au.draw_bboxes(imgarr,bboxes)
  plt.imshow(labelled)
  
  if showlabels:
    for bbox,label in zip(*(bboxes,classes)):
      ax.text(s=allclasses[label], x=bbox[0]+1, y=bbox[1]-1, fontsize=12, bbox={'facecolor': 'red', 'pad': 1, 'edgecolor':'none'})

  plt.show()
  plt.close()

def convert_tif_to_jpg(fn):
    im = Image.open(fn)
    name = str(fn).rstrip(".tif")
    print(name)
    name = name + '.jpg'
    print(name)
    im.save(name, 'JPEG')
    return name

def write_path_images(path, fn):
    try:
        content = "\n".join([os.path.abspath(os.path.join(path,fn)) for fn in os.listdir(path)])
        with open(fn, 'w') as f:
            f.write(content)
    except Exception as ex:
        print(ex)
        return 
    return fn

################################################################
# Functions: Distributions
################################################################

def plot_scatter_matrix(df):
  scatter_matrix(df)
  plt.show()
  plt.close()

def plot_correlation(df, x, y, ylog=False, xlog=False, fn=None):
  from scipy.stats import pearsonr
  from scipy.stats import spearmanr

  #covariance = np.covariance(df[x],df[y])
  corrp, pvp = pearsonr(df[x],df[y])
  corrs, pvs = spearmanr(df[x],df[y])

  plt.scatter(df[x], df[y])
  plt.xlabel(x)
  plt.ylabel(y)

  plt.text(s='Person corr: {}, p-value:{}'.format(round(corrp,2),round(pvp,5)), x=df[x].min(), y=df[y].max()-df[y].std())
  plt.text(s='Spearmans corr: {}, p-value:{}'.format(round(corrs,2),round(pvs,5)), x=df[x].min(), y=df[y].max()-(df[y].std()*1.5))
  
  if ylog:
    plt.yscale('log')
  if xlog:
    plt.xscale('log')

  if fn is not None:
    plt.tight_layout()
    plt.savefig(fn, bbox_inches='tight')
    print("{} saved!".format(fn))
  
  plt.show()
  plt.close()


def plot_distribution(df, column, nbins=10, labels=None, quantiles=False, ylog=False, xlog=False, fn=None, show=True):

  ### how to bin the data
  if quantiles:
    fnc = pd.qcut
  else:
    fnc = pd.cut

  ### bining the data
  if quantiles:
    out, bins = fnc(df[column], q=nbins, labels=labels, retbins=True, precision=0)
  else:
    out, bins = fnc(df[column], bins=nbins, retbins=True, include_lowest=True, precision=0, right=False)
  
  ### counting data per bin
  counts = out.value_counts(sort=False)

  ### plot
  ax = counts.plot.bar(rot=0, color="b", figsize=(6,4), width=0.8)
  ax.set_xticklabels([xl.get_text() for xl in ax.get_xticklabels()], rotation=90 if labels is None else 0)
  
  ### labels (categories per bin)
  labels = list(string.ascii_uppercase)[:nbins]
  for x, lbl in enumerate(labels):
      plt.text(s=lbl, x=x, y=counts[x]+1)

  plt.title('{} distribution'.format(column))
  if ylog:
    plt.yscale('log')
  if xlog:
    plt.xscale('log')

  if fn is not None:
    plt.tight_layout()
    plt.savefig(fn)
    print("{} saved!".format(fn))

  if show:  
    plt.show()
  plt.close()

def plot_counts(df, x, row, xindex=False, xlabel=None, ylabel=None, fn=None):
  
  rows = df[row].unique()
  fig,axes = plt.subplots(rows.size, 1, figsize=(6,6))
  data = df.copy()

  for r,val in enumerate(rows):
    tmp = data.query("{}==@val".format(row))[x].value_counts()
    tmpy = tmp.values
    if xindex:
      tmpx = np.arange(1,tmp.shape[0]+1)
    else:
      tmpx = tmp.index.values
    if rows.size > 1:
      ax = axes[r]
    else:
      ax = axes
    ax.bar(x=tmpx, height=tmpy, width=0.5, align='center')
    mit = int(tmpx.shape[0]/2)
    ax.text(s='{}={}'.format(row,val), x=tmpx[mit], y=tmpy[mit]+5)
    if ylabel:
      ax.set_ylabel(ylabel)

  if xlabel:
    ax.set_xlabel(xlabel)

  plt.show()
  plt.close()

def plot_bin_distribution(out, labels=None, title='', ylog=True):
  
  counts = out.value_counts(sort=False)
  ax = counts.plot.bar(rot=0, color="b", figsize=(6,4), width=0.8)
  ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
  
  if labels is not None:
    for x, lbl in enumerate(labels):
      plt.text(s=lbl, x=x, y=counts[x]+1)

  plt.title(title)
  if ylog:
    plt.yscale('log')
  plt.show()
  plt.close()


def plot_data_simple(df, x, y, hue, xlog=False):
  fig,ax = plt.subplots(1,1,figsize=(4,4))
  for group, tmp in df.groupby(hue):
    ax.scatter(x=tmp[x], y=tmp[y], alpha=0.5, label='{}={}'.format(hue,group))
  ax.set_xlabel(x)
  ax.set_ylabel(y)
  if xlog:
    ax.set_xscale('log')
  plt.legend()
  plt.show()
  plt.close()

def plot_data_multiple(df, y, cols, hue, nc=4):
  nr = int(np.ceil(len(cols) / nc))
  data = df.copy()

  fig,axes = plt.subplots(nr,nc,figsize=(nc*3,nr*2), sharey=True)
  r = 0
  c = 0
  for x in cols:
    for group, tmp in data.groupby(hue):
      
      if tmp.loc[:,x].dtype != 'O' and tmp[x].max() > 100:
        # to correct for log(0)
        tmp.loc[:,x] = tmp.apply(lambda row:0.1 if row[x] == 0 else row[x], axis=1)

      axes[r,c].scatter(x=tmp[x], y=tmp[y], alpha=0.5, label='{}={}'.format(hue,group))

      if tmp.loc[:,x].dtype != 'O' and tmp[x].max() > 100:
        axes[r,c].set_xscale('log')

    axes[r,c].set_xlabel(x)
    axes[r,c].set_ylabel(y)
    c+=1
    if c == nc:
      r+=1
      c=0
  axes[0,0].legend()
  fig.tight_layout()
  plt.show()
  plt.close()

################################################################
# Functions: Regression performance
################################################################

def plot_pred_true(pred, true, metrics, fn=None):
  from sklearn.metrics import mean_squared_error
  from sklearn.metrics import mean_absolute_error
  from sklearn.metrics import r2_score
  from scipy.stats import pearsonr
  
  metrics_per_output = {}
  with sns.plotting_context("paper", font_scale=FONT_SCALE):
    cols = pred.shape[1]
    fig,axes = plt.subplots(1,cols,figsize=(cols*3.5, 3))
    titles = ['Mean','Standard Deviation'] if cols == 2 else ['Mean']
    
    corr_pred, pv_pred = pearsonr(pred.T[0],pred.T[1])
    corr_true, pv_true = pearsonr(true.T[0],true.T[1])
    
    for c in np.arange(cols):
      ctrue = true.T[c]
      cpred = pred.T[c]
      
      # for each output
      mae = mean_absolute_error(ctrue,cpred)
      mse = mean_squared_error(ctrue,cpred,squared=True)
      rmse = mean_squared_error(ctrue,cpred,squared=False)
      r2 = r2_score(ctrue,cpred)
      
      metrics_per_output[f'y{c}_mae'] = float(mae)
      metrics_per_output[f'y{c}_mse'] = float(mse)
      metrics_per_output[f'y{c}_rmse'] = float(rmse)
      metrics_per_output[f'y{c}_r2'] = float(r2)
      
      m = max([np.max(cpred),np.max(ctrue)])
      ax = axes if cols == 1 else axes[c]
        
      h=ax.scatter(x=cpred, y=ctrue,  color='blue', alpha=0.5)
      ax.plot([0,m],[0,m],lw=0.5,ls='--',c='k')
      ax.set_title(titles[c])
      ax.set_xlabel("Predicted")
      ax.text(s=f"MAE={mae:.2f}\nMSE={mse:.2f}\nRMSE={rmse:.2f}\nR2={r2:.2f}", va='top', ha='left', x=0, y=np.max(ctrue))
      
      if c==0:
        ax.set_ylabel("True")
      else:
        ax.set_ylabel('')
      
        ### Legend (general all outputs)
        if c==cols-1:
          text = []
          for k,v in metrics.items():
            print(k, v)
            if v is not None:
              try:
                k = k.name
              except:
                pass
              text.append(f"{k} = {round(v,2)}")
          
          text.append(f"Corr_true={corr_true:.2f},{pv_true:.2f}")
          text.append(f"Corr_pred={corr_pred:.2f},{pv_pred:.2f}")
          ax.legend([h]*len(text), text, title='Evaluation', markerscale=0, handlelength=0, loc='center left', bbox_to_anchor=(1, 0.5))
      
    #plt.suptitle("Out-of-sample Performance", y=1.05)
      
    if fn is not None:
      plt.savefig(fn, bbox_inches='tight')
      print('{} saved!'.format(fn))

    plt.close()
    return metrics_per_output

################################################################
# Functions: Classification performance
################################################################

def plot_confusion_matrix(ytest, ypred, labels=None, norm=False, vmin=None, vmax=None, cbar=True, fn=None):
    ses = False
    
    if labels is None:
      labels = sorted(set(ytest) | set(ypred)) 
      
    if ('poor' in set(ytest) or 'poor' in set(ypred)) and len(labels)==4:
      ses = True
      labels = [l for l in ['poor','lower_middle','upper_middle','rich'] if l in labels]

    cf_matrix = confusion_matrix(ytest, ypred, labels=labels, normalize='true' if norm else None) # true, pred
    
    with sns.plotting_context("paper", font_scale=FONT_SCALE):
      w = len(labels)*(2 if cbar else 1.6)
      h = len(labels)*1

      plt.figure(figsize=(w,h))
      ax = sns.heatmap(cf_matrix, annot=True, fmt='.2g', cmap='Blues', vmin=vmin, vmax=vmax, cbar=cbar)
      ax.set_xlabel('Pred')
      ax.set_ylabel('True')
      
      if type(labels[0]) == str and ses:
        if len([1 for l in labels if len(l)>5]) > 0:
          labels = [''.join([w[0].upper() for w in l.split('_')]) for l in labels]
          

      ax.set_xticklabels(labels)
      ax.set_yticklabels(labels, rotation=90 if ses else 0)

      if fn is not None:
        plt.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))

      plt.show()
      plt.close()


################################################################
# Functions: CNN performance
################################################################

def plot_learning_curves(history,fn=None):
  '''
  Metric values per epoch
  '''
  
  print(f"metrics: {history.history.keys()}")
  
  metrics = set([m.replace("val_","") for m in history.history.keys()])
  
  fig,axes = plt.subplots(len(metrics),1,figsize=(6,len(metrics)*2),sharex=True,sharey=False)
  for i, k in enumerate(metrics):
    ax = axes if len(metrics)==1 else axes[i]
    
    # train
    ax.plot(history.history[k], color='blue', label='train')
    
    # val
    key = f'val_{k}'
    if key in history.history:
      ax.plot(history.history[key], color='orange', label='val')
    
    ax.set_title(k)
  plt.legend()
  
  if fn is not None:
    plt.savefig(fn, bbox_inches='tight')
    print('{} saved!'.format(fn))

  plt.close() 


################################################################
# Functions: XGBoost performance
################################################################

def plot_xgboost_feature_importance(model, figsize=(10, 10), max_num_features=30, height=0.7, fn=None):
  ### https://towardsdatascience.com/be-careful-when-interpreting-your-features-importance-in-xgboost-6e16132588e7
  from xgboost import plot_importance
  fig, ax = plt.subplots(figsize=figsize)
  plot_importance(model, max_num_features=max_num_features, height=height, ax=ax, importance_type=XGBOOST_IMPORTANCE_TYPE)
  ax.grid(False)

  if fn is not None:
    fig.savefig(fn, bbox_inches='tight')
    print('{} saved!'.format(fn))

  plt.close()

def plot_feature_importance(df_importance, figsize=(10, 10), max_num_features=30, fn=None):
  tmp = df_importance.nlargest(max_num_features, 'importance')
  fig, ax = plt.subplots(figsize=figsize)
  ax = sns.barplot(x="importance", y="feature_name", data=tmp, orient='h', ax=ax, color=BLUE)
  
  ax.set_xlabel(f"Relative importance ({XGBOOST_IMPORTANCE_TYPE})")
  ax.set_ylabel("Features")

  if fn is not None:
    fig.savefig(fn, bbox_inches='tight')
    print('{} saved!'.format(fn))

  plt.close()

