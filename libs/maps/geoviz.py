import numpy as np
import pandas as pd
import seaborn as sns
import shapefile as shp
import matplotlib.pyplot as plt
from shapely.geometry import Point
from matplotlib.lines import Line2D
from collections import defaultdict
from mpl_toolkits.axes_grid1 import make_axes_locatable

class GeoViz(object):

  def __init__(self, shapefn, dbasefn=None, datapoints=None):
    if dbasefn is None:
      self.sf = shp.Reader(shapefn)
    else:
      fshp = open(shapefn, "rb")
      fdbf = open(dbasefn, "rb")
      self.sf = shp.Reader(shp=fshp, dbf=fdbf)
    self.datapoints = datapoints # (lat,lon) array
    
  def plot(self, title='', x_lim = None, y_lim = None, legendtitle=None, color='red', markersize=10, figsize = (11,9), fn=None):
    '''
    Plot map with lim coordinates
    '''
    fig, ax = plt.subplots(figsize=figsize)

    ### map shape
    id=0
    for shape in self.sf.shapeRecords():
      x = [i[0] for i in shape.shape.points[:]]
      y = [i[1] for i in shape.shape.points[:]]
      ax.plot(x, y, 'k')

      if (x_lim == None) & (y_lim == None):
        x0 = np.mean(x)
        y0 = np.mean(y)
        #plt.text(x0, y0, id, fontsize=10)
      id = id+1

    if (x_lim != None) & (y_lim != None):     
      ax.set_xlim(x_lim)
      ax.set_ylim(y_lim)

    ### points
    if self.datapoints is not None:
      if self.datapoints.shape[1] == 2:
        # only points
        self._plot_only_points(ax,color=color, markersize=markersize, legendtitle=legendtitle)
      elif self.datapoints.shape[1] == 3:
        # point with category
        self._plot_with_one_category(fig,ax)
      elif self.datapoints.shape[1] == 4:
        # point with category and size
        self._plot_with_one_category_and_size(ax)

    ### appearance
    ax.set_axis_off()
    ax.set_title(title)

    ### save
    if fn is not None:
      plt.save(fn)
      print("{} saved!".format(fn))

    plt.show()
    plt.close()

  def _plot_only_points(self, ax, color='red', markersize=10, legendtitle=''):
    # only points
    label = '{} {}'.format(len(self.datapoints),legendtitle)
    plt.scatter(self.datapoints[:,1],self.datapoints[:,0], c=color, label=label, s=markersize)
    plt.legend(loc='upper left')

  def _plot_with_one_category(self, fig, ax):
    categories = np.unique(self.datapoints[:,2])

    if categories.size > 10:
      im = ax.scatter(self.datapoints[:,1],self.datapoints[:,0], c=self.datapoints[:,2], cmap='hot_r') #coolwarm
      divider = make_axes_locatable(ax)
      cax = divider.new_vertical(size="2%", pad=0.25, pack_start=True)
      fig.add_axes(cax)
      fig.colorbar(im, cax=cax, orientation="horizontal", label="{} categories".format(np.unique(self.datapoints[:,2]).size))
    else:
      for cat in categories:
        tmp = self.datapoints[np.where(self.datapoints[:,2]==cat)]
        plt.scatter(tmp[:,1], tmp[:,0], label=cat)
      plt.legend(loc='upper left')

  def _plot_with_one_category_and_size(self, ax):
    categories = np.unique(self.datapoints[:,2])
    for cat in categories:
        tmp = self.datapoints[np.where(self.datapoints[:,2]==cat)]
        plt.scatter(tmp[:,1],tmp[:,0], s=tmp[:,3], label=cat, alpha=0.3)
    lgnd = plt.legend(title = 'Classes')
    lgnd.legendHandles[0]._sizes = [30]
    lgnd.legendHandles[1]._sizes = [30]


  @staticmethod
  def plot_coords(ob):
    x, y = ob.xy
    plt.plot(x, y, 'o', zorder=1)
    plt.show()
    plt.close()

  @staticmethod
  def plot_multiple_coords(obs, lat=None, lon=None, junctions=None, buildings=None):
    
    fig,ax = plt.subplots(1,1,figsize=(6,6))

    if type(obs) == list:
      for i,ob in enumerate(obs):
        x, y = ob.xy
        ax.plot(x, y, '.', label=str(i))
    elif type(obs) == defaultdict:
      for name, ways in obs.items():
        xs = []
        ys = []
        for ob in ways:
          x, y = ob.xy
          xs.extend(x)
          ys.extend(y)
        ax.plot(xs, ys, '.', label=name)

    if lat is not None and lon is not None:
      x, y = Point(lon,lat).xy
      ax.plot(x, y, 'x', color='red', label='centroid', zorder=100000, mew=3)

    if junctions is not None and len(junctions)>0:
      xs = []
      ys = []
      for p in junctions:
        x, y = p.xy
        xs.append(x)
        ys.append(y)
      ax.plot(xs, ys, 'o', label='junctions', color='black')

    if buildings is not None and len(buildings)>0:
      for i,poly in enumerate(buildings):
        x,y = poly.exterior.xy
        ax.fill(x, y, alpha=1, color='grey', ec='none', zorder=10000, label='building' if i==0 else None)
        #ax.plot(x, y, 's', color='#6699cc', alpha=0.5, linewidth=1, solid_capstyle='round', zorder=10000, label='building' if i==0 else None)

    ax.set_aspect('equal', adjustable='box')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    plt.close()