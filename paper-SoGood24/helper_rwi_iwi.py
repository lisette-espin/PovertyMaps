import os 
import seaborn as sns
import numpy as np
import glob
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr
from scipy import stats
import powerlaw
import matplotlib as mpl
import matplotlib.ticker as mticker
from sklearn.metrics import mean_squared_error
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
import matplotlib.transforms as transforms

import sys
sys.path.append('../libs/')

from utils.constants import COUNTRIES
from utils.constants import PROJ_DEG
from utils.constants import PROJ_MET
from utils import ios
from maps import geo

from handlers import RELOCATION_BEST
from handlers import RECENCY_BEST

import warnings
warnings.simplefilter("ignore", UserWarning)


NORM_MEAN = 'mean'
NORM_RESCALE = 'rescale'
NORM_RESCALE_MEAN = 'rescale_mean'

MCHI = 'M1'
MLEE = 'M2'
MESP = 'M3'

def load_rwi_lee_iwi(country, model, features, rwi_pred_path, lee_pred_path, iwi_pred_path, rescale=False):
    model = model.replace('$','').replace('_','')
    fn_rwi = glob.glob(os.path.join(rwi_pred_path, f"{COUNTRIES[country]['code3'].lower()}_relative_wealth_index.csv"))
    fn_iwi_pred = glob.glob(os.path.join(iwi_pred_path, f"{COUNTRIES[country]['code']}_{model}_*_{features}.csv"))
    fn_iwi_feat = glob.glob(os.path.join(iwi_pred_path, f"{COUNTRIES[country]['code']}_features_{features}.csv"))
    fn_lee_pred = glob.glob(os.path.join(lee_pred_path, country, f"{country}_estimated_wealth_index.csv.zip"))
    
    try:
        # RWI
        gdf_rwi = ios.read_geo_csv(fn_rwi[0], lon='longitude', lat='latitude', index_col=None)
        # print("RWI:\n", gdf_rwi)
    except Exception as ex:
        gdf_rwi = None
        print('[ERROR] RWI', country, ex)
        
    try:
        # IWI
        df_pre = ios.load_csv(fn_iwi_pred[0], index_col=0)
        df_feat = ios.load_csv(fn_iwi_feat[0], index_col=None)
        df_iwi = df_pre.set_index('OSMID').join(df_feat.loc[:,['OSMID','lat','lon']].set_index('OSMID'))
        gdf_iwi = geo.get_GeoDataFrame(df_iwi, lon='lon', lat='lat', crs=PROJ_DEG)
        # print("IWI:\n", gdf_iwi)
        del(df_pre)
        del(df_feat)
        del(df_iwi)
    except Exception as ex:
        gdf_iwi = None
        print('[ERROR] IWI', country, ex)
        
    try:
        # IWI-LEE
        df_lee = ios.load_csv(fn_lee_pred[0], index_col=None)
        gdf_lee = geo.get_GeoDataFrame(df_lee, lon='lon', lat='lat', crs=PROJ_DEG)
        gdf_lee = gdf_lee[['lon','lat','geometry','estimated_IWI']]
        # print("LEE:\n", gdf_lee)
    except Exception as ex:
        gdf_lee = None
        print('[ERROR] LEE', country, ex)
        
    if rescale and gdf_rwi is not None and gdf_iwi is not None:
        do_rescale(gdf_rwi, gdf_iwi)
        
    return gdf_rwi, gdf_lee, gdf_iwi
   


def gini_coefficient(X):
    """Calculate the Gini coefficient of a numpy array."""
    # https://github.com/oliviaguest/gini/blob/master/gini.py
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    if type(X)==pd.Series:
        X = X.to_numpy(copy=True)
    X = X.flatten()
    if np.amin(X) < 0:
        # Values cannot be negative:
        X -= np.amin(X)
    # Values cannot be 0:
    X += 0.0000001
    # Values must be sorted:
    X = np.sort(X)
    # Index per array element:
    index = np.arange(1, X.shape[0] + 1)
    # Number of array elements:
    n = X.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * X)) / (n * np.sum(X)))

def get_suptitle(kind):
    if kind is None:
        return 'RAW VALUES'
    elif kind == NORM_MEAN:
        return 'COLOR MEAN-CENTERED'
    elif kind == NORM_RESCALE:
        return 'COLOR RWI-RESCALED'
    elif kind == NORM_RESCALE_MEAN:
        return 'COLOR RWI-RESCALED & MEAN-CENTERED'
    return ''
    
def plot_maps(metadata, country, fig, axes, **kwargs):
    years = kwargs.pop('years', None)
    title = kwargs.pop('title', None)
    suptitle = kwargs.pop('suptitle', False)
    kind = kwargs.pop('kind', False)
    samecolorbar = kwargs.pop('samecolorbar',False)
    
    plt.suptitle(get_suptitle(kind) if suptitle else '', y=0.75 if title else 0.7)
    
    cmap = kwargs.get('cmap', 'RdBu')
    
    #hue_neg, hue_pos = 172, 35
    #cmap = sns.diverging_palette(hue_neg, hue_pos, 50, 50, 1, center="dark", as_cmap=True)
    
    # my_gradient = LinearSegmentedColormap.from_list('my_gradient', (
    #                 # Edit this gradient at https://eltos.github.io/gradient/#0:00493E-49.5:D0ECE8-50:9D9D9D-50.5:F6E6C1-100:673B07
    #                 (0.000, (0.000, 0.286, 0.243)),
    #                 (0.495, (0.816, 0.925, 0.910)),
    #                 (0.500, (0.616, 0.616, 0.616)),
    #                 (0.505, (0.965, 0.902, 0.757)),
    #                 (1.000, (0.404, 0.231, 0.027))))
    # kwargs['cmap'] = my_gradient
    
    legend = kwargs.pop('legend', True)
    legend_kwds = kwargs.get('legend_kwds', {})
    
    if samecolorbar:
        min_iwi = min(metadata['iwi']['data'][metadata['iwi']['mean']].min(),metadata['lee']['data'][metadata['lee']['mean']].min())
        max_iwi = max(metadata['iwi']['data'][metadata['iwi']['mean']].max(),metadata['lee']['data'][metadata['lee']['mean']].max())
        mean_iwi = max(metadata['iwi']['data'][metadata['iwi']['mean']].mean(),metadata['lee']['data'][metadata['lee']['mean']].mean())
        #std_iwi = max(metadata['iwi']['data'][metadata['iwi']['mean']].std(),metadata['lee']['data'][metadata['lee']['mean']].std())
        kwargs['vmin'] = min_iwi
        kwargs['vmax'] = max_iwi
    
    legend_done = {'rwi':False, 'iwi':False, 'lee':False}
    for key in ['iwi', 'rwi', 'lee']: # iwi first
        obj = metadata[key]
        ax = axes[obj['index']]
        data = obj['data'].copy()
            
        metric = 'iwi' if key=='lee' else key
        
        post = "" if not years else f"\n({years[country][key]})"
        # ax.set_title(f"{metric.upper()} by {obj['source']}{post}" if title else '')
        ax.set_title(obj['source'] if title else '')
        # ax.set_aspect('equal', 'box')
        ax.set_axis_off()
        
        if kind == NORM_MEAN:
            # for each country, using their own wealth scale, we center the white color into the mean of their own wealth scores.
            n, vmin, vcenter, vmax = data.shape[0], data[obj['mean']].min(), data[obj['mean']].mean(), data[obj['mean']].max()
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            data.plot(column=obj['mean'], ax=ax, norm=norm, legend=True, **kwargs)
                
        elif kind == NORM_RESCALE:
            # we rescale RWI scores only, and plot as "raw"
            col = 'rwi_rescaled' if key == 'rwi' else obj['mean']
            data.plot(column=col, ax=ax,  legend=legend, **kwargs)
        
        elif kind == NORM_RESCALE_MEAN:
            # first we rescale RWI scores
            col = 'rwi_rescaled' if key == 'rwi' else obj['mean']
            
            # at this point, all model's results are in IWI domain/scale.
            # 2. for each country, we center the white color into the mean of M3's IWI (Ours)
            if key == 'iwi':
                vcenter = data[col].mean()
            
            vmin, vmax = data[col].min(), data[col].max()
            if samecolorbar:
                vmin = min_iwi
                vmax = max_iwi
                
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            data.plot(column=col, ax=ax, norm=norm, legend=True, **kwargs)
            

        else:
            # here we don'r alter anything
            data.plot(column=obj['mean'], ax=ax, legend=legend,  **kwargs)
        
        # Legend titles
        cbax = [ax2 for ax2 in fig.axes if ax2 not in axes and ax2!=ax][-1] #to get only the colorbar axis
        #cbtitle = 'IWI pred.' if key in ['iwi','lee'] or kind in [NORM_RESCALE, NORM_RESCALE_MEAN] else 'RWI pred.' if kind not in [NORM_RESCALE_MEAN, NORM_RESCALE_MEAN] and key=='rwi' else 'IWI pred.'
        cbtitle = '$IWI$ pred.' if key in ['iwi','lee'] else '$\hat{IWI}$ pred.' if kind in [NORM_RESCALE, NORM_RESCALE_MEAN] and key == 'rwi' else '$RWI$ pred.' if kind not in [NORM_RESCALE_MEAN, NORM_RESCALE_MEAN] and key=='rwi' else 'NA.'
        cby = 1.12 if kind in [NORM_RESCALE, NORM_RESCALE_MEAN] and key == 'rwi' else 1.1
        #cbax.text(s=cbtitle, x=1.9, y=1.1, ha='center', va='top', fontsize=12, color='grey', transform=cbax.transAxes)        
        cbax.text(s=cbtitle, x=1.9, y=cby, ha='center', va='top', fontsize=12, color='grey', transform=cbax.transAxes)        

            
                
def plot_pdf(metadata, country, axes, **kwargs):
    
    for key, obj in metadata.items():
    
        ax = axes[obj['index']]
        data = obj['data']
        values = data[obj['mean']]
        
        values.plot(kind='density', ax=ax)
        
        ax.set_ylabel('p(X=x)' if obj['index']==0 else '')
        ax.spines[['right', 'top']].set_visible(False)
        
        #if obj['index']>0:
        #    ax.set_yticklabels([])
        #ax.set_xticklabels([])
        
        n = values.shape[0]
        m = values.mean()
        s = values.std()
        gc = gini_coefficient(values) * 100
        t = ax.text(s=f'$n=${n}', x=1.0, y=0.48, ha='right', va='bottom', transform=ax.transAxes)
        #t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white'))
        t = ax.text(s=f'$\mu=${m:.2f}, $\sigma=${s:.2f}', x=1.0, y=0.3, ha='right', va='bottom', transform=ax.transAxes)
        #t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white'))
        t = ax.text(s=f'Gini={gc:.2f}', x=1.0, y=0.12, ha='right', va='bottom', transform=ax.transAxes)
        #t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white'))
        
        ax.set_xlim((values.min(), values.max()))
        ax.set_xticklabels([])
    
def plot_ecdf(metadata, country, axes, **kwargs):

    for key, obj in metadata.items():
    
        ax = axes[obj['index']]
        data = obj['data']
        values = data[obj['mean']]
        
        ecdf = ECDF(values) 
        ax.plot(ecdf.x, ecdf.y)
        
        #ax.set_xlabel(key.upper())
        ax.set_ylabel("p(X≤x)" if obj['index']==0 else '')
        ax.spines[['right', 'top']].set_visible(False)
        
        #if obj['index']>0:
        #    ax.set_yticklabels([])
        
        # m = values.mean()
        # s = values.std()
        # gc = gini_coefficient(values)
        # ax.text(s=f'$\mu=${m:.2f}, $\sigma=${s:.2f}', x=1.0, y=0.12, ha='right', va='bottom', transform=ax.transAxes)
        # ax.text(s=f'Gini={gc:.2f}', x=1.0, y=0, ha='right', va='bottom', transform=ax.transAxes)

        ax.set_xlim((values.min(), values.max()))
        # ax.set_xticklabels([])
        
        ## POVERTY LINE: IWI-50 (Headcount 2.00$)
        ## https://hdr.undp.org/system/files/documents/03iwiundpeschborn2013.pdf
        ## https://www.jstor.org/stable/24721406?seq=16
        iwipl = 35
        x_intersection = np.percentile(values, iwipl) # 35th percentile (median)
        y_intersection = np.interp(x_intersection, ecdf.x, ecdf.y)
        ax.plot([x_intersection, x_intersection], [0, y_intersection], ls='--', c='grey')
        ax.plot([0, x_intersection], [y_intersection, y_intersection], ls='--', c='grey')
        ax.scatter([x_intersection], [y_intersection], color='black', zorder=5)
        smooth = 1 if values.max()-values.min()<30 else 2
        ax.text(x_intersection+smooth, y_intersection, f'IWI-{iwipl}\nPoverty Line', fontsize=12, rotation=0, va='top', ha='left')
    
def plot_ccdf(metadata, country, axes, **kwargs):

    for key, obj in metadata.items():
    
        ax = axes[obj['index']]
        data = obj['data']
        values = data[obj['mean']]
        
        ecdf = ECDF(values) 
        ax.plot(ecdf.x, 1-ecdf.y)
        
        ax.set_ylabel("p(X>x)" if obj['index']==0 else '')
        ax.spines[['right', 'top']].set_visible(False)
        
        ax.set_xlim((values.min(), values.max())) 
        # ax.set_ylim(-0.2, 1.2)
        ax.set_yscale("log")
        
        ## POVERTY LINE: IWI-50 (Headcount 2.00$)
        ## https://hdr.undp.org/system/files/documents/03iwiundpeschborn2013.pdf
        # x_intersection = 50
        # y_intersection = np.interp(x_intersection, ecdf.x, ecdf.y)
        # ax.plot([x_intersection, x_intersection], [0, y_intersection], ls='--', c='grey')
        # ax.plot([0, x_intersection], [y_intersection, y_intersection], ls='--', c='grey')
        # ax.scatter([x_intersection], [y_intersection], color='black', zorder=5)
        # ax.text(x_intersection, y_intersection / 2, 'Poverty Line', fontsize=12, rotation=90, va='center', ha='right')
        
        
def plot_powerlaw(metadata, country, axes, **kwargs):

    for key, obj in metadata.items():

        ax = axes[obj['index']]
        data = obj['data']
        values = data[obj['mean']]

        fit = powerlaw.Fit(values, discrete=False)
        fit.plot_ccdf(ax=ax, linewidth=3, label='Empirical Data')
        fit.power_law.plot_ccdf(ax=ax, color='r', linestyle='--', label='Power law fit')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc=3)

        ax.set_xlabel(obj['mean'].upper() if key=='rwi' else key.upper())
        ax.set_ylabel("p(X>x)" if obj['index']==0 else '')
        ax.spines[['right', 'top']].set_visible(False)

        ax.text(s=f"$\\alpha=${fit.power_law.alpha:.2f}", x=1.0, y=1.0, ha='right', va='top', transform=ax.transAxes)
        ax.set_xlim((values.min(), values.max()))
        
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        ax.set_xscale("linear")
        ax.set_yscale("log")
        

def get_metadata_template(gdf_rwi, gdf_lee, gdf_iwi, rescale=False):
    return {'rwi':{'index':0, 'data':gdf_rwi, 'mean':'rwi' if not rescale else 'rwi_rescaled', 'std_col':'error', 'source':'M1', 'metric':'RWI'}, #'Chi et al. 2021'
            'lee':{'index':1, 'data':gdf_lee, 'mean':'estimated_IWI', 'std_col':None, 'source':'M2', 'metric':'IWI'},  #'Lee & Braithwaite 2022'
            'iwi':{'index':2, 'data':gdf_iwi, 'mean':'pred_mean_wi', 'std_col':'pred_std_wi', 'source':'M3', 'metric':'IWI'}} #'Espin-Noboa et al. 2023'

        
def plot_comparison_maps(gdf_rwi, gdf_lee, gdf_iwi, country, output_dir=None, **kwargs):
    figsize = kwargs.pop('figsize', (15,5))
    fig, axes = plt.subplots(1,3,figsize=figsize)
    kind = kwargs.get('kind','raw')
    
    metadata = get_metadata_template(gdf_rwi, gdf_lee, gdf_iwi)
    
    plot_maps(metadata, country, fig, axes,  **kwargs)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.margins(x=0, y=0)
    
    for ax in axes.flatten():
        ax.collections[0].set_rasterized(True)
        
    if output_dir is not None:
        fn = os.path.join(output_dir, f"maps_{COUNTRIES[country]['code3']}_{kind}.pdf")
        fig.savefig(fn, dpi=300, bbox_inches='tight')
        print(f'{fn} saved!')
        
    plt.show()
    plt.close()
    
def do_rescale(gdf_rwi, gdf_iwi):
    # offset = abs(gdf_rwi.rwi.min())
    a, b = gdf_rwi.rwi.min(), gdf_rwi.rwi.max()
    c, d = gdf_iwi.pred_mean_wi.min(), gdf_iwi.pred_mean_wi.max()
    gdf_rwi.loc[:,'rwi_rescaled'] = gdf_rwi.rwi.apply(lambda v: ((v - a) * (d - c) / (b - a)) + c)


def plot_comparison_dist(gdf_rwi, gdf_lee, gdf_iwi, country, output_dir=None, **kwargs):
    title = kwargs.pop('title', None)
    suptitle = kwargs.pop('suptitle', False)
    years = kwargs.pop('years', None)
    
    figsize = kwargs.pop('figsize', (10,5))
    fig, axes = plt.subplots(2,3,figsize=figsize)
    
    metadata = get_metadata_template(gdf_rwi, gdf_lee, gdf_iwi, rescale=True)    
    
    plot_pdf(metadata, country, axes[0,:], **kwargs)
    plot_ecdf(metadata, country, axes[1,:], **kwargs)
    # plot_ccdf(metadata, country, axes[1,:], **kwargs)
    #plot_powerlaw(metadata, country, axes[2,:], **kwargs)

    # minv = []
    # maxv = []
    # for key, obj in metadata.items():
    #     values = obj['data'][obj['mean']]
    #     minv.append(values.min())
    #     maxv.append(values.max())
    # for ax in axes[1,:]:
    #     ax.set_xlim((min(minv), max(maxv))) 
    
    if title: 
        for key, obj in metadata.items():
            post = "" if not years else f"\n({years[country][key]})"
            metric = 'iwi' if key == 'lee' else key
            # axes[0,obj['index']].set_title(f"{metric.upper()} by {obj['source']}{post}" if title else '')
            axes[0,obj['index']].set_title(obj['source'] if title else '')

    from matplotlib import ticker
    import matplotlib
    for key, obj in metadata.items():
        ax = axes[-1,obj['index']]
        ax.set_xlabel('$\hat{IWI}$' if key=='rwi' else '$IWI$')
        
    
    if suptitle:
        plt.suptitle("WEALTH DISTRIBUTION")

        
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1) 
    if output_dir is not None:
        fn = os.path.join(output_dir, f"dist_{COUNTRIES[country]['code3']}.pdf")
        fig.savefig(fn, dpi=300, bbox_inches='tight')
        print(f'{fn} saved!')
    
    plt.show()
    plt.close()
    
    
def get_significance_stars(p_value):
    """
    Returns a string of asterisks representing the statistical significance level of the given p-value.

    Args:
    - p_value (float): The p-value to evaluate.

    Returns:
    - str: A string of asterisks (*) indicating the significance level.
    """
    if p_value <= 0.001:
        return '***'
    elif p_value <= 0.01:
        return '**'
    elif p_value <= 0.05:
        return '*'
    else:
        return ''

def plot_comparison_overlapping_cells(gdf_rwi, gdf_lee, gdf_iwi, country, sources=('iwi','rwi'), output_dir=None, **kwargs):
    suptitle = kwargs.pop('suptitle', False)
    max_distance = kwargs.pop('max_distance', 10)
    
    figsize = kwargs.pop('figsize', (10,5))
    fig, axes = plt.subplots(1,2,figsize=figsize)
    
    sources = ('iwi','rwi') if 'rwi' in sources and 'iwi' in sources else \
              ('iwi','lee') if 'iwi' in sources and 'lee' in sources else \
              ('lee','rwi') if 'rwi' in sources and 'lee' in sources else None
    
    if sources is None or sources[0]==sources[1]:
        # (iwi, rwi), (iwi, lee), (lee, rwi)
        raise Exception("Invalid sources. Allowed any pair of: iwi, rwi, lee")
        
    gdf_other = gdf_rwi if sources[1]=='rwi' else gdf_lee
    gdf_main = gdf_iwi if sources[0]=='iwi' else gdf_lee
    other_col = 'rwi_rescaled' if sources[1]=='rwi' else 'estimated_IWI'
    main_col = 'pred_mean_wi' if sources[0]=='iwi'else 'estimated_IWI'
    other_name = '$\hat{IWI}_{'+MCHI+'}$' if sources[1]=='rwi' else '$IWI_{'+MLEE+'}$'
    main_name = '$IWI_{'+MESP+'}$' if sources[0]=='iwi' else '$IWI_{'+MLEE+'}$'
    
    # spatial join
    tmp = gpd.sjoin_nearest(gdf_other.to_crs(PROJ_MET), gdf_main.to_crs(PROJ_MET), how='inner', 
                            max_distance=max_distance, distance_col='distance')
    tmp.loc[:,'diff'] = tmp.apply(lambda row: row[other_col] - row[main_col], axis=1) 
    # negative: <0 main > other
    # positive: >0 other > main
    
    if tmp.shape[0] != tmp.index.nunique() or tmp.shape[0] != tmp.index_right.nunique():
        print("[WARNING]", tmp.shape[0], tmp.index.nunique(), tmp.index_right.nunique(), "computing mean...")
        
        
    # left scatter
    ax = axes[0]
    
    for color, query in {'blue':"diff>0", 'black':"diff==0", 'red':"diff<0"}.items():
        _t = tmp.query(query)
        ax.scatter(_t[other_col], _t[main_col], color=color, 
                   label=f"{other_name} > {main_name}" if color=='blue' else f'{other_name} < {main_name}' if color=='red' else f"{other_name} = {main_name}")
    
    ax.legend(loc=2)
    ax.set_xlabel(other_name)
    ax.set_ylabel(main_name)
    mi = min(tmp[other_col].min(),tmp[main_col].min())
    ma = max(tmp[other_col].max(),tmp[main_col].max())
    ax.plot([mi,ma],[mi,ma],ls='--', lw=1, c='grey')
    ax.spines[['right', 'top']].set_visible(False)
    
    n = tmp.shape[0]
    ro,pv = pearsonr(tmp[other_col], tmp[main_col])
    ps = get_significance_stars(pv)
    # Gini
    gother = gini_coefficient(tmp[other_col]) * 100
    gmain = gini_coefficient(tmp[main_col]) * 100
    #RMSE
    rmse = mean_squared_error(tmp[other_col], tmp[main_col], squared=False)
    t = ax.text(s='$RMSE=$' + f'{rmse:.2f}', va='bottom', ha='right', transform=ax.transAxes, x=1, y=0.45)
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white'))
    t = ax.text(s='Gini$_{OTHER}=$'.replace('OTHER',other_name.replace('_','').replace('$','').replace('RWI','').replace('IWI','')) + f'{gother:.2f}', va='bottom', ha='right', transform=ax.transAxes, x=1, y=0.35)
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white'))
    t = ax.text(s='Gini$_{MAIN}=$'.replace('MAIN',main_name.replace('_','').replace('$','').replace('RWI','').replace('IWI','')) + f'{gmain:.2f}', va='bottom', ha='right', transform=ax.transAxes, x=1, y=0.25)
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white'))
    # n and correlation
    t = ax.text(s=f'n={n}', va='bottom', ha='right', transform=ax.transAxes, x=1, y=0.15)
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white'))
    t = ax.text(s=f"$\\rho=${ro:.2f}{ps}", va='bottom', ha='right', transform=ax.transAxes, x=1, y=0.05)
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white'))
    
    # right map
    ax = axes[1]
    legend_kwds = kwargs.pop('legend_kwds',None)
    markersize = kwargs.pop('markersize',1)
    cmap='coolwarm_r'
    dmin = tmp['diff'].min()
    dmax = max(tmp['diff'].max(),0.01)
    norm = TwoSlopeNorm(vmin=dmin, vcenter=0, vmax=dmax)
    tmp.to_crs(PROJ_DEG).plot(column='diff', cmap=cmap, legend=True, norm=norm, ax=ax, legend_kwds=legend_kwds,  markersize=markersize)
    gdf_other.plot(color='lightgrey', zorder=0, alpha=0.1, markersize=markersize-(markersize/2.), ax=ax)
    ax.set_aspect('equal', 'box')
    ax.set_axis_off()
    
    # Legend titles
    cbax = [ax2 for ax2 in fig.axes if ax2 not in axes and ax2!=ax][-1] #to get only the colorbar axis
    cbtitle = 'IWI Diff.'
    cbax.text(s=cbtitle, x=1.9, y=1.2, ha='center', va='top', fontsize=12, color='grey', transform=cbax.transAxes)        

            
    if suptitle:
        plt.suptitle(F"OVERLAPPING CELLS WITHIN {max_distance}m", y=0.95)
    
    
    for ax in axes.flatten():
        for collections in ax.collections:
            collections.set_rasterized(True)
        # ax.collections[1].set_rasterized(True)
        
        
    plt.tight_layout()
    if output_dir is not None:
        fn = os.path.join(output_dir, f"samecells_{COUNTRIES[country]['code3']}_{'_'.join(sources)}.pdf")
        fig.savefig(fn, dpi=300, bbox_inches='tight')
        print(f'{fn} saved!')
        
    plt.show()
    plt.close()
    
    
def plot_comparison_admin_maps(gdf_rwi, gdf_lee, gdf_iwi, country, boundary_fn, admin_level, output_dir=None, **kwargs):
    years = kwargs.pop('years', None)
    title = kwargs.pop('title', False)
    suptitle = kwargs.pop('suptitle', False)
    max_distance = kwargs.pop('max_distance', 10)
    kind = kwargs.pop('kind', None)
    legend = kwargs.pop('legend', True)
    samecolorbar = kwargs.pop('samecolorbar',False)
    
    figsize = kwargs.pop('figsize', (15,5))
    fig, axes = plt.subplots(1, 3, figsize=figsize) #, gridspec_kw={'width_ratios': [1,1]}, constrained_layout=True)
    gdf_country = None
    
    metadata = get_metadata_template(gdf_rwi, gdf_lee, gdf_iwi, rescale=True)    

    if samecolorbar:
        try:
            # country
            cc3 = COUNTRIES[country]['code3'].upper()
            fn = boundary_fn.replace("<country>", country).replace('<ccode3>',cc3).replace('<ADM>',str(admin_level))
            gdf_country = gpd.read_file(fn)
            min_iwi = []
            max_iwi = []
            for key in ['iwi','lee']:
                obj = metadata[key]
                col = obj['mean']
                data = obj['data']

                tmp = geo.distribute_data_in_grid(data, gdf_country, column=col, aggfnc='mean', lsuffix='data', how='right')
                min_iwi.append(tmp[col].min())
                max_iwi.append(tmp[col].max())
            
            min_iwi = min(min_iwi)
            max_iwi = max(max_iwi)
            kwargs['vmin'] = min_iwi
            kwargs['vmax'] = max_iwi
        except Exception as ex:
            print(f"[ERROR] {ex}")

        
    try:
        # country
        cc3 = COUNTRIES[country]['code3'].upper()
        fn = boundary_fn.replace("<country>", country).replace('<ccode3>',cc3).replace('<ADM>',str(admin_level))
        gdf_country = gpd.read_file(fn)
        
        for key in ['iwi', 'rwi', 'lee']: # iwi always first
            obj = metadata[key]
            ax = axes[obj['index']]
            
            col = 'rwi_rescaled' if key=='rwi' else obj['mean']
            data = obj['data']
        
            tmp = geo.distribute_data_in_grid(data, gdf_country, column=col, aggfnc='mean', lsuffix='data', how='right')
            
            if kind == NORM_RESCALE_MEAN:
                
                # at this point, all model's results are in IWI domain/scale.
                # 2. for each country, we center the white color into the mean of M3's IWI (Ours)
                if key == 'iwi':
                    vcenter = tmp[col].mean()

                vmin, vmax = tmp[col].min(), tmp[col].max()
                if samecolorbar:
                    vmin = min_iwi
                    vmax = max_iwi

                norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

                # n, vmin, vcenter, vmax, vstd = tmp.shape[0], tmp[col].min(), tmp[col].mean(), tmp[col].max(), tmp[col].std()
                # if samecolorbar:
                #     vmin = min_iwi
                #     vmax = max_iwi
                # norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

                tmp.plot(column=col, ax=ax, norm=norm, legend=legend, **kwargs)
                
            else:
                tmp.plot(column=col, ax=ax, legend=legend, **kwargs)
                
            # Legend titles
            cbax = [ax2 for ax2 in fig.axes if ax2 not in axes and ax2!=ax][-1] #to get only the colorbar axis
            cbtitle = '$IWI$ pred.' if key in ['iwi','lee'] else '$\hat{IWI}$ pred.' if kind in [NORM_RESCALE, NORM_RESCALE_MEAN] and key == 'rwi' else '$RWI$ pred.' if kind not in [NORM_RESCALE_MEAN, NORM_RESCALE_MEAN] and key=='rwi' else 'NA.'
            cby = 1.14 if kind in [NORM_RESCALE, NORM_RESCALE_MEAN] and key == 'rwi' else 1.12
            cbax.text(s=cbtitle, x=1.9, y=cby, ha='center', va='top', fontsize=12, color='grey', transform=cbax.transAxes)        

            if title:
                post = "" if not years else f"\n({years[country][key]})"
                metric = 'iwi' if key == 'lee' else key
                # ax.set_title(f"{metric.upper()} by {obj['source']}{post}")
                ax.set_title(obj['source'])
                
            # ax.set_aspect('equal', 'box')
            ax.set_axis_off()
        
        if suptitle:
            ks = get_suptitle(kind)
            ks = '' if ks in ['',None] else f" | {ks}"
            plt.suptitle(f"MEAN WEALTH | ADMIN_LEVEL={admin_level}{ks}", y=0.75 if title else 0.7)
            
        for ax in axes.flatten():
            try:
                ax.collections[0].set_rasterized(True)
            except:
                pass
            
        # plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.tight_layout()
        # plt.margins(x=0, y=0)

        if output_dir is not None:
            fn = os.path.join(output_dir, f"maps_{COUNTRIES[country]['code3']}_{kind}_admin{admin_level}.pdf")
            fig.savefig(fn, dpi=300, bbox_inches='tight')
            print(f'{fn} saved!')

        plt.show()
        plt.close()
        
    except Exception as ex:
        print(ex)
       
    
def show_preliminaries(gini_fn, gdp_fn, survey_years, output_dir=None):
    df_gini = ios.load_csv(gini_fn, index_col=0, verbose=True, skiprows=[0,1,2])
    df_gdp = ios.load_csv(gdp_fn, index_col=0, verbose=True,  skiprows=[0,1,2])
    
    # removing the last column that is added because the files end with a comma (this pandas believes is a column)
    df_gini.drop(columns=[c for c in df_gini.columns if c.startswith('Unnamed')], inplace=True)
    df_gdp.drop(columns=[c for c in df_gdp.columns if c.startswith('Unnamed')], inplace=True)
    
    # collecting gini and gdp per country and year
    df_preliminaries = pd.DataFrame(columns=['country','year','gini','gdp','in_rwi','in_lee','in_iwi'])
    for country, obj in survey_years.items():
        y_rwi = str(obj['rwi']).replace(',','-').split('-') if obj['rwi'] is not None else []
        y_lee = str(obj['lee']).replace(',','-').split('-') if obj['lee'] is not None else []
        y_iwi = str(obj['iwi']).replace(',','-').split('-') if obj['iwi'] is not None else []
        
        y_rwi = sorted(set([int(y) if int(y)>=2000 else int(f"20{y}") for y in y_rwi]))
        y_lee = sorted(set([int(y) if int(y)>=2000 else int(f"20{y}") for y in y_lee]))
        y_iwi = sorted(set([int(y) if int(y)>=2000 else int(f"20{y}") for y in y_iwi]))
        
        years = sorted(set(y_rwi + y_lee + y_iwi))
        # years = sorted(set([int(y) if int(y)>=2000 else int(f"20{y}") for y in years]))

        # GINI
        tmp = df_gini.loc[country].dropna().reset_index().loc[3:].rename(columns={'index':'year', country:'gini'}) # loc 3, unnecesary rows
        tmp.loc[:,'year'] = tmp.year.astype(int)
        ginis = tmp.sort_values('year', ascending=True).reset_index(drop=True).tail(len(years)).reset_index(drop=True)

        # print(country, len(years), ginis.shape, ginis.year.values, years)
        
        # GDP
        for i, y in enumerate(years):
            year = int(y)

            gini = ginis.loc[i, 'gini'] if i < ginis.shape[0] else None
            gdp  = df_gdp[str(year)].get(country, None)
            
            if gini not in [None,np.nan,'']:
                gini = f"{gini} ({ginis.loc[i, 'year']})"

            # gini = df_gini[year].get(country, None)
#             tries = 0
#             if np.isnan(gini):
#                 prev_year = int(year)-1
#                 while np.isnan(gini):
#                     gini = df_gini[str(prev_year)].get(country, None)

#                     if not np.isnan(gini):
#                         gini = f"{gini} ({prev_year})"
#                         break

#                     if tries >= 10:
#                         break

#                     tries += 1
#                     prev_year -= 1

            obj = {'country':country, 'year':year, 'gini':gini, 'gdp':gdp, 'in_rwi':year in y_rwi, 'in_lee':year in y_lee, 'in_iwi':year in y_iwi}
            df_preliminaries = pd.concat([df_preliminaries, pd.DataFrame(obj, index=[1])], ignore_index=True)

    df_preliminaries.rename(columns={'year':'year_survey', 'gini':'_gini'}, inplace=True) 
    df_preliminaries[['actual_gini','actual_year']] = df_preliminaries.apply(lambda row: str(row._gini).split(' (') if '(' in str(row._gini) else [row._gini, row.year_survey], axis=1, result_type="expand")
    df_preliminaries.actual_year = df_preliminaries.actual_year.apply(lambda v: None if v in [None,np.nan,''] else str(int(v)) if type(v)!=str else str(v).replace(')',''))
    df_preliminaries.actual_gini = df_preliminaries.actual_gini.astype(float)
    
    if output_dir is not None:
        fn = os.path.join(output_dir, 'data_preliminaries.tex')
        df_preliminaries.to_latex(fn, index=False, float_format="{:.1f}".format)
        print(f'{fn} saved!')
    
    return df_preliminaries

def plot_preliminaries(df_preliminaries, output_dir=None, **kwargs):
    
    def duplicate(ya,i,y_actual):
        for j,yb in enumerate(y_actual):
            if j<i:
                if ya == yb:
                    return True
        return False
    
    data = df_preliminaries.copy()
    
    suptitle = kwargs.pop('suptitle', False)
    
    metrics = ['gdp','actual_gini']
    xmetrics = ["Survey's year", "WB's year"]
    metrics_str = ['GDP-PC-USD','GINI']
    nrows = len(metrics)
    
    ncountries = data.country.nunique()
    figsize = kwargs.pop('figsize', (5,5))
    col_order = kwargs.pop('col_order', data.country.unique())
    
    fig, axes = plt.subplots(nrows,ncountries,figsize=figsize,sharex=False,sharey=False)
    colors = mpl.cm.tab10(range(10))
    
    # subplot: country
    # groups (legend): survey year
    # x-axis: gini, gdp
    

    width = 0.25  # the width of the bars
    counter = 0
    for col_index, country in enumerate(col_order):  
        axes[0,col_index].set_title(country)
        axes[-1,col_index].set_xlabel("Year")
        
        #axes[-1,col_index].set_xlabel("Survey's year")
        
        df = data.query("country==@country").sort_values('year_survey', ascending=True)
        for row_index, metric in enumerate(metrics):
            ax = axes[row_index, col_index]
            tmp = df[['year_survey',metric,'actual_year']].values
            x, y_survey, x_actual = tmp[:,0].astype(str), tmp[:,1], tmp[:,2].astype(str)
            
            if metric == 'actual_gini':
                bar_plot = ax.bar(x_actual, y_survey, color='grey')
                ax.bar_label(bar_plot, label_type='center', rotation=90, color='white', fmt='%.1f')
                
#                 label_color = {ya:colors[i] for i, ya in enumerate(sorted(set(y_actual)))}
#                 # barcolors = [label_color[ya] for ya in y_actual]
#                 barlabels = [str(ya) if not duplicate(ya,i,y_actual) else f"_{ya}" for i, ya in enumerate(y_actual)]
#                 bar_plot = ax.bar(x, int(y_actual), label=barlabels, color='grey')
#                 ax.bar_label(bar_plot, label_type='edge', padding=-30, color='white', fmt='%.1f')
                # ax.legend(handles=bar_plot, labels=barlabels, title="WB Gini's year", loc='lower left')

            else:
                bar_plot = ax.bar(x, y_survey, color='grey')
                ax.bar_label(bar_plot, label_type='center', rotation=90, color='white', fmt='%.1f')
            
            # ax.set_xlabel(xmetrics[row_index])
            if col_index==0:
                ax.set_ylabel(metrics_str[row_index])
            # if row_index < len(metrics)-1:
                # ax.set_xticklabels([])
                
    # Add some text for labels, title and custom x-axis tick labels, etc.
    if suptitle:
        plt.suptitle('Preliminaries of survey data used as ground-truth')
    
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.25, hspace=0.35)
    
    if output_dir is not None:
        fn = os.path.join(output_dir, 'data_preliminaries.pdf')
        fig.savefig(fn, dpi=300, bbox_inches='tight')
        print(f'{fn} saved!')
        
    plt.show()
    plt.close()
    
    
# def show_expected_differences(df_preliminaries, output_dir):
#     df_expectations = pd.DataFrame(columns=['country','Gini (older) -> (newer)', 'GDP (older) -> (newer)'])
#     ginismooth = 10 # (10) gini 0 to 100
#     gdpsmooth = 0.1 # normalized becasue GDP and IWI, RWI are in different ranges | Old:(100) gdp min 400 max 8000

#     for country, df in df_preliminaries.groupby('country'):
#         id_min = df.year_survey.astype(int).idxmin()
#         id_max = df.year_survey.astype(int).idxmax()
#         ginidiff = df.loc[id_max,'actual_gini']-df.loc[id_min,'actual_gini'] # (df.loc[id_max,'actual_gini']-df.loc[id_min,'actual_gini']) / (df.loc[id_max,'actual_gini']+df.loc[id_min,'actual_gini'])
#         gdpdiff = (df.loc[id_max,'gdp']-df.loc[id_min,'gdp']) / (df.loc[id_max,'gdp']+df.loc[id_min,'gdp'])
        
#         # print(country, df.loc[id_min,'actual_gini'], df.loc[id_max,'actual_gini'], df.loc[id_min,'actual_gini']-df.loc[id_max,'actual_gini'], ginidiff)
#         # print(country, df.loc[id_min,'gdp'], df.loc[id_max,'gdp'], df.loc[id_min,'gdp']-df.loc[id_max,'gdp'], gdpdiff)
#         # print()
        
#         # higher Gini: newer > older 
#         # lower Gini: newer < older 
#         obj = {'country':country, 
#                'Gini (older) -> (newer)': 'higher Gini (more inequality)' if ginidiff>ginismooth else 'lower Gini (less inequality)' if ginidiff<-ginismooth else 'similar Gini',
#                'GDP (older) -> (newer)': 'higher GDP (richer)' if gdpdiff>gdpsmooth else 'lower GDP (poorer)' if gdpdiff<-gdpsmooth else 'similar GDP'}
#         df_expectations = pd.concat([df_expectations, pd.DataFrame(obj, index=[1])], ignore_index=True)
 
#     df_expectations.set_index('country', inplace=True)
    
#     if output_dir is not None:
#         fn = os.path.join(output_dir, 'data_expectations.tex')
#         df_expectations.to_latex(fn, index=True, float_format="{:.1f}".format)
#         print(f'{fn} saved!')

#     return df_expectations

def show_expected_differences(df_preliminaries, output_dir):
    df_expectations = pd.DataFrame()
    gini_smooth = 10 # (10) gini 0 to 100
    gdp_smooth = 0.1 # normalized becasue GDP and IWI, RWI are in different ranges | Old:(100) gdp min 400 max 8000


    for country, df in df_preliminaries.groupby('country'):

        tmp_rwi = df.query("in_rwi == True")
        tmp_lee = df.query("in_lee == True")
        tmp_iwi = df.query("in_iwi == True")

        # IWI
        iwi_idxmax = tmp_iwi.dropna().actual_year.astype(int).idxmax()
        
        print('IWI:', country, tmp_iwi.dropna().actual_year.max())
        print('RWI:', country, tmp_rwi.dropna().actual_year.max())
        print('LEE:', country, tmp_lee.dropna().actual_year.max())
        
        # RWI vs IWI
        rwi_idxmax = tmp_rwi.dropna().actual_year.astype(int).idxmax()
        gini_diff_rwi_iwi = tmp_rwi.loc[rwi_idxmax,'actual_gini'] - tmp_iwi.loc[iwi_idxmax,'actual_gini']
        str_gini_diff_rwi_iwi = 'higher Gini (more inequality)' if gini_diff_rwi_iwi>gini_smooth else 'lower Gini (less inequality)' if gini_diff_rwi_iwi<-gini_smooth else 'similar or slightly higher Gini' if gini_diff_rwi_iwi > 0 else 'similar or slightly lower Gini' if gini_diff_rwi_iwi < 0 else 'very similar Gini'
        gdp_diff_rwi_iwi = (tmp_rwi.loc[rwi_idxmax,'gdp'] - tmp_iwi.loc[iwi_idxmax,'gdp']) / (tmp_rwi.loc[rwi_idxmax,'gdp'] + tmp_iwi.loc[iwi_idxmax,'gdp'])
        str_gdp_diff_rwi_iwi = 'higher GDP (richer)' if gdp_diff_rwi_iwi>gdp_smooth else 'lower GDP (poorer)' if gdp_diff_rwi_iwi<-gdp_smooth else 'similar or slightly higher GDP' if gdp_diff_rwi_iwi > 0 else 'similar or slightly lower GDP' if gdp_diff_rwi_iwi < 0 else 'similar GDP'

        # IWI-LEE vs IWI
        if tmp_lee.shape[0] > 0:
            lee_idxmax = tmp_lee.dropna().actual_year.astype(int).idxmax()
            gini_diff_lee_iwi = tmp_lee.loc[lee_idxmax,'actual_gini'] - tmp_iwi.loc[iwi_idxmax,'actual_gini']
            str_gini_diff_lee_iwi = 'higher Gini (more inequality)' if gini_diff_lee_iwi>gini_smooth else 'lower Gini (less inequality)' if gini_diff_lee_iwi<-gini_smooth else 'similar or slightly higher Gini' if gini_diff_lee_iwi > 0 else 'similar or slightly lower Gini' if gini_diff_lee_iwi < 0 else 'very similar Gini'
            gdp_diff_lee_iwi = (tmp_lee.loc[lee_idxmax,'gdp'] - tmp_iwi.loc[iwi_idxmax,'gdp']) / (tmp_lee.loc[lee_idxmax,'gdp'] + tmp_iwi.loc[iwi_idxmax,'gdp'])
            str_gdp_diff_lee_iwi = 'higher GDP (richer)' if gdp_diff_lee_iwi>gdp_smooth else 'lower GDP (poorer)' if gdp_diff_lee_iwi<-gdp_smooth else 'similar or slightly higher GDP' if gdp_diff_lee_iwi > 0 else 'similar or slightly lower GDP' if gdp_diff_lee_iwi < 0 else 'similar GDP'
        else:
            gini_diff_lee_iwi = None
            str_gini_diff_lee_iwi = 'NA'
            gdp_diff_lee_iwi = None
            str_gdp_diff_lee_iwi = 'NA'
            
        obj = {'country':country, 
               'gini_rwi_iwi':str_gini_diff_rwi_iwi,
               'gini_lee_iwi':str_gini_diff_lee_iwi,
               'gdp_rwi_iwi':str_gdp_diff_rwi_iwi,
               'gdp_lee_iwi':str_gdp_diff_lee_iwi,
               'gini_rwi_iwi_value':gini_diff_rwi_iwi,
               'gini_lee_iwi_value':gini_diff_lee_iwi,
               'gdp_rwi_iwi_value':gdp_diff_rwi_iwi,
               'gdp_lee_iwi_value':gdp_diff_lee_iwi,
              }


        df_expectations = pd.concat([df_expectations, pd.DataFrame(obj, index=[0])], ignore_index=True)

    df_expectations.set_index('country', inplace=True)
    
    if output_dir is not None:
        fn = os.path.join(output_dir, 'data_expectations.tex')
        df_expectations.to_latex(fn, index=True, float_format="{:.1f}".format)
        print(f'{fn} saved!')

    return df_expectations

def show_final_summary(df_preliminaries, df_expectations, survey_years, best_models, features_source, rwi_path, lee_pred_path, iwi_pred_path, max_distance, output_dir=None):
    df_summary = pd.DataFrame(columns=['country',
                                       'chi_years', 'chi_n', 'chi_mean', 'chi_std', 'chi_gini',
                                       'lee_years', 'lee_n', 'lee_mean', 'lee_std', 'lee_gini',
                                       'us_years', 'us_n', 'us_mean', 'us_std', 'us_gini',
                                       'diff_chi_gini_expected', 'diff_chi_gini_obtained',
                                       'diff_lee_gini_expected', 'diff_lee_gini_obtained',
                                       'diff_chi_wealth_expected', 'diff_chi_wealth_obtained', 'ks_rwi_iwi', #'ztest_chi_wealth_obtained',
                                       'diff_lee_wealth_expected', 'diff_lee_wealth_obtained', 'ks_lee_iwi', #'ztest_lee_wealth_obtained',
                                       'overlap_chi_n', 'overlap_chi_rmse', 'overlap_chi_mean_rwi',
                                       'overlap_chi_mean_iwi', 'overlap_chi_mean_diff', 'overlap_chi_mean_diff_norm', 'overlap_chi_mean_ztest', 'overlap_chi_mean_ro', 'overlap_chi_gini_rwi', 'overlap_chi_gini_iwi', 'overlap_chi_gini_diff',
                                       'overlap_lee_n', 'overlap_lee_rmse', 'overlap_lee_mean_rwi', 'overlap_lee_mean_iwi', 'overlap_lee_mean_diff', 'overlap_lee_mean_diff_norm', 'overlap_lee_mean_ztest', 'overlap_lee_mean_ro', 'overlap_lee_gini_rwi', 'overlap_lee_gini_iwi', 'overlap_lee_gini_diff'])
    
    for country, df_pre in df_preliminaries.groupby('country'):
        df_ex = df_expectations.loc[country]
        
        try:
            model = best_models[country]
        except Exception as ex:
            print(country, ex)
            model = 'CB'
        print(country, model)

        gdf_rwi, gdf_lee, gdf_iwi = load_rwi_lee_iwi(country, model, features_source, rwi_path, lee_pred_path, iwi_pred_path, rescale=True)
        
        gini_rwi = gini_coefficient(gdf_rwi.rwi_rescaled) * 100
        gini_lee = gini_coefficient(gdf_lee.estimated_IWI) * 100
        gini_iwi = gini_coefficient(gdf_iwi.pred_mean_wi) * 100
        
        # RWI (Chi)
        n_rwi = gdf_rwi.shape[0]
        x_rwi = gdf_rwi.rwi_rescaled.mean()
        s_rwi = gdf_rwi.rwi_rescaled.std()
        sn_rwi = (1/n_rwi) * sum((gdf_rwi.rwi_rescaled-x_rwi)**2)
        
        # IWI (Lee)
        n_lee = gdf_lee.shape[0]
        x_lee = gdf_lee.estimated_IWI.mean()
        s_lee = gdf_lee.estimated_IWI.std()
        sn_lee = (1/n_lee) * sum((gdf_lee.estimated_IWI-x_lee)**2)
        
        # IWI (us)
        n_iwi = gdf_iwi.shape[0]
        x_iwi = gdf_iwi.pred_mean_wi.mean()
        s_iwi = gdf_iwi.pred_mean_wi.std()
        sn_iwi = (1/n_iwi) * sum((gdf_iwi.pred_mean_wi-x_iwi)**2) # variance
        
        ### RWI vs IWI ###
        #ztest_rwi_iwi = (x_iwi - x_rwi) / np.sqrt(((sn_iwi**2)/n_iwi) + ((sn_rwi**2)/n_rwi))
        ks_rwi_iwi, ks_pv_rwi_iwi = stats.ks_2samp(gdf_rwi.rwi_rescaled, gdf_iwi.pred_mean_wi)
        
        # overlap
        df_overlap_rwi = gpd.sjoin_nearest(gdf_rwi.to_crs(PROJ_MET), gdf_iwi.to_crs(PROJ_MET), how='inner', max_distance=max_distance, distance_col='distance')
        df_overlap_rwi.loc[:,'diff'] = df_overlap_rwi.apply(lambda row: row.pred_mean_wi-row.rwi_rescaled, axis=1) # negative: rwi > iwi 
        on_chi = df_overlap_rwi.shape[0]
        ox1_chi = df_overlap_rwi.pred_mean_wi.mean()
        ox2_chi = df_overlap_rwi.rwi_rescaled.mean()
        os1_chi = df_overlap_rwi.pred_mean_wi.std()
        os2_chi = df_overlap_rwi.rwi_rescaled.std()
        ro_chi,pv_chi = pearsonr(df_overlap_rwi.rwi_rescaled, df_overlap_rwi.pred_mean_wi)
        ps_chi = get_significance_stars(pv_chi)
        ogini_iwi_chi =  gini_coefficient(df_overlap_rwi.pred_mean_wi) * 100
        ogini_rwi_chi =  gini_coefficient(df_overlap_rwi.rwi_rescaled) * 100
        osn1_chi = os1_chi / np.sqrt(on_chi)
        osn2_chi = os2_chi / np.sqrt(on_chi)
        oztest_chi = (ox1_chi - ox2_chi) / np.sqrt(osn1_chi**2 + osn2_chi**2)
        rmse_chi = mean_squared_error(df_overlap_rwi.rwi_rescaled, df_overlap_rwi.pred_mean_wi, squared=False)
        
        ### IWI (Lee) vs IWI (us) ###
        # ztest_lee_iwi = (x_iwi - x_lee) / np.sqrt((sn_iwi/n_iwi) + (sn_lee/n_lee))
        ks_lee_iwi, ks_pv_lee_iwi = stats.ks_2samp(gdf_lee.estimated_IWI, gdf_iwi.pred_mean_wi)
        
        # overlap
        df_overlap_lee = gpd.sjoin_nearest(gdf_lee.to_crs(PROJ_MET), gdf_iwi.to_crs(PROJ_MET), how='inner', max_distance=max_distance, distance_col='distance')
        df_overlap_lee.loc[:,'diff'] = df_overlap_lee.apply(lambda row: row.pred_mean_wi-row.estimated_IWI, axis=1) # negative: iwi (lee) > iwi (us) 
        on_lee = df_overlap_lee.shape[0]
        ox1_lee = df_overlap_lee.pred_mean_wi.mean()
        ox2_lee = df_overlap_lee.estimated_IWI.mean()
        os1_lee = df_overlap_lee.pred_mean_wi.std()
        os2_lee = df_overlap_lee.estimated_IWI.std()
        ro_lee,pv_lee = pearsonr(df_overlap_lee.estimated_IWI, df_overlap_lee.pred_mean_wi)
        ps_lee = get_significance_stars(pv_lee)
        ogini_iwi_lee =  gini_coefficient(df_overlap_lee.pred_mean_wi) * 100
        ogini_rwi_lee =  gini_coefficient(df_overlap_lee.estimated_IWI) * 100
        osn1_lee = os1_lee / np.sqrt(on_lee)
        osn2_lee = os2_lee / np.sqrt(on_lee)
        oztest_lee = (ox1_lee - ox2_lee) / np.sqrt(osn1_lee**2 + osn2_lee**2)
        rmse_lee = mean_squared_error(df_overlap_lee.estimated_IWI, df_overlap_lee.pred_mean_wi, squared=False)

        
        obj = {'country': country,
               
               'chi_years': survey_years[country]['rwi'],
               'chi_n': n_rwi,
               'chi_mean': x_rwi,    # rescaled
               'chi_std': s_rwi,     # rescaled
               'chi_gini': gini_rwi, # rescaled
               
               'lee_years': survey_years[country]['lee'],
               'lee_n': n_lee,
               'lee_mean': x_lee,
               'lee_std': s_lee,
               'lee_gini':gini_lee,
               
               'us_years': survey_years[country]['iwi'],
               'us_n': n_iwi,
               'us_mean': x_iwi,
               'us_std': s_iwi,
               'us_gini':gini_iwi,
               
               'diff_chi_gini_expected': df_ex.gini_rwi_iwi_value,
               'diff_chi_gini_obtained': gini_rwi - gini_iwi,
               
               'diff_lee_gini_expected': df_ex.gini_lee_iwi_value,
               'diff_lee_gini_obtained': gini_lee - gini_iwi,
               
               'diff_chi_wealth_expected': df_ex.gdp_rwi_iwi_value,
               'diff_chi_wealth_obtained': (x_rwi - x_iwi) / (x_rwi + x_iwi),
               # 'ztest_chi_wealth_obtained': ztest_rwi_iwi,
               'ks_rwi_iwi':f"{ks_rwi_iwi:.2f} {get_significance_stars(ks_pv_rwi_iwi)}",
               
               'diff_lee_wealth_expected': df_ex.gdp_lee_iwi_value,
               'diff_lee_wealth_obtained': (x_lee - x_iwi) / (x_lee + x_iwi),
               # 'ztest_lee_wealth_obtained': ztest_lee_iwi,
               'ks_lee_iwi':f"{ks_lee_iwi:.2f} {get_significance_stars(ks_pv_lee_iwi)}",
               
               'overlap_chi_n': on_chi,
               'overlap_chi_rmse':rmse_chi,
               'overlap_chi_mean_rwi': ox2_chi,
               'overlap_chi_mean_iwi': ox1_chi,
               'overlap_chi_mean_diff': ox1_chi - ox2_chi,
               'overlap_chi_mean_diff_norm': (ox1_chi - ox2_chi) / (ox1_chi + ox2_chi),
               'overlap_chi_mean_ztest': oztest_chi,
               'overlap_chi_mean_ro': f"{ro_chi:.2f} {ps_chi}",
               'overlap_chi_gini_rwi': ogini_rwi_chi,
               'overlap_chi_gini_iwi': ogini_iwi_chi,
               'overlap_chi_gini_diff': ogini_rwi_chi - ogini_iwi_chi, 
               
               'overlap_lee_n': on_lee,
               'overlap_lee_rmse':rmse_lee,
               'overlap_lee_mean_rwi': ox2_lee,
               'overlap_lee_mean_iwi': ox1_lee,
               'overlap_lee_mean_diff': ox1_lee - ox2_lee,
               'overlap_lee_mean_diff_norm': (ox1_lee - ox2_lee) / (ox1_lee + ox2_lee),
               'overlap_lee_mean_ztest': oztest_lee,
               'overlap_lee_mean_ro': f"{ro_lee:.2f} {ps_lee}",
               'overlap_lee_gini_rwi': ogini_rwi_lee,
               'overlap_lee_gini_iwi': ogini_iwi_lee,
               'overlap_lee_gini_diff': ogini_rwi_lee - ogini_iwi_lee, 
              }
        
            
        df_summary = pd.concat([df_summary, pd.DataFrame(obj, index=[1])], ignore_index=True)
        
    if output_dir is not None:
        fn = os.path.join(output_dir, 'summary.tex')
        df_summary.set_index('country').to_latex(fn, index=True, float_format="{:.2f}".format)
        print(f'{fn} saved!')
        
    return df_summary