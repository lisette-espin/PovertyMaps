import os 
import glob
import powerlaw
import itertools
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import geopandas as gpd
from scipy import stats
import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
from scipy.stats import gaussian_kde
import matplotlib.transforms as transforms
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import jensenshannon
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statsmodels.distributions.empirical_distribution import ECDF


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
NORM_MEAN_GROUP = 'mean_group'
NORM_RESCALE = 'rescale'
NORM_RESCALE_MEAN = 'rescale_mean'

MLEE = 'M1'
MESP = 'M2'
MCHI = 'M3'

MODEL_ORDER = [MLEE, MESP, MCHI]
MODEL_GROUPS = [[MLEE, MESP], [MCHI]]
MODEL_PAIRS = [(MLEE, MESP), (MLEE, MCHI), (MESP, MCHI)]
MODEL_PAIRS_TALK = [(MLEE, MESP)]

def sns_reset():
    sns.reset_orig()
    # matplotlib.use('agg')

def sns_paper_style():
    sns_reset()
    # sns.set_context("paper", font_scale=1.5) 
    #rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5}) 
    rc('font', family = 'serif')
    sns.set_style("white")
    sns.set_context("paper", font_scale = 1.6, rc={"grid.linewidth": 0.6})
    
def sns_talk_style():
    sns_reset()
    sns.set_context("talk")
    plt.rcParams['ytick.left'] = False
    plt.rcParams['xtick.bottom'] = False
    
def set_style(font_scale=1.5):
    sns_reset()
    sns.reset_orig()
    sns.set_context("poster", font_scale=font_scale) #rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5}) 
    rc('font', family = 'serif')
    
def set_latex():    
    # rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

    
    
def load_rwi_lee_iwi(country, model, features, rwi_pred_path, lee_pred_path, iwi_pred_path, rescale=False):
    model = model.replace('$','').replace('_','')
    fn_rwi = glob.glob(os.path.join(rwi_pred_path, f"{COUNTRIES[country]['code3'].lower()}_relative_wealth_index.csv"))
    fn_iwi_pred = glob.glob(os.path.join(iwi_pred_path, f"{COUNTRIES[country]['code']}_{model}_*_{features}.csv"))
    fn_iwi_feat = glob.glob(os.path.join(iwi_pred_path, f"{COUNTRIES[country]['code']}_features_{features}.csv"))
    fn_lee_pred = glob.glob(os.path.join(lee_pred_path, country, f"{country}_estimated_wealth_index.csv.zip"))
    
    try:
        # RWI-CHI
        gdf_rwi = ios.read_geo_csv(fn_rwi[0], lon='longitude', lat='latitude', index_col=None)
        # print("RWI:\n", gdf_rwi)
    except Exception as ex:
        gdf_rwi = None
        print('[ERROR] RWI', country, ex)
        
    try:
        # IWI-ESPIN
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
        do_rescale(gdf_rwi, 'rwi', gdf_iwi, 'pred_mean_wi', MESP)
        
    if rescale and gdf_rwi is not None and gdf_lee is not None:
        do_rescale(gdf_rwi, 'rwi', gdf_lee, 'estimated_IWI', MLEE)
        
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
    
    # same legend
    if kind in [NORM_MEAN_GROUP]:
        try:
            # country
            min_iwi = []
            max_iwi = []
            for model in [MLEE, MESP]:
                obj = metadata[model]
                col = obj['mean']
                data = obj['data']
                min_iwi.append(data[col].min())
                max_iwi.append(data[col].max())
                if model == MESP:
                    mean = data[col].mean()
            min_iwi = min(min_iwi)
            max_iwi = max(max_iwi)
            vlim_m2_m3 = (min_iwi, mean, max_iwi)
        except Exception as ex:
            print(f"[ERROR] {ex}")
            
    plt.suptitle(get_suptitle(kind) if suptitle else '', y=0.75 if title else 0.7)
    
    cmap_rwi = 'RdBu'
    cmap = kwargs.get('cmap', 'RdBu')
    
    legend = kwargs.pop('legend', True)
    legend_kwds = kwargs.get('legend_kwds', {})
    
    for model in MODEL_ORDER:
        obj = metadata[model]
        ax = axes[obj['index']]
        data = obj['data'].copy()
        
        post = "" if not years else f"\n({years[country][model]})"
        ax.set_title(obj['source'] if title else '')
        ax.set_axis_off()       
        
        kwargs['cmap'] = cmap if model in [MLEE, MESP] else cmap_rwi
        
        if kind == NORM_MEAN:
            # for each country, using their own wealth scale, we center the white color into the mean of their own wealth scores.
            n, vmin, vcenter, vmax = data.shape[0], data[obj['mean']].min(), data[obj['mean']].mean(), data[obj['mean']].max()
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            data.plot(column=obj['mean'], ax=ax, norm=norm, legend=True, **kwargs)
        if kind == NORM_MEAN_GROUP:
            if model in [MLEE, MESP]:
                vmin, vcenter, vmax = vlim_m2_m3
            else:
                n, vmin, vcenter, vmax = data.shape[0], data[obj['mean']].min(), data[obj['mean']].mean(), data[obj['mean']].max()
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            data.plot(column=obj['mean'], ax=ax, norm=norm, legend=True, **kwargs)
                
        else:
            # here we don't alter anything
            data.plot(column=obj['mean'], ax=ax, legend=legend,  **kwargs)
        
        # Legend titles
        cbax = [ax2 for ax2 in fig.axes if ax2 not in axes and ax2!=ax][-1] #to get only the colorbar axis
        cbtitle = '$IWI$ pred.' if model in [MLEE, MESP] else 'RWI pred.' if model == MCHI else None
        cbax.text(s=cbtitle, x=1.9, y=1.1, ha='center', va='top', fontsize=12, color='grey', transform=cbax.transAxes)        

            
def plot_pdf(metadata, country, axes, **kwargs):
    
    #ymin = float(min([gaussian_kde(obj['data'][obj['mean']]).evaluate(obj['data'][obj['mean']]).min() for key, obj in metadata.items()]))
    #ymax = float(max([gaussian_kde(obj['data'][obj['mean']]).evaluate(obj['data'][obj['mean']]).max() for key, obj in metadata.items()]))

    for group in MODEL_GROUPS:
        gi, ga = None, None
        
        for model in group:
            
            obj = metadata[model]
            
            ax = axes[obj['index']]
            data = obj['data']
            values = data[obj['mean']].astype(float)
            # values.plot(kind='density', ax=ax)

            # Create the KDE model from the data
            kde = gaussian_kde(values)
            x = np.linspace(min(values), max(values), len(values))
            density = kde(x)
            ax.plot(x, density)
            ymin = min(density)
            ymax = max(density)
            
            ax.set_ylabel('p(X=x)' if obj['index']==0 else '')
            ax.spines[['right', 'top']].set_visible(False)

            n = values.shape[0]
            m = values.mean()
            s = values.std()
            gc = gini_coefficient(values) * 100
            t = ax.text(s=f'$n=${n}', x=1.0, y=0.48, ha='right', va='bottom', transform=ax.transAxes)
            t = ax.text(s=f'$\mu=${m:.2f}, $\sigma=${s:.2f}', x=1.0, y=0.3, ha='right', va='bottom', transform=ax.transAxes)
            if model != MCHI:
                t = ax.text(s=f'Gini={gc:.2f}', x=1.0, y=0.12, ha='right', va='bottom', transform=ax.transAxes)
            
            gi = ymin if gi is None else min((ymin,gi))
            ga = ymax if ga is None else max((ymax,ga))

        # same y-axis
        for model in group:
            obj = metadata[model]
            ax = axes[obj['index']]
            ax.set_ylim((gi,ga))
    
def plot_ecdf(metadata, country, axes, **kwargs):

    
    for model, obj in metadata.items():
    
        ax = axes[obj['index']]
        data = obj['data']
        values = data[obj['mean']]
        
        ecdf = ECDF(values) 
        ax.plot(ecdf.x, ecdf.y)
        
        ax.set_ylabel("p(X≤x)" if obj['index']==0 else '')
        ax.spines[['right', 'top']].set_visible(False)
        
        ## POVERTY LINE: IWI-50 (Headcount 2.00$)
        ## https://hdr.undp.org/system/files/documents/03iwiundpeschborn2013.pdf
        ## https://www.jstor.org/stable/24721406?seq=16
        iwipl = 35
        x_intersection = np.percentile(values, iwipl) # 35th percentile (median)
        y_intersection = np.interp(x_intersection, ecdf.x, ecdf.y)
        ax.plot([x_intersection, x_intersection], [0, y_intersection], ls='--', c='grey')
        ax.plot([min(0,values.min()), x_intersection], [y_intersection, y_intersection], ls='--', c='grey')
        ax.scatter([x_intersection], [y_intersection], color='black', zorder=5)
        smooth = 3 if model in [MLEE, MESP] else 0.2
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
        

def get_metadata_template(gdf_rwi, gdf_lee, gdf_iwi):
    return {MCHI:{'index':0, 'data':gdf_rwi, 'mean':'rwi', f'mean_rescaled_{MLEE}':f'rwi_rescaled_{MLEE}', f'mean_rescaled_{MESP}':f'rwi_rescaled_{MESP}', 'std_col':'error', 'source':MCHI, 'metric':'RWI'}, #'Chi et al. 2021'
            MLEE:{'index':1, 'data':gdf_lee, 'mean':'estimated_IWI', 'std_col':None, 'source':MLEE, 'metric':'IWI'},  #'Lee & Braithwaite 2022'
            MESP:{'index':2, 'data':gdf_iwi, 'mean':'pred_mean_wi', 'std_col':'pred_std_wi', 'source':MESP, 'metric':'IWI'}} #'Espin-Noboa et al. 2023'

        
def plot_comparison_maps(gdf_rwi, gdf_lee, gdf_iwi, country, output_dir=None, **kwargs):
    dpi = kwargs.pop('dpi', 300)
    
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
        fig.savefig(fn, dpi=dpi, bbox_inches='tight')
        print(f'{fn} saved!')
        
    plt.show()
    plt.close()
    
    
    
def do_rescale(gdf, col, gdf_baseline, col_baseline, name):
    # offset = abs(gdf_rwi.rwi.min())
    a, b = gdf[col].min(), gdf[col].max()
    c, d = gdf_baseline[col_baseline].min(), gdf_baseline[col_baseline].max()
    gdf.loc[:,f'rwi_rescaled_{name}'] = gdf.rwi.apply(lambda v: ((v - a) * (d - c) / (b - a)) + c)


    
def plot_comparison_dist(gdf_rwi, gdf_lee, gdf_iwi, country, output_dir=None, **kwargs):
    dpi = kwargs.pop('dpi', 300)
    
    title = kwargs.pop('title', None)
    suptitle = kwargs.pop('suptitle', False)
    years = kwargs.pop('years', None)
    
    figsize = kwargs.pop('figsize', (10,5))
    nr = 2 
    nc = 3
    fig, axes = plt.subplots(nr, nc, figsize=figsize, sharex=False, sharey=False)
    
    metadata = get_metadata_template(gdf_rwi, gdf_lee, gdf_iwi)    
    
    plot_pdf(metadata, country, axes[0,:], **kwargs)
    plot_ecdf(metadata, country, axes[1,:], **kwargs)
    
    if title: 
        for key, obj in metadata.items():
            post = "" if not years else f"\n({years[country][key]})"
            axes[0,obj['index']].set_title(obj['source'] if title else '')

    # same x-axis
    for group in MODEL_GROUPS:
        gmi=None
        gma=None
        for model in group:
            obj = metadata[model]
            data = obj['data'][obj['mean']]
            mi,ma = data.min(), data.max()
            gmi = mi if gmi is None else min([mi,gmi])
            gma = ma if gma is None else max([ma,gma])
        for model in group:
            smooth = 2 if model in [MLEE, MESP] else 0.02
            obj = metadata[model]
            for rid in range(nr):
                axes[rid,obj['index']].set_xlim((gmi-smooth, gma+smooth))
            
    # for model, obj in metadata.items():
    #     xlim = (obj['data'][obj['mean']].min(), obj['data'][obj['mean']].max())
    #     smooth = 2 if model in [MLEE, MESP] else 1
    #     for rid in range(nr):
    #         axes[rid,obj['index']].set_xlim((xlim[0]-smooth, xlim[1]+smooth))
        
    # x-axis label
    from matplotlib import ticker
    import matplotlib
    for model, obj in metadata.items():
        # bottom
        ax = axes[-1,obj['index']]
        ax.set_xlabel('RWI' if model in [MCHI] else 'IWI')
        # not bottom
        for ax in axes[0:-1,obj['index']]:
            ax.set_xticklabels([])
    
    if suptitle:
        plt.suptitle("WEALTH DISTRIBUTION")
        
    plt.tight_layout()
    #plt.subplots_adjust(hspace=0.1, wspace=0.05) 
    if output_dir is not None:
        fn = os.path.join(output_dir, f"dist_{COUNTRIES[country]['code3']}.pdf")
        fig.savefig(fn, dpi=dpi, bbox_inches='tight')
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

def plot_comparison_overlapping_cells(gdf_rwi, gdf_lee, gdf_iwi, country, sources=(MCHI, MLEE), output_dir=None, **kwargs):
    dpi = kwargs.pop('dpi', 300)
    
    suptitle = kwargs.pop('suptitle', False)
    max_distance = kwargs.pop('max_distance', 10)
    
    figsize = kwargs.pop('figsize', (10,5))
    fig, axes = plt.subplots(1,2,figsize=figsize)
    
    if sources is None or sources[0]==sources[1]:
        raise Exception("Invalid sources. Allowed any pair of: M1, M2, ME")
    
    ### data
    model1 = sources[0]
    model2 = sources[1]
    gdf1, metric1 = get_data_by_model(gdf_rwi, gdf_lee, gdf_iwi, model1)
    gdf2, metric2 = get_data_by_model(gdf_rwi, gdf_lee, gdf_iwi, model2)
    name1 = '$\hat{IWI}_{'+model1+'}$' if model1 == MCHI else '$IWI_{'+model1+'}$'
    name2 = '$RWI_{'+model2+'}$' if model2 == MCHI else '$IWI_{'+model2+'}$'
    
    # rescaling if needed
    # metric1 = f'{metric1}_rescaled_{model2}' if model1 == MCHI else metric1
    
    # overlaps
    df_overlap = gpd.sjoin_nearest(gdf1.to_crs(PROJ_MET), gdf2.to_crs(PROJ_MET), how='inner', max_distance=max_distance, distance_col='distance')
    df_overlap.loc[:,'diff'] = df_overlap.apply(lambda row: row[metric1]-row[metric2], axis=1) # negative: metric2 > metric1

    # removing duplicates
    df_overlap = df_overlap.loc[df_overlap.groupby(df_overlap.index)['distance'].idxmin()]
    df_overlap = df_overlap.loc[df_overlap.groupby(df_overlap.index_right)['distance'].idxmin()]


    ### spatial join
    #tmp = gpd.sjoin_nearest(gdf1.to_crs(PROJ_MET), gdf2.to_crs(PROJ_MET), how='inner', max_distance=max_distance, distance_col='distance')
    #tmp.loc[:,'diff'] = tmp.apply(lambda row: row[metric1] - row[metric2], axis=1) 
    # negative: <0 model2 > model1
    # positive: >0 model1 > model2
    
    
    if df_overlap.shape[0] != df_overlap.index.nunique() or df_overlap.shape[0] != df_overlap.index_right.nunique():
        print("[WARNING]", df_overlap.shape[0], df_overlap.index.nunique(), df_overlap.index_right.nunique(), "computing mean...")
        
    #### left scatter ####
    ax = axes[0]
    smooth = 0.1
    for color, query in {'blue':f"diff>@smooth", 'red':f"diff<-@smooth", 'black':f"diff<=@smooth and diff>=-@smooth"}.items():
        _t = df_overlap.query(query)
        ax.scatter(_t[metric1], _t[metric2], color=color, 
                   alpha=1.,
                   label=f"{name1} > {name2}" if color=='blue' else f'{name1} < {name2}' if color=='red' else f"{name1} = {name2}")
    
    ax.legend(loc=2, borderpad=0.1, labelspacing=0.1, handletextpad=0.0)
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    
    mi=0
    ma=100
    ax.set_xlim((mi,ma))
    ax.set_ylim((mi,ma))
    
    ax.plot([mi,ma],[mi,ma],ls='--', lw=1, c='grey')
    ax.spines[['right', 'top']].set_visible(False)
    
    n = df_overlap.shape[0]
    ro,pv = pearsonr(df_overlap[metric1], df_overlap[metric2])
    ps = get_significance_stars(pv)
    # Gini
    g1 = gini_coefficient(df_overlap[metric1]) * 100
    g2 = gini_coefficient(df_overlap[metric2]) * 100
    #RMSE
    rmse = mean_squared_error(df_overlap[metric1], df_overlap[metric2], squared=False)
    t = ax.text(s='$RMSE=$' + f'{rmse:.2f}', va='bottom', ha='right', transform=ax.transAxes, x=1, y=0.45, bbox=dict(facecolor='white', alpha=0.8, boxstyle="square,pad=0.05"))
    # t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white'))
    

    t = ax.text(s='Gini$_{V1}=$'.replace('V1', name1.replace('$','')) + f'{g1:.2f}', va='bottom', ha='right', transform=ax.transAxes, x=1, y=0.35, bbox=dict(facecolor='white', alpha=0.8, boxstyle="square,pad=0.05"))
    # t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white'))
    t = ax.text(s='Gini$_{V2}=$'.replace('V2', name2.replace('$','')) + f'{g2:.2f}', va='bottom', ha='right', transform=ax.transAxes, x=1, y=0.25, bbox=dict(facecolor='white', alpha=0.8, boxstyle="square,pad=0.05"))
    # t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white'))
    # n and correlation
    t = ax.text(s=f'n={n}', va='bottom', ha='right', transform=ax.transAxes, x=1, y=0.15, bbox=dict(facecolor='white', alpha=0.8, boxstyle="square,pad=0.05"))
    # t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white'))
    t = ax.text(s=f"$\\rho=${ro:.2f}{ps}", va='bottom', ha='right', transform=ax.transAxes, x=1, y=0.05, bbox=dict(facecolor='white', alpha=0.8, boxstyle="square,pad=0.05"))
    # t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white'))
    
    
    #### right map ###
    ax = axes[1]
    legend_kwds = kwargs.pop('legend_kwds',None)
    markersize = kwargs.pop('markersize',1)
    cmap='coolwarm_r'
    dmin = df_overlap['diff'].min()
    dmax = max(df_overlap['diff'].max(),0.01)
    norm = TwoSlopeNorm(vmin=dmin, vcenter=0, vmax=dmax)
    df_overlap.to_crs(PROJ_DEG).plot(column='diff', cmap=cmap, legend=True, norm=norm, ax=ax, legend_kwds=legend_kwds,  markersize=markersize)
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
        
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    if output_dir is not None:
        fn = os.path.join(output_dir, f"samecells_{COUNTRIES[country]['code3']}_{'_'.join(sources)}.pdf")
        fig.savefig(fn, dpi=dpi, bbox_inches='tight')
        print(f'{fn} saved!')
        
    plt.show()
    plt.close()
    
    
def plot_comparison_admin_maps(gdf_rwi, gdf_lee, gdf_iwi, country, boundary_fn, admin_level, output_dir=None, **kwargs):
    dpi = kwargs.pop('dpi', 300)
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
    
    metadata = get_metadata_template(gdf_rwi, gdf_lee, gdf_iwi)
    
    cmap_rwi = 'RdBu'
    cmap = kwargs.get('cmap', 'RdBu')
    
    # same legend
    if kind in [NORM_MEAN_GROUP]:
        try:
            # country
            cc3 = COUNTRIES[country]['code3'].upper()
            fn = boundary_fn.replace("<country>", country).replace('<ccode3>',cc3).replace('<ADM>',str(admin_level))
            gdf_country = gpd.read_file(fn)
            min_iwi = []
            max_iwi = []
            for model in [MLEE, MESP]:
                obj = metadata[model]
                col = obj['mean']
                data = obj['data']
                tmp = geo.distribute_data_in_grid(data, gdf_country, column=col, aggfnc='mean', lsuffix='data', how='right')
                min_iwi.append(tmp[col].min())
                max_iwi.append(tmp[col].max())
                if model == MESP:
                    mean = tmp[col].mean()
                
            min_iwi = min(min_iwi)
            max_iwi = max(max_iwi)
            vlim_m2_m3 = (min_iwi, mean, max_iwi)
        except Exception as ex:
            print(f"[ERROR] {ex}")

    try:
        # country
        cc3 = COUNTRIES[country]['code3'].upper()
        fn = boundary_fn.replace("<country>", country).replace('<ccode3>',cc3).replace('<ADM>',str(admin_level))
        gdf_country = gpd.read_file(fn)
        
        
        for model in MODEL_ORDER:
            obj = metadata[model]
            ax = axes[obj['index']]
            
            data = obj['data']
            metric = obj['mean']
            
            tmp = geo.distribute_data_in_grid(data, gdf_country, column=metric, aggfnc='mean', lsuffix='data', how='right')
            
            kwargs['cmap'] = cmap if model in [MLEE, MESP] else cmap_rwi
            
            if kind == NORM_MEAN:
                n, vmin, vcenter, vmax, vstd = tmp.shape[0], tmp[metric].min(), tmp[metric].mean(), tmp[metric].max(), tmp[metric].std()
                norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                tmp.plot(column=metric, ax=ax, norm=norm, legend=legend, **kwargs)
                
            elif kind == NORM_MEAN_GROUP:
                
                if model in [MLEE, MESP]:
                    vmin, vcenter, vmax = vlim_m2_m3
                else:
                    n, vmin, vcenter, vmax, vstd = tmp.shape[0], tmp[metric].min(), tmp[metric].mean(), tmp[metric].max(), tmp[metric].std()
                    
                norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                tmp.plot(column=metric, ax=ax, norm=norm, legend=legend, **kwargs)
                
            else:
                tmp.plot(column=metric, ax=ax, legend=legend, **kwargs)
                
            # Legend titles
            cbax = [ax2 for ax2 in fig.axes if ax2 not in axes and ax2!=ax][-1] #to get only the colorbar axis
            val = 'RWI' if model == MCHI else 'IWI'
            cbtitle = f'{val} pred.'
            cbax.text(s=cbtitle, x=1.9, y=1.12, ha='center', va='top', fontsize=12, color='grey', transform=cbax.transAxes)        

            if title:
                post = "" if not years else f"\n({years[country][model]})"
                ax.set_title(obj['source'])
                
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
            fig.savefig(fn, dpi=dpi, bbox_inches='tight')
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
    df_preliminaries = pd.DataFrame(columns=['country','year','gini','gdp',F'in_{MLEE}',f'in_{MESP}',f'in_{MCHI}'])
    for country, obj in survey_years.items():
        y_lee = str(obj[MLEE]).replace(',','-').split('-') if obj[MLEE] is not None else []
        y_iwi = str(obj[MESP]).replace(',','-').split('-') if obj[MESP] is not None else []
        y_rwi = str(obj[MCHI]).replace(',','-').split('-') if obj[MCHI] is not None else []
        
        y_lee = sorted(set([int(y) if int(y)>=2000 else int(f"20{y}") for y in y_lee]))
        y_iwi = sorted(set([int(y) if int(y)>=2000 else int(f"20{y}") for y in y_iwi]))
        y_rwi = sorted(set([int(y) if int(y)>=2000 else int(f"20{y}") for y in y_rwi]))
        
        years = sorted(set(y_rwi + y_lee + y_iwi))

        # GINI
        tmp = df_gini.loc[country].dropna().reset_index().loc[3:].rename(columns={'index':'year', country:'gini'}) # loc 3, unnecesary rows
        tmp.loc[:,'year'] = tmp.year.astype(int)
        ginis = tmp.sort_values('year', ascending=True).reset_index(drop=True).tail(len(years)).reset_index(drop=True)

        
        # GDP
        for i, y in enumerate(years):
            year = int(y)

            gini = ginis.loc[i, 'gini'] if i < ginis.shape[0] else None
            gdp  = df_gdp[str(year)].get(country, None)
            
            if gini not in [None,np.nan,'']:
                gini = f"{gini} ({ginis.loc[i, 'year']})"


            obj = {'country':country, 'year':year, 'gini':gini, 'gdp':gdp, f'in_{MLEE}':year in y_lee, f'in_{MESP}':year in y_iwi, f'in_{MCHI}':year in y_rwi}
            df_preliminaries = pd.concat([df_preliminaries, pd.DataFrame(obj, index=[1])], ignore_index=True)

    df_preliminaries.rename(columns={'year':'year_survey', 'gini':'_gini'}, inplace=True) 
    df_preliminaries[['actual_gini','actual_year']] = df_preliminaries.apply(lambda row: str(row._gini).split(' (') if '(' in str(row._gini) else 
                                                                             [row._gini, row.year_survey], axis=1, result_type="expand")
    df_preliminaries.actual_year = df_preliminaries.actual_year.apply(lambda v: None if v in [None,np.nan,''] else str(int(v)) if type(v)!=str else str(v).replace(')',''))
    df_preliminaries.actual_gini = df_preliminaries.actual_gini.astype(float)
    
    if output_dir is not None:
        fn = os.path.join(output_dir, 'data_preliminaries.tex')
        df_preliminaries.to_latex(fn, index=False, float_format="{:.1f}".format)
        print(f'{fn} saved!')
    
    return df_preliminaries

def plot_preliminaries(df_preliminaries, output_dir=None, **kwargs):
    
    dpi = kwargs.pop('dpi', 300)
    
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
        fig.savefig(fn, dpi=dpi, bbox_inches='tight')
        print(f'{fn} saved!')
        
    plt.show()
    plt.close()
    
    
    
def show_expected_differences(df_preliminaries, output_dir):
    df_expectations = pd.DataFrame()
    gini_smooth = 10 # (10) gini 0 to 100
    gdp_smooth = 0.1 # normalized becasue GDP and IWI, RWI are in different ranges | Old:(100) gdp min 400 max 8000


    for country, df in df_preliminaries.groupby('country'):
        
        obj = {'country':country}
        
        for (model1, model2) in MODEL_PAIRS:
            tmp1 = df.query(f"in_{model1} == True")
            tmp2 = df.query(f"in_{model2} == True")
            

            if tmp1.shape[0] <= 0 or tmp2.shape[0] <= 0:
                gini_diff = None
                str_gini_diff = 'NA'
                gdp_diff = None
                str_gdp_diff = 'NA'
            else:
                gdp1 = tmp1.gdp.mean()
                gdp2 = tmp2.gdp.mean()
                gdp_diff = (gdp1 - gdp2) / (gdp1 + gdp2)
                
                if sum(tmp1[f'in_{model1}']) == sum(tmp2[f'in_{model2}']) and sum(tmp2[f'in_{model2}']) == tmp2.shape[0]:
                    gini_diff = 0
                    
                elif len(set(tmp1.dropna().actual_year.unique())-set(tmp1.dropna().year_survey.unique())) == 0 or \
                len(set(tmp2.dropna().actual_year.unique())-set(tmp2.dropna().year_survey.unique())) == 0:
                    gini1 = tmp1.dropna().actual_gini.mean()
                    gini2 = tmp2.dropna().actual_gini.mean()
                    gini_diff = gini1 - gini2
                    
                else:
                    gini1 = None
                    gini2 = None
                    gini_diff = None
                    
                # idxmax1 = tmp1.dropna().actual_year.astype(int).idxmax()
                # idxmax2 = tmp2.dropna().actual_year.astype(int).idxmax()
                # gini_diff = tmp1.loc[idxmax1,'actual_gini'] - tmp2.loc[idxmax2,'actual_gini']
                str_gini_diff = None if gini_diff is None else \
                f'higher Gini (more inequality in {model1})' if gini_diff>gini_smooth else \
                f'lower Gini (less inequality in {model1})' if gini_diff<-gini_smooth else \
                f'similar or slightly higher Gini (slightly more inequality in {model1})' if gini_diff > 0 else \
                f'similar or slightly lower Gini (slightly less inequality in {model1})' if gini_diff < 0 else \
                'very similar Gini'
                # gdp_diff = (tmp1.loc[idxmax1,'gdp'] - tmp2.loc[idxmax2,'gdp']) / (tmp1.loc[idxmax1,'gdp'] + tmp2.loc[idxmax2,'gdp'])
                str_gdp_diff = f'higher GDP ({model1} is richer)' if gdp_diff>gdp_smooth else \
                f'lower GDP ({model1} is poorer)' if gdp_diff<-gdp_smooth else \
                f'similar or slightly higher GDP ({model1} is slightly richer)' if gdp_diff > 0 else \
                f'similar or slightly lower GDP ({model1} is slightly poorer)' if gdp_diff < 0 else \
                'similar GDP'
            
            obj[f'gini_{model1}_{model2}'] = str_gini_diff
            obj[f'gdp_{model1}_{model2}'] = str_gdp_diff
            obj[f'gini_{model1}_{model2}_value'] = gini_diff
            obj[f'gdp_{model1}_{model2}_value'] = gdp_diff

        df_expectations = pd.concat([df_expectations, pd.DataFrame(obj, index=[0])], ignore_index=True)

    df_expectations.set_index('country', inplace=True)
    
    if output_dir is not None:
        fn = os.path.join(output_dir, 'data_expectations.tex')
        df_expectations.to_latex(fn, index=True, float_format="{:.1f}".format)
        print(f'{fn} saved!')

    return df_expectations


def get_data_by_model(gdf_rwi, gdf_lee, gdf_iwi, model):
    gdf = gdf_rwi if model == MCHI else gdf_lee if model == MLEE else gdf_iwi if model == MESP else None
    metric = 'rwi' if model == MCHI else 'estimated_IWI' if model == MLEE else 'pred_mean_wi' if model == MESP else None
    return gdf, metric
    
def show_final_summary(df_preliminaries, df_expectations, survey_years, best_models, features_source, rwi_path, lee_pred_path, iwi_pred_path, max_distance, output_dir=None):
    df_summary = pd.DataFrame(columns=[])
    
    for country, df_pre in df_preliminaries.groupby('country'):
        df_ex = df_expectations.loc[country]
        
        try:
            model = best_models[country]
        except Exception as ex:
            print(country, ex)
            model = 'CB'
        print(country, model)

        # all data per model
        gdf_rwi, gdf_lee, gdf_iwi = load_rwi_lee_iwi(country, model, features_source, rwi_path, lee_pred_path, iwi_pred_path, rescale=True)
        
        # init results
        obj = {'country':country}
        
        # results per model (original scales)
        for model in MODEL_ORDER:
            gdf, metric = get_data_by_model(gdf_rwi, gdf_lee, gdf_iwi, model)
            n = gdf.shape[0]        # number of places
            x = gdf[metric].mean()  # mean predicted wealth
            s = gdf[metric].std()   # mean standard deviation of wealth
            sv = (1/n) * sum((gdf[metric] - x)**2)  # mean of sample variance of wealth
            g = gini_coefficient(gdf[metric]) * 100 # gini coefficient of predicted wealth
            res = stats.skewtest(gdf[metric])
            pvs = get_significance_stars(res.pvalue)
            obj[f'{model}_years'] = survey_years[country][model]
            obj[f'{model}_n'] = n
            obj[f'{model}_mean'] = x
            obj[f'{model}_std'] = s
            obj[f'{model}_var'] = sv
            obj[f'{model}_gini'] = g
            obj[f'{model}_skewness_stat'] = res.statistic 
            obj[f'{model}_skewness_pv'] = pvs 
            obj[f'{model}_skewness_stat_str'] = f"{res.statistic:.2f} {pvs}"
            
        # rescaled MCHI
        for model2 in [MLEE, MESP]:
            metric1 = f'rwi_rescaled_{model2}'
            n = gdf_rwi.shape[0]                         # number of places
            x = gdf_rwi[metric1].mean()                  # mean predicted wealth
            s = gdf_rwi[metric1].std()                   # mean standard deviation of wealth
            sv = (1/n) * sum((gdf_rwi[metric1]-x)**2)    # mean of sample variance of wealth
            g = gini_coefficient(gdf_rwi[metric1]) * 100 # gini coefficient of predicted wealth
            obj[f'{MCHI}_rs{model2}_mean'] = x
            obj[f'{MCHI}_rs{model2}_std'] = s
            obj[f'{MCHI}_rs{model2}_var'] = sv
            obj[f'{MCHI}_rs{model2}_gini'] = g
            
        # comparison
        for (model1, model2) in MODEL_PAIRS:
            gdf1, metric1 = get_data_by_model(gdf_rwi, gdf_lee, gdf_iwi, model1)
            gdf2, metric2 = get_data_by_model(gdf_rwi, gdf_lee, gdf_iwi, model2)
            
            # rescaling wealth metric for MCHI (rwi)
            metric1 = f'{metric1}_rescaled_{model2}' if model1 == MCHI else metric1
            metric2 = f'{metric2}_rescaled_{model1}' if model2 == MCHI else metric2
            
            # full distributions
            ks, ks_pv = stats.ks_2samp(gdf1[metric1], gdf2[metric2])
            obj[f'ks_{model1}_{model2}'] = f"{ks:.2f} {get_significance_stars(ks_pv)}"
            
            obj[f'diff_{model1}_{model2}_gini_expected'] = df_ex[f'gini_{model1}_{model2}_value']
            obj[f'diff_{model1}_{model2}_gini_obtained'] = obj[f'{model1}_gini'] - obj[f'{model2}_gini']
            
            obj[f'diff_{model1}_{model2}_wealth_expected'] = df_ex[f'gdp_{model1}_{model2}_value']
            obj[f'diff_{model1}_{model2}_wealth_obtained'] = (obj[f'{model1}_mean'] - obj[f'{model2}_mean']) / (obj[f'{model1}_mean'] + obj[f'{model2}_mean'])

            # overlaps
            df_overlap = gpd.sjoin_nearest(gdf1.to_crs(PROJ_MET), gdf2.to_crs(PROJ_MET), how='inner', max_distance=max_distance, distance_col='distance')
            df_overlap.loc[:,'diff'] = df_overlap.apply(lambda row: row[metric1]-row[metric2], axis=1) # negative: metric2 > metric1
            
            # removing duplicates
            #print(df_overlap.shape[0], df_overlap.index.nunique(), df_overlap.index_right.nunique())
            df_overlap = df_overlap.loc[df_overlap.groupby(df_overlap.index)['distance'].idxmin()]
            df_overlap = df_overlap.loc[df_overlap.groupby(df_overlap.index_right)['distance'].idxmin()]
            #print(df_overlap.shape[0], df_overlap.index.nunique(), df_overlap.index_right.nunique())
            
            on = df_overlap.shape[0]              # number of overlapped cells
            pct1 = on*100/gdf1.shape[0]
            pct2 = on*100/gdf2.shape[0]
            ox1 = df_overlap[metric1].mean()      # mean predicted wealth data1
            ox2 = df_overlap[metric2].mean()      # mean predicted welath data2 
            os1 = df_overlap[metric1].std()       # std.dev. data1
            os2 = df_overlap[metric2].std()       # std.dev. data2
            ro, pv = pearsonr(df_overlap[metric1], df_overlap[metric2]) # pearson correlation between two predicted values
            ps = get_significance_stars(pv)       # significance test (string)
            ogini1 =  gini_coefficient(df_overlap[metric1]) * 100  # gini coefficient predicted wealth distribution (only overlap) data1
            ogini2 =  gini_coefficient(df_overlap[metric2]) * 100  # gini coefficient predicted wealth distribution (only overlap) data2
            osn1 = os1 / np.sqrt(on)
            osn2 = os2 / np.sqrt(on)
            oztest = (ox1 - ox2) / np.sqrt(osn1**2 + osn2**2)
            rmse = mean_squared_error(df_overlap[metric1], df_overlap[metric2], squared=False)
        
            obj[f'overlap_{model1}_{model2}_n'] = on
            obj[f'overlap_{model1}_{model2}_pct_{model1}'] = pct1
            obj[f'overlap_{model1}_{model2}_pct_{model2}'] = pct2
            obj[f'overlap_{model1}_{model2}_rmse'] = rmse,
            obj[f'overlap_{model1}_{model2}_mean_wealth_{model1}'] = ox2,
            obj[f'overlap_{model1}_{model2}_mean_wealth_{model2}'] = ox1,
            obj[f'overlap_{model1}_{model2}_mean_wealth_diff'] = ox1 - ox2,
            obj[f'overlap_{model1}_{model2}_mean_wealth_diff_norm'] = (ox1 - ox2) / (ox1 + ox2),
            obj[f'overlap_{model1}_{model2}_expected_wealth_diff_norm'] = df_ex[f'gdp_{model1}_{model2}_value']
            obj[f'overlap_{model1}_{model2}_mean_wealth_ztest'] = oztest,  # ztest
            obj[f'overlap_{model1}_{model2}_mean_ro'] = f"{ro:.2f} {ps}",  # pearson 
            obj[f'overlap_{model1}_{model2}_gini_{model1}'] = ogini1,      # gini wealth model1
            obj[f'overlap_{model1}_{model2}_gini_{model2}'] = ogini2,      # gini wealth model2
            obj[f'overlap_{model1}_{model2}_gini_diff'] = ogini1 - ogini2, # negative: metric2 > metric1
            obj[f'overlap_{model1}_{model2}_expected_gini_diff'] = df_ex[f'gini_{model1}_{model2}_value']
            
        df_summary = pd.concat([df_summary, pd.DataFrame(obj, index=[1])], ignore_index=True)
        
    if output_dir is not None:
        fn = os.path.join(output_dir, 'summary.tex')
        df_summary.set_index('country').to_latex(fn, index=True, float_format="{:.2f}".format)
        print(f'{fn} saved!')
        
    return df_summary
                
def plot_wealth_distributions(countries_order, best_models, features_source, m1_path, m2_path, m3_path, model_pairs=MODEL_PAIRS, output_dir=None, **kwargs):
    h = 1.2  # height
    w = 4.0  # width
    nc = len(model_pairs) # M1-M2 vs M1-M3
    nr = len(countries_order) # countries
    nbins = 100
    colors = {MLEE:'tab:green', MESP:'tab:orange', MCHI:'tab:blue'}
    log = True
    figsize = kwargs.pop('figsize', (nc*w, nr*h))
    yscale = kwargs.pop('yscale', 'linear')
    
    
    fig, axes = plt.subplots(nr, nc, figsize=figsize, sharex=True, sharey=True)
    for r, country in enumerate(countries_order):

        # DATA
        gdf_rwi, gdf_lee, gdf_iwi = load_rwi_lee_iwi(country, best_models[country], features_source, m1_path, m2_path, m3_path, rescale=True)

        # Removing "Marion Island, South Africa" (lat: <-46, eg.lat,lon -46.876666	37.855896) - for vis purposes
        if country == 'South Africa':
            lat_marion_island = -46 
            gdf_lee_mod = gdf_lee.query("lat>@lat_marion_island")
            gdf_iwi_mod = gdf_iwi.query("lat>@lat_marion_island")


        for c, (model1, model2) in enumerate(model_pairs):
            ax = axes[r, c] if nc > 1 else axes[r]

            ### Data
            gdf1, metric1 = get_data_by_model(gdf_rwi, gdf_lee, gdf_iwi, model1)
            gdf2, metric2 = get_data_by_model(gdf_rwi, gdf_lee, gdf_iwi, model2)
            
            # rescaled
            # metric1 = f'{metric1}_rescaled_{model2}' if model1 == MCHI else metric1
            metric2 = f'{metric2}_rescaled_{model1}' if model2 == MCHI else metric2
            
            
            ### Plots
            sns.kdeplot(gdf1[metric1], label=model1, color=colors[model1], ax=ax)
            sns.kdeplot(gdf2[metric2], label=model2, color=colors[model2], ax=ax)
            
            ### Gini
            g1 = f"{gini_coefficient(gdf1[metric1]) * 100:.1f}"
            g2 = f"{gini_coefficient(gdf2[metric2]) * 100:.1f}"
            
            ### Jensen-Shannon distance
            kde1 = gaussian_kde(gdf1[metric1])
            kde2 = gaussian_kde(gdf2[metric2])
            common_support = np.linspace(min(gdf1[metric1].min(), gdf2[metric2].min()), max(gdf1[metric1].max(), gdf2[metric2].max()), 100)
            pdf1 = kde1(common_support)
            pdf2 = kde2(common_support)
            pdf1 /= np.sum(pdf1)
            pdf2 /= np.sum(pdf2)
            js = jensenshannon(pdf1, pdf2)

                           
            ### Labels
            ax.text(s=f"JS={js:.1f}", x=0.99, y=0.95, ha='right', va='top', transform=ax.transAxes, size='x-small')
            ax.text(s=u"Gini$_{" + model1 + "}$=" + g1, x=0.99, y=0.75, ha='right', va='top', transform=ax.transAxes, size='x-small')
            ax.text(s=u"Gini$_{" + model2 + "}$=" + g2, x=0.99, y=0.55, ha='right', va='top', transform=ax.transAxes, size='x-small')
            
            
            ### Aesthetics
            ax.spines.top.set_visible(False)
            ax.spines.right.set_visible(False)
            
            if yscale == 'log':
                ax.set_yscale(yscale)
                ax.set_yticks([10**-3,  10**-2,  10**-1])
                ax.set_ylim(20**-3, None)

            if r == 0:
                ax.set_title(f"{model1} vs {model2}")

            if r == nr-1:
                if c == int(nc/2):
                    ax.set_xlabel(u"Predicted Wealth ($w$)")
                else:
                    ax.set_xlabel('')
            ax.set_xlabel(u"Predicted Wealth ($w$)")

            if r == nr/2 and c==0:
                ax.set_ylabel(u'P(X=$w$)')
                ax.yaxis.set_label_coords(-0.25, 1.)
            else:
                ax.set_ylabel('')
                
            if c == nc-1:
                ax.text(s=COUNTRIES[country]['code3'], x=1.2, y=0.95, ha='right', va='top', transform=ax.transAxes)

    # Custom legend handles using Line2D to create line markers
    legend_elements =  [Line2D([0], [0], color=color, lw=2, label=model) for model,color in colors.items()]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.11 if nc==3 else 1.16 if nc==2 else 1.25 if nc==1 else 1.0, 1.))

    plt.tight_layout() 
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    if output_dir is not None:
        dpi = 300
        fn = os.path.join(output_dir, 'plot_distributions_all_models.pdf')
        fig.savefig(fn, dpi=dpi, bbox_inches='tight')
        print(f'{fn} saved!')

    plt.show()
    plt.close()

    
     


def plot_maps_HQ_DIST(country, models, gdf_rwi, gdf_lee, gdf_iwi, boundary_fn, admin_level, output_dir):
    

    gdf1, metric1 = get_data_by_model(gdf_rwi, gdf_lee, gdf_iwi, MLEE)
    gdf2, metric2 = get_data_by_model(gdf_rwi, gdf_lee, gdf_iwi, MESP)
    cmap = 'BrBG'
    markersize = 4
    dpi = 300
    
    if country == 'South Africa':
        lat_marion_island = -46 
        gdf1 = gdf1.query("lat>@lat_marion_island")
        gdf2 = gdf2.query("lat>@lat_marion_island")

        
    # Country boundary
    cc3 = COUNTRIES[country]['code3'].upper()
    fn = boundary_fn.replace("<country>", country).replace('<ccode3>',cc3).replace('<ADM>',str(admin_level))
    gdf_country = gpd.read_file(fn)
    
    # Create figure and axes for a 2x2 grid of subplots
    fig, axs = plt.subplots(2,2, figsize=(10,10), 
                        facecolor='w',
                        constrained_layout=True, 
                        sharex=True, sharey=True, 
                        subplot_kw=dict(aspect='equal'))
    
    # Set the same aspect ratio for all maps to ensure symmetry
    for ax in axs.flat:
        ax.set_aspect('equal')
        ax.set_axis_off()
        
    # First row data
    vmin_row1 = min(gdf1[metric1].min(), gdf2[metric2].min())
    vmax_row1 = max(gdf1[metric1].max(), gdf2[metric2].max())
    mean_row1 = gdf2[metric2].mean()  # Mean for plot (0, 1)

    # Second row data
    gdfadm1 = geo.distribute_data_in_grid(gdf1, gdf_country, column=metric1, aggfnc='mean', lsuffix='data', how='right')
    gdfadm2 = geo.distribute_data_in_grid(gdf2, gdf_country, column=metric2, aggfnc='mean', lsuffix='data', how='right')
    vmin_row2 = min(gdfadm1[metric1].min(), gdfadm2[metric2].min())
    vmax_row2 = max(gdfadm1[metric1].max(), gdfadm2[metric2].max())
    mean_row2 = gdfadm2[metric2].mean()  # Mean for plot (0, 1)
    
    # Plot: Top
    norm = TwoSlopeNorm(vmin=vmin_row1, vcenter=mean_row1, vmax=vmax_row1)
    gdf1.plot(column=metric1, ax=axs[0, 0], norm=norm, cmap=cmap, markersize=markersize)
    gdf2.plot(column=metric2, ax=axs[0, 1], norm=norm, cmap=cmap, markersize=markersize)
    cbtop = fig.colorbar(axs[0,1].collections[0], ax=axs[0,:], orientation='vertical', shrink=0.4, anchor=(-6.,0.2))
    axs[0,0].set_title(MLEE)
    axs[0,1].set_title(MESP)
    
    # Plot: bottom
    norm = TwoSlopeNorm(vmin=vmin_row2, vcenter=mean_row2, vmax=vmax_row2)
    gdfadm1.plot(column=metric1, ax=axs[1, 0], norm=norm, cmap=cmap)
    gdfadm2.plot(column=metric2, ax=axs[1, 1], norm=norm, cmap=cmap)
    cbottom = fig.colorbar(axs[1,1].collections[0], ax=axs[1,:], orientation='vertical', shrink=0.4, anchor=(1.8,0.5))
    
    for ax in axs.flatten():
        for collections in ax.collections:
            collections.set_rasterized(True)
    
    # Adjust spacing
    plt.tight_layout()
    plt.margins(x=0, y=0)
    plt.subplots_adjust(wspace=0., hspace=0.)
    plt.gca().set_aspect('equal')
    
    if output_dir is not None:
        fn = os.path.join(output_dir, f"maps_HQ_DIST_{COUNTRIES[country]['code3']}.pdf")
        fig.savefig(fn, dpi=dpi, bbox_inches='tight')
        print(f'{fn} saved!')
        
    # Show the plot
    plt.show()
    plt.close()
    
    
def plot_dist_combined(country, models, gdf_rwi, gdf_lee, gdf_iwi, boundary_fn, admin_level, output_dir):

    colors = {MLEE:'tab:blue', MESP:'tab:orange'}
    dpi = 300
    
    # Create figure and axes for a 2x2 grid of subplots
    fig, axes = plt.subplots(2,1, figsize=(5,7), 
                        facecolor='w',
                        constrained_layout=True, 
                        sharex=True, sharey=False)
    
    x_intersection_max = None
    for model in [MLEE, MESP]:
        
        gdf, metric = get_data_by_model(gdf_rwi, gdf_lee, gdf_iwi, model)
        
        #PDF
        ax = axes[0]
        values = gdf[metric]
        sns.kdeplot(values, label=model, color=colors[model], ax=ax)
        ax.set_ylabel('p(X=x)')
        ax.spines[['right', 'top']].set_visible(False)
        ax.legend()
        # ax.set_yscale('log')
        # ax.set_yticks([10**-3,  10**-2,  10**-1])
        # ax.set_ylim(20**-3, None)
    
        # ECDF
        ax = axes[1]
        sns.ecdfplot(values, label=model, color=colors[model], ax=ax)
        ax.set_ylabel('p(X$\leq$x)')
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_ylim(0.0, 1.01)
        ax.set_xlabel('Predicted IWI')
        
        iwipl = 35
        ecdf = ECDF(values) 
        x_intersection = np.percentile(values, iwipl) # 35th percentile (median)
        y_intersection = np.interp(x_intersection, ecdf.x, ecdf.y)
        ax.plot([x_intersection, x_intersection], [0, y_intersection], ls='--', c=colors[model])
        ax.plot([min(0,values.min()), x_intersection], [y_intersection, y_intersection], ls='--', c=colors[model])
        ax.scatter([x_intersection], [y_intersection], color='black', zorder=5)
        x_intersection_max = x_intersection if x_intersection_max is None else max((x_intersection, x_intersection_max))
        
    smooth = 3
    ax.text(x_intersection_max+smooth, y_intersection, f'IWI-{iwipl}\nPoverty Line', fontsize=12, rotation=0, va='top', ha='left')
                
            
    # Adjust spacing
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0, hspace=0.1)
    
    if output_dir is not None:
        fn = os.path.join(output_dir, f"dist_{COUNTRIES[country]['code3']}.pdf")
        fig.savefig(fn, dpi=dpi, bbox_inches='tight')
        print(f'{fn} saved!')
        
        
    plt.show()
    plt.close()
    
    
    
    




    

# def plot_maps(metadata, country, fig, axes, **kwargs):
#     years = kwargs.pop('years', None)
#     title = kwargs.pop('title', None)
#     suptitle = kwargs.pop('suptitle', False)
#     kind = kwargs.pop('kind', False)
#     samecolorbar = kwargs.pop('samecolorbar',False)
    
#     plt.suptitle(get_suptitle(kind) if suptitle else '', y=0.75 if title else 0.7)
    
#     cmap = kwargs.get('cmap', 'RdBu')
    
#     #hue_neg, hue_pos = 172, 35
#     #cmap = sns.diverging_palette(hue_neg, hue_pos, 50, 50, 1, center="dark", as_cmap=True)
    
#     # my_gradient = LinearSegmentedColormap.from_list('my_gradient', (
#     #                 # Edit this gradient at https://eltos.github.io/gradient/#0:00493E-49.5:D0ECE8-50:9D9D9D-50.5:F6E6C1-100:673B07
#     #                 (0.000, (0.000, 0.286, 0.243)),
#     #                 (0.495, (0.816, 0.925, 0.910)),
#     #                 (0.500, (0.616, 0.616, 0.616)),
#     #                 (0.505, (0.965, 0.902, 0.757)),
#     #                 (1.000, (0.404, 0.231, 0.027))))
#     # kwargs['cmap'] = my_gradient
    
#     legend = kwargs.pop('legend', True)
#     legend_kwds = kwargs.get('legend_kwds', {})
    
#     if samecolorbar:
#         min_iwi = min(metadata['iwi']['data'][metadata['iwi']['mean']].min(),metadata['lee']['data'][metadata['lee']['mean']].min())
#         max_iwi = max(metadata['iwi']['data'][metadata['iwi']['mean']].max(),metadata['lee']['data'][metadata['lee']['mean']].max())
#         mean_iwi = max(metadata['iwi']['data'][metadata['iwi']['mean']].mean(),metadata['lee']['data'][metadata['lee']['mean']].mean())
#         #std_iwi = max(metadata['iwi']['data'][metadata['iwi']['mean']].std(),metadata['lee']['data'][metadata['lee']['mean']].std())
#         kwargs['vmin'] = min_iwi
#         kwargs['vmax'] = max_iwi
    
#     legend_done = {'rwi':False, 'iwi':False, 'lee':False}
#     for key in ['iwi', 'rwi', 'lee']: # iwi first
#         obj = metadata[key]
#         ax = axes[obj['index']]
#         data = obj['data'].copy()
            
#         metric = 'iwi' if key=='lee' else key
        
#         post = "" if not years else f"\n({years[country][key]})"
#         # ax.set_title(f"{metric.upper()} by {obj['source']}{post}" if title else '')
#         ax.set_title(obj['source'] if title else '')
#         # ax.set_aspect('equal', 'box')
#         ax.set_axis_off()
        
#         if kind == NORM_MEAN:
#             # for each country, using their own wealth scale, we center the white color into the mean of their own wealth scores.
#             n, vmin, vcenter, vmax = data.shape[0], data[obj['mean']].min(), data[obj['mean']].mean(), data[obj['mean']].max()
#             norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
#             data.plot(column=obj['mean'], ax=ax, norm=norm, legend=True, **kwargs)
                
#         elif kind == NORM_RESCALE:
#             # we rescale RWI scores only, and plot as "raw"
#             col = 'rwi_rescaled' if key == 'rwi' else obj['mean']
#             data.plot(column=col, ax=ax,  legend=legend, **kwargs)
        
#         elif kind == NORM_RESCALE_MEAN:
#             # first we rescale RWI scores
#             col = 'rwi_rescaled' if key == 'rwi' else obj['mean']
            
#             # at this point, all model's results are in IWI domain/scale.
#             # 2. for each country, we center the white color into the mean of M3's IWI (Ours)
#             if key == 'iwi':
#                 vcenter = data[col].mean()
            
#             vmin, vmax = data[col].min(), data[col].max()
#             if samecolorbar:
#                 vmin = min_iwi
#                 vmax = max_iwi
                
#             norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
#             data.plot(column=col, ax=ax, norm=norm, legend=True, **kwargs)
            

#         else:
#             # here we don'r alter anything
#             data.plot(column=obj['mean'], ax=ax, legend=legend,  **kwargs)
        
#         # Legend titles
#         cbax = [ax2 for ax2 in fig.axes if ax2 not in axes and ax2!=ax][-1] #to get only the colorbar axis
#         #cbtitle = 'IWI pred.' if key in ['iwi','lee'] or kind in [NORM_RESCALE, NORM_RESCALE_MEAN] else 'RWI pred.' if kind not in [NORM_RESCALE_MEAN, NORM_RESCALE_MEAN] and key=='rwi' else 'IWI pred.'
#         cbtitle = '$IWI$ pred.' if key in ['iwi','lee'] else '$\hat{IWI}$ pred.' if kind in [NORM_RESCALE, NORM_RESCALE_MEAN] and key == 'rwi' else '$RWI$ pred.' if kind not in [NORM_RESCALE_MEAN, NORM_RESCALE_MEAN] and key=='rwi' else 'NA.'
#         cby = 1.12 if kind in [NORM_RESCALE, NORM_RESCALE_MEAN] and key == 'rwi' else 1.1
#         #cbax.text(s=cbtitle, x=1.9, y=1.1, ha='center', va='top', fontsize=12, color='grey', transform=cbax.transAxes)        
#         cbax.text(s=cbtitle, x=1.9, y=cby, ha='center', va='top', fontsize=12, color='grey', transform=cbax.transAxes)        



# def plot_comparison_admin_maps(gdf_rwi, gdf_lee, gdf_iwi, country, boundary_fn, admin_level, output_dir=None, **kwargs):
#     dpi = kwargs.pop('dpi', 300)
    
#     years = kwargs.pop('years', None)
#     title = kwargs.pop('title', False)
#     suptitle = kwargs.pop('suptitle', False)
#     max_distance = kwargs.pop('max_distance', 10)
#     kind = kwargs.pop('kind', None)
#     legend = kwargs.pop('legend', True)
#     samecolorbar = kwargs.pop('samecolorbar',False)
    
#     figsize = kwargs.pop('figsize', (15,5))
#     fig, axes = plt.subplots(1, 3, figsize=figsize) #, gridspec_kw={'width_ratios': [1,1]}, constrained_layout=True)
#     gdf_country = None
    
#     metadata = get_metadata_template(gdf_rwi, gdf_lee, gdf_iwi, rescale=True)    

#     if samecolorbar:
#         try:
#             # country
#             cc3 = COUNTRIES[country]['code3'].upper()
#             fn = boundary_fn.replace("<country>", country).replace('<ccode3>',cc3).replace('<ADM>',str(admin_level))
#             gdf_country = gpd.read_file(fn)
#             min_iwi = []
#             max_iwi = []
#             for key in ['iwi','lee']:
#                 obj = metadata[key]
#                 col = obj['mean']
#                 data = obj['data']

#                 tmp = geo.distribute_data_in_grid(data, gdf_country, column=col, aggfnc='mean', lsuffix='data', how='right')
#                 min_iwi.append(tmp[col].min())
#                 max_iwi.append(tmp[col].max())
            
#             min_iwi = min(min_iwi)
#             max_iwi = max(max_iwi)
#             kwargs['vmin'] = min_iwi
#             kwargs['vmax'] = max_iwi
#         except Exception as ex:
#             print(f"[ERROR] {ex}")

        
#     try:
#         # country
#         cc3 = COUNTRIES[country]['code3'].upper()
#         fn = boundary_fn.replace("<country>", country).replace('<ccode3>',cc3).replace('<ADM>',str(admin_level))
#         gdf_country = gpd.read_file(fn)
        
#         for key in ['iwi', 'rwi', 'lee']: # iwi always first
#             obj = metadata[key]
#             ax = axes[obj['index']]
            
#             col = 'rwi_rescaled' if key=='rwi' else obj['mean']
#             data = obj['data']
        
#             tmp = geo.distribute_data_in_grid(data, gdf_country, column=col, aggfnc='mean', lsuffix='data', how='right')
            
#             if kind == NORM_RESCALE_MEAN:
                
#                 # at this point, all model's results are in IWI domain/scale.
#                 # 2. for each country, we center the white color into the mean of M3's IWI (Ours)
#                 if key == 'iwi':
#                     vcenter = tmp[col].mean()

#                 vmin, vmax = tmp[col].min(), tmp[col].max()
#                 if samecolorbar:
#                     vmin = min_iwi
#                     vmax = max_iwi

#                 norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

#                 # n, vmin, vcenter, vmax, vstd = tmp.shape[0], tmp[col].min(), tmp[col].mean(), tmp[col].max(), tmp[col].std()
#                 # if samecolorbar:
#                 #     vmin = min_iwi
#                 #     vmax = max_iwi
#                 # norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

#                 tmp.plot(column=col, ax=ax, norm=norm, legend=legend, **kwargs)
                
#             else:
#                 tmp.plot(column=col, ax=ax, legend=legend, **kwargs)
                
#             # Legend titles
#             cbax = [ax2 for ax2 in fig.axes if ax2 not in axes and ax2!=ax][-1] #to get only the colorbar axis
#             cbtitle = '$IWI$ pred.' if key in ['iwi','lee'] else '$\hat{IWI}$ pred.' if kind in [NORM_RESCALE, NORM_RESCALE_MEAN] and key == 'rwi' else '$RWI$ pred.' if kind not in [NORM_RESCALE_MEAN, NORM_RESCALE_MEAN] and key=='rwi' else 'NA.'
#             cby = 1.14 if kind in [NORM_RESCALE, NORM_RESCALE_MEAN] and key == 'rwi' else 1.12
#             cbax.text(s=cbtitle, x=1.9, y=cby, ha='center', va='top', fontsize=12, color='grey', transform=cbax.transAxes)        

#             if title:
#                 post = "" if not years else f"\n({years[country][key]})"
#                 metric = 'iwi' if key == 'lee' else key
#                 # ax.set_title(f"{metric.upper()} by {obj['source']}{post}")
#                 ax.set_title(obj['source'])
                
#             # ax.set_aspect('equal', 'box')
#             ax.set_axis_off()
        
#         if suptitle:
#             ks = get_suptitle(kind)
#             ks = '' if ks in ['',None] else f" | {ks}"
#             plt.suptitle(f"MEAN WEALTH | ADMIN_LEVEL={admin_level}{ks}", y=0.75 if title else 0.7)
            
#         for ax in axes.flatten():
#             try:
#                 ax.collections[0].set_rasterized(True)
#             except:
#                 pass
            
#         # plt.subplots_adjust(wspace=0.1, hspace=0.1)
#         plt.tight_layout()
#         # plt.margins(x=0, y=0)

#         if output_dir is not None:
#             fn = os.path.join(output_dir, f"maps_{COUNTRIES[country]['code3']}_{kind}_admin{admin_level}.pdf")
#             fig.savefig(fn, dpi=dpi, bbox_inches='tight')
#             print(f'{fn} saved!')

#         plt.show()
#         plt.close()
        
#     except Exception as ex:
#         print(ex)
    
    
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


# def show_expected_differences(df_preliminaries, output_dir):
#     df_expectations = pd.DataFrame()
#     gini_smooth = 10 # (10) gini 0 to 100
#     gdp_smooth = 0.1 # normalized becasue GDP and IWI, RWI are in different ranges | Old:(100) gdp min 400 max 8000


#     for country, df in df_preliminaries.groupby('country'):

#         tmp_rwi = df.query("in_rwi == True")
#         tmp_lee = df.query("in_lee == True")
#         tmp_iwi = df.query("in_iwi == True")

#         # IWI
#         iwi_idxmax = tmp_iwi.dropna().actual_year.astype(int).idxmax()
        
#         print('IWI:', country, tmp_iwi.dropna().actual_year.max())
#         print('RWI:', country, tmp_rwi.dropna().actual_year.max())
#         print('LEE:', country, tmp_lee.dropna().actual_year.max())
            
#         # RWI vs IWI
#         rwi_idxmax = tmp_rwi.dropna().actual_year.astype(int).idxmax()
#         gini_diff_rwi_iwi = tmp_rwi.loc[rwi_idxmax,'actual_gini'] - tmp_iwi.loc[iwi_idxmax,'actual_gini']
#         str_gini_diff_rwi_iwi = 'higher Gini (more inequality)' if gini_diff_rwi_iwi>gini_smooth else 'lower Gini (less inequality)' if gini_diff_rwi_iwi<-gini_smooth else 'similar or slightly higher Gini' if gini_diff_rwi_iwi > 0 else 'similar or slightly lower Gini' if gini_diff_rwi_iwi < 0 else 'very similar Gini'
#         gdp_diff_rwi_iwi = (tmp_rwi.loc[rwi_idxmax,'gdp'] - tmp_iwi.loc[iwi_idxmax,'gdp']) / (tmp_rwi.loc[rwi_idxmax,'gdp'] + tmp_iwi.loc[iwi_idxmax,'gdp'])
#         str_gdp_diff_rwi_iwi = 'higher GDP (richer)' if gdp_diff_rwi_iwi>gdp_smooth else 'lower GDP (poorer)' if gdp_diff_rwi_iwi<-gdp_smooth else 'similar or slightly higher GDP' if gdp_diff_rwi_iwi > 0 else 'similar or slightly lower GDP' if gdp_diff_rwi_iwi < 0 else 'similar GDP'

#         # IWI-LEE vs IWI
#         if tmp_lee.shape[0] > 0:
#             lee_idxmax = tmp_lee.dropna().actual_year.astype(int).idxmax()
#             gini_diff_lee_iwi = tmp_lee.loc[lee_idxmax,'actual_gini'] - tmp_iwi.loc[iwi_idxmax,'actual_gini']
#             str_gini_diff_lee_iwi = 'higher Gini (more inequality)' if gini_diff_lee_iwi>gini_smooth else 'lower Gini (less inequality)' if gini_diff_lee_iwi<-gini_smooth else 'similar or slightly higher Gini' if gini_diff_lee_iwi > 0 else 'similar or slightly lower Gini' if gini_diff_lee_iwi < 0 else 'very similar Gini'
#             gdp_diff_lee_iwi = (tmp_lee.loc[lee_idxmax,'gdp'] - tmp_iwi.loc[iwi_idxmax,'gdp']) / (tmp_lee.loc[lee_idxmax,'gdp'] + tmp_iwi.loc[iwi_idxmax,'gdp'])
#             str_gdp_diff_lee_iwi = 'higher GDP (richer)' if gdp_diff_lee_iwi>gdp_smooth else 'lower GDP (poorer)' if gdp_diff_lee_iwi<-gdp_smooth else 'similar or slightly higher GDP' if gdp_diff_lee_iwi > 0 else 'similar or slightly lower GDP' if gdp_diff_lee_iwi < 0 else 'similar GDP'
#         else:
#             gini_diff_lee_iwi = None
#             str_gini_diff_lee_iwi = 'NA'
#             gdp_diff_lee_iwi = None
#             str_gdp_diff_lee_iwi = 'NA'
            
#         obj = {'country':country, 
#                'gini_rwi_iwi':str_gini_diff_rwi_iwi,
#                'gini_lee_iwi':str_gini_diff_lee_iwi,
#                'gdp_rwi_iwi':str_gdp_diff_rwi_iwi,
#                'gdp_lee_iwi':str_gdp_diff_lee_iwi,
#                'gini_rwi_iwi_value':gini_diff_rwi_iwi,
#                'gini_lee_iwi_value':gini_diff_lee_iwi,
#                'gdp_rwi_iwi_value':gdp_diff_rwi_iwi,
#                'gdp_lee_iwi_value':gdp_diff_lee_iwi,
#               }


#         df_expectations = pd.concat([df_expectations, pd.DataFrame(obj, index=[0])], ignore_index=True)

#     df_expectations.set_index('country', inplace=True)
    
#     if output_dir is not None:
#         fn = os.path.join(output_dir, 'data_expectations.tex')
#         df_expectations.to_latex(fn, index=True, float_format="{:.1f}".format)
#         print(f'{fn} saved!')

#     return df_expectations
                
# #         # RWI (Chi) wrt M3
# #         col = f'rwi_rescaled_{MESP}'
# #         n_rwi = gdf_rwi.shape[0]
# #         x_rwi_wrt_iwi = gdf_rwi[col].mean()
# #         s_rwi_wrt_iwi = gdf_rwi[col].std()
# #         sn_rwi_wrt_iwi = (1/n_rwi) * sum((gdf_rwi[col]-x_rwi_wrt_iwi)**2)
        
# #         # RWI (Chi) wrt M2
# #         col = f'rwi_rescaled_{MLEE}'
# #         n_rwi = gdf_rwi.shape[0]
# #         x_rwi_wrt_lee = gdf_rwi[col].mean()
# #         s_rwi_wrt_lee = gdf_rwi[col].std()
# #         sn_rwi_wrt_lee = (1/n_rwi) * sum((gdf_rwi[col]-x_rwi_wrt_lee)**2)
        
    
#         ### RWI vs IWI ###
#         col = f'rwi_rescaled_{MESP}'
        
#         # basic stats
#         n_rwi = gdf_rwi.shape[0]
#         x_rwi_wrt_iwi = gdf_rwi[col].mean()
#         s_rwi_wrt_iwi = gdf_rwi[col].std()
#         sn_rwi_wrt_iwi = (1/n_rwi) * sum((gdf_rwi[col]-x_rwi_wrt_iwi)**2)
        
#         # two-sample Kolmogorov-Smirnov test for goodness of fit
#         ks_rwi_iwi, ks_pv_rwi_iwi = stats.ks_2samp(gdf_rwi[col], gdf_iwi.pred_mean_wi)
        
#         # overlap
#         df_overlap_rwi_iwi = gpd.sjoin_nearest(gdf_rwi.to_crs(PROJ_MET), gdf_iwi.to_crs(PROJ_MET), how='inner', max_distance=max_distance, distance_col='distance')
#         df_overlap_rwi_iwi.loc[:,'diff'] = df_overlap_rwi_iwi.apply(lambda row: row.pred_mean_wi-row[col], axis=1) # negative: rwi > iwi 
#         on_chi_iwi = df_overlap_rwi_iwi.shape[0]
#         ox1_chi_iwi = df_overlap_rwi_iwi.pred_mean_wi.mean()
#         ox2_chi_iwi = df_overlap_rwi_iwi[col].mean()
#         os1_chi_iwi = df_overlap_rwi_iwi.pred_mean_wi.std()
#         os2_chi_iwi = df_overlap_rwi_iwi[col].std()
#         ro_chi_iwi,pv_chi_iwi = pearsonr(df_overlap_rwi_iwi[col], df_overlap_rwi_iwi.pred_mean_wi)
#         ps_chi_iwi = get_significance_stars(pv_chi_iwi)
#         ogini_iwi_chi =  gini_coefficient(df_overlap_rwi_iwi.pred_mean_wi) * 100
#         ogini_rwi_chi =  gini_coefficient(df_overlap_rwi_iwi[col]) * 100
#         osn1_chi = os1_chi / np.sqrt(on_chi)
#         osn2_chi = os2_chi / np.sqrt(on_chi)
#         oztest_chi = (ox1_chi - ox2_chi) / np.sqrt(osn1_chi**2 + osn2_chi**2)
#         rmse_chi = mean_squared_error(df_overlap_rwi_iwi[col], df_overlap_rwi_iwi.pred_mean_wi, squared=False)
        
        
#         ### RWI vs LEE ###
#         col = f'rwi_rescaled_{MLEE}'
        
#         # basic stats
#         n_rwi = gdf_rwi.shape[0]
#         x_rwi_wrt_lee = gdf_rwi[col].mean()
#         s_rwi_wrt_lee = gdf_rwi[col].std()
#         sn_rwi_wrt_lee = (1/n_rwi) * sum((gdf_rwi[col]-x_rwi_wrt_lee)**2)
        
#         # two-sample Kolmogorov-Smirnov test for goodness of fit
#         ks_rwi_lee, ks_pv_rwi_lee = stats.ks_2samp(gdf_rwi[col], gdf_lee.estimated_IWI)
        
#         # overlap
#         df_overlap_rwi_lee = gpd.sjoin_nearest(gdf_rwi.to_crs(PROJ_MET), gdf_lee.to_crs(PROJ_MET), how='inner', max_distance=max_distance, distance_col='distance')
#         df_overlap_rwi_lee.loc[:,'diff'] = df_overlap_rwi_lee.apply(lambda row: row.estimated_IWI-row[col], axis=1) # negative: rwi > lee 
#         on_chi_lee = df_overlap_rwi_lee.shape[0]
#         ox1_chi_lee = df_overlap_rwi_lee.pred_mean_wi.mean()
#         ox2_chi_lee = df_overlap_rwi_lee[col].mean()
#         os1_chi_lee = df_overlap_rwi_lee.pred_mean_wi.std()
#         os2_chi_lee = df_overlap_rwi_lee[col].std()
#         ro_chi_lee,pv_chi_lee = pearsonr(df_overlap_rwi_lee[col], df_overlap_rwi_lee.estimated_IWI)
#         ps_chi_lee = get_significance_stars(pv_chi_lee)
#         ogini_lee_chi =  gini_coefficient(df_overlap_rwi_lee.estimated_IWI) * 100
#         ogini_rwi_chi =  gini_coefficient(df_overlap_rwi_lee[col]) * 100
#         osn1_chi_lee = os1_chi_lee / np.sqrt(on_chi_lee)
#         osn2_chi_lee = os2_chi_lee / np.sqrt(on_chi_lee)
#         oztest_chi_lee = (ox1_chi_lee - ox2_chi_lee) / np.sqrt(osn1_chi_lee**2 + osn2_chi_lee**2)
#         rmse_chi_lee = mean_squared_error(df_overlap_rwi_lee[col], df_overlap_rwi_lee.estimated_IWI, squared=False)
        
        
        
#         ### IWI (Lee) vs IWI (us) ###
#         # ztest_lee_iwi = (x_iwi - x_lee) / np.sqrt((sn_iwi/n_iwi) + (sn_lee/n_lee))
#         ks_lee_iwi, ks_pv_lee_iwi = stats.ks_2samp(gdf_lee.estimated_IWI, gdf_iwi.pred_mean_wi)
        
#         # overlap
#         df_overlap_lee = gpd.sjoin_nearest(gdf_lee.to_crs(PROJ_MET), gdf_iwi.to_crs(PROJ_MET), how='inner', max_distance=max_distance, distance_col='distance')
#         df_overlap_lee.loc[:,'diff'] = df_overlap_lee.apply(lambda row: row.pred_mean_wi-row.estimated_IWI, axis=1) # negative: iwi (lee) > iwi (us) 
#         on_lee = df_overlap_lee.shape[0]
#         ox1_lee = df_overlap_lee.pred_mean_wi.mean()
#         ox2_lee = df_overlap_lee.estimated_IWI.mean()
#         os1_lee = df_overlap_lee.pred_mean_wi.std()
#         os2_lee = df_overlap_lee.estimated_IWI.std()
#         ro_lee,pv_lee = pearsonr(df_overlap_lee.estimated_IWI, df_overlap_lee.pred_mean_wi)
#         ps_lee = get_significance_stars(pv_lee)
#         ogini_iwi_lee =  gini_coefficient(df_overlap_lee.pred_mean_wi) * 100
#         ogini_rwi_lee =  gini_coefficient(df_overlap_lee.estimated_IWI) * 100
#         osn1_lee = os1_lee / np.sqrt(on_lee)
#         osn2_lee = os2_lee / np.sqrt(on_lee)
#         oztest_lee = (ox1_lee - ox2_lee) / np.sqrt(osn1_lee**2 + osn2_lee**2)
#         rmse_lee = mean_squared_error(df_overlap_lee.estimated_IWI, df_overlap_lee.pred_mean_wi, squared=False)

        
#         obj = {'country': country,
               
#                'chi_years': survey_years[country]['rwi'],
#                'chi_n': n_rwi,
#                'chi_mean': x_rwi,    # rescaled
#                'chi_std': s_rwi,     # rescaled
#                'chi_gini': gini_rwi, # rescaled
               
#                'lee_years': survey_years[country]['lee'],
#                'lee_n': n_lee,
#                'lee_mean': x_lee,
#                'lee_std': s_lee,
#                'lee_gini':gini_lee,
               
#                'us_years': survey_years[country]['iwi'],
#                'us_n': n_iwi,
#                'us_mean': x_iwi,
#                'us_std': s_iwi,
#                'us_gini':gini_iwi,
               
#                'diff_chi_gini_expected': df_ex.gini_rwi_iwi_value,
#                'diff_chi_gini_obtained': gini_rwi - gini_iwi,
               
#                'diff_lee_gini_expected': df_ex.gini_lee_iwi_value,
#                'diff_lee_gini_obtained': gini_lee - gini_iwi,
               
#                'diff_chi_wealth_expected': df_ex.gdp_rwi_iwi_value,
#                'diff_chi_wealth_obtained': (x_rwi - x_iwi) / (x_rwi + x_iwi),
#                # 'ztest_chi_wealth_obtained': ztest_rwi_iwi,
#                'ks_rwi_iwi':f"{ks_rwi_iwi:.2f} {get_significance_stars(ks_pv_rwi_iwi)}",
               
#                'diff_lee_wealth_expected': df_ex.gdp_lee_iwi_value,
#                'diff_lee_wealth_obtained': (x_lee - x_iwi) / (x_lee + x_iwi),
#                # 'ztest_lee_wealth_obtained': ztest_lee_iwi,
               # 'ks_lee_iwi':f"{ks_lee_iwi:.2f} {get_significance_stars(ks_pv_lee_iwi)}",
               
#                'overlap_chi_n': on_chi,
#                'overlap_chi_rmse':rmse_chi,
#                'overlap_chi_mean_rwi': ox2_chi,
#                'overlap_chi_mean_iwi': ox1_chi,
#                'overlap_chi_mean_diff': ox1_chi - ox2_chi,
#                'overlap_chi_mean_diff_norm': (ox1_chi - ox2_chi) / (ox1_chi + ox2_chi),
#                'overlap_chi_mean_ztest': oztest_chi,
#                'overlap_chi_mean_ro': f"{ro_chi:.2f} {ps_chi}",
#                'overlap_chi_gini_rwi': ogini_rwi_chi,
#                'overlap_chi_gini_iwi': ogini_iwi_chi,
#                'overlap_chi_gini_diff': ogini_rwi_chi - ogini_iwi_chi, 
               
#                'overlap_lee_n': on_lee,
#                'overlap_lee_rmse':rmse_lee,
#                'overlap_lee_mean_rwi': ox2_lee,
#                'overlap_lee_mean_iwi': ox1_lee,
#                'overlap_lee_mean_diff': ox1_lee - ox2_lee,
#                'overlap_lee_mean_diff_norm': (ox1_lee - ox2_lee) / (ox1_lee + ox2_lee),
#                'overlap_lee_mean_ztest': oztest_lee,
#                'overlap_lee_mean_ro': f"{ro_lee:.2f} {ps_lee}",
#                'overlap_lee_gini_rwi': ogini_rwi_lee,
#                'overlap_lee_gini_iwi': ogini_iwi_lee,
#                'overlap_lee_gini_diff': ogini_rwi_lee - ogini_iwi_lee, 
#               }
        
            
#         df_summary = pd.concat([df_summary, pd.DataFrame(obj, index=[1])], ignore_index=True)
        
#     if output_dir is not None:
#         fn = os.path.join(output_dir, 'summary.tex')
#         df_summary.set_index('country').to_latex(fn, index=True, float_format="{:.2f}".format)
#         print(f'{fn} saved!')
        
#     return df_summary