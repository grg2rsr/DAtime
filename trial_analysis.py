"""
 
                                         
   _ __  _   _ _ __ _ __   ___  ___  ___ 
  | '_ \| | | | '__| '_ \ / _ \/ __|/ _ \
  | |_) | |_| | |  | |_) | (_) \__ \  __/
  | .__/ \__,_|_|  | .__/ \___/|___/\___|
  |_|              |_|                   
 
"""


# %%
%matplotlib qt5

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 331
import seaborn as sns

import sys,os
import pickle
from copy import copy
from tqdm import tqdm
from pathlib import Path

import scipy as sp
import numpy as np
import pandas as pd
import neo
import elephant as ele
# import spikeglx as glx
import quantities as pq

from helpers import *

# %% 

folder = Path("/home/georg/data/2019-12-03_JP1355_GR_full_awake_2/stim3_g0/")
# folder = Path("/home/georg/data/2020-03-04_GR_JP2111_full/stim1_g0")

os.chdir(folder)
import analysis_params

bin_file = folder / analysis_params.bin_file
path = bin_file.with_suffix('.neo.zfr.sliced')

with open(path,'rb') as fH:
    print("reading ", path)
    Segs = pickle.load(fH)
    print("... done")

# read stim related info
stim_path = folder / analysis_params.stim_file
stim_map_path = folder / analysis_params.stim_map_file

StimMap = pd.read_csv(stim_map_path, delimiter=',')
StimsDf = pd.read_csv(stim_path, index_col=0, delimiter=',')


"""
 
                         _   
    ___ ___  _   _ _ __ | |_ 
   / __/ _ \| | | | '_ \| __|
  | (_| (_) | |_| | | | | |_ 
   \___\___/ \__,_|_| |_|\__|
                             
 
"""
# %% 

nTrials = len(Segs)
nUnits = len(Segs[0].spiketrains)

# gather pre spikes
nSpikes = sp.zeros((nUnits,nTrials,2)) # last axis is pre, post
for i in tqdm(range(nTrials),desc="counting spikes"):
    seg = Segs[i]
    try:
        vpl_stim, = select(seg.epochs, 'VPL_stims')
        t_post = vpl_stim.times[-1] + vpl_stim.durations[-1]
    except:
        t_post = 0 * pq.s # this calculates the diff through DA only stim

    # pre
    seg_sliced = seg.time_slice(-1*pq.s, 0*pq.s)
    nSpikes[:,i,0] = [len(st) for st in seg_sliced.spiketrains]

    # post
    seg_sliced = seg.time_slice(t_post, t_post + 2*pq.s)
    nSpikes[:,i,1] = [len(st)/2 for st in seg_sliced.spiketrains]

    # seg_sliced = seg.time_slice(t_post + 0.1*pq.s, t_post + 2.9*pq.s)
    # nSpikes[:,i,1] = [len(st)/2.8 for st in seg_sliced.spiketrains]

# save the result
out_path = bin_file.with_suffix('.pre_post_spikes.npy')
sp.save(out_path,nSpikes)


"""
 
       _                  
    __| | ___  _ __   ___ 
   / _` |/ _ \| '_ \ / _ \
  | (_| | (_) | | | |  __/
   \__,_|\___/|_| |_|\___|
                          
 
"""






"""
 
   _                 _ 
  | | ___   __ _  __| |
  | |/ _ \ / _` |/ _` |
  | | (_) | (_| | (_| |
  |_|\___/ \__,_|\__,_|
                       
 
"""
# %%
path = bin_file.with_suffix('.pre_post_spikes.npy')
nSpikes = sp.load(path)
dSpikes = nSpikes[:,:,1] - nSpikes[:,:,0]


"""
 
             __                            _   
   _ __ ___ / _| ___  _ __ _ __ ___   __ _| |_ 
  | '__/ _ \ |_ / _ \| '__| '_ ` _ \ / _` | __|
  | | |  __/  _| (_) | |  | | | | | | (_| | |_ 
  |_|  \___|_|  \___/|_|  |_| |_| |_|\__,_|\__|
                                               
 
"""

# %% 
nTrials = len(Segs)
unit_ids = [st.annotations['id'] for st in Segs[0].spiketrains]

Data = pd.DataFrame(dSpikes.T,columns=unit_ids,index=range(nTrials))

# lagged prev reg
# StimsDf['prev_blue'] = sp.roll(StimsDf['blue'],-1)

Df = pd.concat([StimsDf,Data],axis=1)
Dfm = pd.melt(Df,id_vars=StimsDf.columns,var_name='unit_id',value_name='dSpikes')
Dfm['stim_id'] = pd.Categorical(Dfm['stim_id'])

Dfm['pre'] = pd.melt(pd.DataFrame(nSpikes[:,:,0].T))['value']
Dfm['post'] = pd.melt(pd.DataFrame(nSpikes[:,:,1].T))['value']

Df = Dfm.drop('dSpikes',axis=1)
Dfmm = pd.melt(Df,id_vars=Df.columns[:-2],var_name='when',value_name='nSpikes')

cat = pd.Categorical(Dfmm['when'])
# cat = cat.as_ordered(cat)
cat = cat.reorder_categories(['pre','post'])
Dfmm['when'] = cat

"""
 
       _        _                           _      _     
   ___| |_ __ _| |_ ___ _ __ ___   ___   __| | ___| |___ 
  / __| __/ _` | __/ __| '_ ` _ \ / _ \ / _` |/ _ \ / __|
  \__ \ || (_| | |_\__ \ | | | | | (_) | (_| |  __/ \__ \
  |___/\__\__,_|\__|___/_| |_| |_|\___/ \__,_|\___|_|___/
                                                         
 
"""
# %% linear regression for finding how many dSpikes per stim class
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy

formula = "nSpikes ~ 1 + when"

StatsDf = pd.DataFrame(columns=['unit_id','stim_id','p','m'])
Df_pvalues = []
Df_params = []
for stim_id in pd.unique(Dfm['stim_id']):
    for unit in tqdm(unit_ids):
        data = Dfmm.groupby(['unit_id','stim_id','opto']).get_group((unit,stim_id,'red'))

        dmatrix = patsy.dmatrices(formula, data=data)[1]

        model = smf.ols(formula=formula, data=data)
        res = model.fit()

        pvalues = res.pvalues
        params = res.params

        pvalues['unit'] = unit
        params['unit'] = unit

        pvalues['stim_id'] = stim_id
        params['stim_id'] = stim_id

        Df_pvalues.append(pvalues)
        Df_params.append(params)

Df_pvalues = pd.DataFrame(Df_pvalues)
Df_params = pd.DataFrame(Df_params)

# %% distilling this to df stats
key  = 'when[T.post]'
StatsDf = pd.concat([Df_params[['unit','stim_id',key]],Df_pvalues[key]],axis=1)
StatsDf.columns = ['unit_id','stim_id','m','p']
StatsDf[['unit_id','stim_id']] = StatsDf[['unit_id','stim_id']].astype('int')

Sts = Segs[0].spiketrains

depthDf = pd.DataFrame([(st.annotations['id'],st.annotations['area']) for st in Sts],columns=['unit_id','area'])
StatsDf = pd.merge(StatsDf,depthDf,on='unit_id')

alpha = 0.05
StatsDf['sig'] = StatsDf['p'] < alpha
StatsDf['upmod'] = sp.logical_and(StatsDf['sig'].values,(StatsDf['m'] > 0).values)

out_path = bin_file.with_suffix('.stim_stats.csv')
StatsDf.to_csv(out_path)


StatsDf.groupby(('stim_id','area')).sum()

# %% or load

out_path = bin_file.with_suffix('.stim_stats.csv')
StatsDf = pd.read_csv(out_path)

"""
 
       _        _               _     
   ___| |_ __ _| |_ ___  __   _(_)___ 
  / __| __/ _` | __/ __| \ \ / / / __|
  \__ \ || (_| | |_\__ \  \ V /| \__ \
  |___/\__\__,_|\__|___/   \_/ |_|___/
                                      
 
"""

# %% connected dots plot
fig, axes = plt.subplots(ncols=3,sharey=True,figsize=[2,5])
stim_id = 1
nStims  = pd.unique(StimsDf['stim_id']).shape[0]

for k, stim_id in enumerate(range(nStims)):

    data = Dfm.groupby(['stim_id','opto']).get_group((stim_id,'red'))

    sig_mod_unit_ids = pd.unique(StatsDf.groupby(('stim_id','upmod')).get_group((stim_id,True))['unit_id'])

    d = data[['unit_id','pre','post']].groupby('unit_id').mean()
    for i,row in d.iterrows():
        unit_id = row.name
        if unit_id in sig_mod_unit_ids:
            color = 'red'
        else:
            color = 'gray'
        axes[k].plot([0,1],[row['pre'],row['post']],lw=1,color=color,alpha=0.25)


fig.tight_layout()

# %%
# adding area info to Dfm to make my life easier
for i,row in StatsDf.groupby('stim_id').get_group(stim_id).iterrows():
    Dfm.loc[Dfm['unit_id'] == row['unit_id'],'area'] = row['area']

# %% STRIP PLOT 
stim_id = 1
area = 'STR'

sort_ids = Dfm.groupby(('area','unit_id')).mean().loc[area,'dSpikes'].sort_values().index

# get corresponding indices to unit_ids
all_ids = [st.annotations['id'] for st in Segs[0].spiketrains]
sort_ix = [all_ids.index(id) for id in sort_ids]

nUnits = len(sort_ids)

StatsDf['downmod'] = sp.logical_and(StatsDf['m'] < 0, StatsDf['p'] < 0.05)
stats = StatsDf.groupby(('area','stim_id')).get_group((area,stim_id))
stats['colors'] = 'gray'

cmap = plt.get_cmap('PiYG')

stats.loc[stats['upmod']==True,'colors'] = 'deepskyblue'
stats.loc[stats['downmod']==True,'colors'] = '#d62728'
stats.index = stats['unit_id']
colors = [stats.loc[id,'colors'] for id in sort_ids]

data = Dfm.groupby(('area','stim_id','opto')).get_group((area,stim_id,'red'))


gkw = dict(width_ratios=(1,0.025))
fig, axes = plt.subplots(nrows=2,ncols=2,figsize=[9,3],gridspec_kw=gkw)

plot = sns.stripplot(ax=axes[0,0], x='unit_id', order=sort_ids, y='dSpikes',data=data,size=1)
# axes.axhline(0,lw=1,linestyle=':',color='k',alpha=0.5)

for i,collection in enumerate(plot.collections):
    collection.set_facecolor(colors[i])

axes[0,0].set_ylim(-45.45)
axes[0,0].set_ylabel('∆spikes')
axes[0,0].set_xlabel('')
axes[0,0].set_xticks([])
sns.despine(ax=axes[0,0],bottom=True)

# fig, axes = plt.subplots(nrows=3,ncols=2,figsize=[5,5],gridspec_kw=gkw)
# fig, axes = plt.subplots(ncols=2, figsize=[11,3], gridspec_kw=gkw)

# get corresponding indices to unit_ids
all_ids = [st.annotations['id'] for st in Segs[0].spiketrains]
sort_ix = [all_ids.index(id) for id in sort_ids]
stim_inds = StimsDf.groupby(('stim_id','opto')).get_group((stim_id,'red')).index

im = axes[1,0].matshow(dSpikes[sort_ix,:][:,stim_inds].T,vmin=-15,vmax=15,cmap='PiYG')
cbar = fig.colorbar(im, cax=axes[1,1], shrink=0.75,label='∆spikes')

axes[1,0].set_xlabel('units')
axes[1,0].set_ylabel('stim #')
axes[1,0].set_aspect('auto')

axes[0,1].remove()
fig.tight_layout()
fig.savefig('/home/georg/Desktop/ciss/stripandpepper.png',dpi=331)













"""
 
                                 _   _       _ _                 _     
    __ ___   ____ _    __ _  ___| |_(_)_   _(_) |_ _   _  __   _(_)___ 
   / _` \ \ / / _` |  / _` |/ __| __| \ \ / / | __| | | | \ \ / / / __|
  | (_| |\ V / (_| | | (_| | (__| |_| |\ V /| | |_| |_| |  \ V /| \__ \
   \__,_| \_/ \__, |  \__,_|\___|\__|_| \_/ |_|\__|\__, |   \_/ |_|___/
              |___/                                |___/               
 
"""
# %% 

nUnits = len(Segs[0].spiketrains)
nTrials = len(Segs)

Rates = sp.zeros((nUnits,nTrials),dtype='object')
ix = sp.arange(nTrials)

for i,j in enumerate(ix):
    Rates[:,i] = Segs[j].analogsignals

# %% splitting
import colorcet as cc
# plotting helper
def plot_average_rates(Rates, stim_inds, unit_inds, order=None, axes=None):
    """ takes the Rates array with all info and plots only the subset """
    Rates_ = Rates[:,stim_inds]
    Rates_ = Rates_[unit_inds,:]

    nUnits = Rates_.shape[0]
    nTrials = Rates_.shape[1]

    r_avgs = sp.zeros((Rates_[0,0].shape[0],nUnits))
    for u in range(nUnits):
        r = average_asigs(Rates_[u,:])
        r_avgs[:,u] = r.magnitude.flatten()

    if order is None:
        sort_inds = sp.argsort(sp.argmax(r_avgs,0))[::-1]
    else:
        sort_inds = order 

    ext = (r.times[0],r.times[-1],0,nUnits)
    im = axes.matshow(r_avgs.T[sort_inds,:],cmap='viridis',origin='bottom',extent=ext,vmin=-1,vmax=3)
    axes.set_aspect('auto')

    return axes, im, r_avgs, sort_inds

# gather indices
stim_id = 1
area = 'STR'
stim_inds = StimsDf.groupby(('opto','stim_id')).get_group(('red',stim_id)).index
stim_inds_opto = StimsDf.groupby(('opto','stim_id')).get_group(('both',stim_id)).index

sig_mod_unit_ids = StatsDf.groupby(('area','stim_id','sig')).get_group((area,stim_id,True))['unit_id'].unique()

# get corresponding indices to unit_ids
all_ids = [st.annotations['id'] for st in Segs[0].spiketrains]
sig_mod_unit_ix = [all_ids.index(id) for id in sig_mod_unit_ids]

gkw = dict(height_ratios=(1,0.075))
fig, axes = plt.subplots(nrows=2,ncols=3,figsize=[9,3.6],gridspec_kw=gkw)

ax, im, r_avgs_vpl, order = plot_average_rates(Rates, stim_inds, sig_mod_unit_ix, axes=axes[0,0])
fig.colorbar(im,cax=axes[1,0],orientation='horizontal',label="firing rate(z)", shrink=0.8)

ax, im, r_avgs_da, order = plot_average_rates(Rates, stim_inds_opto, sig_mod_unit_ix, order=order, axes=axes[0,1])
fig.colorbar(im,cax=axes[1,1],orientation='horizontal',label="firing rate(z)", shrink=0.8)

r_avgs_d = r_avgs_da - r_avgs_vpl
r = Rates[0,0]
ext = (r.times[0],r.times[-1],0,r_avgs_d.shape[1])
im = axes[0,2].matshow(r_avgs_d.T[order,:],cmap='PiYG',extent=ext,origin='bottom',vmin=-1,vmax=1)
axes[0,2].set_aspect('auto')
fig.colorbar(im,cax=axes[1,2],orientation='horizontal',label="difference", shrink=0.8)

vpl_stim, = select(Segs[stim_inds[0]].epochs,'VPL_stims')
da_stim, = select(Segs[stim_inds_opto[0]].epochs,'DA_stims')

for ax in axes[0,:]:
    add_epoch(ax,vpl_stim,color='firebrick', linewidth=0.5)
    add_epoch(ax,vpl_stim,color='firebrick',above=True, linewidth=1)

# add_epoch(axes[0,1],da_stim,color='darkcyan',above=True)
# add_epoch(axes[0,2],da_stim,color='darkcyan',above=True)


for ax in axes[0,:]:
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('time (s)')
axes[0,0].set_ylabel('units')


axes[0,0].set_title('VPL',fontsize=10)
axes[0,1].set_title('VPL/SNc',fontsize=10)
axes[0,2].set_title('VPL/SNc - VPL stim',fontsize=10)

axes[0,1].set_yticklabels([])
axes[0,2].set_yticklabels([])

for ax in axes[0,:]:
    ax.set_xlim(-0.0,1)

fig.tight_layout()
# fig.subplots_adjust(wspace=0.01,hspace=0.3)
# fig.savefig('/home/georg/Desktop/ciss/avg_activity.png',dpi=331)    












"""
 
   _       __ _                          
  | | ___ / _| |_ _____   _____ _ __ ___ 
  | |/ _ \ |_| __/ _ \ \ / / _ \ '__/ __|
  | |  __/  _| || (_) \ V /  __/ |  \__ \
  |_|\___|_|  \__\___/ \_/ \___|_|  |___/
                                         
 
"""


# %% cmass sorting - just keep for later
cs = sp.cumsum(r_avgs - np.min(r_avgs,axis=0)[sp.newaxis,:],0)
mid_val = (cs[-1,:] - cs[0,:])/2
mid_inds = sp.argmin(sp.absolute(cs - mid_val[sp.newaxis,:]),axis=0)
sort_inds = sp.argsort(mid_inds)[::-1]
all_inds = all_inds[sort_inds]

