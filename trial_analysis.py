"""
 
                                         
   _ __  _   _ _ __ _ __   ___  ___  ___ 
  | '_ \| | | | '__| '_ \ / _ \/ __|/ _ \
  | |_) | |_| | |  | |_) | (_) \__ \  __/
  | .__/ \__,_|_|  | .__/ \___/|___/\___|
  |_|              |_|                   
 
"""


# %%
%matplotlib qt5
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 166
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
import utils

# %% file dialog
bin_path = utils.get_file_dialog()

# %% previous
bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-20_1a_JJP-00875_wt/stim1_g0/stim1_g0_t0.imec.ap.bin")
bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-22_3b_JJP-00869_wt/stim1_g0/stim1_g0_t0.imec.ap.bin")
bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-22_4a_JJP-00871_wt/stim1_g0/stim1_g0_t0.imec.ap.bin")
bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-23_4b_JJP-00871_wt/stim1_g0/stim1_g0_t0.imec.ap.bin")
bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-25_7a_JJP-00874_dh/stim1_g0/stim1_g0_t0.imec.ap.bin")
bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-26_7b_JJP-00874_dh/stim1_g0/stim1_g0_t0.imec.ap.bin")
bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-26_8a_JJP-00876_dh/stim1_g0/stim1_g0_t0.imec.ap.bin")

# from here 1.0
bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-27_8b_JJP-00876_dh/stim1_g0/stim1_g0_imec0/stim1_g0_t0.imec0.ap.bin")
# bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-29_10a_JJP-00870_dh/stim2_g0/stim2_g0_imec0/stim2_g0_t0.imec0.ap.bin")

# local 
# bin_path = Path("/media/georg/data/batch10/2020-06-27_8b_JJP-00876_dh/stim1_g0/stim1_g0_imec0/stim1_g0_t0.imec0.ap.bin")

folder = bin_path.parent

# %% read data

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

TrialInfo = pd.read_csv(folder / "TrialInfo.csv",index_col=0)
UnitInfo = pd.read_csv(folder / "UnitInfo.csv")

"""
 
 #### ########  ######## ##    ## ######## #### ######## ##    ##    ##     ##  #######  ########      ######  ######## ##       ##        ######  
  ##  ##     ## ##       ###   ##    ##     ##  ##        ##  ##     ###   ### ##     ## ##     ##    ##    ## ##       ##       ##       ##    ## 
  ##  ##     ## ##       ####  ##    ##     ##  ##         ####      #### #### ##     ## ##     ##    ##       ##       ##       ##       ##       
  ##  ##     ## ######   ## ## ##    ##     ##  ######      ##       ## ### ## ##     ## ##     ##    ##       ######   ##       ##        ######  
  ##  ##     ## ##       ##  ####    ##     ##  ##          ##       ##     ## ##     ## ##     ##    ##       ##       ##       ##             ## 
  ##  ##     ## ##       ##   ###    ##     ##  ##          ##       ##     ## ##     ## ##     ##    ##    ## ##       ##       ##       ##    ## 
 #### ########  ######## ##    ##    ##    #### ##          ##       ##     ##  #######  ########      ######  ######## ######## ########  ######  
 
"""

# %% by counting spikes pre / post

nTrials = TrialInfo.shape[0]
nUnits = UnitInfo.shape[0]

extra_offset = 0.1 *pq.s # to avoid counting the artifact

# gather pre spikes
nSpikes = sp.zeros((nUnits,nTrials,2)) # last axis is pre, post
for i in tqdm(range(nTrials),desc="counting spikes"):
    seg = Segs[i]
    try:
        vpl_stim, = select(seg.epochs, 'VPL_stims')
        t_post = vpl_stim.times[-1] + vpl_stim.durations[-1] + extra_offset
    except:
        t_post = 0 * pq.s # this calculates the diff through DA only stim

    # pre
    seg_sliced = seg.time_slice(-1*pq.s, 0*pq.s - extra_offset)
    nSpikes[:,i,0] = [len(st)/0.9 for st in seg_sliced.spiketrains]

    # post
    # seg_sliced = seg.time_slice(t_post + extra_offset, t_post + 2*pq.s + extra_offset)
    # nSpikes[:,i,1] = [len(st)/2 for st in seg_sliced.spiketrains]

    seg_sliced = seg.time_slice(t_post + 0.1*pq.s, t_post + 2.9*pq.s)
    nSpikes[:,i,1] = [len(st)/2.8 for st in seg_sliced.spiketrains]

dSpikes = nSpikes[:,:,1] - nSpikes[:,:,0]

# save the result
out_path = bin_file.with_suffix('.pre_post_spikes.npy')
sp.save(out_path,nSpikes)


"""
 
 ########   #######  ##    ## ######## 
 ##     ## ##     ## ###   ## ##       
 ##     ## ##     ## ####  ## ##       
 ##     ## ##     ## ## ## ## ######   
 ##     ## ##     ## ##  #### ##       
 ##     ## ##     ## ##   ### ##       
 ########   #######  ##    ## ######## 
 
"""










"""
 
 ##        #######     ###    ########  
 ##       ##     ##   ## ##   ##     ## 
 ##       ##     ##  ##   ##  ##     ## 
 ##       ##     ## ##     ## ##     ## 
 ##       ##     ## ######### ##     ## 
 ##       ##     ## ##     ## ##     ## 
 ########  #######  ##     ## ########  
 
"""

# %%
path = bin_file.with_suffix('.pre_post_spikes.npy')
nSpikes = sp.load(path)
dSpikes = nSpikes[:,:,1] - nSpikes[:,:,0]

# %% dSpikes vis
fig, axes = plt.subplots(figsize=[5,5])
sort_inds = sp.argsort(sp.average(dSpikes,axis=1))
axes.matshow(dSpikes[sort_inds,:],cmap='PiYG',vmin=-3,vmax=3)
axes.set_xlabel('trial #')
axes.set_ylabel('units')

# %%
"""
 
 ########  ######## ########  #######  ########  ##     ##    ###    ######## 
 ##     ## ##       ##       ##     ## ##     ## ###   ###   ## ##      ##    
 ##     ## ##       ##       ##     ## ##     ## #### ####  ##   ##     ##    
 ########  ######   ######   ##     ## ########  ## ### ## ##     ##    ##    
 ##   ##   ##       ##       ##     ## ##   ##   ##     ## #########    ##    
 ##    ##  ##       ##       ##     ## ##    ##  ##     ## ##     ##    ##    
 ##     ## ######## ##        #######  ##     ## ##     ## ##     ##    ##    

melting dataset for linear regression w statsmodels 
"""

nTrials = len(Segs)
unit_ids = [st.annotations['id'] for st in Segs[0].spiketrains]

Data = pd.DataFrame(dSpikes.T,columns=unit_ids,index=range(nTrials))

# lagged prev reg
# TrialInfo['prev_blue'] = sp.roll(TrialInfo['blue'],-1)

Df = pd.concat([TrialInfo,Data],axis=1)
Dfm = pd.melt(Df,id_vars=TrialInfo.columns,var_name='unit_id',value_name='dSpikes')
Dfm['stim_id'] = pd.Categorical(Dfm['stim_id'])

Dfm['pre'] = pd.melt(pd.DataFrame(nSpikes[:,:,0].T))['value']
Dfm['post'] = pd.melt(pd.DataFrame(nSpikes[:,:,1].T))['value']

Df = Dfm.drop('dSpikes',axis=1)
Dfmm = pd.melt(Df,id_vars=Df.columns[:-2],var_name='when',value_name='nSpikes')

cat = pd.Categorical(Dfmm['when'])
# cat = cat.as_ordered(cat)
cat = cat.reorder_categories(['pre','post'])
Dfmm['when'] = cat


# %% linear regression for finding how many dSpikes per stim class
"""
 
  ######  ########    ###    ########  ######  ##     ##  #######  ########  ######## ##        ######  
 ##    ##    ##      ## ##      ##    ##    ## ###   ### ##     ## ##     ## ##       ##       ##    ## 
 ##          ##     ##   ##     ##    ##       #### #### ##     ## ##     ## ##       ##       ##       
  ######     ##    ##     ##    ##     ######  ## ### ## ##     ## ##     ## ######   ##        ######  
       ##    ##    #########    ##          ## ##     ## ##     ## ##     ## ##       ##             ## 
 ##    ##    ##    ##     ##    ##    ##    ## ##     ## ##     ## ##     ## ##       ##       ##    ## 
  ######     ##    ##     ##    ##     ######  ##     ##  #######  ########  ######## ########  ######  
 
"""

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

# this is dangerous but possible bc the order is preserved
StatsDf['area'] = UnitInfo['area']

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

StatsDf.groupby(['stim_id','area']).sum()

# %% to check - go into this via inspection
# StatsDf
unit_id = 6
stim_id = 1

stim_inds = TrialInfo.groupby('stim_id').get_group(stim_id).index

from helpers import select
Sts = []
for i in stim_inds:
    Sts.append(select(Segs[i].spiketrains, unit_id, key='id')[0])

# plot raster
fig, axes = plt.subplots()
ysep = 0.1
for i,st in enumerate(Sts):
    axes.plot(st.times.magnitude, sp.ones(st.times.shape[0]) + i*ysep,'.',color='k')

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
stim_id = 0
nStims  = pd.unique(TrialInfo['stim_id']).shape[0]

for k, stim_id in enumerate(range(nStims)):

    data = Dfm.groupby(['stim_id','opto']).get_group((stim_id,'red'))

    sig_mod_unit_ids = pd.unique(StatsDf.groupby(['stim_id','upmod']).get_group((stim_id,True))['unit_id'])

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
# stim_id = 0
# # adding area info to Dfm to make my life easier
# for i,row in StatsDf.groupby('stim_id').get_group(stim_id).iterrows():
#     Dfm.loc[Dfm['unit_id'] == row['unit_id'],'area'] = row['area']

# %% 
for i, row in UnitInfo.iterrows():
    Dfm.loc[Dfm['unit_id'] == row['id'],'area'] = row['area']

# %% NEW STRIP PLOT 
area = 'STR'
stim_id = 0
sort_ids = Dfm.groupby(['opto','stim_id','area']).get_group(('red',stim_id,area)).groupby('unit_id').mean()['dSpikes'].sort_values().index

# get corresponding indices to unit_ids
all_ids = [st.annotations['id'] for st in Segs[0].spiketrains]
sort_ix = [all_ids.index(id) for id in sort_ids]

nUnits = len(sort_ids)

StatsDf['downmod'] = sp.logical_and(StatsDf['m'] < 0, StatsDf['p'] < 0.05)
stats = StatsDf.groupby(['area','stim_id']).get_group((area,stim_id))
stats['colors'] = 'gray'

cmap = plt.get_cmap('PiYG')

stats.loc[stats['upmod']==True,'colors'] = '#d62728'
stats.loc[stats['downmod']==True,'colors'] = '#1f77b4'
stats.index = stats['unit_id']
colors = [stats.loc[id,'colors'] for id in sort_ids]

data = Dfm.groupby(['area','stim_id','opto']).get_group((area,stim_id,'red'))

# gkw = dict(width_ratios=(1,0.025))
fig, axes = plt.subplots(figsize=[6,3])

plot = sns.stripplot(ax=axes, x='unit_id', order=sort_ids, y='dSpikes',data=data,size=1)
axes.axhline(0,lw=1,color='k',alpha=0.4,zorder=100)

for i,collection in enumerate(plot.collections):
    collection.set_facecolor(colors[i])

axes.set_ylim(-45.45)
axes.set_ylabel('∆spikes')
axes.set_xlabel('units')
axes.set_xticks([])
sns.despine(ax=axes,bottom=True)


xlim = axes.get_xlim()



# %% looking into the issue of wrong sorting (as bruno pointed out)

K = StatsDf.groupby(['stim_id','area']).get_group((1,'STR'))
K.index = K.unit_id

T = Dfm.groupby(['stim_id','opto']).get_group((1,'red')).groupby('unit_id').mean()

dSpikes_mean = T.loc[K.index,'dSpikes'] # the mean dSpikes

K['dSpikes_mean'] = dSpikes_mean

for i,id in enumerate(sort_ids):
    y = dSpikes_mean.loc[id]
    axes.plot([i],[y],'.',color=colors[i],markersize=5,alpha=0.5,zorder=100)

axes.set_xlim(xlim)

# fig.savefig('/home/georg/Desktop/ciss/stripplot_new.png',dpi=331)

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
# import colorcet as cc
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
    im = axes.matshow(r_avgs.T[sort_inds,:],cmap='viridis',origin='lower',extent=ext,vmin=-1,vmax=2.5)
    axes.set_aspect('auto')

    return axes, im, r_avgs, sort_inds

# gather indices
stim_id = 0
area = 'STR'
stim_inds = TrialInfo.groupby(['opto','stim_id']).get_group(('red',stim_id)).index
stim_inds_opto = TrialInfo.groupby(['opto','stim_id']).get_group(('both',stim_id)).index

sig_mod_unit_ids = StatsDf.groupby(['area','stim_id','sig']).get_group((area,stim_id,True))['unit_id'].unique()

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
im = axes[0,2].matshow(r_avgs_d.T[order,:],cmap='PiYG',extent=ext,origin='lower',vmin=-1,vmax=1)
axes[0,2].set_aspect('auto')
fig.colorbar(im,cax=axes[1,2],orientation='horizontal',label="difference", shrink=0.8)

vpl_stim, = select(Segs[stim_inds[0]].epochs,'VPL_stims')
da_stim, = select(Segs[stim_inds_opto[0]].epochs,'DA_stims')

for ax in axes[0,:]:
    add_epoch(ax,vpl_stim,color='firebrick', linewidth=0.5)
    add_epoch(ax,vpl_stim,color='firebrick',above=True, linewidth=1)

add_epoch(axes[0,1],da_stim,color='darkcyan',above=True)
add_epoch(axes[0,2],da_stim,color='darkcyan',above=True)


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
    ax.set_xlim(-0.25,2.75)

fig.tight_layout()
# fig.subplots_adjust(wspace=0.01,hspace=0.3)
fname = 'avg_activity_stim_%i_'%stim_id + area+'.png'
plot_dir = bin_path.parent / 'plots'
os.makedirs(plot_dir, exist_ok=True)
fig.savefig(plot_dir / fname ,dpi=331)












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


# # %% STRIP PLOT 
# area = 'STR'
# stim_id = 1
# sort_ids = Dfm.groupby(('area','unit_id')).mean().loc[area,'dSpikes'].sort_values().index

# # get corresponding indices to unit_ids
# all_ids = [st.annotations['id'] for st in Segs[0].spiketrains]
# sort_ix = [all_ids.index(id) for id in sort_ids]

# nUnits = len(sort_ids)

# StatsDf['downmod'] = sp.logical_and(StatsDf['m'] < 0, StatsDf['p'] < 0.05)
# stats = StatsDf.groupby(('area','stim_id')).get_group((area,stim_id))
# stats['colors'] = 'gray'

# stats.loc[stats['upmod']==True,'colors'] = 'deepskyblue'
# stats.loc[stats['downmod']==True,'colors'] = '#d62728'
# stats.index = stats['unit_id']
# colors = [stats.loc[id,'colors'] for id in sort_ids]

# data = Dfm.groupby(('area','stim_id','opto')).get_group((area,stim_id,'red'))


# gkw = dict(width_ratios=(1,0.025))
# fig, axes = plt.subplots(nrows=2,ncols=2,figsize=[9,3],gridspec_kw=gkw)

# plot = sns.stripplot(ax=axes[0,0], x='unit_id', order=sort_ids, y='dSpikes',data=data,size=1)
# # axes.axhline(0,lw=1,linestyle=':',color='k',alpha=0.5)

# for i,collection in enumerate(plot.collections):
#     collection.set_facecolor(colors[i])

# axes[0,0].set_ylim(-45.45)
# axes[0,0].set_ylabel('∆spikes')
# axes[0,0].set_xlabel('')
# axes[0,0].set_xticks([])
# sns.despine(ax=axes[0,0],bottom=True)

# # fig, axes = plt.subplots(nrows=3,ncols=2,figsize=[5,5],gridspec_kw=gkw)
# # fig, axes = plt.subplots(ncols=2, figsize=[11,3], gridspec_kw=gkw)

# # get corresponding indices to unit_ids
# all_ids = [st.annotations['id'] for st in Segs[0].spiketrains]
# sort_ix = [all_ids.index(id) for id in sort_ids]
# stim_inds = TrialInfo.groupby(('stim_id','opto')).get_group((stim_id,'red')).index

# im = axes[1,0].matshow(dSpikes[sort_ix,:][:,stim_inds].T,vmin=-15,vmax=15,cmap='PiYG')
# cbar = fig.colorbar(im, cax=axes[1,1], shrink=0.75,label='∆spikes')

# axes[1,0].set_xlabel('units')
# axes[1,0].set_ylabel('stim #')
# axes[1,0].set_aspect('auto')

# axes[0,1].remove()
# fig.tight_layout()
# fig.savefig('/home/georg/Desktop/ciss/stripandpepper.png',dpi=331)


# %% ISI check 
nTrials = len(Segs)
nUnits = len(Segs[0].spiketrains)

# gather pre spikes
Dists = sp.zeros((nUnits,nTrials,2)) # last axis is pre, post
for i in tqdm(range(5),desc="ISI calculations"): # over nTrials
    seg = Segs[i]
    try:
        vpl_stim, = select(seg.epochs, 'VPL_stims')
        t_post = vpl_stim.times[-1] + vpl_stim.durations[-1]
    except:
        t_post = 0 * pq.s # this calculates the diff through DA only stim

    seg.time_slice(-1*pq.s,0*pq.s).spiketrains


# %% or: more unbiased: euclid dists
nTrials = len(Segs)
nUnits = len(Segs[0].spiketrains)

# gather pre spikes
Dists = sp.zeros((nUnits,nTrials,2)) # last axis is pre, post
for i in tqdm(range(5),desc="euclid dists"): # over nTrials
    seg = Segs[i]
    try:
        vpl_stim, = select(seg.epochs, 'VPL_stims')
        t_post = vpl_stim.times[-1] + vpl_stim.durations[-1] + 0.1*pqs
    except:
        t_post = 0 * pq.s # this calculates the diff through DA only stim

    # euclidean distances
    asigs_pre_pre = sp.stack([asig.time_slice(-1*pq.s, -0.5*pq.s).magnitude for asig in seg.analogsignals],axis=0)[:,:,0]
    asigs_pre = sp.stack([asig.time_slice(-0.5*pq.s, 0*pq.s).magnitude for asig in seg.analogsignals],axis=0)[:,:,0]
    asigs_post = sp.stack([asig.time_slice(t_post, t_post + 0.5*pq.s).magnitude for asig in seg.analogsignals],axis=0)[:,:,0]
   
    Dists[:,i,0] = sp.sqrt(sp.sum((asigs_pre_pre - asigs_pre)**2,axis=1))
    Dists[:,i,1] = sp.sqrt(sp.sum((asigs_post - asigs_pre)**2,axis=1))


# save the result
# out_path = bin_file.with_suffix('.eDists.npy')
# sp.save(out_path,Dists)