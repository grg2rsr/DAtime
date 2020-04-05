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
StimsDf = pd.read_csv(stim_path, delimiter=',')


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
for i in tqdm(range(nTrials)):
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
    # seg_sliced = seg.time_slice(t_post, t_post + 1*pq.s)
    seg_sliced = seg.time_slice(t_post + 0.1*pq.s, t_post + 2.9*pq.s)
    nSpikes[:,i,1] = [len(st)/2.8 for st in seg_sliced.spiketrains]

# save the result
out_path = bin_file.with_suffix('.pre_post_spikes2.npy')
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


"""
 
                                    _    
   _ __  _ __ ___   _ __   ___  ___| |_  
  | '_ \| '__/ _ \ | '_ \ / _ \/ __| __| 
  | |_) | | |  __/ | |_) | (_) \__ \ |_  
  | .__/|_|  \___| | .__/ \___/|___/\__| 
  |_|              |_|                   
 
"""
# %%
dSpikes = nSpikes[:,:,1] - nSpikes[:,:,0]

# %% plot salt n pepper image
nStimClasses = len(StimsDf.groupby('stim_id'))

fig, axes = plt.subplots(ncols=nStimClasses, figsize=[5.285, 4.775], sharey=True)
for k in range(nStimClasses):
    inds = StimsDf.groupby(['stim_id','opto']).get_group((k,'red')).index
    im = axes[k].matshow(dSpikes[:,inds],vmin=-15,vmax=15,cmap='PiYG')
    
axes[0].set_ylabel('cell id')
axes[1].set_xlabel('stim #')
fig.suptitle('∆spikes in 1s, post - pre')

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75,label='∆spikes')

for ax in axes:
    ax.set_xticklabels([])

# this could be kept as a helper
# sts = [st for st in seg.spiketrains if st.annotations['id'] in stim_inds]

# %% flatten
k = 0
inds = StimsDf.groupby(['stim_id','opto']).get_group((k,'red')).index

Df = pd.DataFrame(dSpikes[:,inds],index=range(nUnits),columns=range(len(inds)))

Dfm = pd.melt(Df.reset_index(),id_vars='index')
Dfm.columns=['unit','trial','dSpikes']

sort_inds = Dfm.groupby('unit').mean().sort_values('dSpikes').index
sns.barplot(data=Dfm,x='unit',y='dSpikes',order=sort_inds,errwidth=0.1,**dict(linewidth=0.1))


# %% pack all stuff
unit_ids = [st.annotations['id'] for st in Segs[0].spiketrains]

StimsDf['prev_blue'] = sp.roll(StimsDf['blue'].values,-1)

Df = pd.DataFrame(dSpikes.T,columns=unit_ids,index=range(nTrials))
Df = pd.concat([StimsDf[['stim_id','prev_blue']],Df],axis=1)

Dfm = pd.melt(Df,id_vars=['stim_id','prev_blue'],var_name='unit_id',value_name='dSpikes')












# %%

"""
 
                                 _   _       _ _                 _     
    __ ___   ____ _    __ _  ___| |_(_)_   _(_) |_ _   _  __   _(_)___ 
   / _` \ \ / / _` |  / _` |/ __| __| \ \ / / | __| | | | \ \ / / / __|
  | (_| |\ V / (_| | | (_| | (__| |_| |\ V /| | |_| |_| |  \ V /| \__ \
   \__,_| \_/ \__, |  \__,_|\___|\__|_| \_/ |_|\__|\__, |   \_/ |_|___/
              |___/                                |___/               
 
"""

# %% annotate depth
# FIXME this requires going back all the way
Sts = Segs[0].spiketrains
d = [st.annotations['depth'] - 4000 for st in Sts]
depth_sep = analysis_params.depth_sep

for st in Sts:
    depth = st.annotations['depth'] - 4000
    if depth < depth_sep:
        st.annotate(area='STR')
    else:
        st.annotate(area='CX')

# reformatting from new data format 
# and shuffeling!

nUnits = len(Segs[0].spiketrains)
nTrials = len(Segs)

Rates = sp.zeros((nUnits,nTrials),dtype='object')
ix = sp.arange(nTrials)
# sp.random.shuffle(ix)

for i,j in enumerate(ix):
    Rates[:,i] = Segs[j].analogsignals

# %% selection
stim_k = 1 # the stim to analyze 
stim_inds = StimsDf.groupby(['stim_id','opto']).get_group((stim_k,'red')).index
Rates_ = Rates[:,stim_inds]

opto_inds = StimsDf.groupby(['stim_id','opto']).get_group((stim_k,'both')).index
Rates_opto_ = Rates[:,opto_inds]

nTrials = stim_inds.shape[0]
nUnits = Rates_.shape[0]

# FIXME hacked stim for now
dur, n, f = StimMap.iloc[stim_k][['dur','n','f']]
times = sp.arange(n) * 1/f * pq.s + analysis_params.VPL_onset * pq.s
epoch = neo.core.Epoch(times=times ,durations=dur*pq.s)

vpl_stim = epoch

# %% average rate plot: split by cx vs str
str_inds = sp.where([s.annotations['area'] == 'STR' for s in Segs[0].spiketrains])[0]
cx_inds = sp.where([s.annotations['area'] == 'CX' for s in Segs[0].spiketrains])[0]

# xval sort
def xval_sort(inds,k):
    """ poor mans xval sorter """
    r_avgs = sp.zeros((Rates_[0,0].shape[0],inds.shape[0],2))
    for i,u in enumerate(inds):
        r = average_asigs(Rates_[u,:int(nTrials/k)])
        r_avgs[:,i,0] = r.magnitude.flatten()

        r = average_asigs(Rates_[u,int(nTrials/k):])
        r_avgs[:,i,1] = r.magnitude.flatten()

    # sorting: xval sorting
    sort_inds = sp.argsort(sp.argmax(r_avgs[:,:,0],0))[::-1]
    return inds[sort_inds]

k = 3
str_inds = xval_sort(str_inds, k)
cx_inds = xval_sort(cx_inds, k)
all_inds = xval_sort(sp.arange(nUnits),k)

# %% no xval
# averaging the rates
nUnits = Rates_.shape[0]
nTrials = Rates_.shape[1]

r_avgs = sp.zeros((Rates_[0,0].shape[0],nUnits))
for u in range(nUnits):
    r = average_asigs(Rates_[u,:])
    r_avgs[:,u] = r.magnitude.flatten()


r_avgs_opto = sp.zeros((Rates_opto_[0,0].shape[0],nUnits))
for u in range(nUnits):
    r = average_asigs(Rates_opto_[u,:])
    r_avgs_opto[:,u] = r.magnitude.flatten()

def peak_sort(inds, r_avgs):
    sort_inds = sp.argsort(sp.argmax(r_avgs[:,inds],0))[::-1]
    return inds[sort_inds]

all_inds = sp.arange(nUnits)
all_inds = peak_sort(all_inds, r_avgs)

str_inds = peak_sort(str_inds, r_avgs)   
cx_inds = peak_sort(cx_inds, r_avgs)   

str_inds_opto = peak_sort(str_inds, r_avgs_opto)
cx_inds_opto = peak_sort(cx_inds, r_avgs_opto)


# %% cmass sorting - just keep for later
cs = sp.cumsum(r_avgs - np.min(r_avgs,axis=0)[sp.newaxis,:],0)
mid_val = (cs[-1,:] - cs[0,:])/2
mid_inds = sp.argmin(sp.absolute(cs - mid_val[sp.newaxis,:]),axis=0)
sort_inds = sp.argsort(mid_inds)[::-1]
all_inds = all_inds[sort_inds]

# %% plotting all together
fig, axes = plt.subplots()
ext = (r.times[0],r.times[-1],0,len(all_inds))
im = axes.matshow(r_avgs.T[all_inds,:],cmap='inferno',origin='bottom',extent=ext,vmin=-1,vmax=3)

axes.set_aspect('auto')
add_stim(axes,vpl_stim,DA=False)


# %% plot CX and STR seperated
fig, axes = plt.subplots(nrows=2,sharex=True,gridspec_kw=dict(height_ratios=(1,len(str_inds)/len(cx_inds))))
# norm = mpl.colors.DivergingNorm(0,vmin=-1,vmax=2)
# im = plt.matshow(r_avgs.T[inds,:],cmap='inferno',origin='bottom',extent=ext,norm=norm)

ext = (r.times[0],r.times[-1],0,len(cx_inds))
im = axes[0].matshow(r_avgs.T[cx_inds,:],cmap='inferno',origin='bottom',extent=ext,vmin=-1,vmax=3)

ext = (r.times[0],r.times[-1],0,len(str_inds))
im = axes[1].matshow(r_avgs.T[str_inds,:],cmap='inferno',origin='bottom',extent=ext,vmin=-1,vmax=3)

for ax in axes:
    ax.set_aspect('auto')
    add_stim(ax,vpl_stim,DA=False)

# plt.colorbar(im)


# %% plot CX and STR seperated --- opto
fig, axes = plt.subplots(nrows=2,sharex=True,gridspec_kw=dict(height_ratios=(1,len(str_inds)/len(cx_inds))))
# norm = mpl.colors.DivergingNorm(0,vmin=-1,vmax=2)
# im = plt.matshow(r_avgs.T[inds,:],cmap='inferno',origin='bottom',extent=ext,norm=norm)

ext = (r.times[0],r.times[-1],0,len(cx_inds))
im = axes[0].matshow(r_avgs_opto.T[cx_inds,:],cmap='inferno',origin='bottom',extent=ext,vmin=-1,vmax=3)

ext = (r.times[0],r.times[-1],0,len(str_inds))
im = axes[1].matshow(r_avgs_opto.T[str_inds,:],cmap='inferno',origin='bottom',extent=ext,vmin=-1,vmax=3)

for ax in axes:
    ax.set_aspect('auto')
    add_stim(ax,vpl_stim,DA=False)

# plt.colorbar(im)


























# %%

# %% all cells
# vpl data
nUnits = Rates_.shape[0]
nTrials = Rates_.shape[1]

r_avgs = sp.zeros((Rates_[0,0].shape[0],nUnits,2))
for u in range(nUnits):
    r = average_asigs(Rates_[u,:int(nTrials/3)])
    r_avgs[:,u,0] = r.magnitude.flatten()
    r = average_asigs(Rates_[u,int(nTrials/3):])
    r_avgs[:,u,1] = r.magnitude.flatten()

# sorting: xval sorting
inds = sp.argsort(sp.argmax(r_avgs[:,:,0],0))[::-1]

# r_avgs = sp.average(r_avgs,axis=2)
r_avgs = r_avgs[:,:,1]

# %%
ext = (r.times[0],r.times[-1],0,nUnits)
# norm = mpl.colors.DivergingNorm(0,vmin=-1,vmax=2)
# im = plt.matshow(r_avgs.T[inds,:],cmap='inferno',origin='bottom',extent=ext,norm=norm)
im = plt.matshow(r_avgs.T[inds,:],cmap='inferno',origin='bottom',extent=ext,vmin=-1,vmax=3)
plt.gca().set_aspect('auto')
add_stim(plt.gca(),vpl_stim,DA=False)
plt.colorbar(im)

# %%
# opto data
nUnits = Rates_opto_.shape[0]
r_avgs_opto = sp.zeros((Rates_opto_[0,0].shape[0],nUnits))
for u in range(nUnits):
    r = average_asigs(Rates_opto_[u,:])
    r_avgs_opto[:,u] = r.magnitude.flatten()

# inds = sp.argsort(sp.argmax(r_avgs,0))
ext = (r.times[0],r.times[-1],0,nUnits)
plt.matshow(r_avgs_opto.T[inds,:],origin='bottom',extent=ext,vmin=-1,vmax=2)
plt.gca().set_aspect('auto')
add_stim(plt.gca(),vpl_stim)

D = r_avgs_opto - r_avgs
v = 2
plt.matshow(D.T[inds,:],origin='bottom',extent=ext,vmin=-v,vmax=v,cmap='PiYG')
plt.gca().set_aspect('auto')
add_stim(plt.gca(),vpl_stim)

# %% 
fig, axes = plt.subplots()
colors = sns.color_palette('viridis',n_colors=nUnits)
ysep = 0.2
tvec = Rates_[0,0].times.magnitude.flatten()
for u in range(nUnits):
    axes.plot(r_avgs[:,inds[u]]+u*ysep,color=colors[u],zorder=-1*u)
    # axes.fill_between(tvec,r_avgs[:,inds[u]]+u*ysep,u*ysep,color=colors[u],zorder=-1*u)
