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
 
   _                 _ 
  | | ___   __ _  __| |
  | |/ _ \ / _` |/ _` |
  | | (_) | (_| | (_| |
  |_|\___/ \__,_|\__,_|
                       
 
"""
# %%
path = bin_file.with_suffix('.pre_post_spikes.npy')
nSpikes = sp.load(path)

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
